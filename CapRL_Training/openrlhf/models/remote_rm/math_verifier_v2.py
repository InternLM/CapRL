import json
import os
import random
import re
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import Levenshtein
from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

app = Flask(__name__)

problem_to_answer = {}


def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return ""


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()

def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1



def find_similar_problem(problem):
    max_sim = -1
    target_problem = None
    for p in problem_to_answer.keys():
        sim = Levenshtein.ratio(problem, p)
        if sim > max_sim:
            max_sim = sim
            target_problem = p
    return target_problem

def math_parse(content):
    return parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

def verify_math(input_queue, output_queue):
    while True:
        content, sol = input_queue.get()
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            pattern = re.compile(r"</think>\s*<answer>.*?\\boxed{(.*)}.*?</answer>", re.DOTALL)
            matches = re.findall(pattern, content)
            if len(matches) > 0:
                answer_parsed = matches[-1]
            else:
                answer_parsed = ''
            math_answer = math_parse(f'${answer_parsed}$')
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            try:
                reward = max(float(verify(answer_parsed, gold_parsed)), float(verify(math_answer, gold_parsed)))
            except Exception as e:
                reward = 1.0
                print("Failed to verify: ", e)
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", sol)

        output_queue.put(reward)


@app.route("/get_reward", methods=["POST"])
def get_reward():
    # 获取请求中的 JSON 数据
    data = request.get_json()
    # 检查是否有 'query' 字段
    if "query" not in data:
        return jsonify({"error": "queries field is required"}), 400
    rewards = []
    f_rewards = []
    a_rewards = []
    for q,problem in zip(data["query"],data["prompts"]):
        if problem is None:
            return jsonify({"error": f"problem not found from {q}"}), 400
        if problem not in problem_to_answer:
            # This should not happen
            print(f"problem not exists: {problem}")
            problem = find_similar_problem(problem)
        answer = problem_to_answer[problem]
        response = get_response_from_query(q) or q
        #print (q, response)
        if response is None:
            return jsonify({"error": f"response not found from {q}"}), 400
        format_reward = float(verify_format(response))
        input_queue.put((response, answer))
        acc_reward = float(output_queue.get())
        do_print = random.randint(1, 20) == 1
        if do_print:
            info=f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\n"
            info = re.sub(r"<\|.*?\|>","",info)
            print(info)
            
        rewards.append(0 * format_reward + acc_reward)
        f_rewards.append(format_reward)
        a_rewards.append(acc_reward)
    # 返回包含 rewards 的响应
    return jsonify({"rewards": rewards, "f_rewards": f_rewards, "a_rewards": a_rewards})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Datasets to use (comma separated)", required=True
    )
    parser.add_argument(
        "--prompt-template", type=str, default=None, help="Prompt template", required=True
    )
    parser.add_argument(
        "--input_key", type=str, default="prompt", help="The key name of prompt."
    )
    args = parser.parse_args()
    
    # Split dataset paths and load all datasets
    dataset = []
    for dataset_path in args.dataset.split(','):
        dataset_path = dataset_path.strip()
        if dataset_path.endswith("json"):
            with open(dataset_path, "r") as f:
                dataset.extend(json.load(f))
        elif dataset_path.endswith("jsonl"):
            with open(dataset_path, "r") as f:
                dataset.extend([json.loads(l) for l in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for dataset: {dataset_path}")

    format_pattern = r"^<think>(?:(?!</think>).)*</think>\s*<answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template=="chatml":
        problem_pattern = r"<\|im_start\|>user\n(.*?)<\|im_end\|>"
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template=="qwen1":
        problem_pattern = r"｜User｜>(.*?)<｜Assistant｜>"
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template=="base":
        problem_pattern = r"User: (.*?)\n\nAssistant:"
        response_prefix = r"Assistant: "
    else:
        raise ValueError(f"Unknown chat format: {args.dataset}")
    print("load dataset success")
    for item in dataset:
        problem = item[args.input_key]
        answer = item["answer"].strip()
        # we require the answer to be in latex format
        if answer[0] != "$":
            answer = "$" + answer + "$"
        problem_to_answer[problem] = answer

    # math_verify can only run in main thread
    input_queue = Queue()
    output_queue = Queue()
    p = Process(target=verify_math, args=(input_queue, output_queue))
    p.start()

    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    p.kill()
