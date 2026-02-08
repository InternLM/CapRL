import argparse
import itertools
import json
import re
import time
from flask import Flask, jsonify, request

import torch
import uvicorn
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch.multiprocessing as mp
from openrlhf.utils.logging_utils import init_logger

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
# from qwen_vl_utils import process_vision_info  # This import was unused, kept as is.
from PIL import Image
import os
import random
from tqdm import tqdm
import numpy as np
import httpx  # 新增: 用于异步HTTP请求
import asyncio  # 新增: 用于并发管理

logger = init_logger(__name__)


def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    return text


def get_response_from_query(q: str):
    response_prefix = r"<\|im_start\|>assistant\n"
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end():]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def shuffle_options(question, answer):
    question = question.replace('\n   - E) Can not answer based on the caption', '')
    question = question.replace('\n   - F) Can not answer based on the caption', '')
    lines = question.split('\n')
    q_text = lines[0]
    options = lines[1:]

    pattern = r'-\s*([A-F])\)\s*(.+)'
    original_options = {}
    options = [o for o in options if len(o)]
    for opt in options:
        match = re.search(pattern, opt.strip())
        if match:
            label = match.group(1)
            content = match.group(2)
            original_options[label] = content
        else:
            print(lines)
            raise ValueError(f"ERROR: {opt}")

    correct_answer_label = answer
    if correct_answer_label not in original_options:
        raise ValueError(f"{correct_answer_label} not in the candidates")
    correct_answer_text = original_options[correct_answer_label]

    shuffled_items = list(original_options.items())
    random.shuffle(shuffled_items)

    new_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    new_options = {}
    new_answer = ''
    for i, (_, content) in enumerate(shuffled_items):
        label = new_labels[i]
        new_options[label] = content
        if content == correct_answer_text:
            new_answer = label

    new_question_lines = [q_text]
    for label in new_options:
        new_question_lines.append(f"   - {label}) {new_options[label]}")

    return ['\n'.join(new_question_lines) + '\n   - F) Can not answer based on the caption', new_answer]


sampling_params = SamplingParams(
    n=1,
    temperature=0.6,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=10,
    stop_token_ids=[],
)


def parse_easy(answer, gt):
    pattern1 = re.compile(r'[A-I]')
    res = pattern1.findall(answer)
    if len(res) > 0:
        res = res[0]
        return 1 if res == gt else 0
    else:
        return 0


prompt_v1 = '''<|im_start|>user
{}
Please answer the question based on the caption. 
The final answer should be a single option letter.{}
You should answer the question with the following format: The answer is X. Here X is the correct option latter<|im_end|>
<|im_start|>assistant
The answer is'''


class RewardModelProxy:
    def __init__(self, args):
        self.custom_reward_func = None
        reward_func = args.reward_func
        self.qa_num = args.qa_num
        self.shuffle_qa = args.shuffle_qa
        self.all_qa = args.all_qa

        assert not (self.shuffle_qa and self.all_qa)

        # vLLM将根据 'tensor_parallel_size' 和 CUDA_VISIBLE_DEVICES 自动管理GPU
        self.llm = LLM(
            model=args.reward_pretrain,
            tensor_parallel_size=args.tp,  # 使用命令行传入的tp参数
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(args.reward_pretrain, trust_remote_code=True)

    def get_reward(self, prompts, queries, labels):
        prompt = '''<|im_start|>user
You will be given an image caption describing the visual content.  
Your task is to answer the multiple-choice question **strictly based on the caption**, even if the answer may seem obvious from prior knowledge or question wording.

Ignore any external knowledge. Do not make assumptions beyond what the caption explicitly or implicitly states.

Example 1:
Caption: <Caption Start> A woman in a red coat is walking a black dog across a snowy park. <Caption End>  
Question: What color is the dog?
- A) Brown  
- B) White  
- C) Black  
- D) Gray
- E) Can not answer based on the caption

The answer is C.

Example 2:
Caption: <Caption Start> A child is waving a British flag during a parade. <Caption End>  
Question: What color is the flag?
- A) Red  
- B) Blue  
- C) Red, white, and blue  
- D) White
- E) Can not answer based on the caption

The answer is E.

Now, answer the question based on the following caption:

Caption: <Caption Start> {} <Caption End>  
Question: {}  <|im_end|>
<|im_start|>assistant
The answer is'''

        inputs = []
        answers = []

        if self.all_qa:
            qa_num = []
            for cap, qa in prompts:
                qa_num.append(len(qa))
                for q, a in qa:
                    temp = prompt.format(cap, q)
                    inputs.append({"prompt": temp})
                    answers.append(a)
            # llm.generate现在接收字符串列表
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
            generated_text = [out.outputs[0].text for out in outputs]
            temp_rewards = []
            for answer, gt in zip(generated_text, answers):
                #print (answer, gt, parse_easy(answer, gt))
                temp_rewards.append(parse_easy(answer, gt))
            rewards = []
            for qa_n in qa_num:
                rewards.append(np.mean(temp_rewards[:qa_n]) * 2)
                temp_rewards = temp_rewards[qa_n:]

        else:
            qa_num = self.qa_num
            for cap, qa in prompts:
                # print("caption:\n",cap)
                for i in range(qa_num):
                    q, a = random.choice(qa)
                    if self.shuffle_qa:
                        q, a = shuffle_options(q, a)
                    temp = prompt.format(cap, q)
                    inputs.append({"prompt": temp})
                    answers.append(a)
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
            generated_text = [out.outputs[0].text for out in outputs]
            rewards = []
            for answer, gt in zip(generated_text, answers):
                # print(answer, gt, parse_easy(answer, gt))
                rewards.append(parse_easy(answer, gt))
            rewards = (torch.Tensor(rewards).view(-1, qa_num).mean(dim=-1).view(-1) * 2).tolist()

        # --- 完善 print ---
        if inputs:
            print("=================================")
            print("input:", inputs[0])  # <-- 修改：访问 dict 中的 prompt
            print("output:", generated_text[0])
            print("rewards", rewards)  # <-- 修改：澄清 all reward 的含义
            print("=================================")
            # --- 结束 print ---
        return rewards, generated_text


# --- 工作进程函数 (Worker Process) ---
# 每个工作进程运行这个函数，加载模型并启动一个FastAPI服务
def run_worker_server(args, rank, gpu_ids):
    """
    Args:
        rank (int): 当前工作进程在其节点上的本地排名，用于确定端口号。
        gpu_ids (list[int]): 分配给此工作进程的GPU ID列表。
    """
    # 为当前进程设置可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # 强制PyTorch重新初始化CUDA上下文
    torch.cuda.init()

    # 确保每个工作进程有唯一的端口号
    worker_port = args.worker_base_port + rank

    print(f"Starting worker (local rank {rank}) on GPUs {gpu_ids} at port {worker_port}...")

    # 在这里初始化模型，确保它在正确的GPU上下文中
    reward_model = RewardModelProxy(args)
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/get_reward")
    async def get_reward_endpoint(request: Request):
        data = await request.json()
        t = time.time()
        prompts = data.get('prompts')
        queries = data.get("query")
        labels = data.get("labels")
        rewards, generated_text = reward_model.get_reward(prompts, queries, labels)
        # 返回的rank是本地rank，主要用于日志调试
        result = {"rewards": rewards, "local_rank": rank}

        if sum(rewards) == 0.:
            print(f"--- Rank {rank} REWARDS ARE 0 ---", flush=True)
            print("Associated Generated Texts:", flush=True)
            print(generated_text, flush=True)
            print("-------------------------", flush=True)

        reward_mean = sum(rewards) / len(rewards)
        print(f"Worker {rank} (GPUs: {gpu_ids}) processed {len(prompts)} samples in {time.time() - t:.3f}s, mean reward: {reward_mean:.3f}", flush=True)
        return JSONResponse(result)

    uvicorn.run(app, host="0.0.0.0", port=worker_port, log_level="info")


# --- 主进程函数 (Master Process) ---
def run_master_server(args):
    app = FastAPI()

    # 计算每个worker host上有多少个worker instance
    num_hosts = len(args.worker_hosts)
    if args.num_workers % num_hosts != 0:
        raise ValueError(
            f"Total number of workers ({args.num_workers}) must be divisible by the number of worker hosts ({num_hosts}).")
    num_workers_per_host = args.num_workers // num_hosts

    # 构建所有worker的完整URL列表
    worker_urls = []
    for host in args.worker_hosts:
        for i in range(num_workers_per_host):
            port = args.worker_base_port + i
            worker_urls.append(f"http://{host}:{port}/get_reward")

    async def send_to_worker(session, url, chunk_data, chunk_index):
        """异步发送数据到单个工作进程"""
        try:
            response = await session.post(url, json=chunk_data, timeout=300)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request to {url} (chunk {chunk_index}) failed: {e}")
            return None

    @app.post("/get_reward")
    async def distributed_get_reward(request: Request):
        """接收请求，分发任务，合并结果"""
        data = await request.json()
        prompts = data.get('prompts', [])
        queries = data.get("query", [])
        labels = data.get("labels", [])

        num_prompts = len(prompts)
        if num_prompts == 0:
            return JSONResponse({"rewards": []})

        t_start = time.time()

        # 将数据均分给每个工作进程
        total_workers = len(worker_urls)
        chunk_size = (num_prompts + total_workers - 1) // total_workers
        chunks = []
        for i in range(total_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_prompts)
            if start_idx >= num_prompts:
                break

            chunk_data = {
                "prompts": prompts[start_idx:end_idx],
                "query": queries,
                "labels": labels
            }
            chunks.append(chunk_data)

        async with httpx.AsyncClient() as session:
            tasks = [send_to_worker(session, worker_urls[i], chunks[i], i) for i in range(len(chunks))]
            worker_responses = await asyncio.gather(*tasks)

        all_rewards = []

        for i, response in enumerate(worker_responses):
            if response and 'rewards' in response:
                all_rewards.extend(response['rewards'])
            else:
                # -------------------------------------------------- #
                # 方案A：直接抛出异常，让整个batch失败
                # -------------------------------------------------- #
                logger.error(f"Worker for chunk {i} (URL: {worker_urls[i]}) failed. Aborting batch.")
                # 返回一个HTTP 500错误
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Worker for chunk {i} failed.", "failed_url": worker_urls[i]}
                )

        total_time = time.time() - t_start
        print(f"Master: Distributed {num_prompts} samples to {len(chunks)} workers. Total time: {total_time:.3f}s",
              flush=True)

        return JSONResponse({"rewards": all_rewards})

    print(f"Starting master server on http://{args.master_host}:{args.port}")
    print(f"Distributing tasks to {args.num_workers} workers across {num_hosts} hosts.")
    print(f"Worker URLs: {worker_urls}")
    uvicorn.run(app, host=args.master_host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--role", type=str, required=True, choices=["master", "worker"],
                        help="The role of this instance.")

    # Reward Model (for workers)
    parser.add_argument("--reward_pretrain", type=str, help="Path to the pretrained reward model.")
    parser.add_argument("--max_len", type=int, default="2048")
    parser.add_argument("--shuffle_qa", action="store_true", default=False)
    parser.add_argument("--all_qa", action="store_true", default=False)
    parser.add_argument("--qa_num", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size for vLLM engine on each worker.")

    # Master Server Configuration
    parser.add_argument("--master_host", type=str, default="0.0.0.0", help="IP for the MASTER server to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the MASTER server.")
    parser.add_argument("--worker_hosts", type=str, nargs='+',
                        help="A list of all worker IPs or hostnames for the master to connect to.")

    # Worker Server Configuration
    parser.add_argument("--worker_base_port", type=int, default=8001, help="Base port for WORKER servers on each host.")
    parser.add_argument("--num_workers", type=int, help="TOTAL number of worker instances across all hosts.")

    parser.add_argument("--reward_func", type=str, default=None)

    args = parser.parse_args()

    if args.role == "master":
        if not args.worker_hosts or not args.num_workers:
            parser.error("--role 'master' requires --worker_hosts and --num_workers.")
        run_master_server(args)

    elif args.role == "worker":
        mp.set_start_method('spawn', force=True)
        if not args.reward_pretrain:
            parser.error("--role 'worker' requires --reward_pretrain.")

        # 检查并启动工作进程
        num_gpus_available = torch.cuda.device_count()
        if num_gpus_available < args.tp:
            raise ValueError(
                f"Not enough GPUs available for tensor parallelism. Required: {args.tp}, Found: {num_gpus_available}")

        if num_gpus_available % args.tp != 0:
            raise ValueError(
                f"Number of available GPUs ({num_gpus_available}) is not divisible by the tensor parallel size ({args.tp}).")

        # 计算在这个节点上可以启动多少个worker instance
        num_local_workers = num_gpus_available // args.tp
        print(f'Found {num_gpus_available} GPUs. Starting {num_local_workers} worker processes with TP={args.tp}...')

        # 创建GPU ID分组
        gpu_ids = list(range(num_gpus_available))
        gpu_groups = [gpu_ids[i:i + args.tp] for i in range(0, len(gpu_ids), args.tp)]

        processes = []
        for i in range(num_local_workers):
            # 每个进程获得一个唯一的本地rank和一组GPU
            local_rank = i
            gpu_group = gpu_groups[i]
            p = mp.Process(target=run_worker_server, args=(args, local_rank, gpu_group))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Shutting down worker processes...")
            for p in processes:
                p.terminate()
                p.join()
            print("All workers terminated.")