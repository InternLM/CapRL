#!/usr/bin/env python3

from __future__ import annotations
import random
import argparse
import base64
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from tqdm import tqdm
from openai import OpenAI


BATCH_SIZE = 100         
MAX_WORKERS = 32         
ROTATE_NUM = 4
PROMPT='{}. Answer the question with only the correct letter'

def shuffle_options(question, answer):
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
            print (lines)
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
        
    return ['\n'.join(new_question_lines), new_answer] 


def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image;base64,{encoded}"


def vlm_client_qa(client: OpenAI, prompt: str, image_path: str, max_tokens: int = 4) -> str:
    if image_path is not None:
        message_payload = [
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_base64(image_path)},
            },
            {"type": "text", "text": prompt},
        ]
    else:
        message_payload = [
            {"type": "text", "text": prompt},
        ]        

    resp = client.chat.completions.create(
        model="qwen-vl",
        messages=[
            {"role": "user", "content": message_payload},
        ],
        temperature=.1,
        max_tokens=max_tokens,
        extra_body={
            "repetition_penalty": 1.05,
            },
    )
    return resp.choices[0].message.content


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(samples: List[Dict[str, Any]], outfile: Path) -> None:
    if not samples:
        return
    with outfile.open("a", encoding="utf-8") as f:
        for item in samples:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def load_progress(progress_file: Path) -> int:
    if not progress_file.exists():
        return 0
    with progress_file.open("r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
            return int(obj.get("index", 0))
        except json.JSONDecodeError:
            return 0


def save_progress(progress_file: Path, index: int) -> None:
    with progress_file.open("w", encoding="utf-8") as f:
        json.dump({"index": index}, f, ensure_ascii=False)


def process_sample(client: OpenAI, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        answers = []
        qas = sample['qa_list']
        image_path = sample['image_path']
        
        v_outputs=[]
        l_outputs=[]
        for qa in qas:
            question = qa['question']
            answer = qa['answer']
            question = question.replace('\n   - E) Can not answer based on the caption', '')
            for _ in range(ROTATE_NUM):
                q, a = shuffle_options(question, answer)

                answers.append(a)
                prompt_v = q + '\n   - E) None of the above'
                prompt_v = PROMPT.format(prompt_v)

                prompt_l = PROMPT.format(q)

                v_output = vlm_client_qa(client, prompt_v, image_path)
                l_output = vlm_client_qa(client, prompt_l, image_path=None)
                v_outputs.append(v_output)
                l_outputs.append(l_output)


        vis_results = [[ans in out] for ans, out in zip(answers, v_outputs)]
        nlp_results = [[ans in out] for ans, out in zip(answers, l_outputs)]
        return [sample, vis_results, nlp_results]
    
    except Exception as exc:
        logging.warning("%s -> %s", image_path, exc)
        return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch qa generator using vLLM VLM")
    parser.add_argument("--data-path", type=Path, default="/path/2_extract_qa/2_output_sample.json", help="file for process")
    parser.add_argument("--part-id", type=int, default=0, help="current process part id (0-based)")
    parser.add_argument("--all-parts", type=int, default=1, help="current process total parts")
    parser.add_argument("--vlm-port", type=int, default=21000, help="vLLM port")
    parser.add_argument("--json-folder", type=Path, default="/path/3_answer_qa/3_qa_file_output_folder", help="jsonl output folder")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # 1. load dataset and split
    data = load_dataset(args.data_path)
    part_size = math.ceil(len(data) / args.all_parts)
    start, end = args.part_id * part_size, min((args.part_id + 1) * part_size, len(data))
    current_part = data[start:end]
    logging.info("Loaded %d samples; processing slice %d‑%d", len(data), start, end - 1)

    # 2. prepare output files and resume if needed
    args.json_folder.mkdir(parents=True, exist_ok=True)
    output_file = args.json_folder / f"part_{args.part_id}.jsonl"
    progress_file = args.json_folder / f"progress_part_{args.part_id}.jsonl"
    resume_index = load_progress(progress_file)
    current_part = current_part[resume_index:]
    logging.info("Resume from index %d (%d samples remain)", resume_index, len(current_part))

    # 3. initialize openai client
    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{args.vlm_port}/v1")

    # 4. multithreading processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for idx, sample in enumerate(tqdm(current_part, desc="Captioning")):
            futures.append(executor.submit(process_sample, client, sample))

            if (idx + 1) % BATCH_SIZE == 0 or idx + 1 == len(current_part):
                completed = [f.result() for f in futures if f.result() is not None]
                save_jsonl(completed, output_file)
                save_progress(progress_file, resume_index + idx + 1)
                futures.clear()
                logging.info("Saved %d samples (progress %d/%d)", len(completed), idx + 1, len(current_part))

    logging.info("All done! Output → %s", output_file)


if __name__ == "__main__":
    main()
