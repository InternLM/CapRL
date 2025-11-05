#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

BATCH_SIZE = 100
MAX_WORKERS = 32
SYS_PROMPT='''Your task is to generate five multiple-choice questions and their answers about the object based on the provided image.
The questions should be challenging and focus on the image content. Your answer should strictly follow the following format:
#### 1. **Which method achieves the highest accuracy (Acc) on the FF++ (HQ) dataset?**
   - A) Method "a"
   - B) Method "b"
   - C) Method "c"
   - D) Ours

**Answer:** D) Ours   
------
#### 2. **What is the primary color of the kayak in the image?**
   - A) Blue
   - B) Red
   - C) Black
   - D) White

**Answer:** B) Red 
------
You should strictly follow the above format and should not generate irrelevant sentences. All the questions should be answered based on the image.
'''
PROMPT='Here is the image'

def _encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image;base64,{encoded}"


def vlm_client_qa(client: OpenAI, prompt: str, image_path: str, max_tokens: int = 2048) -> str:
    message_payload = [
        {
            "type": "image_url",
            "image_url": {"url": _encode_image_base64(image_path)},
        },
        {"type": "text", "text": prompt},
    ]
    

    resp = client.chat.completions.create(
        model="qwen-vl",
        messages=[
            {"role": "system", "content": SYS_PROMPT},
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
        new_sample={}
        image_path = sample
        response = vlm_client_qa(client, PROMPT, image_path)
        new_sample["qa_response"] = response
        new_sample["image_path"] = sample
        return new_sample
    except Exception as exc:
        logging.warning("%s -> %s", image_path, exc)
        return None

# ────────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch qa generator using vLLM VLM")
    parser.add_argument("--data-path", type=Path, default="/path/1_input_sample.json", help="file including image paths")
    parser.add_argument("--part-id", type=int, default=0, help="current process part id (0-based)")
    parser.add_argument("--all-parts", type=int, default=1, help="current process total parts")
    parser.add_argument("--vlm-port", type=int, default=21000, help="vLLM port")
    parser.add_argument("--json-folder", type=Path, default="/path/1_generate_qa/1_qa_file_output_folder", help="jsonl output folder")
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
        for idx, sample in enumerate(tqdm(current_part, desc="generating")):
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
