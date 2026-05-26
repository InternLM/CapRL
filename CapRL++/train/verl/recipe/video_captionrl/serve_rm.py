"""
Reward server for video_captionrl. It exposes /get_reward and is used by
reward_fn.py through REWARD_REMOTE_URL.

Score modes:
  qa       : caption-based multiple-choice QA accuracy
  vl_judge : direct video-caption scoring with a multimodal judge

Tasks:
  image: image caption QA without timestamp format reward
  video: video caption QA with optional timestamp format reward
"""
import argparse
import json
import logging
import os
import random
import re
import time

import asyncio
import numpy as np
import torch
import torch.multiprocessing as mp
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import httpx
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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
    ends_of_sentence = ["<|im_end|>", "", "<|endoftext|>"]
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
        logger.warning(
            "shuffle_options: label %r not in parsed options %s; skipping shuffle (question/label mismatch or format).",
            correct_answer_label,
            sorted(original_options.keys()),
        )
        return [question + "\n   - F) Can not answer based on the caption", answer]
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

PROMPT_IMAGE = """<|im_start|>user
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
The answer is"""

PROMPT_VIDEO = """<|im_start|>user
        You will be given a video caption describing visual content and events over time.  
        Your task is to answer the multiple-choice question **strictly based on the caption**, even if the answer may seem obvious from prior knowledge or question wording.

        Ignore any external knowledge. Do not make assumptions beyond what the caption explicitly or implicitly states. Pay close attention to timestamps (e.g., [mm:ss]) and the sequence of events.

        Example 1:
        Caption: <Caption Start> [00:00 - 00:05] A man in a blue shirt is running down a wet street. [00:06] He suddenly stops and [00:08] drops a brown leather wallet on the ground. <Caption End>  
        Question: What happens immediately after the man stops running at [00:06]?
        - A) He takes off his blue shirt  
        - B) He drops a brown leather wallet  
        - C) He picks up an umbrella  
        - D) He looks at his watch
        - E) Can not answer based on the caption

        The answer is B.

        Example 2:
        Caption: <Caption Start> [00:15] A silver sedan runs a red light and [00:18] crashes into a fire hydrant. [00:20] Water sprays high into the air. <Caption End>  
        Question: Why did the silver sedan run the red light?
        - A) The brakes failed  
        - B) The driver was texting  
        - C) The driver was speeding to a hospital  
        - D) It was raining heavily
        - E) Can not answer based on the caption

        The answer is E.

        Now, answer the question based on the following caption:

        Caption: <Caption Start> {} <Caption End>  
        Question: {}  <|im_end|>
        <|im_start|>assistant
        The answer is"""


def parse_easy(answer, gt):
    pattern1 = re.compile(r'[A-I]')
    res = pattern1.findall(answer)
    if len(res) > 0:
        res = res[0]
        return 1 if res == gt else 0
    return 0


# Timestamp format reward for [mm:ss] and [mm:ss - mm:ss] brackets.
_TIMESTAMP_BRACKET = re.compile(
    r"\[(\d{1,3}):(\d{2})(?:\s*-\s*(\d{1,3}):(\d{2}))?\]",
    re.IGNORECASE,
)


def _parse_ts_groups(m: re.Match, max_minute: int = 599) -> tuple:
    """Return (start_sec, end_sec_or_none, ok)."""
    mm, ss = int(m.group(1)), int(m.group(2))
    if not (0 <= ss <= 59) or mm < 0 or mm > max_minute:
        return 0, None, False
    t0 = mm * 60 + ss
    if m.group(3) is None:
        return t0, None, True
    emm, ess = int(m.group(3)), int(m.group(4))
    if not (0 <= ess <= 59) or emm < 0 or emm > max_minute:
        return t0, None, False
    t1 = emm * 60 + ess
    if t1 < t0:
        return t0, t1, False
    return t0, t1, True


def compute_format_reward(
    text: str,
    max_minute: int = 599,
) -> float:
    """
    Score timestamp bracket formatting for video captions in [0, 1].

    Let S be the timestamp-like brackets matched by the regex, and N_all=|S|.
    N_valid counts brackets satisfying logical constraints: valid seconds,
    minute upper bound, and interval end time no earlier than start time.

        0.5 * N_valid / max(N_all, 1) + 0.5 * I_chrono

    I_chrono is 1 when at least one valid timestamp exists and valid start
    times are monotonically non-decreasing in caption order; otherwise 0.
    """
    if not text or not str(text).strip():
        return 0.0
    s = str(text)
    matches = list(_TIMESTAMP_BRACKET.finditer(s))
    if not matches:
        return 0.0

    starts_in_order = []
    valid = 0
    for m in matches:
        t0, t1, ok = _parse_ts_groups(m, max_minute=max_minute)
        if not ok:
            continue
        valid += 1
        starts_in_order.append(t0)

    validity = valid / max(len(matches), 1)
    chrono = 0.0
    if valid > 0 and all(starts_in_order[i] <= starts_in_order[i + 1] for i in range(len(starts_in_order) - 1)):
        chrono = 1.0

    return float(0.5 * validity + 0.5 * chrono)


# ===================== VL Judge prompt =====================
VL_JUDGE_PROMPT = """You are an expert video-caption evaluator. You will be given a video and a text caption that was generated to describe the video. Your job is to judge how well the caption describes the video.

Evaluate the caption on the following criteria:
1. **Factuality** (0-1): Are the described objects, actions, and scene elements actually present in the video?
2. **Temporal Alignment** (0-1): Are the timestamps and chronological order of events correct?
3. **Coverage** (0-1): Does the caption cover the key events and important details in the video?
4. **Hallucination Penalty** (0-1): 1 means no hallucination; 0 means severe fabrication of events not in the video.

Output ONLY a JSON object with these four scores and a brief reason. Do NOT output anything else.

Example output:
{{"factuality": 0.8, "temporal_alignment": 0.7, "coverage": 0.6, "hallucination_penalty": 0.9, "reason": "Caption describes main events correctly but misses a key action at 00:15."}}

Caption to evaluate:
{}"""

VL_JUDGE_SAMPLING_PARAMS = SamplingParams(
    n=1,
    temperature=0.2,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=256,
    stop_token_ids=[],
)


def _parse_judge_score(text: str) -> float:
    """Parse the JSON output from judge model and return a fused score in [0, 1]."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            try:
                obj = json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                logger.warning("Failed to parse judge JSON: %s", text[:200])
                return 0.0
        else:
            logger.warning("No JSON found in judge output: %s", text[:200])
            return 0.0

    factuality = float(obj.get("factuality", 0.0))
    temporal = float(obj.get("temporal_alignment", 0.0))
    coverage = float(obj.get("coverage", 0.0))
    halluc = float(obj.get("hallucination_penalty", 0.0))

    factuality = max(0.0, min(1.0, factuality))
    temporal = max(0.0, min(1.0, temporal))
    coverage = max(0.0, min(1.0, coverage))
    halluc = max(0.0, min(1.0, halluc))

    score = 0.30 * factuality + 0.25 * temporal + 0.25 * coverage + 0.20 * halluc
    return float(max(0.0, min(1.0, score)))


class RewardModelProxy:
    def __init__(self, args):
        self.args = args
        self.custom_reward_func = None
        self.task = getattr(args, "task", "video")
        self.score_mode = getattr(args, "score_mode", "qa")
        self.qa_num = args.qa_num
        self.shuffle_qa = args.shuffle_qa
        self.all_qa = args.all_qa
        self.format_reward_weight = float(getattr(args, "format_reward_weight", 0.0))
        self.format_max_minute = int(getattr(args, "format_max_minute", 599))
        if self.score_mode == "qa":
            assert not (self.shuffle_qa and self.all_qa)

        if self.score_mode == "vl_judge":
            self.llm = LLM(
                model=args.reward_pretrain,
                tensor_parallel_size=args.tp,
                trust_remote_code=True,
                max_model_len=getattr(args, "judge_max_model_len", 18000),
                limit_mm_per_prompt={"video": 1},
            )
        else:
            self.llm = LLM(
                model=args.reward_pretrain,
                tensor_parallel_size=args.tp,
                trust_remote_code=True,
            )
        self.processor = AutoProcessor.from_pretrained(args.reward_pretrain, trust_remote_code=True)

    def _get_response_length(self, response):
        if response is None:
            return 0
        response_text = response if isinstance(response, str) else str(response)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(response_text, add_special_tokens=False))
            except Exception:
                pass
        return len(response_text.split())

    @staticmethod
    def _load_video_frames(video_path: str, target_fps: float = 2.0):
        """Load video at target_fps using decord, return (ndarray[N,H,W,3], metadata)."""
        try:
            import decord
            try:
                decord.bridge.set_bridge("numpy")
            except Exception:
                decord.bridge.set_bridge("native")
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            native_fps = float(vr.get_avg_fps())
            duration = total_frames / native_fps if native_fps > 0 else 0.0

            step = max(1, int(round(native_fps / target_fps)))
            indices = list(range(0, total_frames, step))
            if len(indices) < 2:
                indices = [0, min(1, total_frames - 1)]

            batch = vr.get_batch(indices)
            if hasattr(batch, 'asnumpy'):
                frames = batch.asnumpy()
            else:
                frames = np.array(batch)

            metadata = {
                "fps": native_fps,
                "duration": duration,
                "total_num_frames": total_frames,
                "frames_indices": indices,
                "video_backend": "decord",
                "do_sample_frames": False,
            }
            return frames, metadata
        except Exception as e:
            logger.error("Failed to load video %s with decord: %s", video_path, e)
            return None, None

    def get_reward_vl_judge(self, samples):
        """
        Score [{"caption": ..., "video_path": ...}, ...] with a multimodal judge.
        Qwen3-VL under vLLM expects each video item as (ndarray, metadata_dict),
        so frames are loaded with decord before generation.
        """
        from concurrent.futures import ThreadPoolExecutor

        video_paths = []
        for sample in samples:
            vp = sample.get("video_path", "")
            video_paths.append(vp if (vp and os.path.isfile(vp)) else None)

        video_results = [None] * len(samples)
        valid_indices = [i for i, vp in enumerate(video_paths) if vp is not None]
        if valid_indices:
            with ThreadPoolExecutor(max_workers=min(8, len(valid_indices))) as pool:
                futures = {
                    pool.submit(self._load_video_frames, video_paths[i], 2.0): i
                    for i in valid_indices
                }
                for future in futures:
                    idx = futures[future]
                    frames, meta = future.result()
                    if frames is not None:
                        video_results[idx] = (frames, meta)

        inputs = []
        for i, sample in enumerate(samples):
            caption = sample.get("caption", "")
            judge_text = VL_JUDGE_PROMPT.format(caption)
            video_item = video_results[i]
            has_video = video_item is not None

            if not has_video and video_paths[i] is not None:
                logger.warning("Video not loaded: %s; scoring text-only.", sample.get("video_path", ""))

            if has_video:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": judge_text},
                        ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": judge_text}],
                    }
                ]

            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            llm_input = {"prompt": prompt_text}
            if has_video:
                llm_input["multi_modal_data"] = {"video": [video_item]}
            inputs.append(llm_input)

        outputs = self.llm.generate(inputs, sampling_params=VL_JUDGE_SAMPLING_PARAMS, use_tqdm=False)
        generated_texts = [out.outputs[0].text for out in outputs]

        judge_rewards = []
        for text in generated_texts:
            score = _parse_judge_score(text)
            judge_rewards.append(score)

        rewards = list(judge_rewards)

        if inputs:
            print("=================================", flush=True)
            cap0 = samples[0].get("caption", "") if samples else ""
            print("[vl_judge] caption (first):", cap0, flush=True)
            vp0 = samples[0].get("video_path", "") if samples else ""
            if vp0:
                print("[vl_judge] video_path (first):", vp0, flush=True)
            print("[vl_judge] input prompt tail (first, last 512 chars):", inputs[0].get("prompt", "")[-512:], flush=True)
            print("[vl_judge] output (first):", generated_texts[0][:300], flush=True)
            print("[vl_judge] rewards:", rewards, flush=True)
            print("=================================", flush=True)

        return rewards, generated_texts, {"judge_rewards": judge_rewards}

    def _to_json_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(v) for v in obj]
        return obj

    def _append_jsonl_records(self, path, records):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                safe = self._to_json_serializable(record)
                f.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def _build_sample_records(
        self,
        prompts,
        input_meta,
        generated_text,
        rewards,
        qa_num_per_caption=None,
        fixed_qa_num=None,
        qa_rewards=None,
        format_rewards=None,
    ):
        records = []
        if self.all_qa:
            idx = 0
            for i, (cap, qa) in enumerate(prompts):
                n = qa_num_per_caption[i]
                questions = [
                    {"question": input_meta[idx + j][1], "correct_answer": input_meta[idx + j][2], "model_answer": generated_text[idx + j].strip()}
                    for j in range(n)
                ]
                idx += n
                cap_text = cap if isinstance(cap, str) else str(cap)
                row = {"caption": cap_text, "response_length": self._get_response_length(cap_text), "reward": float(rewards[i]), "questions": questions}
                if qa_rewards is not None:
                    row["qa_reward"] = float(qa_rewards[i])
                if format_rewards is not None:
                    row["format_reward"] = float(format_rewards[i])
                records.append(row)
        else:
            for i, (cap, _) in enumerate(prompts):
                start = i * fixed_qa_num
                end = start + fixed_qa_num
                questions = [
                    {"question": input_meta[j][1], "correct_answer": input_meta[j][2], "model_answer": generated_text[j].strip()}
                    for j in range(start, end)
                ]
                cap_text = cap if isinstance(cap, str) else str(cap)
                row = {"caption": cap_text, "response_length": self._get_response_length(cap_text), "reward": float(rewards[i]), "questions": questions}
                if qa_rewards is not None:
                    row["qa_reward"] = float(qa_rewards[i])
                if format_rewards is not None:
                    row["format_reward"] = float(format_rewards[i])
                records.append(row)
        return records

    def get_reward(self, prompts, queries, labels):
        prompt = PROMPT_VIDEO if self.task == "video" else PROMPT_IMAGE
        inputs = []
        answers = []
        input_meta = []

        if self.all_qa:
            qa_num = []
            for cap, qa in prompts:
                qa_num.append(len(qa))
                for q, a in qa:
                    temp = prompt.format(cap, q)
                    inputs.append({"prompt": temp})
                    answers.append(a)
                    input_meta.append((cap, q, a))
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
            generated_text = [out.outputs[0].text for out in outputs]
            temp_rewards = []
            for answer, gt in zip(generated_text, answers):
                temp_rewards.append(parse_easy(answer, gt))
            rewards = []
            for qa_n in qa_num:
                rewards.append(np.mean(temp_rewards[:qa_n]))
                temp_rewards = temp_rewards[qa_n:]
            qa_num_per_caption = qa_num
            fixed_qa_num = None
        else:
            qa_num = self.qa_num
            for cap, qa in prompts:
                for i in range(qa_num):
                    q, a = random.choice(qa)
                    if self.shuffle_qa:
                        q, a = shuffle_options(q, a)
                    temp = prompt.format(cap, q)
                    inputs.append({"prompt": temp})
                    answers.append(a)
                    input_meta.append((cap, q, a))
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
            generated_text = [out.outputs[0].text for out in outputs]
            rewards = []
            for answer, gt in zip(generated_text, answers):
                rewards.append(parse_easy(answer, gt))
            rewards = torch.Tensor(rewards).view(-1, qa_num).mean(dim=-1).view(-1).tolist()
            qa_num_per_caption = None
            fixed_qa_num = qa_num

        qa_rewards = list(rewards)
        # Timestamp format reward is only used for video tasks.
        w_fmt = self.format_reward_weight if self.task == "video" else 0.0
        format_rewards = None
        if w_fmt > 0.0:
            format_rewards = [
                compute_format_reward(
                    cap if isinstance(cap, str) else str(cap),
                    max_minute=self.format_max_minute,
                )
                for cap, _ in prompts
            ]
            rewards = [q + w_fmt * f for q, f in zip(qa_rewards, format_rewards)]

        sample_records = self._build_sample_records(
            prompts=prompts,
            input_meta=input_meta,
            generated_text=generated_text,
            rewards=rewards,
            qa_num_per_caption=qa_num_per_caption,
            fixed_qa_num=fixed_qa_num,
            qa_rewards=qa_rewards if w_fmt > 0.0 else None,
            format_rewards=format_rewards,
        )

        zero_log_path = getattr(self.args, "zero_reward_log_path", None)
        if zero_log_path and sample_records and sum(rewards) == 0.0:
            try:
                self._append_jsonl_records(zero_log_path, sample_records)
                logger.info(f"Saved {len(prompts)} zero-reward sample(s) to {zero_log_path}")
            except Exception as e:
                logger.warning(f"Failed to write zero-reward samples to {zero_log_path}: {e}")

        save_longest = getattr(self.args, "save_longest_response_log", False)
        longest_log_path = getattr(self.args, "longest_response_log_path", None)
        if save_longest and longest_log_path and sample_records:
            try:
                max_len = max(int(record["response_length"]) if record.get("response_length") is not None else 0 for record in sample_records)
                longest_records = []
                for record in sample_records:
                    rlen = record.get("response_length")
                    if rlen is not None and int(rlen) == max_len:
                        new_record = dict(record)
                        new_record["batch_max_response_length"] = max_len
                        longest_records.append(new_record)
                if longest_records:
                    self._append_jsonl_records(longest_log_path, longest_records)
                    logger.info(f"Saved {len(longest_records)} longest-response sample(s) (response_length={max_len}) to {longest_log_path}")
            except Exception as e:
                logger.warning(f"Failed to write longest-response samples to {longest_log_path}: {e}", exc_info=True)

        if inputs:
            print("=================================")
            print("input:", inputs[0])
            print("output:", generated_text[0])
            print("rewards", rewards)
            if w_fmt > 0.0:
                print("qa_rewards", qa_rewards, "format_rewards", format_rewards, "format_weight", w_fmt)
            print("=================================")
        extra = {"qa_rewards": qa_rewards, "format_rewards": format_rewards}
        return rewards, generated_text, extra


def run_worker_server(args, rank, gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    torch.cuda.init()
    worker_port = args.worker_base_port + rank
    print(f"Starting worker (local rank {rank}) on GPUs {gpu_ids} at port {worker_port}...", flush=True)

    if getattr(args, "zero_reward_log_path", None):
        base = args.zero_reward_log_path
        args.zero_reward_log_path = (base[:-5] if base.endswith(".jsonl") else base) + f"_rank{rank}.jsonl"
    if getattr(args, "save_longest_response_log", False) and getattr(args, "longest_response_log_path", None):
        base = args.longest_response_log_path
        args.longest_response_log_path = (base[:-5] if base.endswith(".jsonl") else base) + f"_rank{rank}.jsonl"

    reward_model = RewardModelProxy(args)
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.post("/get_reward")
    async def get_reward_endpoint(request: Request):
        try:
            data = await request.json()
        except ClientDisconnect:
            logger.warning(f"Worker {rank}: client disconnected before body read, returning 503 for retry.")
            return JSONResponse(status_code=503, content={"rewards": [], "error": "client_disconnect"})
        t = time.time()

        req_score_mode = data.get("score_mode", reward_model.score_mode)

        if req_score_mode == "vl_judge":
            samples = data.get("samples", [])
            try:
                rewards, generated_text, extra = reward_model.get_reward_vl_judge(samples)
            except Exception as e:
                logger.error(f"Worker {rank}: get_reward_vl_judge failed: {e}", exc_info=True)
                n = len(samples)
                rewards = [0.0] * n
                generated_text = [""] * n
                extra = {"judge_rewards": rewards}
            result = {"rewards": rewards, "local_rank": rank, "judge_rewards": extra["judge_rewards"]}
            n_samples = len(samples)
        else:
            prompts = data.get('prompts')
            queries = data.get("query")
            labels = data.get("labels")
            try:
                rewards, generated_text, extra = reward_model.get_reward(prompts, queries, labels)
            except Exception as e:
                logger.error(f"Worker {rank}: get_reward failed: {e}", exc_info=True)
                n = len(prompts) if prompts else 0
                rewards = [0.0] * n
                generated_text = [""] * n
                extra = {"qa_rewards": rewards, "format_rewards": None}
            result = {"rewards": rewards, "local_rank": rank, "qa_rewards": extra["qa_rewards"]}
            if extra.get("format_rewards") is not None:
                result["format_rewards"] = extra["format_rewards"]
            n_samples = len(prompts) if prompts else 0

        if sum(rewards) == 0.0:
            print(f"--- Rank {rank} REWARDS ARE 0 ---", flush=True)
            print("Associated Generated Texts:", generated_text[:3], flush=True)
            print("-------------------------", flush=True)
        print(f"Worker {rank} (GPUs: {gpu_ids}) [{req_score_mode}] processed {n_samples} samples in {time.time() - t:.3f}s, mean reward: {sum(rewards)/max(len(rewards),1):.3f}", flush=True)
        return JSONResponse(result)

    uvicorn.run(app, host="0.0.0.0", port=worker_port, log_level="info")


def run_master_server(args):
    app = FastAPI()
    num_hosts = len(args.worker_hosts)
    if args.num_workers % num_hosts != 0:
        raise ValueError(f"Total number of workers ({args.num_workers}) must be divisible by the number of worker hosts ({num_hosts}).")
    num_workers_per_host = args.num_workers // num_hosts
    worker_urls = []
    for host in args.worker_hosts:
        for i in range(num_workers_per_host):
            port = args.worker_base_port + i
            worker_urls.append(f"http://{host}:{port}/get_reward")

    async def send_to_worker(session, url, chunk_data, chunk_index):
        try:
            response = await session.post(url, json=chunk_data, timeout=600)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request to {url} (chunk {chunk_index}) failed: {e}")
            return None

    @app.post("/get_reward")
    async def distributed_get_reward(request: Request):
        try:
            data = await request.json()
        except ClientDisconnect:
            logger.warning("Master: client disconnected before body read, returning 503 for retry.")
            return JSONResponse(status_code=503, content={"rewards": [], "error": "client_disconnect"})

        req_score_mode = data.get("score_mode", getattr(args, "score_mode", "qa"))

        if req_score_mode == "vl_judge":
            samples = data.get("samples", [])
            num_samples = len(samples)
            if num_samples == 0:
                return JSONResponse({"rewards": []})
            t_start = time.time()
            total_workers = len(worker_urls)
            chunk_size = (num_samples + total_workers - 1) // total_workers
            chunks = []
            for i in range(total_workers):
                s = i * chunk_size
                e = min(s + chunk_size, num_samples)
                if s >= num_samples:
                    break
                chunks.append({"score_mode": "vl_judge", "samples": samples[s:e]})

            async with httpx.AsyncClient() as session:
                tasks_list = [send_to_worker(session, worker_urls[i], chunks[i], i) for i in range(len(chunks))]
                worker_responses = await asyncio.gather(*tasks_list)

            all_rewards = []
            all_judge_rewards = []
            for i, response in enumerate(worker_responses):
                if response and "rewards" in response:
                    all_rewards.extend(response["rewards"])
                    jr = response.get("judge_rewards")
                    if jr is not None:
                        all_judge_rewards.extend(jr)
                else:
                    logger.error(f"Worker for chunk {i} (URL: {worker_urls[i]}) failed. Aborting batch.")
                    return JSONResponse(status_code=500, content={"error": f"Worker for chunk {i} failed.", "failed_url": worker_urls[i]})

            print(f"Master [vl_judge]: Distributed {num_samples} samples to {len(chunks)} workers. Total time: {time.time() - t_start:.3f}s", flush=True)
            out = {"rewards": all_rewards}
            if len(all_judge_rewards) == num_samples:
                out["judge_rewards"] = all_judge_rewards
            return JSONResponse(out)

        # QA mode (original)
        prompts = data.get('prompts', [])
        queries = data.get("query", [])
        labels = data.get("labels", [])
        num_prompts = len(prompts)
        if num_prompts == 0:
            return JSONResponse({"rewards": []})
        t_start = time.time()
        total_workers = len(worker_urls)
        chunk_size = (num_prompts + total_workers - 1) // total_workers
        chunks = []
        for i in range(total_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_prompts)
            if start_idx >= num_prompts:
                break
            chunks.append({"prompts": prompts[start_idx:end_idx], "query": queries, "labels": labels})

        async with httpx.AsyncClient() as session:
            tasks_list = [send_to_worker(session, worker_urls[i], chunks[i], i) for i in range(len(chunks))]
            worker_responses = await asyncio.gather(*tasks_list)

        all_rewards = []
        all_qa_rewards = []
        all_format_rewards = []
        for i, response in enumerate(worker_responses):
            if response and 'rewards' in response:
                all_rewards.extend(response['rewards'])
                if response.get("qa_rewards") is not None:
                    all_qa_rewards.extend(response["qa_rewards"])
                fr = response.get("format_rewards")
                if fr is not None:
                    all_format_rewards.extend(fr)
            else:
                logger.error(f"Worker for chunk {i} (URL: {worker_urls[i]}) failed. Aborting batch.")
                return JSONResponse(status_code=500, content={"error": f"Worker for chunk {i} failed.", "failed_url": worker_urls[i]})

        print(f"Master [qa]: Distributed {num_prompts} samples to {len(chunks)} workers. Total time: {time.time() - t_start:.3f}s", flush=True)
        out = {"rewards": all_rewards}
        if len(all_qa_rewards) == num_prompts:
            out["qa_rewards"] = all_qa_rewards
        if len(all_format_rewards) == num_prompts:
            out["format_rewards"] = all_format_rewards
        return JSONResponse(out)

    print(f"Starting master server on http://{args.master_host}:{args.port}")
    print(f"Distributing tasks to {args.num_workers} workers across {num_hosts} hosts.")
    print(f"Worker URLs: {worker_urls}")
    uvicorn.run(app, host=args.master_host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, required=True, choices=["master", "worker"])
    parser.add_argument("--reward_pretrain", type=str, help="Path to the pretrained reward model.")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--shuffle_qa", action="store_true", default=False)
    parser.add_argument("--all_qa", action="store_true", default=False)
    parser.add_argument("--qa_num", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--master_host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker_hosts", type=str, nargs='+')
    parser.add_argument("--worker_base_port", type=int, default=8001)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--reward_func", type=str, default=None)
    parser.add_argument("--zero_reward_log_path", type=str, default=None)
    parser.add_argument("--save_longest_response_log", action="store_true")
    parser.add_argument("--longest_response_log_path", type=str, default=None)
    parser.add_argument(
        "--format_reward_weight",
        type=float,
        default=0.0,
        help="Fuse with QA reward as final=qa + w*format. Set 0 to disable.",
    )
    parser.add_argument(
        "--format_max_minute",
        type=int,
        default=599,
        help="Maximum allowed minute component in a timestamp bracket.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["image", "video"],
        default="video",
        help="image: image caption QA; video: video caption QA with optional timestamp format reward.",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        choices=["qa", "vl_judge"],
        default="qa",
        help="qa: caption QA accuracy; vl_judge: direct video-caption LLM-as-a-judge scoring.",
    )
    parser.add_argument(
        "--judge_max_model_len",
        type=int,
        default=18000,
        help="vLLM max_model_len for vl_judge mode.",
    )
    args = parser.parse_args()

    if args.role == "master":
        if not args.worker_hosts or not args.num_workers:
            parser.error("--role 'master' requires --worker_hosts and --num_workers.")
        run_master_server(args)
    elif args.role == "worker":
        mp.set_start_method('spawn', force=True)
        if not args.reward_pretrain:
            parser.error("--role 'worker' requires --reward_pretrain.")
        num_gpus_available = torch.cuda.device_count()
        if num_gpus_available < args.tp:
            raise ValueError(f"Not enough GPUs. Required: {args.tp}, Found: {num_gpus_available}")
        if num_gpus_available % args.tp != 0:
            raise ValueError(f"GPUs ({num_gpus_available}) not divisible by tp ({args.tp}).")
        num_local_workers = num_gpus_available // args.tp
        print(f'Found {num_gpus_available} GPUs. Starting {num_local_workers} worker processes with TP={args.tp}...')
        gpu_ids = list(range(num_gpus_available))
        gpu_groups = [gpu_ids[i:i + args.tp] for i in range(0, len(gpu_ids), args.tp)]
        processes = [mp.Process(target=run_worker_server, args=(args, i, gpu_groups[i])) for i in range(num_local_workers)]
        for p in processes:
            p.start()
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                p.join()
            print("All workers terminated.")
