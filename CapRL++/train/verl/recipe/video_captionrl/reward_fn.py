import os
import re
import json
import random
import itertools
import aiohttp
import asyncio
import requests
import subprocess
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

# Remote reward server endpoint. When set, this module posts rewards to it
# instead of using local vLLM inference.
REWARD_REMOTE_URL = os.environ.get("REWARD_REMOTE_URL", "").strip()

# qa: caption QA accuracy; vl_judge: direct video-caption scoring.
REWARD_SCORE_MODE = os.environ.get("REWARD_SCORE_MODE", "qa").strip().lower()

# Avoid large local logs when using a remote reward server unless explicitly enabled.
REWARD_LOG_TO_FILE = os.environ.get("REWARD_LOG_TO_FILE", "0" if REWARD_REMOTE_URL else "1").lower() in ("1", "true", "yes")
REWARD_LOG_DIR = os.environ.get("REWARD_LOG_DIR", "/tmp/video_captionrl_rewards")
if REWARD_LOG_TO_FILE:
    os.makedirs(REWARD_LOG_DIR, exist_ok=True)
    REWARD_LOG_FILE = os.path.join(REWARD_LOG_DIR, f"rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
else:
    REWARD_LOG_FILE = None

URLS = os.environ.get("REWARD_VLLM_URLS", "http://reward-node:8000/v1/chat/completions")
REWARD_VLLM_URLS = [u.strip() for u in URLS.split(",") if u.strip()]
REWARD_MODEL = os.environ.get("REWARD_VLLM_MODEL", "/models/Qwen3-VL-4B")

QA_NUM = int(os.environ.get("REWARD_QA_NUM", "8"))
SHUFFLE_QA = os.environ.get("REWARD_SHUFFLE_QA", "true").lower() in ("1", "true", "yes")
ALL_QA = os.environ.get("REWARD_ALL_QA", "false").lower() in ("1", "true", "yes")

_url_cycle = itertools.cycle(REWARD_VLLM_URLS)

CANNOT_ANSWER_TEXT = "Can not answer based on the caption"

REWARD_DEBUG = os.environ.get("REWARD_DEBUG", "").lower() in ("1", "true", "yes")
_reward_debug_logged = False

import threading
import uuid

JUDGE_BATCH_SIZE = int(os.environ.get("REWARD_JUDGE_BATCH_SIZE", "64"))
JUDGE_BATCH_TIMEOUT = float(os.environ.get("REWARD_JUDGE_BATCH_TIMEOUT", "2.0"))

# Batch single-sample remote calls to reduce HTTP overhead.
QA_BATCH_SIZE = int(os.environ.get("REWARD_QA_BATCH_SIZE", str(JUDGE_BATCH_SIZE)))
QA_BATCH_TIMEOUT = float(os.environ.get("REWARD_QA_BATCH_TIMEOUT", str(JUDGE_BATCH_TIMEOUT)))


class _JudgeBatcher:
    """Thread-safe request batcher for vl_judge mode."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Dict[str, dict] = {}
        self._results: Dict[str, Any] = {}
        self._events: Dict[str, threading.Event] = {}
        self._batch_ids: list = []
        self._timer: Optional[threading.Timer] = None

    def submit(self, caption: str, video_path: Optional[str]) -> dict:
        req_id = uuid.uuid4().hex
        event = threading.Event()

        with self._lock:
            self._pending[req_id] = {"caption": caption, "video_path": video_path or ""}
            self._events[req_id] = event
            self._batch_ids.append(req_id)
            batch_full = len(self._batch_ids) >= JUDGE_BATCH_SIZE

            if self._timer is None and not batch_full:
                self._timer = threading.Timer(JUDGE_BATCH_TIMEOUT, self._flush)
                self._timer.daemon = True
                self._timer.start()

            if batch_full:
                self._flush_locked()

        event.wait(timeout=1800)
        with self._lock:
            result = self._results.pop(req_id, None)
            self._events.pop(req_id, None)
        if result is None:
            return {"score": 0.0, "judge_reward": 0.0, "length_reward": 0.0, "cap_tokens": 0}
        return result

    def _flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if not self._batch_ids:
            return

        ids = list(self._batch_ids)
        samples = [self._pending.pop(rid) for rid in ids]
        self._batch_ids.clear()

        threading.Thread(target=self._do_request, args=(ids, samples), daemon=True).start()

    def _do_request(self, ids: list, samples: list):
        captions = [s["caption"] for s in samples]
        vpaths = [s["video_path"] for s in samples]
        try:
            final, judge = _call_judge_reward_server(captions, vpaths, REWARD_REMOTE_URL)
        except Exception as e:
            print(f"[reward_fn] Batched judge request failed: {e}", flush=True)
            final = [0.0] * len(ids)
            judge = list(final)

        final_adj, r_ls, cap_lens = _apply_length_to_final_scores(captions, final)

        with self._lock:
            for i, rid in enumerate(ids):
                self._results[rid] = {
                    "score": float(final_adj[i]),
                    "judge_reward": float(judge[i]),
                    "length_reward": float(r_ls[i]),
                    "cap_tokens": int(cap_lens[i]),
                }
                if rid in self._events:
                    self._events[rid].set()


_judge_batcher: Optional[_JudgeBatcher] = None


def _get_judge_batcher() -> _JudgeBatcher:
    global _judge_batcher
    if _judge_batcher is None:
        _judge_batcher = _JudgeBatcher()
    return _judge_batcher


class _QABatcher:
    """Thread-safe request batcher for qa mode when compute_score is invoked per-sample (e.g. NaiveRewardManager)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Dict[str, dict] = {}
        self._results: Dict[str, Any] = {}
        self._events: Dict[str, threading.Event] = {}
        self._batch_ids: list = []
        self._timer: Optional[threading.Timer] = None

    def submit(self, solution_str: str, ground_truth: Any) -> dict:
        req_id = uuid.uuid4().hex
        event = threading.Event()

        with self._lock:
            self._pending[req_id] = {"solution_str": solution_str, "ground_truth": ground_truth}
            self._events[req_id] = event
            self._batch_ids.append(req_id)
            batch_full = len(self._batch_ids) >= QA_BATCH_SIZE

            if self._timer is None and not batch_full:
                self._timer = threading.Timer(QA_BATCH_TIMEOUT, self._flush)
                self._timer.daemon = True
                self._timer.start()

            if batch_full:
                self._flush_locked()

        event.wait(timeout=1800)
        with self._lock:
            result = self._results.pop(req_id, None)
            self._events.pop(req_id, None)
        if result is None:
            return {"score": 0.0, "qa_reward": 0.0, "length_reward": 0.0, "cap_tokens": 0}
        return result

    def _flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

        if not self._batch_ids:
            return

        ids = list(self._batch_ids)
        samples = [self._pending.pop(rid) for rid in ids]
        self._batch_ids.clear()

        threading.Thread(target=self._do_request, args=(ids, samples), daemon=True).start()

    def _do_request(self, ids: list, samples: list):
        solution_str_list = [s["solution_str"] for s in samples]
        ground_truth_list = [s["ground_truth"] for s in samples]
        try:
            final, qa, fmt = _call_openrlhf_reward_server(
                solution_str_list, ground_truth_list, REWARD_REMOTE_URL
            )
        except Exception as e:
            print(f"[reward_fn] Batched QA request failed: {e}", flush=True)
            final = [0.0] * len(ids)
            qa = list(final)
            fmt = None

        final_adj, r_ls, cap_lens = _apply_length_to_final_scores(solution_str_list, final)

        with self._lock:
            for i, rid in enumerate(ids):
                row: Dict[str, Any] = {
                    "score": float(final_adj[i]),
                    "qa_reward": float(qa[i]),
                    "length_reward": float(r_ls[i]),
                    "cap_tokens": int(cap_lens[i]),
                }
                if fmt is not None:
                    row["format_reward"] = float(fmt[i])
                self._results[rid] = row
                if rid in self._events:
                    self._events[rid].set()


_qa_batcher: Optional[_QABatcher] = None


def _get_qa_batcher() -> _QABatcher:
    global _qa_batcher
    if _qa_batcher is None:
        _qa_batcher = _QABatcher()
    return _qa_batcher

# Piecewise caption length reward. Thresholds and weight come from environment variables.
_length_tokenizer = None
_length_tokenizer_path: Optional[str] = None
_length_disabled_warned = False


def _piecewise_length_reward(cap_tokens: int, l1: int, l2: int) -> float:
    """R_l: cap<=l1 -> 1.0; (l1,l2] decays linearly to 0; >l2 -> 0."""
    if cap_tokens <= l1:
        return 1.0
    if l2 <= l1:
        return 0.0 if cap_tokens > l1 else 1.0
    if cap_tokens <= l2:
        return 1.0 - (cap_tokens - l1) / (l2 - l1)
    return 0.0


def _get_length_tokenizer():
    """Load the tokenizer used for length reward."""
    global _length_tokenizer, _length_tokenizer_path
    path = os.environ.get("REWARD_LENGTH_TOKENIZER_PATH", "").strip() or os.environ.get(
        "REWARD_VLLM_MODEL", ""
    ).strip()
    if not path:
        return None
    if _length_tokenizer is not None and _length_tokenizer_path == path:
        return _length_tokenizer
    from transformers import AutoTokenizer

    _length_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    _length_tokenizer_path = path
    return _length_tokenizer


def _count_caption_tokens(text: Optional[str]) -> int:
    tok = _get_length_tokenizer()
    if tok is None:
        return 0
    s = text if isinstance(text, str) else str(text or "")
    return len(tok.encode(s, add_special_tokens=False))


def _length_reward_weight() -> float:
    global _length_disabled_warned
    w = float(os.environ.get("REWARD_LENGTH_WEIGHT", "0"))
    if w == 0.0:
        return 0.0
    path = os.environ.get("REWARD_LENGTH_TOKENIZER_PATH", "").strip() or os.environ.get(
        "REWARD_VLLM_MODEL", ""
    ).strip()
    if not path:
        if not _length_disabled_warned:
            print(
                "[reward_fn] REWARD_LENGTH_WEIGHT>0 but no REWARD_LENGTH_TOKENIZER_PATH "
                "(or REWARD_VLLM_MODEL); length reward disabled.",
                flush=True,
            )
            _length_disabled_warned = True
        return 0.0
    return w


def _length_l1_l2() -> Tuple[int, int]:
    l1 = int(os.environ.get("REWARD_LENGTH_L1", "2048"))
    l2 = int(os.environ.get("REWARD_LENGTH_L2", "3072"))
    return l1, l2


def _apply_length_to_final_scores(
    solution_str_list: List[str], base_final: List[float]
) -> Tuple[List[float], List[float], List[int]]:
    """Add weighted length reward to the already fused remote score."""
    w = _length_reward_weight()
    l1, l2 = _length_l1_l2()
    cap_lens = [_count_caption_tokens(s) for s in solution_str_list]
    r_ls = [_piecewise_length_reward(n, l1, l2) for n in cap_lens]
    if w == 0.0:
        return list(base_final), r_ls, cap_lens
    adj = [float(b) + w * r for b, r in zip(base_final, r_ls)]
    return adj, r_ls, cap_lens


def _next_url():
    return next(_url_cycle)


def _parse_easy(answer_text: str, gt: str) -> int:
    """Parse the first option letter from model output."""
    if not answer_text:
        return 0
    pattern = re.compile(r'[A-I]')
    res = pattern.findall(answer_text)
    if len(res) > 0:
        return 1 if res[0] == gt else 0
    return 0


def _shuffle_options(question: str, answer: str) -> Tuple[str, str]:
    """Shuffle options while preserving the correct answer label."""
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

    correct_answer_label = answer
    if correct_answer_label not in original_options:
        return question + '\n   - F) Can not answer based on the caption', answer

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

    return '\n'.join(new_question_lines) + '\n   - F) Can not answer based on the caption', new_answer


PROMPT_TEMPLATE = '''You will be given an image caption describing the visual content.  
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
Question: {}  

You must output **exactly one line** in the format:
The answer is X.
where X is a single capital letter from A to F. Do not output anything else.'''


def _build_prompt(caption: str, question: str) -> str:
    """Build the QA prompt."""
    return PROMPT_TEMPLATE.format(caption.strip(), question.strip())


async def _vllm_chat(prompt: str) -> str:
    """Call a local vLLM OpenAI-compatible endpoint."""
    payload = {
        "model": REWARD_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": 10,
    }
    url = _next_url()
    try:
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        print(f"[reward_fn] vLLM HTTP {resp.status}: url={url}, body={text[:500]}")
                        return ""
                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError as e:
                        print(f"[reward_fn] vLLM response not JSON: url={url}, error={e}")
                        return ""
                    if "choices" not in result or not result["choices"]:
                        return ""
                    choice = result["choices"][0]
                    return choice.get("message", {}).get("content") or choice.get("content") or ""
            except aiohttp.ClientError as e:
                print(f"[reward_fn] vLLM connection error to {url}: {e}")
                return ""
    except Exception as e:
        print(f"[reward_fn] unexpected error when calling vLLM at {url}: {e}")
        return ""


async def _compute_single_score(solution_str: str, ground_truth: List[Dict]) -> Dict[str, Any]:
    """
    Compute a QA reward for one caption.
    ground_truth may be List[Tuple[question_str, answer_str]] or List[Dict].
    """
    if not ground_truth:
        return {"score": 0.0, "correct_count": 0, "total_count": 0, "details": []}

    qa_list_raw = []
    for item in ground_truth:
        if isinstance(item, dict):
            q = item.get("question", "")
            a = item.get("answer", "A")
            choices = item.get("choices", [])
            if choices:
                choice_lines = []
                for i, c in enumerate(choices):
                    c = (c or "").strip()
                    if re.match(r'^[A-F]\)', c, re.IGNORECASE):
                        choice_lines.append(f"   - {c}")
                    else:
                        label = chr(ord('A') + i)
                        choice_lines.append(f"   - {label}) {c}")
                q_full = q.strip() + "\n" + "\n".join(choice_lines)
            else:
                q_full = q.strip()
            qa_list_raw.append((q_full, a.strip().upper()))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            qa_list_raw.append((item[0], item[1].strip().upper()))

    if not qa_list_raw:
        return {"score": 0.0, "correct_count": 0, "total_count": 0, "details": []}

    inputs = []
    answers = []
    questions = []
    details = []

    if ALL_QA:
        for q, a in qa_list_raw:
            prompt = _build_prompt(solution_str, q)
            inputs.append(prompt)
            answers.append(a)
            questions.append(q)
    else:
        for _ in range(QA_NUM):
            q, a = random.choice(qa_list_raw)
            if SHUFFLE_QA:
                q, a = _shuffle_options(q, a)
            prompt = _build_prompt(solution_str, q)
            inputs.append(prompt)
            answers.append(a)
            questions.append(q)

    tasks = [_vllm_chat(p) for p in inputs]
    outputs = await asyncio.gather(*tasks)

    correct_list = []
    global _reward_debug_logged
    for i, (output_text, gt, q_text) in enumerate(zip(outputs, answers, questions)):
        is_correct = _parse_easy(output_text, gt)
        correct_list.append(is_correct)
        
        if REWARD_DEBUG and not _reward_debug_logged:
            _reward_debug_logged = True
            print("[reward_fn] DEBUG: prompt_tail=", inputs[i][-300:])
            print("[reward_fn] DEBUG: output=", repr(output_text), "expected=", gt, "correct=", is_correct)
        
        details.append({
            "question": q_text,
            "answer": gt,
            "prediction": output_text.strip() if output_text else "",
            "is_correct": bool(is_correct),
        })

    score = (sum(correct_list) / len(correct_list)) if correct_list else 0.0

    return {
        "score": score,
        "correct_count": sum(correct_list),
        "total_count": len(correct_list),
        "details": details,
    }


def _log_reward(caption: str, result: Dict[str, Any]):
    """Write reward details when local logging is enabled."""
    if not REWARD_LOG_TO_FILE or REWARD_LOG_FILE is None:
        return
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "caption": caption[:2000],
        "score": result["score"],
        "correct_count": result["correct_count"],
        "total_count": result["total_count"],
        "details": result.get("details", []),
        "length_reward": result.get("length_reward"),
        "cap_tokens": result.get("cap_tokens"),
    }
    try:
        with open(REWARD_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Warning: Failed to log reward: {e}")


def _ground_truth_to_qa_list(ground_truth: Union[List, Any]) -> List[Tuple[str, str]]:
    """Convert VERL ground truth into [(question_full, answer), ...]."""
    qa_list_raw = []
    for item in (ground_truth if isinstance(ground_truth, list) else [ground_truth]):
        if isinstance(item, dict):
            q = item.get("question", "")
            a = (item.get("answer", "A") or "A").strip().upper()
            choices = item.get("choices", [])
            if choices:
                choice_lines = []
                for i, c in enumerate(choices):
                    c = (c or "").strip()
                    if re.match(r"^[A-F]\)", c, re.IGNORECASE):
                        choice_lines.append(f"   - {c}")
                    else:
                        choice_lines.append(f"   - {chr(ord('A') + i)}) {c}")
                q_full = q.strip() + "\n" + "\n".join(choice_lines)
            else:
                q_full = q.strip()
            qa_list_raw.append((q_full, a))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            qa_list_raw.append((item[0], str(item[1]).strip().upper()))
    return qa_list_raw


def _call_openrlhf_reward_server(
    solution_str_list: List[str], ground_truth_list: List[Any], url: str
) -> Tuple[List[float], List[float], Optional[List[float]]]:
    """
    Call the QA reward server.
    Payload: {"prompts": [[caption, [[q,a],[q,a],...]], ...], "query": [], "labels": []}.
    Returns (final_rewards, qa_rewards, format_rewards_or_None).
    """
    prompts_payload = []
    for solution_str, ground_truth in zip(solution_str_list, ground_truth_list):
        qa_list = _ground_truth_to_qa_list(ground_truth)
        if not qa_list:
            prompts_payload.append([solution_str, []])
        else:
            prompts_payload.append([solution_str, qa_list])
    payload = {"prompts": prompts_payload, "query": [], "labels": []}
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=1800)
        resp.raise_for_status()
        data = resp.json()
        rewards = data.get("rewards", [])
        if len(rewards) != len(solution_str_list):
            raise ValueError(
                f"Reward server returned {len(rewards)} rewards for {len(solution_str_list)} samples"
            )
        final = [float(r) for r in rewards]
        qa = data.get("qa_rewards")
        if qa is not None and len(qa) == len(solution_str_list):
            qa = [float(x) for x in qa]
        else:
            qa = list(final)
        fmt = data.get("format_rewards")
        if fmt is not None and len(fmt) == len(solution_str_list):
            fmt = [float(x) for x in fmt]
        else:
            fmt = None
        return final, qa, fmt
    except Exception as e:
        print(f"[reward_fn] OpenRLHF reward server error: {e}")
        raise

def _call_judge_reward_server(
    solution_str_list: List[str],
    video_path_list: List[Optional[str]],
    url: str,
) -> Tuple[List[float], List[float]]:
    """
    Call the vl_judge reward server.
    Payload: {"score_mode": "vl_judge", "samples": [{"caption": ..., "video_path": ...}, ...]}.
    Returns (final_rewards, judge_rewards).
    """
    samples = []
    for caption, vpath in zip(solution_str_list, video_path_list):
        samples.append({"caption": caption, "video_path": vpath or ""})
    payload = {"score_mode": "vl_judge", "samples": samples}
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=1800)
        resp.raise_for_status()
        data = resp.json()
        rewards = data.get("rewards", [])
        if len(rewards) != len(solution_str_list):
            raise ValueError(
                f"Judge reward server returned {len(rewards)} rewards for {len(solution_str_list)} samples"
            )
        final = [float(r) for r in rewards]
        judge = data.get("judge_rewards")
        if judge is not None and len(judge) == len(solution_str_list):
            judge = [float(x) for x in judge]
        else:
            judge = list(final)
        return final, judge
    except Exception as e:
        print(f"[reward_fn] Judge reward server error: {e}")
        raise

def compute_score(
    solution_str: Union[str, List[str], None] = None,
    ground_truth: Union[Any, List[Any], None] = None,
    extra_info: Dict[str, Any] = None,
    **kwargs,
) -> Union[float, Dict[str, float], List[Dict[str, float]]]:
    """Reward function entry point for verl."""
    if kwargs.get("solution_strs") is not None:
        solution_str = kwargs["solution_strs"]
    if kwargs.get("ground_truths") is not None:
        ground_truth = kwargs["ground_truths"]
    extra_infos = kwargs.get("extra_infos", None)

    is_single = isinstance(solution_str, str)
    if is_single:
        solution_str_list = [solution_str]
        ground_truth_list = [ground_truth]
        extra_info_list = [extra_info or {}]
    else:
        solution_str_list = solution_str
        ground_truth_list = ground_truth if ground_truth is not None else [None] * len(solution_str_list)
        if extra_infos is not None:
            extra_info_list = list(extra_infos)
        elif extra_info is not None:
            extra_info_list = [extra_info] * len(solution_str_list)
        else:
            extra_info_list = [{}] * len(solution_str_list)

    def _extract_video_paths() -> List[Optional[str]]:
        paths: List[Optional[str]] = []
        for ei in extra_info_list:
            if not isinstance(ei, dict):
                paths.append(None)
                continue
            vp = ei.get("video_path") or None
            if vp is None:
                vps = ei.get("video_paths")
                if isinstance(vps, list) and vps:
                    vp = vps[0] if isinstance(vps[0], str) else None
            paths.append(vp)
        return paths

    if REWARD_SCORE_MODE == "vl_judge":
        video_path_list = _extract_video_paths()

        if not REWARD_REMOTE_URL:
            print("[reward_fn] WARNING: vl_judge mode without REWARD_REMOTE_URL; returning 0 scores.", flush=True)
            zero = {"score": 0.0, "judge_reward": 0.0, "length_reward": 0.0, "cap_tokens": 0}
            return zero if is_single else [dict(zero) for _ in solution_str_list]

        if is_single:
            batcher = _get_judge_batcher()
            return batcher.submit(solution_str_list[0], video_path_list[0])

        final, judge = _call_judge_reward_server(solution_str_list, video_path_list, REWARD_REMOTE_URL)
        final_adj, r_ls, cap_lens = _apply_length_to_final_scores(solution_str_list, final)
        return [
            {
                "score": float(final_adj[i]),
                "judge_reward": float(judge[i]),
                "length_reward": float(r_ls[i]),
                "cap_tokens": int(cap_lens[i]),
            }
            for i in range(len(solution_str_list))
        ]

    def _remote_result_to_output(
        final: List[float],
        qa: List[float],
        fmt: Optional[List[float]],
        single: bool,
        caps: List[str],
    ):
        final_adj, r_ls, cap_lens = _apply_length_to_final_scores(caps, final)
        if single:
            out: Dict[str, Any] = {
                "score": float(final_adj[0]),
                "qa_reward": float(qa[0]),
                "length_reward": float(r_ls[0]),
                "cap_tokens": int(cap_lens[0]),
            }
            if fmt is not None:
                out["format_reward"] = float(fmt[0])
            return out
        rows = []
        for i, r in enumerate(final_adj):
            row: Dict[str, Any] = {
                "score": float(r),
                "qa_reward": float(qa[i]),
                "length_reward": float(r_ls[i]),
                "cap_tokens": int(cap_lens[i]),
            }
            if fmt is not None:
                row["format_reward"] = float(fmt[i])
            rows.append(row)
        return rows

    if REWARD_REMOTE_URL:
        if is_single:
            return _get_qa_batcher().submit(solution_str_list[0], ground_truth_list[0])
        final, qa, fmt = _call_openrlhf_reward_server(solution_str_list, ground_truth_list, REWARD_REMOTE_URL)
        return _remote_result_to_output(final, qa, fmt, is_single, solution_str_list)

    async def _batch_compute():
        tasks = [
            _compute_single_score(sol, gt)
            for sol, gt in zip(solution_str_list, ground_truth_list)
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            results = loop.run_until_complete(_batch_compute())
        except Exception:
            results = asyncio.run(_batch_compute())
    else:
        results = asyncio.run(_batch_compute())

    w_eff = _length_reward_weight()
    l1, l2 = _length_l1_l2()
    for caption, result in zip(solution_str_list, results):
        n = _count_caption_tokens(caption)
        rl = _piecewise_length_reward(n, l1, l2)
        result["length_reward"] = rl
        result["cap_tokens"] = n
        result["score"] = float(result["score"]) + w_eff * rl

    for caption, result in zip(solution_str_list, results):
        _log_reward(caption, result)

    if is_single:
        r0 = results[0]
        return {
            "score": float(r0["score"]),
            "length_reward": float(r0.get("length_reward", 0.0)),
            "cap_tokens": int(r0.get("cap_tokens", 0)),
        }
    return [
        {
            "score": float(r["score"]),
            "length_reward": float(r.get("length_reward", 0.0)),
            "cap_tokens": int(r.get("cap_tokens", 0)),
        }
        for r in results
    ]


def start_vllm_openai_cluster(
    model_path: Optional[str] = None,
    num_gpus: int = 8,
    base_port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 12288,
):
    """
    Start local vLLM OpenAI-compatible API servers for reward computation.
    """
    if model_path is None:
        model_path = REWARD_MODEL

    procs = []
    for local_rank in range(num_gpus):
        port = base_port + local_rank
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(local_rank)

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--served-model-name",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-model-len",
            str(max_model_len),
            "--trust-remote-code",
            "--disable-log-requests",
        ]

        print(f"[reward_fn] starting vLLM on GPU {local_rank}, port {port}")
        print("[reward_fn] cmd:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))
        time.sleep(5)

    print(
        f"[reward_fn] started {len(procs)} vLLM instances on ports "
        f"{base_port}-{base_port + len(procs) - 1}"
    )
    print("[reward_fn] press Ctrl+C to stop all instances.")

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("[reward_fn] received Ctrl+C, terminating all vLLM processes...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utilities for video_captionrl reward model (vLLM cluster starter)."
    )
    subparsers = parser.add_subparsers(dest="command")

    p_vllm = subparsers.add_parser(
        "start_vllm",
        help="Start a local multi-GPU vLLM OpenAI API cluster for reward computation.",
    )
    p_vllm.add_argument(
        "--model",
        type=str,
        default=None,
        help="vLLM model path. Defaults to REWARD_VLLM_MODEL or REWARD_MODEL.",
    )
    p_vllm.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of vLLM instances to start.",
    )
    p_vllm.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="First vLLM port; later instances increment by 1.",
    )
    p_vllm.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.85,
        help="vLLM --gpu-memory-utilization.",
    )
    p_vllm.add_argument(
        "--max-model-len",
        type=int,
        default=12288,
        help="vLLM --max-model-len.",
    )

    args = parser.parse_args()

    if args.command == "start_vllm":
        model_path = args.model or os.environ.get("REWARD_VLLM_MODEL") or REWARD_MODEL
        start_vllm_openai_cluster(
            model_path=model_path,
            num_gpus=args.num_gpus,
            base_port=args.base_port,
            gpu_memory_utilization=args.gpu_mem_util,
            max_model_len=args.max_model_len,
        )
    else:
        parser.print_help()
