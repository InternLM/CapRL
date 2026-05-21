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

# ========== 远程 Reward Server ==========
# 在 reward 节点执行 verl/recipe/video_captionrl/scripts/start_reward_server.sh（或 --task image|video），
# 训练节点设 REWARD_REMOTE_URL=http://<reward_ip>:18889/get_reward，本模块 POST 到该地址；
# 不再走本地 vLLM，默认不写本地大 jsonl，日志在 reward 终端查看
REWARD_REMOTE_URL = os.environ.get("REWARD_REMOTE_URL", "").strip()

# ========== 评分模式 ==========
# qa: 原有 caption + QA 正确率;  vl_judge: 视频+caption 直接打分 (LLM-as-a-judge)
REWARD_SCORE_MODE = os.environ.get("REWARD_SCORE_MODE", "qa").strip().lower()

# ========== 日志配置 ==========
# 使用远程 server 时默认不写本地 jsonl（避免数据量过大）；设为 1 可强制写
REWARD_LOG_TO_FILE = os.environ.get("REWARD_LOG_TO_FILE", "0" if REWARD_REMOTE_URL else "1").lower() in ("1", "true", "yes")
REWARD_LOG_DIR = os.environ.get("REWARD_LOG_DIR", "/tmp/video_captionrl_rewards")
if REWARD_LOG_TO_FILE:
    os.makedirs(REWARD_LOG_DIR, exist_ok=True)
    REWARD_LOG_FILE = os.path.join(REWARD_LOG_DIR, f"rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
else:
    REWARD_LOG_FILE = None

# 多 URL 轮询（逗号分隔）
URLS = os.environ.get("REWARD_VLLM_URLS", "http://reward-node:8000/v1/chat/completions")
REWARD_VLLM_URLS = [u.strip() for u in URLS.split(",") if u.strip()]
REWARD_MODEL = os.environ.get("REWARD_VLLM_MODEL", "/models/Qwen3-VL-4B")

# ========== 与 OpenRLHF 对齐的配置 ==========
QA_NUM = int(os.environ.get("REWARD_QA_NUM", "8"))
SHUFFLE_QA = os.environ.get("REWARD_SHUFFLE_QA", "true").lower() in ("1", "true", "yes")
ALL_QA = os.environ.get("REWARD_ALL_QA", "false").lower() in ("1", "true", "yes")

_url_cycle = itertools.cycle(REWARD_VLLM_URLS)

CANNOT_ANSWER_TEXT = "Can not answer based on the caption"

REWARD_DEBUG = os.environ.get("REWARD_DEBUG", "").lower() in ("1", "true", "yes")
_reward_debug_logged = False

# ========== vl_judge 请求聚合器 ==========
# 多个并发 compute_score 调用（来自 reward_loop 的 asyncio.gather）会把请求放入队列，
# 凑够 batch 或超时后统一 POST，大幅减少 HTTP 开销并让 server 端批量推理。
import threading
import uuid

JUDGE_BATCH_SIZE = int(os.environ.get("REWARD_JUDGE_BATCH_SIZE", "64"))
JUDGE_BATCH_TIMEOUT = float(os.environ.get("REWARD_JUDGE_BATCH_TIMEOUT", "2.0"))

# QA 远程模式下的单样本调用聚合（与 JUDGE_* 语义一致；未设置时沿用 judge 的默认值）
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

# ========== 长度奖励 R_L（分段）：cap 为 caption 的 token 长度 ==========
# l1,l2 与权重由环境变量配置（见 start_reward_serve_rm.sh / 训练脚本）
_length_tokenizer = None
_length_tokenizer_path: Optional[str] = None
_length_disabled_warned = False


def _piecewise_length_reward(cap_tokens: int, l1: int, l2: int) -> float:
    """R_l: cap<=l1 -> 1.0；(l1,l2] 线性降到 0；>l2 -> 0。"""
    if cap_tokens <= l1:
        return 1.0
    if l2 <= l1:
        return 0.0 if cap_tokens > l1 else 1.0
    if cap_tokens <= l2:
        return 1.0 - (cap_tokens - l1) / (l2 - l1)
    return 0.0


def _get_length_tokenizer():
    """与 actor 对齐时请设置 REWARD_LENGTH_TOKENIZER_PATH 为 Caption 模型 HF 目录。"""
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
    """score = R_acc + w_fmt*R_format + w_l*R_l 中与远程 fused 一致：在已融合 final 上加 w_l*R_l。"""
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
    """与 OpenRLHF parse_easy 完全一致"""
    if not answer_text:
        return 0
    pattern = re.compile(r'[A-I]')
    res = pattern.findall(answer_text)
    if len(res) > 0:
        return 1 if res[0] == gt else 0
    return 0


def _shuffle_options(question: str, answer: str) -> Tuple[str, str]:
    """与 OpenRLHF shuffle_options 完全一致"""
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
        # 如果找不到答案，返回原样
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


# 与 OpenRLHF 完全一致的 prompt 模板（内容与远端 serve_rm.py 对齐），并额外显式约束输出格式
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
    """构建与 OpenRLHF 一致的 prompt"""
    return PROMPT_TEMPLATE.format(caption.strip(), question.strip())


async def _vllm_chat(prompt: str) -> str:
    """调用 vLLM，与 OpenRLHF sampling_params 对齐"""
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
    计算单个 caption 的得分，与 OpenRLHF get_reward 完全对齐
    ground_truth: List[Tuple[question_str, answer_str]] 或 List[Dict]
    """
    if not ground_truth:
        return {"score": 0.0, "correct_count": 0, "total_count": 0, "details": []}

    # 转换 ground_truth 格式：支持 List[Dict] 或 List[Tuple]
    qa_list_raw = []
    for item in ground_truth:
        if isinstance(item, dict):
            q = item.get("question", "")
            a = item.get("answer", "A")
            choices = item.get("choices", [])
            # 构建与 OpenRLHF 一致的 question 格式
            if choices:
                choice_lines = []
                for i, c in enumerate(choices):
                    c = (c or "").strip()
                    # 如果已有标签则保留，否则加标签
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
        # 使用全部题目
        for q, a in qa_list_raw:
            prompt = _build_prompt(solution_str, q)
            inputs.append(prompt)
            answers.append(a)
            questions.append(q)
    else:
        # 有放回抽样 qa_num 道题
        for _ in range(QA_NUM):
            q, a = random.choice(qa_list_raw)
            if SHUFFLE_QA:
                q, a = _shuffle_options(q, a)
            prompt = _build_prompt(solution_str, q)
            inputs.append(prompt)
            answers.append(a)
            questions.append(q)

    # 并发调用 vLLM
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
            # 题干 + 选项（即传给模型的 question 部分）
            "question": q_text,
            # 标准答案（选项字母）
            "answer": gt,
            # 模型原始输出（不再截断，通常只含一个选项字母）
            "prediction": output_text.strip() if output_text else "",
            # 是否答对
            "is_correct": bool(is_correct),
        })

    # 使用原始正确率作为 reward（不再额外缩放，后续 advantage 会做归一化）
    score = (sum(correct_list) / len(correct_list)) if correct_list else 0.0

    return {
        "score": score,
        "correct_count": sum(correct_list),
        "total_count": len(correct_list),
        "details": details,
    }


def _log_reward(caption: str, result: Dict[str, Any]):
    """记录 reward 到本地文件（仅当 REWARD_LOG_TO_FILE=1 时写入）"""
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
    """将 VERL 的 ground_truth（List[Dict]）转为 OpenRLHF server 需要的 [(question_full, answer), ...]"""
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
    调用 OpenRLHF reward server（run_reward_8gpu.sh 起的 serve_rm）。
    请求体与 OpenRLHF 一致: {"prompts": [[caption, [[q,a],[q,a],...]], ...], "query": [], "labels": []}
    返回: (final_rewards, qa_rewards, format_rewards_or_None)；旧版 server 无 qa/format 字段时 qa=final。
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
    调用 vl_judge 模式的 reward server。
    请求体: {"score_mode": "vl_judge", "samples": [{"caption": ..., "video_path": ...}, ...]}
    返回: (final_rewards, judge_rewards)
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


# def _call_openrlhf_reward_server(solution_str_list: List[str], ground_truth_list: List[Any], url: str) -> List[float]:
#     prompts_payload = []
#     for solution_str, ground_truth in zip(solution_str_list, ground_truth_list):
#         qa_list = _ground_truth_to_qa_list(ground_truth)
#         if not qa_list:
#             prompts_payload.append([solution_str, []])
#         else:
#             prompts_payload.append([solution_str, qa_list])
#     payload = {"prompts": prompts_payload, "query": [], "labels": []}
    
#     max_retries = 5
#     last_error = None
    
#     for attempt in range(max_retries):
#         try:
#             print(f"[DEBUG reward_fn] Attempt {attempt+1}: Sending request to {url} with {len(prompts_payload)} samples", flush=True)
#             resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=1800)
#             print(f"[DEBUG reward_fn] Response status: {resp.status_code}", flush=True)
            
#             if resp.status_code != 200:
#                 # 打印出错的样本内容
#                 print(f"[DEBUG reward_fn] ===== ERROR SAMPLE =====", flush=True)
#                 print(f"[DEBUG reward_fn] solution_str (first 500 chars): {solution_str_list[0][:500] if solution_str_list else 'EMPTY'}", flush=True)
#                 print(f"[DEBUG reward_fn] ground_truth: {ground_truth_list[0] if ground_truth_list else 'EMPTY'}", flush=True)
#                 print(f"[DEBUG reward_fn] Error response: {resp.text[:1000]}", flush=True)
#                 print(f"[DEBUG reward_fn] ===== END ERROR SAMPLE =====", flush=True)
                
#                 last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
#                 if attempt < max_retries - 1:
#                     time.sleep(2 * (attempt + 1))
#                     continue
#                 resp.raise_for_status()
            
#             data = resp.json()
#             rewards = data.get("rewards", [])
#             if len(rewards) != len(solution_str_list):
#                 raise ValueError(f"Reward server returned {len(rewards)} rewards for {len(solution_str_list)} samples")
#             return [float(r) for r in rewards]
            
#         except requests.exceptions.RequestException as e:
#             print(f"[DEBUG reward_fn] Attempt {attempt+1} failed: {e}", flush=True)
#             print(f"[DEBUG reward_fn] Failed sample solution_str: {solution_str_list[0][:500] if solution_str_list else 'EMPTY'}", flush=True)
#             last_error = str(e)
#             if attempt < max_retries - 1:
#                 time.sleep(2 * (attempt + 1))
#                 continue
#             raise
    
#     raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_error}")


def compute_score(
    solution_str: Union[str, List[str], None] = None,
    ground_truth: Union[Any, List[Any], None] = None,
    extra_info: Dict[str, Any] = None,
    **kwargs,
) -> Union[float, Dict[str, float], List[Dict[str, float]]]:
    """verl 框架调用的 reward function。支持 qa / vl_judge 两种评分模式，支持本地 vLLM 或远程 reward server。"""
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

    # ===================== vl_judge 模式 =====================
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

    # ===================== qa 模式（原逻辑） =====================
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

    # 本地 vLLM 计算（与 OpenRLHF 逻辑对齐）
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
    在当前节点上启动一组 vLLM OpenAI API server，供本文件中的 reward 函数使用。
    逻辑等价于原来 CapRL 的 reward_node_start_vllm.sh，但直接用 Python 实现，方便统一管理。
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
        # 给每个进程一点启动时间，避免端口竞争和显存瞬时峰值
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

    # 单节点多卡 vLLM OpenAI API server（供 reward_fn 调用）
    p_vllm = subparsers.add_parser(
        "start_vllm",
        help="Start a local multi-GPU vLLM OpenAI API cluster for reward computation.",
    )
    p_vllm.add_argument(
        "--model",
        type=str,
        default=None,
        help="vLLM 模型路径；默认使用 REWARD_VLLM_MODEL 环境变量或本文件中的 REWARD_MODEL。",
    )
    p_vllm.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="要启动的 vLLM 实例数量（每个实例绑定一张 GPU）。",
    )
    p_vllm.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="第一个 vLLM 实例监听的端口，后续依次加 1。",
    )
    p_vllm.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.85,
        help="vLLM 的 --gpu-memory-utilization。",
    )
    p_vllm.add_argument(
        "--max-model-len",
        type=int,
        default=12288,
        help="vLLM 的 --max-model-len。",
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
