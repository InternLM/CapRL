#!/usr/bin/env python3
"""Standalone MMBench-Video judge for saved Prism inference outputs."""

import argparse
import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import httpx
from openai import OpenAI
from tqdm import tqdm

_http_client = httpx.Client(proxy=None, timeout=60.0)
gpt_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    http_client=_http_client,
)

MMBENCH_VIDEO_SYSTEM_PROMPT = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""

MMV_DIMENSIONS = {
    "CP": ["Video Topic", "Video Emotion", "Video Scene", "Video Style"],
    "FP-S": ["OCR", "Object Recognition", "Attribute Recognition", "Event Recognition", "Human Motion", "Counting"],
    "FP-C": ["Spatial Relationship", "Human-object Interaction", "Human Interaction"],
    "HL": ["Hallucination"],
    "LR": ["Structuralized Image-Text Understanding", "Mathematical Calculation"],
    "AR": ["Physical Property", "Function Reasoning", "Identity Reasoning"],
    "RR": ["Natural Relation", "Physical Relation", "Social Relation"],
    "CSR": ["Common Sense Reasoning"],
    "TR": ["Counterfactual Reasoning", "Causal Reasoning", "Future Prediction"],
}
L3_DIMS = []
for k, v in MMV_DIMENSIONS.items():
    L3_DIMS.extend(v)
MMV_DIMENSIONS["Perception"] = []
MMV_DIMENSIONS["Reasoning"] = []
MMV_DIMENSIONS["Overall"] = []
for k in ["CP", "FP-C", "FP-S", "HL"]:
    MMV_DIMENSIONS["Perception"].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS["Overall"].extend(MMV_DIMENSIONS[k])
for k in ["LR", "AR", "RR", "CSR", "TR"]:
    MMV_DIMENSIONS["Reasoning"].extend(MMV_DIMENSIONS[k])
    MMV_DIMENSIONS["Overall"].extend(MMV_DIMENSIONS[k])  # Overall = Perception ∪ Reasoning


def _parse_score_from_response(content: str) -> int:
    """Parse a 0-3 score from the judge response. Return -1 on failure."""
    content = (content or "").strip()
    m = re.search(r"\b([0-3])\b", content)
    if m:
        return int(m.group(1))
    return -1


def _build_judge_prompt(question: str, answer: str, prediction: str) -> str:
    """Build the VLMEvalKit-style judge prompt."""
    return f"Question: {question}\nGroundtruth answer: {answer}\nCandidate answer: {prediction}\nYour response: "


def _process_one_judge(inputs: Tuple[int, str, str, str, str]) -> Tuple[int, int]:
    """Judge one item and return (idx, score 0-3 or -1)."""
    idx, question, answer, prediction, judge_model = inputs
    if not prediction or "[api_error" in prediction or "[video_" in prediction:
        return idx, -1
    user = _build_judge_prompt(question, answer, prediction)
    messages = [{"role": "system", "content": MMBENCH_VIDEO_SYSTEM_PROMPT}, {"role": "user", "content": user}]
    try:
        out = gpt_client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )
        content = out.choices[0].message.content or ""
        return idx, _parse_score_from_response(content)
    except Exception:
        return idx, -1


def _eval_mmbench_video_with_llm_judge(
    items: List[Tuple[int, str, str, str]],
    judge_model: str,
    max_concurrency: int = 16,
) -> List[int]:
    """Score flattened judge items and restore scores by item index."""
    inputs = [(idx, q, a, p, judge_model) for idx, q, a, p in items]
    results: List[int] = [-1] * len(items)
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {executor.submit(_process_one_judge, inp): inp for inp in inputs}
        for future in tqdm(as_completed(futures), total=len(inputs), desc="GPT Judge"):
            idx, score = future.result()
            results[idx] = score
    return results


@dataclass
class SampleResult:
    sample: Dict[str, Any]
    captions: List[str]
    responses: List[str]
    scores: List[int]  # 0-3 or -1

    @property
    def bon(self) -> float:
        valid = [s for s in self.scores if s >= 0]
        if not valid:
            return 0.0
        return float(max(valid)) / 3.0

    @property
    def m_acc(self) -> float:
        valid = [s for s in self.scores if s >= 0]
        if not valid:
            return 0.0
        return float(np.mean(valid)) / 3.0

    @property
    def best_score(self) -> int:
        """BoN score on the 0-3 scale."""
        valid = [s for s in self.scores if s >= 0]
        return max(valid) if valid else -1

    @property
    def mean_score(self) -> float:
        """Mean score on the 0-3 scale, or -1 if all scores failed."""
        valid = [s for s in self.scores if s >= 0]
        return float(np.mean(valid)) if valid else -1.0


def get_dimension_rating(sample_results: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Dict[str, str]]:
    """Aggregate dimensions following the VLMEvalKit get_dimension_rating logic."""
    coarse_rating: Dict[str, List[int]] = {k: [] for k in MMV_DIMENSIONS}
    fine_rating: Dict[str, List[int]] = {k: [] for k in L3_DIMS}

    for sample, score in sample_results:
        dims_raw = sample.get("dimensions", "[]")
        try:
            cates = ast.literal_eval(dims_raw) if isinstance(dims_raw, str) else dims_raw
        except Exception:
            cates = []
        for c in cates:
            if c in fine_rating:
                fine_rating[c].append(score)
        for d, dim_list in MMV_DIMENSIONS.items():
            if dim_list and any(x in dim_list for x in cates):
                coarse_rating[d].append(score)

    coarse_all = {k: f"{np.mean([max(x, 0) for x in v]):.3f}" if v else "0.000" for k, v in coarse_rating.items()}
    coarse_valid = {
        k: f"{np.mean([x for x in v if x >= 0]):.3f}" if [x for x in v if x >= 0] else "0.000"
        for k, v in coarse_rating.items()
    }
    fine_all = {k: f"{np.mean([max(x, 0) for x in v]):.3f}" if v else "0.000" for k, v in fine_rating.items()}
    fine_valid = {
        k: f"{np.mean([x for x in v if x >= 0]):.3f}" if [x for x in v if x >= 0] else "0.000"
        for k, v in fine_rating.items()
    }
    return dict(coarse_all=coarse_all, coarse_valid=coarse_valid, fine_all=fine_all, fine_valid=fine_valid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone MMBench-Video Prism judge")
    parser.add_argument("--inference-path", type=str, required=True,
                        help="Inference file, e.g. mmbench_video_inference_step{N}.json")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory; defaults to the inference file directory")
    parser.add_argument("--step", type=int, default=None,
                        help="Step id; parsed from the inference file name by default")
    parser.add_argument("--judge-model", type=str, default="gpt-4-turbo")
    parser.add_argument("--judge-concurrency", type=int, default=32)

    args = parser.parse_args()

    if not os.path.isfile(args.inference_path):
        raise FileNotFoundError(f"Inference file not found: {args.inference_path}")

    with open(args.inference_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise ValueError("Inference file is empty")

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.inference_path))
    os.makedirs(output_dir, exist_ok=True)

    step = args.step
    if step is None:
        m = re.search(r"step(\d+)", os.path.basename(args.inference_path))
        step = int(m.group(1)) if m else 0

    samples = [r["sample"] for r in records]
    all_responses = [r["responses"] for r in records]

    flattened: List[Tuple[int, str, str, str]] = []
    sizes: List[int] = []
    for sample, responses in zip(samples, all_responses):
        sizes.append(len(responses))
        for resp in responses:
            flattened.append((
                len(flattened),
                str(sample["question"]),
                str(sample["answer"]),
                str(resp),
            ))

    print(f"Loaded {len(records)} records from {args.inference_path}")
    print(f"Judge model: {args.judge_model}, total items: {len(flattened)}")

    scores_flat = _eval_mmbench_video_with_llm_judge(
        flattened,
        judge_model=args.judge_model,
        max_concurrency=args.judge_concurrency,
    )

    per_sample_scores: List[List[int]] = []
    idx = 0
    for sz in sizes:
        sub = scores_flat[idx : idx + sz]
        idx += sz
        per_sample_scores.append(sub)

    final_results: List[SampleResult] = []
    for rec, scores in zip(records, per_sample_scores):
        final_results.append(
            SampleResult(
                sample=rec["sample"],
                captions=rec["captions"],
                responses=rec["responses"],
                scores=scores,
            )
        )

    bon = float(np.mean([x.bon for x in final_results])) if final_results else 0.0
    m_acc = float(np.mean([x.m_acc for x in final_results])) if final_results else 0.0

    sample_with_score: List[Tuple[Dict[str, Any], float]] = [
        (r.sample, r.mean_score) for r in final_results
    ]
    dimension_rating = get_dimension_rating(sample_with_score)

    details = []
    for x in final_results:
        details.append({
            "sample": x.sample,
            "captions": x.captions,
            "responses": x.responses,
            "scores": x.scores,
            "bon": x.bon,
            "m_acc": x.m_acc,
        })

    summary = {
        "benchmark": "mmbench_video",
        "num_samples": len(final_results),
        "judge_model": args.judge_model,
        "BoN": bon,
        "M_Acc": m_acc,
        **dimension_rating,
    }

    detail_path = os.path.join(output_dir, f"prism_mmbench_video_details_step{step}.json")
    summary_path = os.path.join(output_dir, f"prism_mmbench_video_summary_step{step}.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved details -> {detail_path}")
    print(f"Saved summary -> {summary_path}")
    print(f"Final metrics: BoN={bon:.4f}, M_Acc={m_acc:.4f}")

    print("\n========== VLMEvalKit-aligned dimension scores (coarse_all) ==========")
    for k, v in sorted(dimension_rating.get("coarse_all", {}).items()):
        if k and v != "0.00":
            print(f"  {k}: {v}")
    print("\n========== coarse_valid ==========")
    for k, v in sorted(dimension_rating.get("coarse_valid", {}).items()):
        if k and v != "0.00":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
