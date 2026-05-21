#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prism Stage 1 adapter for Tarsier2 native inference."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from typing import Any, Dict, List

import torch
import yaml
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import eval_prism_video_benchmarks_vlmevalkit as prism_eval


def _process_one_tarsier(model, processor, prompt: str, video_path: str, generate_kwargs: dict) -> str:
    from dataset.utils import format_one_sample

    sample = format_one_sample(video_path, prompt)
    batch_data = processor(sample)
    model_inputs: Dict[str, Any] = {}
    for k, v in batch_data.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**model_inputs, **generate_kwargs)
    input_len = model_inputs["input_ids"][0].shape[0]
    output_text = processor.processor.tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    )
    return output_text.strip()


def _worker_dp(
    rank: int,
    shard: List[Tuple[int, Dict[str, Any]]],
    tarsier_root: str,
    model_name_or_path: str,
    config_path: str,
    benchmark: str,
    generate_kwargs: dict,
    gen_num: int,
    out_path: str,
) -> None:
    sys.path.insert(0, tarsier_root)
    os.chdir(tarsier_root)
    from tasks.utils import load_model_and_processor

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    data_config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model, processor = load_model_and_processor(model_name_or_path, data_config)
    prompt_text = prism_eval.caption_instruction_for_benchmark(benchmark)

    records: List[Dict[str, Any]] = []
    for global_idx, sample in tqdm(shard, desc=f"Tarsier rank {rank}"):
        try:
            video_path = prism_eval.ensure_video_path(sample)
        except Exception as e:
            tqdm.write(f"[rank {rank}] idx={global_idx} ensure_video_path: {e}")
            cap = ""
        else:
            try:
                cap = _process_one_tarsier(model, processor, prompt_text, video_path, generate_kwargs)
            except Exception as e:
                tqdm.write(f"[rank {rank}] idx={global_idx} generate: {e}")
                cap = ""
        records.append({"_gidx": global_idx, "sample": sample, "captions": [cap] * gen_num})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)


def _merge_dp_parts(paths: List[str], gen_num: int) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            flat.extend(json.load(f))
    flat.sort(key=lambda r: int(r["_gidx"]))
    out: List[Dict[str, Any]] = []
    for row in flat:
        caps = row["captions"]
        if len(caps) < gen_num:
            last = caps[-1] if caps else ""
            caps = caps + [last] * (gen_num - len(caps))
        else:
            caps = caps[:gen_num]
        out.append({"sample": row["sample"], "captions": caps})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prism Stage1 with Tarsier2")
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--data-path", type=str, required=True)
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--max-num", type=int, default=-1)
    ap.add_argument("--num-frames", type=int, default=128, help="Recorded in metadata; Tarsier sampling is controlled by its config")
    ap.add_argument("--gen-num", type=int, default=4, help="Number of captions expected by Prism; one Tarsier output is repeated if needed")
    ap.add_argument("--model-name-or-path", type=str, required=True)
    ap.add_argument(
        "--tarsier-root",
        type=str,
        default=os.environ.get("TARSIER_ROOT", ""),
        help="Tarsier repository root containing tasks/, dataset/, and configs/",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Tarsier yaml config; defaults to <tarsier-root>/configs/tarser2_default_config.yaml",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--gpu", type=str, default=None, help="Comma-separated GPU ids for data parallelism; single process if unset")
    args = ap.parse_args()

    tarsier_root = os.path.abspath(args.tarsier_root)
    if not os.path.isdir(tarsier_root):
        print(f"ERROR: tarsier-root not found: {tarsier_root}", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, tarsier_root)

    cfg = args.config or os.path.join(tarsier_root, "configs", "tarser2_default_config.yaml")
    cfg = os.path.abspath(cfg)
    if not os.path.isfile(cfg):
        print(f"ERROR: config not found: {cfg}", file=sys.stderr)
        sys.exit(1)

    bench = args.benchmark.strip().lower()
    samples = prism_eval.load_benchmark_samples(bench, args.data_path)
    if args.max_num > 0:
        samples = samples[: args.max_num]
    n = len(samples)
    print(f"Loaded {n} samples for benchmark={bench}")

    os.makedirs(args.save_dir, exist_ok=True)
    intermediate_path = os.path.join(args.save_dir, f"intermediate_{bench}_step{args.step}.json")

    do_sample = args.temperature > 0
    gen_kw = {
        "do_sample": do_sample,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "use_cache": True,
    }
    if do_sample:
        gen_kw["temperature"] = args.temperature

    gpu_list = None
    if args.gpu:
        gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    n_gpus = len(gpu_list) if gpu_list else 1

    indexed = list(enumerate(samples))

    if n_gpus <= 1:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if gpu_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        os.chdir(tarsier_root)
        from tasks.utils import load_model_and_processor

        data_config = yaml.safe_load(open(cfg, "r", encoding="utf-8"))
        model, processor = load_model_and_processor(args.model_name_or_path, data_config)
        prompt_text = prism_eval.caption_instruction_for_benchmark(bench)
        caption_records: List[Dict[str, Any]] = []
        for sample in tqdm(samples, desc="Tarsier Prism Stage1"):
            try:
                vpath = prism_eval.ensure_video_path(sample)
            except Exception as e:
                tqdm.write(f"ensure_video_path: {e}")
                cap = ""
            else:
                try:
                    cap = _process_one_tarsier(model, processor, prompt_text, vpath, gen_kw)
                except Exception as e:
                    tqdm.write(f"generate: {e}")
                    cap = ""
            one_caption = cap
            caps = [one_caption] * args.gen_num
            caption_records.append({"sample": sample, "captions": caps})
        prism_eval.save_json(intermediate_path, caption_records)
        print(f"Saved intermediate -> {intermediate_path}")
        return

    chunks: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(n_gpus)]
    for i, s in indexed:
        chunks[i % n_gpus].append((i, s))

    tmp_base = intermediate_path + ".dp_part"
    ctx = multiprocessing.get_context("spawn")
    procs: List[multiprocessing.Process] = []
    for rank in range(n_gpus):
        if not chunks[rank]:
            continue
        out_p = f"{tmp_base}.rank_{rank}.json"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[rank]
        p = ctx.Process(
            target=_worker_dp,
            args=(
                rank,
                chunks[rank],
                tarsier_root,
                args.model_name_or_path,
                cfg,
                bench,
                gen_kw,
                args.gen_num,
                out_p,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Tarsier DP worker failed with exit code {p.exitcode}")

    part_paths = [f"{tmp_base}.rank_{r}.json" for r in range(n_gpus) if os.path.isfile(f"{tmp_base}.rank_{r}.json")]
    merged = _merge_dp_parts(part_paths, args.gen_num)
    if len(merged) != n:
        print(f"[WARN] merged records {len(merged)} != num samples {n}", flush=True)
    prism_eval.save_json(intermediate_path, merged)
    for pp in part_paths:
        try:
            os.remove(pp)
        except OSError:
            pass
    print(f"Saved intermediate -> {intermediate_path}")


if __name__ == "__main__":
    main()
