#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prism Stage 1 adapter for official TimeLens inference."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import eval_prism_video_benchmarks_vlmevalkit as prism_eval


def _video_frame_count(video_path: str) -> Optional[int]:
    """Return the decord-readable frame count, or None on failure."""
    try:
        import decord

        vr = decord.VideoReader(video_path)
        n = len(vr)
        return int(n) if n > 0 else None
    except Exception:
        return None


def run_timelens_stage1(
    samples: List[Dict[str, Any]],
    args: Any,
    log_prefix: str = "[Stage1-official]",
) -> List[Dict[str, Any]]:
    """Run the same path used by --caption-backend timelens_official."""
    import time
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    dtype_name = str(getattr(args, "caption_official_dtype", "bfloat16")).lower()
    dtype_map: Dict[str, Any] = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map.get(dtype_name, torch.bfloat16)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": getattr(args, "caption_official_device_map", "auto"),
    }
    model_kwargs["dtype"] = model_dtype
    attn_impl = getattr(args, "caption_official_attn_impl", "")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model_path = getattr(args, "caption_model_path", "") or ""
    if not model_path:
        raise ValueError("caption_model_path is required for TimeLens Stage1")

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    benchmark = str(getattr(args, "benchmark", "")).strip().lower()
    num_frames = int(getattr(args, "num_frames", 128))
    gen_num = int(getattr(args, "gen_num", 4))
    caption_batch_size = int(getattr(args, "caption_batch_size", 8))

    sample_fps_opt = getattr(args, "video_sample_fps", None)
    use_fps_mode = sample_fps_opt is not None and float(sample_fps_opt) > 0
    sample_fps_val = float(sample_fps_opt) if use_fps_mode else 0.0

    def _build_messages(video_path: str) -> List[Dict[str, Any]]:
        if use_fps_mode:
            video_item: Dict[str, Any] = {
                "type": "video",
                "video": video_path,
                "fps": sample_fps_val,
                "max_frames": num_frames,
            }
            vmf = getattr(args, "video_min_frames", None)
            if vmf is not None and int(vmf) > 0:
                video_item["min_frames"] = int(vmf)
        else:
            total_frames = _video_frame_count(video_path)
            if total_frames is None:
                n_use = num_frames
            else:
                n_use = min(total_frames, num_frames)
            video_item = {
                "type": "video",
                "video": video_path,
                "nframes": n_use,
            }
        min_pixels = int(getattr(args, "caption_official_min_pixels", 0))
        total_pixels = int(getattr(args, "caption_official_total_pixels", 0))
        if min_pixels > 0:
            video_item["min_pixels"] = min_pixels
        if total_pixels > 0:
            video_item["total_pixels"] = total_pixels
        return [
            {
                "role": "user",
                "content": [
                    video_item,
                    {"type": "text", "text": prism_eval.caption_instruction_for_benchmark(benchmark)},
                ],
            }
        ]

    input_device = getattr(args, "caption_official_input_device", "auto")
    if input_device == "auto":
        input_device = "cuda" if torch.cuda.is_available() else "cpu"

    caption_records: List[Dict[str, Any]] = []
    total_batches = (len(samples) + caption_batch_size - 1) // caption_batch_size if samples else 0
    start_time = time.time()
    for batch_idx, i in enumerate(range(0, len(samples), caption_batch_size), start=1):
        batch = samples[i : i + caption_batch_size]
        for s in batch:
            video_path = prism_eval.ensure_video_path(s)
            try:
                messages = _build_messages(video_path)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=int(getattr(args, "caption_official_image_patch_size", 16)),
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                video_metadatas = None
                if videos is not None:
                    videos, video_metadatas = zip(*videos)
                    videos, video_metadatas = list(videos), list(video_metadatas)
                inputs = processor(
                    text=[text],
                    images=images,
                    videos=videos,
                    video_metadata=video_metadatas,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                ).to(input_device)

                sample_captions: List[str] = []
                cap_temp = float(getattr(args, "caption_temperature", 0.7))
                do_sample = cap_temp > 0
                gen_kwargs: Dict[str, Any] = {
                    "do_sample": do_sample,
                    "max_new_tokens": int(getattr(args, "caption_max_tokens", 2048)),
                }
                if do_sample:
                    gen_kwargs["temperature"] = cap_temp
                    gen_kwargs["top_p"] = float(getattr(args, "caption_top_p", 0.8))
                with torch.inference_mode():
                    for _ in range(gen_num):
                        output_ids = model.generate(**inputs, **gen_kwargs)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
                        ]
                        text_out = processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )[0].strip()
                        sample_captions.append(text_out if text_out else "[empty_caption]")
                caption_records.append({"sample": s, "captions": sample_captions})
            except Exception as e:
                print(f"[WARN] official Stage1 failed on {video_path}: {e}")
                caption_records.append({"sample": s, "captions": ["[video_official_failed]"] * gen_num})
        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == total_batches:
            prism_eval._log_batch_progress(log_prefix, batch_idx, total_batches, start_time)
    return caption_records


def _worker_timelens_entry(
    rank: int,
    gpu_id: str,
    indexed_shard: List[Tuple[int, Dict[str, Any]]],
    args_dict: Dict[str, Any],
    out_path: str,
) -> None:
    """Worker process with one visible GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    samples = [s for _, s in indexed_shard]
    records = run_timelens_stage1(samples, args, log_prefix=f"[TimeLens rank{rank}]")
    out_rows = []
    for (gidx, _), rec in zip(indexed_shard, records):
        out_rows.append({"_gidx": gidx, "sample": rec["sample"], "captions": rec["captions"]})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False)


def _merge_timelens_dp(part_paths: List[str]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for p in part_paths:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                flat.extend(json.load(f))
    flat.sort(key=lambda x: int(x["_gidx"]))
    return [{"sample": x["sample"], "captions": x["captions"]} for x in flat]


def main() -> None:
    ap = argparse.ArgumentParser(description="Prism Stage1: TimeLens official (no vLLM)")
    ap.add_argument("--benchmark", type=str, required=True)
    ap.add_argument("--data-path", type=str, required=True)
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--max-num", type=int, default=-1)
    ap.add_argument(
        "--num-frames",
        type=int,
        default=128,
        help="Target frame count; also used as max_frames when --video-sample-fps is set.",
    )
    ap.add_argument(
        "--video-sample-fps",
        type=float,
        default=None,
        metavar="FPS",
        help="Positive FPS enables FPS sampling; --num-frames is used as max_frames.",
    )
    ap.add_argument(
        "--video-min-frames",
        type=int,
        default=None,
        help="min_frames for FPS sampling; defaults to qwen_vl_utils behavior.",
    )
    ap.add_argument("--gen-num", type=int, default=4)
    ap.add_argument("--caption-batch-size", type=int, default=8)
    ap.add_argument("--caption-max-tokens", type=int, default=2048)
    ap.add_argument(
        "--caption-temperature",
        type=float,
        default=0.0,
        help="0 uses greedy decoding; values >0 enable sampling",
    )
    ap.add_argument("--caption-top-p", type=float, default=0.8)
    ap.add_argument("--model-name-or-path", type=str, required=True, dest="caption_model_path")
    ap.add_argument("--caption-official-dtype", type=str, default="bfloat16")
    ap.add_argument("--caption-official-attn-impl", type=str, default="flash_attention_2")
    ap.add_argument("--caption-official-device-map", type=str, default="auto")
    ap.add_argument("--caption-official-input-device", type=str, default="auto")
    ap.add_argument("--caption-official-image-patch-size", type=int, default=16)
    ap.add_argument("--caption-official-min-pixels", type=int, default=64 * 32 * 32)
    ap.add_argument("--caption-official-total-pixels", type=int, default=14336 * 32 * 32)
    ap.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Comma-separated GPU ids for data parallelism; single process if unset.",
    )
    args = ap.parse_args()
    if args.video_sample_fps is not None and args.video_sample_fps <= 0:
        ap.error("--video-sample-fps must be positive or omitted")

    bench = args.benchmark.strip().lower()
    samples = prism_eval.load_benchmark_samples(bench, args.data_path)
    if args.max_num > 0:
        samples = samples[: args.max_num]
    print(f"Loaded {len(samples)} samples for benchmark={bench}")

    os.makedirs(args.save_dir, exist_ok=True)
    intermediate_path = os.path.join(args.save_dir, f"intermediate_{bench}_step{args.step}.json")

    args.benchmark = bench
    args_dict = vars(args).copy()

    gpu_list = None
    if args.gpu:
        gpu_list = [x.strip() for x in args.gpu.split(",") if x.strip()]
    n_gpus = len(gpu_list) if gpu_list else 0

    if n_gpus > 1:
        indexed = list(enumerate(samples))
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
            p = ctx.Process(
                target=_worker_timelens_entry,
                args=(rank, gpu_list[rank], chunks[rank], args_dict, out_p),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"TimeLens DP worker failed with exit code {p.exitcode}")
        part_paths = [f"{tmp_base}.rank_{r}.json" for r in range(n_gpus) if os.path.isfile(f"{tmp_base}.rank_{r}.json")]
        records = _merge_timelens_dp(part_paths)
        for pp in part_paths:
            try:
                os.remove(pp)
            except OSError:
                pass
    else:
        if n_gpus == 1 and gpu_list:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list[0]
        if (
            torch.cuda.is_available()
            and torch.cuda.device_count() > 1
            and n_gpus <= 1
        ):
            print(
                "[WARN] More than one GPU is visible, but --gpu was not set. "
                "With device_map=auto, Transformers may shard the model and slow down inference. "
                "Expose one GPU or pass --gpu for data parallel workers.",
                flush=True,
            )
        records = run_timelens_stage1(samples, args, log_prefix="[TimeLens-Stage1]")

    with open(intermediate_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Saved intermediate -> {intermediate_path}")


if __name__ == "__main__":
    main()
