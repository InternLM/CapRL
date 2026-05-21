# Video Evaluation Prism

This repository evaluates video caption models with a two-stage Prism-style pipeline:

1. A video model reads the input video and generates one or more captions.
2. A text-only LLM receives the caption plus the original benchmark question and answers the question.
3. The final answer is scored against the benchmark annotation.

The setup is useful when you want to measure whether generated captions preserve enough visual information for downstream reasoning.
The current implementation supports the Qwen3-VL family, Tarsier2, and TimeLens inference paths.
For Prism evaluation of image caption models, see `/CapRL/Prism_Evaluation`.

## Supported Benchmarks

- Video-MME
- MVBench
- MMVU
- MMBench-Video
- TOMATO
- TimeLens-Bench

## Repository Layout

```text
.
├── eval_prism_video_benchmarks_vlmevalkit.py  # main evaluator
├── run_prism_stage1_tarsier.py                # Tarsier model Stage 1 adapter
├── run_prism_stage1_timelens.py               # TimeLens model Stage 1 adapter
├── scripts/
│   ├── run_vllm_prism.sh                      # end-to-end vLLM server pipeline
│   ├── run_stage2_only.sh                     # rerun Stage 2 from saved captions
│   └── vllm_topology_utils.sh                 # vLLM/Nginx helper functions
├── tools/                                     # judge tools
├── examples/env.example
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

The scripts assume an OpenAI-compatible vLLM server interface. For multi-GPU local serving, install `nginx` and make sure `vllm serve` is available in the active environment.

## Quick Start

Edit or source `examples/env.example`, then run:

```bash
source examples/env.example
bash scripts/run_vllm_prism.sh
```

Required environment variables:

```bash
export CAPTION_MODEL_PATH="/path/to/video-caption-model"
export DOWNSTREAM_MODEL_PATH="/path/to/text-llm"
export DATA_PATH="/path/to/benchmark.tsv"
export BENCHMARK="videomme"
```

Common runtime variables:

```bash
export SAVE_DIR="./outputs/videomme"
export NUM_GPUS="8"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUM_FRAMES_SERVER="128"
export ALLOWED_MEDIA_PATH="/path/to/datasets"
```

`ALLOWED_MEDIA_PATH` must be a parent directory that contains the local video files. vLLM uses it to allow `file://` video access when Stage 1 sends video paths to the server.

## Running the Python Evaluator Directly

Stage 1 only:

```bash
python eval_prism_video_benchmarks_vlmevalkit.py \
  --benchmark videomme \
  --data-path /path/to/Video-MME.tsv \
  --caption-model-path /path/to/video-model \
  --downstream-model-path /path/to/text-llm \
  --save-dir outputs/videomme \
  --vllm-api-base http://127.0.0.1:8000/v1 \
  --caption-model-name /path/to/video-model \
  --stage-num 1
```

Stage 2 only:

```bash
INTERMEDIATE_PATH=outputs/videomme/intermediate_videomme_step0.json \
DOWNSTREAM_MODEL_PATH=/path/to/text-llm \
BENCHMARK=videomme \
bash scripts/run_stage2_only.sh
```

## Dataset Paths

The evaluator tries to use paths in the annotation file first. If a benchmark stores relative paths or URLs, set the corresponding root:

```bash
export VIDEOMME_VIDEO_ROOT="/path/to/Video-MME/videos"
export MVBench_ROOT="/path/to/MVBench"
export MVBENCH_META_ROOT="/path/to/MVBench/json"
export MMVU_VIDEO_ROOT="/path/to/MMVU/videos"
export MMBENCH_VIDEO_ROOT="/path/to/MMBench-Video/videos"
export TOMATO_ROOT="/path/to/TOMATO"
export TOMATO_VIDEO_ROOT="/path/to/TOMATO/videos"
export TIMELENS_VIDEO_ROOT="/path/to/TimeLens-Bench/videos"
```

Benchmark-specific notes:

- Video-MME: `VIDEOMME_VIDEO_ROOT` should contain `<video_id>.mp4` files when the TSV does not provide usable absolute paths.
- MVBench: `MVBench_ROOT` should be the root used to resolve `prefix/video` fields in `MVBench.tsv`; `MVBENCH_META_ROOT` is only used to recover options/task metadata for legacy JSON inputs.
- MMVU: `MMVU_VIDEO_ROOT` should contain the relative paths extracted from `/videos/...` URLs in `validation.json`.
- MMBench-Video: `MMBENCH_VIDEO_ROOT` should contain the video files named by the TSV `video_path` basename.
- TOMATO: set `TOMATO_VIDEO_ROOT` directly, or set `TOMATO_ROOT` and keep videos under `$TOMATO_ROOT/videos`.
- TimeLens-Bench: `TIMELENS_VIDEO_ROOT` should contain `charades/`, `activitynet/`, and `qvhighlights/` subdirectories.

## Judge API

MMVU and MMBench-Video may require an LLM judge during Stage 2. Use any OpenAI-compatible API:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

On machines without network access, run inference with `--skip-judge`, then score the saved inference JSON with:

```bash
python tools/judge_mmvu.py --inference-path outputs/mmvu/mmvu_inference_step0.json
python tools/judge_mmbench_video.py --inference-path outputs/mmbench_video/mmbench_video_inference_step0.json
```

## Outputs

Each run writes:

- `intermediate_<benchmark>_step<N>.json`: Stage 1 captions.
- `prism_<benchmark>_details_step<N>.json`: per-sample captions, answers, and scores.
- `prism_<benchmark>_summary_step<N>.json`: aggregate metrics.

The main metrics are `BoN` and `M_Acc`; benchmark-specific default metrics are also written when available.
