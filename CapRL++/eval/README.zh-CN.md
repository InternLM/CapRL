# Video Evaluation Prism

本仓库用于评测视频 caption 模型，采用两阶段 Prism 流程：

1. 视频模型读取视频，生成一条或多条 caption。
2. 纯文本 LLM 接收 caption 和原 benchmark 问题，并回答问题。
3. 将最终答案与标注答案比较，得到 caption 对下游问答的支持能力。

这个评测关注的是：caption 是否保留了足够的视频信息，使文本模型能够完成原始视频理解任务。
目前支持Qwen3-VL系列，以及Tarsier2和TimeLens的推理。
关于Image Caption的Prism评测，请参考/CapRL/Prism_Evaluation。

## 支持的 Benchmark

- Video-MME
- MVBench
- MMVU
- MMBench-Video
- TOMATO
- TimeLens-Bench

## 目录结构

```text
.
├── eval_prism_video_benchmarks_vlmevalkit.py  # 主评测程序
├── run_prism_stage1_tarsier.py                # Tarsier Model Stage 1 适配器
├── run_prism_stage1_timelens.py               # TimeLens Model Stage 1 适配器
├── scripts/
│   ├── run_vllm_prism.sh                      # vLLM 端到端评测脚本
│   ├── run_stage2_only.sh                     # 从已生成 caption 重跑 Stage 2
│   └── vllm_topology_utils.sh                 # vLLM/Nginx 启动辅助 
├── tools/                                     # judge 工具
├── examples/env.example
└── requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

脚本默认使用 OpenAI 兼容的 vLLM 接口。若使用本地多 GPU 服务，请安装 `nginx`，并确保当前环境可以运行 `vllm serve`。

## 快速开始

先修改或加载 `examples/env.example`：

```bash
source examples/env.example
bash scripts/run_vllm_prism.sh
```

必须设置的变量：

```bash
export CAPTION_MODEL_PATH="/path/to/video-caption-model"
export DOWNSTREAM_MODEL_PATH="/path/to/text-llm"
export DATA_PATH="/path/to/benchmark.tsv"
export BENCHMARK="videomme"
```

常用运行参数：

```bash
export SAVE_DIR="./outputs/videomme"
export NUM_GPUS="8"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUM_FRAMES_SERVER="128"
export ALLOWED_MEDIA_PATH="/path/to/datasets"
```

`ALLOWED_MEDIA_PATH` 必须是本地视频文件所在目录的上级目录。Stage 1 通过 vLLM server 读取 `file://` 视频路径时，vLLM 会用这个参数限制允许访问的本地媒体目录。

## 直接运行 Python 主程序

只跑 Stage 1，生成 caption：

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

从已有 caption 重跑 Stage 2：

```bash
INTERMEDIATE_PATH=outputs/videomme/intermediate_videomme_step0.json \
DOWNSTREAM_MODEL_PATH=/path/to/text-llm \
BENCHMARK=videomme \
bash scripts/run_stage2_only.sh
```

## 数据路径

评测程序会优先使用标注文件中的路径。如果 benchmark 使用相对路径或 URL，可设置对应的视频根目录：

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

各 benchmark 路径说明：

- Video-MME：当 TSV 中没有可用绝对路径时，`VIDEOMME_VIDEO_ROOT` 下应包含 `<video_id>.mp4`。
- MVBench：`MVBench_ROOT` 用于解析 `MVBench.tsv` 中的 `prefix/video` 字段；`MVBENCH_META_ROOT` 只用于 legacy JSON 输入的选项和 task metadata 补全。
- MMVU：`MMVU_VIDEO_ROOT` 用于解析 `validation.json` 中 `/videos/...` URL 对应的相对路径。
- MMBench-Video：`MMBENCH_VIDEO_ROOT` 下应包含 TSV `video_path` basename 对应的视频文件。
- TOMATO：可直接设置 `TOMATO_VIDEO_ROOT`；或者设置 `TOMATO_ROOT`，并将视频放在 `$TOMATO_ROOT/videos`。
- TimeLens-Bench：`TIMELENS_VIDEO_ROOT` 下应包含 `charades/`、`activitynet/`、`qvhighlights/` 子目录。

## Judge API

MMVU 和 MMBench-Video 在进行Stage2时可能需要 LLM judge。可使用任意 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

如果 GPU 节点无法联网，可在推理时使用 `--skip-judge`，之后在可联网机器上打分：

```bash
python tools/judge_mmvu.py --inference-path outputs/mmvu/mmvu_inference_step0.json
python tools/judge_mmbench_video.py --inference-path outputs/mmbench_video/mmbench_video_inference_step0.json
```

## 输出文件

每次运行会写出：

- `intermediate_<benchmark>_step<N>.json`：Stage 1 生成的 captions。
- `prism_<benchmark>_details_step<N>.json`：逐样本 caption、回答和得分。
- `prism_<benchmark>_summary_step<N>.json`：汇总指标。

主要指标为 `BoN` 和 `M_Acc`，同时输出各benchmark的默认指标。
