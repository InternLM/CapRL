
#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Script: eval_unified_submit.sh
# Purpose: 批量提交多组基准测试任务到 rjob 集群，统一传递 bench_idx 参数。
# Usage: 直接执行此脚本即可。
# -----------------------------------------------------------------------------
# set -euo pipefail

# ------------------------- 配置区（可按需调整） ------------------------------
# 基准测试名称列表，对应 bench_idx 下标
TAG_LIST=(
  "chartqa"
  "infovqa"
  "MMStar"
  "mmmu"
  "CharXiv"
  "ChartQA_Pro"
  "MMMU_Pro"
  "seed2k"
  "MathVerse_MINIVInt"
  "MathVision"
  "WeMath"
  "VisuLogic"
)

# 需要运行的 bench_idx 索引，可自定义
bench_idx_list=(0 1 2 3)

# rjob 资源配置
GPU=1
MEMORY=200000   # 单位：MiB
CPU=20
GROUP="mllmexp_gpu"
IMAGE="IMAGE-TYPE"
MOUNT="gpfs://gpfs1/mllm:/mnt/shared-storage-user/mllm"
SCRIPT_PATH="/path/scripts/bash.sh"

# ------------------------------ 主逻辑 --------------------------------------
for bench_idx in "${bench_idx_list[@]}"; do
  TAG="${TAG_LIST[$bench_idx]}"
  JOB_NAME="eval-unified-${TAG}"
  echo "🔧 正在提交作业: ${JOB_NAME} (bench_idx=${bench_idx})"

  rjob submit \
    --name="${JOB_NAME}" \
    --gpu="${GPU}" \
    --memory="${MEMORY}" \
    --cpu="${CPU}" \
    --charged-group="${GROUP}" \
    --private-machine=group \
    --mount="${MOUNT}" \
    --image="${IMAGE}" \
    -P 12 \
    --host-network=true \
    -e DISTRIBUTED_JOB=true \
    -- bash -ex "${SCRIPT_PATH}" \
        "${bench_idx}" &

done

# -------------------------------- 结束 --------------------------------------
# bench idx
# rjob名称
# script 路径
# 节点数量/STEPS
# 模型路径
