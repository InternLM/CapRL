#!/usr/bin/env bash

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

# bench_idx
bench_idx_list=(0 1 2 3)


GPU=1
MEMORY=200000  
CPU=20
GROUP="mllmexp_gpu"
IMAGE="IMAGE-TYPE"
MOUNT="gpfs://gpfs1/mllm:/mnt/shared-storage-user/mllm"
SCRIPT_PATH="/path/scripts/bash.sh"

# --------------------------------------------------------------------
for bench_idx in "${bench_idx_list[@]}"; do
  TAG="${TAG_LIST[$bench_idx]}"
  JOB_NAME="eval-unified-${TAG}"
  echo "🔧 launch: ${JOB_NAME} (bench_idx=${bench_idx})"

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

