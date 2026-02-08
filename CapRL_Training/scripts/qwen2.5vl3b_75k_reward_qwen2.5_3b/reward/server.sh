#!/bin/bash

# ========== Environment Setup ==========
# Adjust CUDA paths according to your environment
# export PATH=/path/to/cuda-12.8/bin:$PATH
# export LD_LIBRARY_PATH=/path/to/cuda-12.8/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/path/to/cuda-12.8

# Activate conda environment
# conda activate CapRL

# Set working directory to CapRL_Training root
WORK_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
cd $WORK_DIR

# ========== Paths (modify these) ==========
REWARD_MODEL="/path/to/Qwen2.5-3B-Instruct"

# ========== Launch Reward Server ==========
# Start worker process (background)
python reward_server/serve_rm.py \
  --num_workers 8 \
  --tp 1 \
  --port 8889 \
  --worker_base_port 8899 \
  --reward_pretrain $REWARD_MODEL \
  --role worker &

# Start master process (foreground)
python reward_server/serve_rm.py \
  --num_workers 8 \
  --tp 1 \
  --port 8889 \
  --worker_base_port 8899 \
  --reward_pretrain $REWARD_MODEL \
  --role master \
  --worker_hosts 0.0.0.0
