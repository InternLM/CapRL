#!/bin/bash

# ========== Resource Config ==========
gpu_count=8
cpu_count=$(( 20 * gpu_count ))
memory_mb=$(( 240000 * gpu_count ))

# ========== Job Config ==========
job_name="reward-server-qwen2-5-3b"
script_path="scripts/qwen2.5vl3b_75k_reward_qwen2.5_3b/reward/server.sh"

echo "Job Name: $job_name"
echo "Script Path: $script_path"

# ========== Submit Job ==========
# Modify the following command according to your cluster scheduler (e.g., slurm, rjob, etc.)
# Example with slurm:
#   srun --job-name=$job_name --gres=gpu:$gpu_count --nodes=1 \
#        --ntasks-per-node=1 bash $script_path

# Example with rjob (internal):
rjob submit --name=$job_name \
  --gpu=$gpu_count \
  --memory=$memory_mb \
  --cpu=$cpu_count \
  -P 1 \
  --host-network=true \
  -e DISTRIBUTED_JOB=true \
  -- bash -ex $script_path
