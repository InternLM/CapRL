#!/bin/bash
# Stage 0: Initial Alignment with LLaVA-558K
# This script uses LLaMA-Factory for training.
# Please install LLaMA-Factory first: https://github.com/hiyouga/LLaMA-Factory

# ========== Paths (modify these) ==========
LLAMAFACTORY_DIR="/path/to/LLaMA-Factory"
MODEL_PATH="/path/to/Qwen2.5_VL_3B_rdm_merger_ori_llm"  # See initiallize_vlm_3b.ipynb

cd $LLAMAFACTORY_DIR

# ========== Conda Environment ==========
# conda activate llamafactory
echo "current conda env:$CONDA_DEFAULT_ENV"

# ========== Job Config ==========
export JOB_NAME=pt_llava558
export Pretrain_SAVE_PATH=Qwen2.5-VL-3B-pretrain-${JOB_NAME}

# ========== Wandb Config ==========
wandb login
export WANDB_MODE=offline
export WANDB_PROJECT=${Pretrain_SAVE_PATH}

# ========== Cache ==========
export tokenized_path=${LLAMAFACTORY_DIR}/cache/caprl_align_cache
mkdir -p "$tokenized_path"

set -x -e

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ========== Distributed Config ==========
export NNODES=${NODE_COUNT:-1}
export GPUS_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_PORT=29501
echo "===================================="
echo "NNODES=${NNODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "===================================="

# ========== Training Hyperparameters ==========
export full_batch_size=256
export per_device_batch_size=8
export gradient_accumulation_steps=$((full_batch_size / (per_device_batch_size * GPUS_PER_NODE * NNODES)))
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"

export lr=1e-3
export image_max_pixels=$((1024 * 28 * 28))
export cutoff_len=6144
export output_dir=ckpt/${Pretrain_SAVE_PATH}
export model_name_or_path=${MODEL_PATH}

# ========== Launch Training ==========
torchrun \
    --nnodes $NNODES --nproc_per_node $GPUS_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    ${LLAMAFACTORY_DIR}/src/train.py \
    --deepspeed ${LLAMAFACTORY_DIR}/examples/deepspeed/ds_z3_config.json \
    --model_name_or_path $model_name_or_path \
    --image_max_pixels ${image_max_pixels} \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --freeze_vision_tower true \
    --freeze_multi_modal_projector false \
    --freeze_language_model true \
    --dataset llava_558k \
    --template qwen2_vl \
    --cutoff_len ${cutoff_len} \
    --overwrite_cache true \
    --tokenized_path ${tokenized_path} \
    --preprocessing_num_workers 64 \
    --dataloader_num_workers 16 \
    --output_dir $output_dir \
    --logging_steps 1 \
    --save_steps 10000 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate ${lr} \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --save_total_limit 1 \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --report_to wandb
