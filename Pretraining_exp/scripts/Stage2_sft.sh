#!/bin/bash
# Stage 2: SFT with general instruction data
# This script uses LLaMA-Factory for training.
# Please install LLaMA-Factory first: https://github.com/hiyouga/LLaMA-Factory

# ========== Paths (modify these) ==========
LLAMAFACTORY_DIR="/path/to/LLaMA-Factory"

cd $LLAMAFACTORY_DIR

# ========== Conda Environment ==========
# conda activate llamafactory
echo "current conda env:$CONDA_DEFAULT_ENV"

# ========== Job Config ==========
export JOB_NAME=caprl_sft
export Pretrain_SAVE_PATH=Qwen2.5-VL-3B-pretrain-caprl_further_pretrain
export Finetune_SAVE_PATH=Qwen2.5-VL-3B-finetune-${JOB_NAME}

# ========== Wandb Config ==========
wandb login
export WANDB_MODE=offline
export WANDB_PROJECT=${Finetune_SAVE_PATH}

# ========== Cache ==========
export tokenized_path=${LLAMAFACTORY_DIR}/cache/caprl_sft_cache
mkdir -p "$tokenized_path"

set -x -e

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ========== Distributed Config ==========
export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR}
export NODE_RANK=${NODE_RANK}
export MASTER_PORT=29503
echo "===================================="
echo "NNODES=${NNODES}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "NODE_RANK=${NODE_RANK}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "===================================="

# ========== Training Hyperparameters ==========
export full_batch_size=128
export per_device_batch_size=4
export gradient_accumulation_steps=$((full_batch_size / (per_device_batch_size * PROC_PER_NODE * NNODES)))
echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"

export lr=2e-5
export image_max_pixels=$((2048 * 28 * 28))
export cutoff_len=6144
export output_dir=ckpt/${Finetune_SAVE_PATH}
export model_name_or_path=ckpt/${Pretrain_SAVE_PATH}

# ========== Launch Training ==========
torchrun \
    --nnodes $NNODES --nproc_per_node $PROC_PER_NODE --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    ${LLAMAFACTORY_DIR}/src/train.py \
    --deepspeed ${LLAMAFACTORY_DIR}/examples/deepspeed/ds_z2_config.json \
    --model_name_or_path $model_name_or_path \
    --image_max_pixels ${image_max_pixels} \
    --trust_remote_code true \
    --stage sft \
    --do_train true \
    --finetuning_type full \
    --freeze_vision_tower false \
    --freeze_multi_modal_projector false \
    --freeze_language_model false \
    --dataset open1m_image,open1m_text \
    --template qwen2_vl \
    --cutoff_len ${cutoff_len} \
    --overwrite_cache true \
    --tokenized_path ${tokenized_path} \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 16 \
    --output_dir $output_dir \
    --logging_steps 1 \
    --plot_loss true \
    --overwrite_output_dir true \
    --per_device_train_batch_size $per_device_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate ${lr} \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --flash_attn fa2 \
    --save_steps 10000 \
    --save_total_limit 1 \
    --report_to wandb
