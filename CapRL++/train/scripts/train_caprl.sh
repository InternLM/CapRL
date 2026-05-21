#!/usr/bin/env bash

set -euo pipefail
set -x

# Single-node Video CapRL++ training with verl.
# Support both video and image caption training.
#
# Start the reward service first:
#   bash scripts/start_reward_serve_rm.sh
#
# Then launch training, for example:
#   CAPTION_MODEL=/path/to/Qwen3-VL-4B-Instruct \
#   DATASET=/path/to/train.jsonl \
#   SAVE_DIR=/path/to/checkpoints/caprl-rloo \
#   REWARD_NODE_IP=127.0.0.1 \
#   bash scripts/train_caprl.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-${TRAIN_ROOT}/verl}"

unset RAY_ADDRESS

cd "${VERL_ROOT}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        echo "CONDA_ENV is set but conda is not available in PATH." >&2
        exit 1
    fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Remote reward service. Use 127.0.0.1 when the reward service runs on the
# same machine.
export REWARD_SCORE_MODE="${REWARD_SCORE_MODE:-qa}"
REWARD_NODE_IP="${REWARD_NODE_IP:-127.0.0.1}"
REWARD_PORT="${REWARD_PORT:-18889}"
export REWARD_REMOTE_URL="${REWARD_REMOTE_URL:-http://${REWARD_NODE_IP}:${REWARD_PORT}/get_reward}"

# Required user-provided paths.
: "${CAPTION_MODEL:?Set CAPTION_MODEL to the initial caption model path or Hugging Face model id.}"
: "${DATASET:?Set DATASET to the training jsonl path.}"
: "${SAVE_DIR:?Set SAVE_DIR to the output checkpoint directory.}"

# Length reward settings. Keep the tokenizer aligned with the caption model
# unless you intentionally use another tokenizer.
export REWARD_LENGTH_TOKENIZER_PATH="${REWARD_LENGTH_TOKENIZER_PATH:-${CAPTION_MODEL}}"
export REWARD_LENGTH_L1="${REWARD_LENGTH_L1:-2048}"
export REWARD_LENGTH_L2="${REWARD_LENGTH_L2:-3072}"
export REWARD_LENGTH_WEIGHT="${REWARD_LENGTH_WEIGHT:-0.2}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-CapRL_video}"
export WANDB_DIR="${WANDB_DIR:-${TRAIN_ROOT}/logs/wandb}"
RUN_NAME="${RUN_NAME:-qwen3_vl_4b_video_rloo}"
mkdir -p "${WANDB_DIR}"

# local training logs are not needed
export VERL_LOG_ADV_STATS="${VERL_LOG_ADV_STATS:-0}"
export REWARD_LOG_TO_FILE="${REWARD_LOG_TO_FILE:-0}"

SAVE_HF_MODEL="${SAVE_HF_MODEL:-True}"
if [[ "${SAVE_HF_MODEL}" == "True" ]]; then
    CHECKPOINT_CONTENTS="['model','hf_model','optimizer','extra']"
else
    CHECKPOINT_CONTENTS="['model','optimizer','extra']"
fi

BATCH_SIZE="${BATCH_SIZE:-128}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-50}"

# When choosing image caption training, add the following parameters:
# data.input_type=image \
# data.prompt_key=prompt \

python -m verl.trainer.main_ppo \
    --config-path="${VERL_ROOT}/recipe/video_captionrl" \
    --config-name=ppo_grpo_qwen3vl \
    data.train_files="[${DATASET}]" \
    data.val_files="[${DATASET}]" \
    data.caption_log_dir=None \
    data.gen_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-4096}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-4096}" \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path="${CAPTION_MODEL}" \
    algorithm.adv_estimator=rloo \
    algorithm.rollout_correction.bypass_mode=true \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-5}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps="${LR_WARMUP_STEPS:-20}" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.use_remove_padding=true \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}" \
    actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY:-fsdp2}" \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.prompt_length="${ROLLOUT_PROMPT_LENGTH:-13000}" \
    actor_rollout_ref.rollout.response_length="${ROLLOUT_RESPONSE_LENGTH:-4096}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN:-18000}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-18000}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.88}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}" \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.actor.checkpoint.save_contents="${CHECKPOINT_CONTENTS}" \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS:-8}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-10}" \
    trainer.max_ckpt_to_keep="${MAX_CKPT_TO_KEEP:-10}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${RUN_NAME}"
