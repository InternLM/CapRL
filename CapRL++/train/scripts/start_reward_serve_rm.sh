#!/usr/bin/env bash

set -euo pipefail
set -x

# Reward service for CapRL++.
#
# Example:
#   REWARD_MODEL=/path/to/Qwen3-4B-Instruct \
#   CUDA_VISIBLE_DEVICES=0 \
#   REWARD_PORT=18889 \
#   bash scripts/start_reward_serve_rm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export VERL_ROOT="${VERL_ROOT:-${TRAIN_ROOT}/verl}"

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

export REWARD_PORT="${REWARD_PORT:-18889}"
export REWARD_WORKER_BASE="${REWARD_WORKER_BASE:-$((REWARD_PORT + 10))}"

# Support both video and image reward scoring.
export REWARD_TASK="${REWARD_TASK:-video}"
export REWARD_SCORE_MODE="${REWARD_SCORE_MODE:-qa}"

# qa mode can use a text LLM (our method); vl_judge mode is used for LVLM-as-a-judge training.
: "${REWARD_MODEL:?Set REWARD_MODEL to the reward model path or Hugging Face model id.}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"

# qa-mode reward settings. They are ignored by vl_judge mode.
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
export REWARD_QA_NUM="${REWARD_QA_NUM:-8}"

exec bash "${VERL_ROOT}/recipe/video_captionrl/scripts/start_reward_server.sh"
