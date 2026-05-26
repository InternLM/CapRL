#!/usr/bin/env bash
# Start the video_captionrl reward server as one master and one or more workers.
#
# Common environment variables:
#   REWARD_SCORE_MODE=qa|vl_judge
#   REWARD_TASK=video|image
#   VERL_ROOT
#   CONDA_ROOT / REWARD_CONDA_ENV
#   REWARD_MODEL
#   REWARD_PORT
#   REWARD_WORKER_BASE
#   REWARD_NUM_WORKERS
#   REWARD_TP
#   REWARD_SHUFFLE_QA
#   REWARD_QA_NUM
#   FORMAT_REWARD_WEIGHT
#   FORMAT_MAX_MINUTE
#   ZERO_REWARD_LOG_PATH
#   CUDA_HOME
#
# Examples:
#   REWARD_TASK=video bash .../start_reward_server.sh
#   REWARD_SCORE_MODE=vl_judge REWARD_MODEL=/path/to/Qwen2.5-VL-72B bash .../start_reward_server.sh
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
SERVE_RM_SCRIPT="${VERL_ROOT}/recipe/video_captionrl/serve_rm.py"

CONDA_ROOT="${CONDA_ROOT:-}"
REWARD_CONDA_ENV="${REWARD_CONDA_ENV:-}"

REWARD_SCORE_MODE="${REWARD_SCORE_MODE:-qa}"
REWARD_TASK="${REWARD_TASK:-video}"
REWARD_PORT="${REWARD_PORT:-18889}"
REWARD_WORKER_BASE="${REWARD_WORKER_BASE:-18899}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-8}"
REWARD_TP="${REWARD_TP:-1}"
REWARD_QA_NUM="${REWARD_QA_NUM:-8}"
FORMAT_MAX_MINUTE="${FORMAT_MAX_MINUTE:-599}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-18000}"

: "${REWARD_MODEL:?Set REWARD_MODEL to the reward model path or Hugging Face model id.}"

if [[ "$REWARD_TASK" == "video" ]]; then
  FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
else
  FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0}"
fi

# ZERO_REWARD_LOG_PATH=/path/to/zero.jsonl

SHUFFLE_QA="${REWARD_SHUFFLE_QA:-1}"
SHUFFLE_QA_ARGS=()
if [[ "$SHUFFLE_QA" == "1" ]]; then
  SHUFFLE_QA_ARGS+=(--shuffle_qa)
fi

MASTER_PID=""

do_cleanup() {
  echo ""
  echo "[cleanup] Stopping reward server processes and freeing ports..."
  if [[ "${CLEANUP_DONE:-0}" == "1" ]]; then return 0; fi
  CLEANUP_DONE=1

  if [[ -n "$MASTER_PID" ]] && kill -0 "$MASTER_PID" 2>/dev/null; then
    kill -9 "$MASTER_PID" 2>/dev/null || true
  fi
  pkill -9 -f "recipe/video_captionrl/serve_rm.py" 2>/dev/null || true
  pkill -9 -f "serve_rm.py" 2>/dev/null || true
  pkill -9 -f "video_captionrl/serve_rm" 2>/dev/null || true
  pkill -9 -f "reward_server/serve_rm" 2>/dev/null || true
  if command -v fuser &>/dev/null; then
    for i in 0 1 2 3 4 5 6 7; do
      fuser -k $((REWARD_WORKER_BASE + i))/tcp 2>/dev/null || true
    done
    fuser -k "$REWARD_PORT/tcp" 2>/dev/null || true
  fi
  sleep 2
  pkill -9 -f "recipe/video_captionrl/serve_rm.py" 2>/dev/null || true
  pkill -9 -f "serve_rm.py" 2>/dev/null || true

  echo "[cleanup] Done."
  echo "[cleanup] Opening an interactive shell to keep the container alive; type exit to quit."
  exec bash -i
}

trap do_cleanup EXIT

pkill -9 -f "recipe/video_captionrl/serve_rm.py" 2>/dev/null || true
pkill -9 -f "serve_rm.py" 2>/dev/null || true
if command -v fuser &>/dev/null; then
  for i in 0 1 2 3 4 5 6 7; do
    fuser -k $((REWARD_WORKER_BASE + i))/tcp 2>/dev/null || true
  done
  fuser -k "$REWARD_PORT/tcp" 2>/dev/null || true
fi
sleep 3

if [[ ! -f "$SERVE_RM_SCRIPT" ]]; then
  echo "ERROR: serve_rm not found: $SERVE_RM_SCRIPT"
  exit 1
fi

if [[ -n "$CONDA_ROOT" && -n "$REWARD_CONDA_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$REWARD_CONDA_ENV"
fi

_reward_setup_cuda() {
  if command -v nvcc &>/dev/null; then
    return 0
  fi
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    return 0
  fi
  local candidates=(
    "/usr/local/cuda-12.8"
    "/usr/local/cuda"
  )
  for d in "${candidates[@]}"; do
    if [[ -x "${d}/bin/nvcc" ]]; then
      export CUDA_HOME="$d"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      echo "[start_reward_server] Using CUDA_HOME=${CUDA_HOME} (nvcc was not on PATH)."
      return 0
    fi
  done
  echo "[start_reward_server] ERROR: nvcc not found. Install CUDA toolkit or set CUDA_HOME to a directory containing bin/nvcc." >&2
  echo "[start_reward_server] Example: export CUDA_HOME=/path/to/cuda && export PATH=\$CUDA_HOME/bin:\$PATH" >&2
  return 1
}
_reward_setup_cuda || exit 1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

COMMON_ARGS=(
  --num_workers "$REWARD_NUM_WORKERS"
  --tp "$REWARD_TP"
  --port "$REWARD_PORT"
  --worker_base_port "$REWARD_WORKER_BASE"
  --reward_pretrain "$REWARD_MODEL"
  --qa_num "$REWARD_QA_NUM"
  "${SHUFFLE_QA_ARGS[@]}"
  --format_reward_weight "$FORMAT_REWARD_WEIGHT"
  --format_max_minute "$FORMAT_MAX_MINUTE"
  --task "$REWARD_TASK"
  --score_mode "$REWARD_SCORE_MODE"
  --judge_max_model_len "$JUDGE_MAX_MODEL_LEN"
)

python "$SERVE_RM_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --role master \
  --worker_hosts 0.0.0.0 &

MASTER_PID=$!
echo "Waiting for master to bind port $REWARD_PORT for up to 60 seconds..."
for _ in $(seq 1 30); do
  sleep 2
  if ! kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "Reward master exited. Press Ctrl+C to run cleanup and exit."
    sleep infinity
    exit 0
  fi
  if ss -tlnp 2>/dev/null | grep -q ":$REWARD_PORT "; then
    echo "Port $REWARD_PORT is listening; starting worker."
    break
  fi
done
if ! ss -tlnp 2>/dev/null | grep -q ":$REWARD_PORT "; then
  echo "Timed out waiting for port $REWARD_PORT. Press Ctrl+C to run cleanup and exit."
  sleep infinity
  exit 1
fi

WORKER_EXTRA=(--role worker)
if [[ -n "${ZERO_REWARD_LOG_PATH:-}" ]]; then
  WORKER_EXTRA+=(--zero_reward_log_path "$ZERO_REWARD_LOG_PATH")
fi

python "$SERVE_RM_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  "${WORKER_EXTRA[@]}"

exit 0
