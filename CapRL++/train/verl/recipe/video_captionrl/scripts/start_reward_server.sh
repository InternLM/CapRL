#!/usr/bin/env bash
# =============================================================================
# VERL Caption RL — 统一 Reward Server 启动脚本（master + workers）
# Python 入口：verl/recipe/video_captionrl/serve_rm.py
#
# 环境变量（常用）：
#   REWARD_SCORE_MODE=qa|vl_judge  评分模式，默认 qa；vl_judge 启用 LLM-as-a-judge
#   REWARD_TASK=video|image     默认 video；image 时使用与 CapRL 一致的图像 caption prompt
#   VERL_ROOT                   默认本仓库上级推断或见下方默认值
#   CONDA_ROOT / REWARD_CONDA_ENV
#   REWARD_MODEL                reward 模型路径
#   REWARD_PORT                 master 端口，默认 18889
#   REWARD_WORKER_BASE          worker 起始端口，默认 18899
#   REWARD_NUM_WORKERS          默认 8
#   REWARD_TP                   vLLM TP，默认 1
#   REWARD_SHUFFLE_QA           1 表示 --shuffle_qa（仅 qa 模式）
#   REWARD_QA_NUM               每条 caption 抽样 QA 题数（仅 qa 模式），默认 8
#   FORMAT_REWARD_WEIGHT        仅 video+qa 默认 0.2；image 默认 0
#   FORMAT_MIN_BRACKETS         默认 3
#   ZERO_REWARD_LOG_PATH        可选，None 表示不写
#   CUDA_HOME                     需含 bin/nvcc；vLLM+FlashInfer JIT 编译采样算子时会调用 nvcc。
#                                 若 Pod 无 /usr/local/cuda，请与训练脚本一致设置（见下方自动探测）。
#
# 用法示例：
#   REWARD_TASK=video bash .../start_reward_server.sh
#   REWARD_SCORE_MODE=vl_judge REWARD_MODEL=/path/to/Qwen2.5-VL-72B bash .../start_reward_server.sh
# =============================================================================
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
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-18000}"

: "${REWARD_MODEL:?Set REWARD_MODEL to the reward model path or Hugging Face model id.}"

if [[ "$REWARD_TASK" == "video" ]]; then
  FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.2}"
  FORMAT_MIN_BRACKETS="${FORMAT_MIN_BRACKETS:-3}"
else
  FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0}"
  FORMAT_MIN_BRACKETS="${FORMAT_MIN_BRACKETS:-3}"
fi

# 若需记录全零 reward 样本，设置路径；留空则不传参（与 argparse default=None 一致）
# ZERO_REWARD_LOG_PATH=/path/to/zero.jsonl

SHUFFLE_QA="${REWARD_SHUFFLE_QA:-1}"
SHUFFLE_QA_ARGS=()
if [[ "$SHUFFLE_QA" == "1" ]]; then
  SHUFFLE_QA_ARGS+=(--shuffle_qa)
fi

MASTER_PID=""

do_cleanup() {
  echo ""
  echo "[清理] 正在彻底清理 reward server 相关进程和端口..."
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

  echo "[清理] 已彻底清理。"
  echo "[清理] 进入 shell 以保持容器不退出；输入 exit 再回车可真正退出。"
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

# vLLM v1 + FlashInfer 首次运行会 JIT 编译 CUDA 扩展，必须能找到 nvcc（错误常见：/usr/local/cuda/bin/nvcc not found）
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
  --format_min_brackets "$FORMAT_MIN_BRACKETS"
  --task "$REWARD_TASK"
  --score_mode "$REWARD_SCORE_MODE"
  --judge_max_model_len "$JUDGE_MAX_MODEL_LEN"
)

python "$SERVE_RM_SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --role master \
  --worker_hosts 0.0.0.0 &

MASTER_PID=$!
echo "等待 master 绑定端口 $REWARD_PORT（最多约 60 秒）..."
for _ in $(seq 1 30); do
  sleep 2
  if ! kill -0 "$MASTER_PID" 2>/dev/null; then
    echo "Reward master 进程已退出，Ctrl+C 可彻底清理并退出。"
    sleep infinity
    exit 0
  fi
  if ss -tlnp 2>/dev/null | grep -q ":$REWARD_PORT "; then
    echo "端口 $REWARD_PORT 已监听，启动 worker。"
    break
  fi
done
if ! ss -tlnp 2>/dev/null | grep -q ":$REWARD_PORT "; then
  echo "超时：端口 $REWARD_PORT 仍未监听，Ctrl+C 可彻底清理并退出。"
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
