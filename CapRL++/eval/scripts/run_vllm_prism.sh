#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${CAPTION_MODEL_PATH:?set CAPTION_MODEL_PATH}"
: "${DOWNSTREAM_MODEL_PATH:?set DOWNSTREAM_MODEL_PATH}"
: "${DATA_PATH:?set DATA_PATH}"

BENCHMARK="${BENCHMARK:-videomme}"
BENCHMARK="$(echo "${BENCHMARK}" | tr '[:upper:]' '[:lower:]')"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/outputs/${BENCHMARK}}"

NUM_GPUS="${NUM_GPUS:-1}"
BASE_PORT="${BASE_PORT:-8010}"
LB_PORT="${LB_PORT:-8000}"
NUM_FRAMES_SERVER="${NUM_FRAMES_SERVER:-128}"
ALLOWED_MEDIA_PATH="${ALLOWED_MEDIA_PATH:-$(dirname "$(realpath "${DATA_PATH}")")}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_NUM="${GPU_NUM:-${NUM_GPUS}}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

STEP="${STEP:-0}"
MAX_NUM="${MAX_NUM:--1}"
GEN_NUM="${GEN_NUM:-4}"
CAPTION_BS="${CAPTION_BS:-32}"
ANSWER_BS="${ANSWER_BS:-32}"
CAPTION_MAX_TOKENS="${CAPTION_MAX_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
API_CONCURRENCY="${API_CONCURRENCY:-32}"
MMVU_JUDGE_MODEL="${MMVU_JUDGE_MODEL:-gpt-4o}"
MMVU_JUDGE_CONCURRENCY="${MMVU_JUDGE_CONCURRENCY:-16}"

VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:${LB_PORT}/v1}"
CAPTION_MODEL_NAME="${CAPTION_MODEL_NAME:-${CAPTION_MODEL_PATH}}"
DOWNSTREAM_MODEL_NAME="${DOWNSTREAM_MODEL_NAME:-${DOWNSTREAM_MODEL_PATH}}"

mkdir -p "${SAVE_DIR}" "${ROOT_DIR}/logs"
source "${ROOT_DIR}/scripts/vllm_topology_utils.sh"

stop_servers() {
    pkill -f "vllm serve" 2>/dev/null || true
    nginx -s stop 2>/dev/null || true
    pkill nginx 2>/dev/null || true
}

start_role() {
    local model_path="$1"
    local role="$2"
    local is_video="$3"
    configure_model_topology "${model_path}" "${role}"
    start_vllm_servers "${model_path}" "${is_video}" "${NUM_FRAMES_SERVER}" "${role}_${BENCHMARK}_${NUM_FRAMES_SERVER}frames"
    wait_for_servers "${TOPOLOGY_STARTUP_TIMEOUT}"
    setup_nginx
}

echo "Prism evaluation"
echo "  benchmark: ${BENCHMARK}"
echo "  caption model: ${CAPTION_MODEL_PATH}"
echo "  downstream model: ${DOWNSTREAM_MODEL_PATH}"
echo "  data: ${DATA_PATH}"
echo "  output: ${SAVE_DIR}"

start_role "${CAPTION_MODEL_PATH}" "caption" "true"

python -u "${ROOT_DIR}/eval_prism_video_benchmarks_vlmevalkit.py" \
  --benchmark "${BENCHMARK}" \
  --data-path "${DATA_PATH}" \
  --caption-model-path "${CAPTION_MODEL_PATH}" \
  --downstream-model-path "${DOWNSTREAM_MODEL_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --step "${STEP}" \
  --max-num "${MAX_NUM}" \
  --num-frames "${NUM_FRAMES_SERVER}" \
  --gen-num "${GEN_NUM}" \
  --gpu-num "${GPU_NUM}" \
  --caption-batch-size "${CAPTION_BS}" \
  --caption-max-tokens "${CAPTION_MAX_TOKENS}" \
  --answer-batch-size "${ANSWER_BS}" \
  --vllm-api-base "${VLLM_API_BASE}" \
  --caption-model-name "${CAPTION_MODEL_NAME}" \
  --api-concurrency "${API_CONCURRENCY}" \
  --mmvu-judge-model "${MMVU_JUDGE_MODEL}" \
  --mmvu-judge-concurrency "${MMVU_JUDGE_CONCURRENCY}" \
  $(caption_extra_args) \
  --stage-num 1

stop_servers
sleep 5

start_role "${DOWNSTREAM_MODEL_PATH}" "downstream" "false"

python -u "${ROOT_DIR}/eval_prism_video_benchmarks_vlmevalkit.py" \
  --benchmark "${BENCHMARK}" \
  --data-path "${DATA_PATH}" \
  --caption-model-path "${CAPTION_MODEL_PATH}" \
  --downstream-model-path "${DOWNSTREAM_MODEL_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --step "${STEP}" \
  --max-num "${MAX_NUM}" \
  --num-frames "${NUM_FRAMES_SERVER}" \
  --gen-num "${GEN_NUM}" \
  --gpu-num "${GPU_NUM}" \
  --answer-batch-size "${ANSWER_BS}" \
  --vllm-api-base "${VLLM_API_BASE}" \
  --downstream-api-base "${VLLM_API_BASE}" \
  --downstream-model-name "${DOWNSTREAM_MODEL_NAME}" \
  --downstream-api-concurrency "${API_CONCURRENCY}" \
  --mmvu-judge-model "${MMVU_JUDGE_MODEL}" \
  --mmvu-judge-concurrency "${MMVU_JUDGE_CONCURRENCY}" \
  --stage-num 2

echo "Done: ${SAVE_DIR}"
