#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${INTERMEDIATE_PATH:?set INTERMEDIATE_PATH}"
: "${DOWNSTREAM_MODEL_PATH:?set DOWNSTREAM_MODEL_PATH}"

BENCHMARK="${BENCHMARK:-videomme}"
BENCHMARK="$(echo "${BENCHMARK}" | tr '[:upper:]' '[:lower:]')"
STEP="${STEP:-0}"
ANSWER_BS="${ANSWER_BS:-32}"
API_CONCURRENCY="${API_CONCURRENCY:-32}"
LB_PORT="${LB_PORT:-8000}"
VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:${LB_PORT}/v1}"
DOWNSTREAM_MODEL_NAME="${DOWNSTREAM_MODEL_NAME:-${DOWNSTREAM_MODEL_PATH}}"

python -u "${ROOT_DIR}/eval_prism_video_benchmarks_vlmevalkit.py" \
  --benchmark "${BENCHMARK}" \
  --data-path "" \
  --caption-model-path "" \
  --downstream-model-path "${DOWNSTREAM_MODEL_PATH}" \
  --save-dir "" \
  --step "${STEP}" \
  --answer-batch-size "${ANSWER_BS}" \
  --downstream-api-base "${VLLM_API_BASE}" \
  --downstream-model-name "${DOWNSTREAM_MODEL_NAME}" \
  --downstream-api-concurrency "${API_CONCURRENCY}" \
  --intermediate-path "${INTERMEDIATE_PATH}" \
  --stage-num 2
