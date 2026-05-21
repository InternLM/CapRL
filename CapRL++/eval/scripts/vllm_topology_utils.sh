#!/usr/bin/env bash

# Shared vLLM server helpers for Prism evaluation scripts.

_topology_lower() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

_topology_model_path_safe() {
    local p="${1:-}"
    echo "${p//\//_}"
}

detect_model_profile() {
    local model_path="$1"
    local text
    text=$(_topology_lower "${model_path}")

    if [[ -n "${VLLM_FORCE_PROFILE:-}" ]]; then
        echo "${VLLM_FORCE_PROFILE}"
        return
    fi

    if [[ "${text}" =~ qwen3\.5|qwen3_5|qwen35 ]]; then
        echo "qwen3.5"
    elif [[ "${text}" =~ 235b ]]; then
        # e.g. Qwen3-VL-235B-A22B-Instruct — needs TP+EP across all GPUs, one engine
        echo "235b_vl"
    elif [[ "${text}" =~ (^|[^0-9])30b([^0-9]|$) ]] || [[ "${text}" =~ (^|[^0-9])32b([^0-9]|$) ]]; then
        echo "30b"
    elif [[ "${text}" =~ (^|[^0-9])8b([^0-9]|$) ]]; then
        echo "8b"
    elif [[ "${text}" =~ (^|[^0-9])4b([^0-9]|$) ]]; then
        echo "4b"
    else
        echo "default"
    fi
}

_get_override_or_default() {
    local specific_name="$1"
    local generic_name="$2"
    local default_value="$3"
    local specific_value="${!specific_name:-}"
    local generic_value="${!generic_name:-}"

    if [ -n "${specific_value}" ]; then
        echo "${specific_value}"
    elif [ -n "${generic_value}" ]; then
        echo "${generic_value}"
    else
        echo "${default_value}"
    fi
}

configure_model_topology() {
    local model_path="$1"
    local role="${2:-caption}"
    local role_upper
    role_upper=$(echo "${role}" | tr '[:lower:]' '[:upper:]')

    TOPOLOGY_PROFILE=$(detect_model_profile "${model_path}")
    TOPOLOGY_MODE="single_gpu_replicas"
    TOPOLOGY_REPLICAS="${NUM_GPUS}"
    TOPOLOGY_MAX_MODEL_LEN="16384"
    TOPOLOGY_STARTUP_TIMEOUT="360"
    TOPOLOGY_STARTUP_STAGGER_SEC="8"

    if [ "${role}" = "downstream" ]; then
        TOPOLOGY_STARTUP_TIMEOUT="300"
    fi

    TOPOLOGY_STARTUP_BATCH_SIZE="${NUM_GPUS}"

    TOPOLOGY_CONDA_ENV=""
    TOPOLOGY_VLLM_EXTRA_ARGS=""
    TOPOLOGY_TP_SIZE=""

    case "${TOPOLOGY_PROFILE}" in
        qwen3.5)
            TOPOLOGY_MODE="dp8"
            TOPOLOGY_REPLICAS="1"
            TOPOLOGY_STARTUP_TIMEOUT="600"
            TOPOLOGY_CONDA_ENV="vllm16"
            TOPOLOGY_VLLM_EXTRA_ARGS="-dp ${NUM_GPUS} --mm-encoder-tp-mode data --mm-processor-cache-type shm --reasoning-parser qwen3 --enable-prefix-caching"
            ;;
        235b_vl)
            # Align with video_test/infer/qwen3vl_batch_infer_vllm.py: LLM(tensor_parallel_size=8, enable_expert_parallel=True, mm_encoder_tp_mode="data")
            # Default conda matches Prism scripts (verl-gad); override with VLLM_CONDA_ENV / VLLM_CAPTION_CONDA_ENV if needed (e.g. vllm16).
            TOPOLOGY_MODE="tp8_moe"
            TOPOLOGY_REPLICAS="1"
            TOPOLOGY_STARTUP_TIMEOUT="1800"
            TOPOLOGY_MAX_MODEL_LEN="32768"
            # TOPOLOGY_CONDA_ENV="verl-gad"
            TOPOLOGY_CONDA_ENV="vllm16"
            TOPOLOGY_TP_SIZE="${NUM_GPUS}"
            TOPOLOGY_VLLM_EXTRA_ARGS="--tensor-parallel-size ${TOPOLOGY_TP_SIZE} --enable-expert-parallel --mm-encoder-tp-mode data --mm-processor-cache-type shm --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.87}"
            ;;
        30b)
            TOPOLOGY_STARTUP_BATCH_SIZE="2"
            if [ "${role}" = "caption" ]; then
                TOPOLOGY_STARTUP_TIMEOUT="1200"
            else
                TOPOLOGY_STARTUP_TIMEOUT="900"
            fi
            TOPOLOGY_STARTUP_STAGGER_SEC="5"
            ;;
        8b|4b|default)
            ;;
    esac

    TOPOLOGY_REPLICAS=$(_get_override_or_default "VLLM_${role_upper}_REPLICAS" "VLLM_REPLICAS" "${TOPOLOGY_REPLICAS}")
    TOPOLOGY_MAX_MODEL_LEN=$(_get_override_or_default "VLLM_${role_upper}_MAX_MODEL_LEN" "VLLM_MAX_MODEL_LEN" "${TOPOLOGY_MAX_MODEL_LEN}")
    TOPOLOGY_STARTUP_TIMEOUT=$(_get_override_or_default "VLLM_${role_upper}_STARTUP_TIMEOUT" "VLLM_STARTUP_TIMEOUT" "${TOPOLOGY_STARTUP_TIMEOUT}")
    TOPOLOGY_STARTUP_STAGGER_SEC=$(_get_override_or_default "VLLM_${role_upper}_STARTUP_STAGGER_SEC" "VLLM_STARTUP_STAGGER_SEC" "${TOPOLOGY_STARTUP_STAGGER_SEC}")
    TOPOLOGY_STARTUP_BATCH_SIZE=$(_get_override_or_default "VLLM_${role_upper}_STARTUP_BATCH_SIZE" "VLLM_STARTUP_BATCH_SIZE" "${TOPOLOGY_STARTUP_BATCH_SIZE}")

    if [ "${TOPOLOGY_REPLICAS}" -gt "${NUM_GPUS}" ]; then
        TOPOLOGY_REPLICAS="${NUM_GPUS}"
    fi

    # Single-endpoint modes must not inherit a stale VLLM_REPLICAS from the environment.
    if [ "${TOPOLOGY_MODE}" = "dp8" ] || [ "${TOPOLOGY_MODE}" = "tp8_moe" ]; then
        TOPOLOGY_REPLICAS="1"
    fi

    TOPOLOGY_CONDA_ENV=$(_get_override_or_default "VLLM_${role_upper}_CONDA_ENV" "VLLM_CONDA_ENV" "${TOPOLOGY_CONDA_ENV}")

    if [ -n "${TOPOLOGY_TP_SIZE:-}" ]; then
        TOPOLOGY_TP_SIZE=$(_get_override_or_default "VLLM_${role_upper}_TP_SIZE" "VLLM_TP_SIZE" "${TOPOLOGY_TP_SIZE}")
        # Re-expand TP-dependent args for 235b_vl (was set before override)
        if [ "${TOPOLOGY_PROFILE}" = "235b_vl" ]; then
            TOPOLOGY_VLLM_EXTRA_ARGS="--tensor-parallel-size ${TOPOLOGY_TP_SIZE} --enable-expert-parallel --mm-encoder-tp-mode data --mm-processor-cache-type shm --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.87}"
        fi
    fi

    echo "[$(date '+%H:%M:%S')] Topology preset (${role}): profile=${TOPOLOGY_PROFILE}, mode=${TOPOLOGY_MODE}, replicas=${TOPOLOGY_REPLICAS}, max_model_len=${TOPOLOGY_MAX_MODEL_LEN}, startup_timeout=${TOPOLOGY_STARTUP_TIMEOUT}s, stagger=${TOPOLOGY_STARTUP_STAGGER_SEC}s, startup_batch=${TOPOLOGY_STARTUP_BATCH_SIZE}${TOPOLOGY_TP_SIZE:+, tp=${TOPOLOGY_TP_SIZE}}${TOPOLOGY_CONDA_ENV:+, conda_env=${TOPOLOGY_CONDA_ENV}}"
}

_active_server_count() {
    if [ -n "${TOPOLOGY_REPLICAS:-}" ]; then
        echo "${TOPOLOGY_REPLICAS}"
    else
        echo "${NUM_GPUS}"
    fi
}

caption_extra_args() {
    local args=""
    if [ -n "${ENABLE_THINKING:-}" ]; then
        args="${args} --enable-thinking ${ENABLE_THINKING}"
    fi
    if [ -n "${CAPTION_EXTRA_BODY:-}" ]; then
        args="${args} --caption-extra-body '${CAPTION_EXTRA_BODY}'"
    fi
    echo "${args}"
}

_resolve_conda_activate() {
    if [ -n "${TOPOLOGY_CONDA_ENV}" ]; then
        if [ -n "${CONDA_SH:-}" ]; then
            echo "source ${CONDA_SH} && conda activate ${TOPOLOGY_CONDA_ENV} && "
        elif [ -n "${CONDA_BASE:-}" ]; then
            echo "source ${CONDA_BASE}/bin/activate ${TOPOLOGY_CONDA_ENV} && "
        else
            echo "conda activate ${TOPOLOGY_CONDA_ENV} && "
        fi
    else
        echo ""
    fi
}

_start_dp8_server() {
    local MODEL_PATH="$1"
    local IS_VIDEO="$2"
    local NUM_FRAMES="$3"
    local LOG_SUFFIX="$4"
    local MODEL_PATH_SAFE
    MODEL_PATH_SAFE=$(_topology_model_path_safe "${MODEL_PATH}")
    local PORT="${LB_PORT}"
    local CONDA_PREFIX
    CONDA_PREFIX=$(_resolve_conda_activate)

    echo "[$(date '+%H:%M:%S')] Starting vLLM server in dp8 mode on port ${PORT}..."

    local VIDEO_ARGS=""
    if [ "${IS_VIDEO}" = "true" ]; then
        VIDEO_ARGS="--limit-mm-per-prompt '{\"video\":1}' --allowed-local-media-path ${ALLOWED_MEDIA_PATH} --media-io-kwargs '{\"video\": {\"num_frames\": ${NUM_FRAMES}}}'"
    fi

    bash -c "${CONDA_PREFIX}if [ -n \"\${CUDA_HOME:-}\" ]; then export PATH=\$CUDA_HOME/bin:\$PATH; fi && export MKL_THREADING_LAYER=GNU && vllm serve \"${MODEL_PATH}\" --port ${PORT} --max-model-len ${TOPOLOGY_MAX_MODEL_LEN} ${VIDEO_ARGS} ${TOPOLOGY_VLLM_EXTRA_ARGS} --trust-remote-code --host 0.0.0.0" \
        > "${ROOT_DIR}/logs/vllm_dp8_${LOG_SUFFIX}__${MODEL_PATH_SAFE}.log" 2>&1 &

    echo "[$(date '+%H:%M:%S')] dp8 server launched (pid=$!), log: ${ROOT_DIR}/logs/vllm_dp8_${LOG_SUFFIX}__${MODEL_PATH_SAFE}.log"
}

# Comma-separated CUDA_VISIBLE_DEVICES for GPUs 0..(tp-1)
_cuda_visible_range() {
    local tp="$1"
    local s="" i
    for ((i = 0; i < tp; i++)); do
        [ -n "${s}" ] && s="${s},"
        s="${s}${i}"
    done
    echo "${s}"
}

# Qwen3-VL-235B-A22B style: one vLLM engine, tensor parallel + expert parallel (multi-GPU).
_start_tp_moe_server() {
    local MODEL_PATH="$1"
    local IS_VIDEO="$2"
    local NUM_FRAMES="$3"
    local LOG_SUFFIX="$4"
    local MODEL_PATH_SAFE
    MODEL_PATH_SAFE=$(_topology_model_path_safe "${MODEL_PATH}")
    local PORT="${LB_PORT}"
    local CONDA_PREFIX
    local TP="${TOPOLOGY_TP_SIZE:-${NUM_GPUS}}"
    CONDA_PREFIX=$(_resolve_conda_activate)
    local CVD
    CVD=$(_cuda_visible_range "${TP}")

    echo "[$(date '+%H:%M:%S')] Starting vLLM server in tp8_moe mode (TP=${TP}, EP=on) on port ${PORT}, CUDA_VISIBLE_DEVICES=${CVD}..."

    local VIDEO_ARGS=""
    if [ "${IS_VIDEO}" = "true" ]; then
        VIDEO_ARGS="--limit-mm-per-prompt '{\"video\":1}' --allowed-local-media-path ${ALLOWED_MEDIA_PATH} --media-io-kwargs '{\"video\": {\"num_frames\": ${NUM_FRAMES}}}'"
    fi

    bash -c "${CONDA_PREFIX}export CUDA_VISIBLE_DEVICES=${CVD} && export VLLM_WORKER_MULTIPROC_METHOD=spawn && if [ -n \"\${CUDA_HOME:-}\" ]; then export PATH=\$CUDA_HOME/bin:\$PATH; fi && export MKL_THREADING_LAYER=GNU && vllm serve \"${MODEL_PATH}\" --port ${PORT} --max-model-len ${TOPOLOGY_MAX_MODEL_LEN} ${VIDEO_ARGS} ${TOPOLOGY_VLLM_EXTRA_ARGS} --trust-remote-code --host 0.0.0.0" \
        > "${ROOT_DIR}/logs/vllm_tp_moe_${LOG_SUFFIX}__${MODEL_PATH_SAFE}.log" 2>&1 &

    echo "[$(date '+%H:%M:%S')] tp8_moe server launched (pid=$!), log: ${ROOT_DIR}/logs/vllm_tp_moe_${LOG_SUFFIX}__${MODEL_PATH_SAFE}.log"
}

_launch_single_server() {
    local gpu_id="$1"
    local MODEL_PATH="$2"
    local IS_VIDEO="$3"
    local NUM_FRAMES="$4"
    local LOG_SUFFIX="$5"
    local MODEL_PATH_SAFE
    MODEL_PATH_SAFE=$(_topology_model_path_safe "${MODEL_PATH}")
    local PORT=$((BASE_PORT + gpu_id))

    echo "[$(date '+%H:%M:%S')] Starting server on GPU ${gpu_id}, port ${PORT}"

    if [ "${IS_VIDEO}" = "true" ]; then
        CUDA_VISIBLE_DEVICES=${gpu_id} vllm serve "${MODEL_PATH}" \
            --port "${PORT}" \
            --max-model-len "${TOPOLOGY_MAX_MODEL_LEN}" \
            --limit-mm-per-prompt '{"video":1}' \
            --allowed-local-media-path "${ALLOWED_MEDIA_PATH}" \
            --media-io-kwargs "{\"video\": {\"num_frames\": ${NUM_FRAMES}}}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --trust-remote-code \
            > "${ROOT_DIR}/logs/vllm_server_gpu${gpu_id}_${LOG_SUFFIX}__${MODEL_PATH_SAFE}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${gpu_id} vllm serve "${MODEL_PATH}" \
            --port "${PORT}" \
            --max-model-len "${TOPOLOGY_MAX_MODEL_LEN}" \
            --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
            --trust-remote-code \
            > "${ROOT_DIR}/logs/vllm_downstream_gpu${gpu_id}_${LOG_SUFFIX}__${MODEL_PATH_SAFE}.log" 2>&1 &
    fi
}

_wait_for_ports() {
    local timeout="$1"
    shift
    local ports=("$@")

    for t in $(seq 1 "${timeout}"); do
        local ready=0
        for port in "${ports[@]}"; do
            local code
            code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:${port}/v1/models" 2>/dev/null || echo "000")
            [ "${code}" = "200" ] && ready=$((ready + 1))
        done
        echo -ne "\r[$(date '+%H:%M:%S')] Progress: ${ready}/${#ports[@]} servers ready... (${t}s elapsed)"

        if [ "${ready}" -eq "${#ports[@]}" ]; then
            echo ""
            return 0
        fi
        sleep 1
    done

    echo ""
    return 1
}

start_vllm_servers() {
    local MODEL_PATH="$1"
    local IS_VIDEO="$2"
    local NUM_FRAMES="$3"
    local LOG_SUFFIX="$4"

    if [ "${TOPOLOGY_MODE}" = "dp8" ]; then
        _start_dp8_server "${MODEL_PATH}" "${IS_VIDEO}" "${NUM_FRAMES}" "${LOG_SUFFIX}"
        return $?
    fi

    if [ "${TOPOLOGY_MODE}" = "tp8_moe" ]; then
        _start_tp_moe_server "${MODEL_PATH}" "${IS_VIDEO}" "${NUM_FRAMES}" "${LOG_SUFFIX}"
        return $?
    fi

    local ACTIVE_SERVERS
    ACTIVE_SERVERS=$(_active_server_count)
    local BATCH_SIZE="${TOPOLOGY_STARTUP_BATCH_SIZE:-${ACTIVE_SERVERS}}"

    if [ "${BATCH_SIZE}" -ge "${ACTIVE_SERVERS}" ]; then
        echo "[$(date '+%H:%M:%S')] Starting all ${ACTIVE_SERVERS} vLLM servers at once..."
        for i in $(seq 0 $((ACTIVE_SERVERS - 1))); do
            _launch_single_server "$i" "${MODEL_PATH}" "${IS_VIDEO}" "${NUM_FRAMES}" "${LOG_SUFFIX}"
            [ $i -lt $((ACTIVE_SERVERS - 1)) ] && sleep "${TOPOLOGY_STARTUP_STAGGER_SEC}"
        done
        return 0
    fi

    echo "[$(date '+%H:%M:%S')] Starting ${ACTIVE_SERVERS} vLLM servers in batches of ${BATCH_SIZE}..."
    local launched=0
    while [ "${launched}" -lt "${ACTIVE_SERVERS}" ]; do
        local batch_end=$((launched + BATCH_SIZE))
        [ "${batch_end}" -gt "${ACTIVE_SERVERS}" ] && batch_end="${ACTIVE_SERVERS}"
        local batch_ports=()

        echo "[$(date '+%H:%M:%S')] Launching batch: GPU ${launched}..$((batch_end - 1))"
        for i in $(seq "${launched}" $((batch_end - 1))); do
            _launch_single_server "$i" "${MODEL_PATH}" "${IS_VIDEO}" "${NUM_FRAMES}" "${LOG_SUFFIX}"
            batch_ports+=($((BASE_PORT + i)))
            [ $i -lt $((batch_end - 1)) ] && sleep "${TOPOLOGY_STARTUP_STAGGER_SEC}"
        done

        echo "[$(date '+%H:%M:%S')] Waiting for batch GPU ${launched}..$((batch_end - 1)) to be ready (timeout ${TOPOLOGY_STARTUP_TIMEOUT}s)..."
        if ! _wait_for_ports "${TOPOLOGY_STARTUP_TIMEOUT}" "${batch_ports[@]}"; then
            echo "ERROR: Batch GPU ${launched}..$((batch_end - 1)) failed to start within ${TOPOLOGY_STARTUP_TIMEOUT}s."
            return 1
        fi
        echo "[$(date '+%H:%M:%S')] Batch GPU ${launched}..$((batch_end - 1)) ready."

        launched="${batch_end}"
    done

    echo "[$(date '+%H:%M:%S')] All ${ACTIVE_SERVERS} servers started successfully."
    return 0
}

wait_for_servers() {
    local TIMEOUT="${1:-${TOPOLOGY_STARTUP_TIMEOUT}}"

    if [ "${TOPOLOGY_MODE}" = "dp8" ] || [ "${TOPOLOGY_MODE}" = "tp8_moe" ]; then
        echo "[$(date '+%H:%M:%S')] Waiting for ${TOPOLOGY_MODE} server on port ${LB_PORT} (timeout ${TIMEOUT}s)..."
        if ! _wait_for_ports "${TIMEOUT}" "${LB_PORT}"; then
            echo "ERROR: ${TOPOLOGY_MODE} vLLM server failed to start within ${TIMEOUT}s."
            return 1
        fi
        return 0
    fi

    local ACTIVE_SERVERS
    ACTIVE_SERVERS=$(_active_server_count)
    local BATCH_SIZE="${TOPOLOGY_STARTUP_BATCH_SIZE:-${ACTIVE_SERVERS}}"

    if [ "${BATCH_SIZE}" -lt "${ACTIVE_SERVERS}" ]; then
        echo "[$(date '+%H:%M:%S')] Batched startup already verified all servers. Quick recheck..."
    fi

    local all_ports=()
    for i in $(seq 0 $((ACTIVE_SERVERS - 1))); do
        all_ports+=($((BASE_PORT + i)))
    done

    if ! _wait_for_ports "${TIMEOUT}" "${all_ports[@]}"; then
        echo "ERROR: Not all vLLM servers started within ${TIMEOUT}s."
        return 1
    fi
    return 0
}

setup_nginx() {
    if [ "${TOPOLOGY_MODE}" = "dp8" ] || [ "${TOPOLOGY_MODE}" = "tp8_moe" ]; then
        echo "[$(date '+%H:%M:%S')] ${TOPOLOGY_MODE} mode: server already on port ${LB_PORT}, Nginx not needed."
        return 0
    fi

    local ACTIVE_SERVERS
    ACTIVE_SERVERS=$(_active_server_count)

    echo "[$(date '+%H:%M:%S')] Setting up Nginx load balancer on port ${LB_PORT}..."

    local UPSTREAM_SERVERS=""
    for i in $(seq 0 $((ACTIVE_SERVERS - 1))); do
        local port=$((BASE_PORT + i))
        UPSTREAM_SERVERS="${UPSTREAM_SERVERS}        server 127.0.0.1:${port};\n"
    done

    cat > /tmp/nginx_vllm_lb.conf << EOF
events {
    worker_connections 4096;
}

http {
    client_max_body_size 100m;
    upstream vllm {
        least_conn;
$(echo -e "$UPSTREAM_SERVERS")
    }

    server {
        listen ${LB_PORT};

        location / {
            proxy_pass http://vllm;
            proxy_http_version 1.1;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }
    }
}
EOF

    nginx -s stop 2>/dev/null || true
    pkill nginx 2>/dev/null || true
    sleep 2

    nginx -c /tmp/nginx_vllm_lb.conf
    sleep 1

    local POST_CODE
    POST_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://127.0.0.1:${LB_PORT}/v1/models" 2>/dev/null || echo "000")
    if [ "${POST_CODE}" != "200" ]; then
        echo "ERROR: Nginx load balancer failed (got HTTP ${POST_CODE})."
        return 1
    fi
    echo "[$(date '+%H:%M:%S')] Nginx ready."
    return 0
}
