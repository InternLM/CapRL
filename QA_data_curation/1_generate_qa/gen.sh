#!/bin/bash

export NODE_RANK=${NODE_RANK}
export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-1}
echo "===================================="
echo "NNODES=${NNODES}"
echo "Current (NODE_RANK): $NODE_RANK"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "===================================="


MODEL_PATH="/path/models--Qwen--Qwen2.5-VL-72B-Instruct/"
SERVE_CMD="vllm serve $MODEL_PATH --trust-remote-code --tensor-parallel-size=2 --pipeline-parallel-size=1 --gpu_memory_utilization=0.95 --served-model-name=qwen-vl --host 0.0.0.0"

echo "VLLM_PORTS_PRISM = $((33500+NODE_RANK))"
VLLM_PORTS_PRISM=$((33500+NODE_RANK))

CUDA_VISIBLE_DEVICES=0,1 nohup bash -c "$SERVE_CMD --port $((VLLM_PORTS_PRISM))" > /dev/null 2>&1 &
VLLM_PIDS_PRISM=$!

echo "Started vllm API on GPU, listening on port $((VLLM_PORTS_PRISM))"

wait_times=0
while ! curl -s http://127.0.0.1:${VLLM_PORTS_PRISM} > /dev/null; do
    wait_times=$((wait_times + 1))
    if [ $wait_times -ge 80 ]; then
        echo "Reaching maximum waiting time for vllm, restarting..."
        kill ${VLLM_PIDS_PRISM} 2>/dev/null
        sleep 10
        VLLM_PORTS_PRISM=$((VLLM_PORTS_PRISM + 100))
        CUDA_VISIBLE_DEVICES=0,1 nohup bash -c "$SERVE_CMD --port ${VLLM_PORTS_PRISM}" > /dev/null 2>&1 &
        VLLM_PIDS_PRISM=$!
        echo "Restarted vllm API on GPU, listening on port ${VLLM_PORTS_PRISM}"
        wait_times=0
    fi
    echo "Waiting for vllm API on VLLM_PORTS_PRISM port ${VLLM_PORTS_PRISM} to start..."
    sleep 10
done


echo "Started for API on VLLM_PORTS_PRISM"

python /path/1_generate_qa/gen_qa.py \
    --part-id "$NODE_RANK" \
    --all-parts "$NNODES" \
    --vlm-port "$VLLM_PORTS_PRISM"
