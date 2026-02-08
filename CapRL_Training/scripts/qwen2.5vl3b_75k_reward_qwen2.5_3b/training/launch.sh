set -x

# Activate conda environment
# conda activate CapRL

# Set working directory to CapRL_Training root
WORK_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
cd $WORK_DIR
export PYTHONPATH="${WORK_DIR}:$PYTHONPATH"

# ========== Paths (modify these) ==========
DATASET="/path/to/your/qa_dataset.json"
PRETRAIN_MODEL="/path/to/Qwen2.5-VL-3B-Instruct"
SAVE_PATH="./outputs/CapRLv2"
MODEL_CPK_NAME="Qwen2.5-VL-3B-75k-reward-qwen2.5-3b"

# ========== Wandb Config ==========
export WANDB_MODE=offline
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="CapRLv2"
export WANDB_DIR="${SAVE_PATH}/wandb/${MODEL_CPK_NAME}"
wandb login
echo "now wandb dir:${WANDB_DIR}"
mkdir -p $WANDB_DIR
mkdir -p "${SAVE_PATH}/${MODEL_CPK_NAME}"

# ========== Distributed Config ==========
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export NODE_RANK=${NODE_RANK:-0}
export CUDA_LAUNCH_BLOCKING=1

ray stop

if [ $NODE_RANK -eq 0 ]; then
    ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --temp-dir ~/.cache/ray
    sleep 5

    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json='{"working_dir": "./", "excludes": [".git"]}' \
       -- python3 -m openrlhf.cli.train_ppo_ray \
       --ref_num_nodes 2 \
       --ref_num_gpus_per_node 8 \
       --remote_rm_url http://<REWARD_SERVER_IP>:8889/get_reward \
       --actor_num_nodes 2 \
       --actor_num_gpus_per_node 8 \
       --vllm_num_engines 8 \
       --vllm_tensor_parallel_size 2 \
       --colocate_all_models \
       --vllm_enable_sleep \
       --vllm_gpu_memory_utilization 0.4 \
       --enable_prefix_caching \
       --pretrain $PRETRAIN_MODEL \
       --save_path $SAVE_PATH/$MODEL_CPK_NAME \
       --remove_pi_old \
       --reward_type "chart" \
       --exp_mode "cap_v2" \
       --cap_v2_multi_qa \
       --cap_v2_norm \
       --cap_v2_only_cap \
       --cap_remote_reward \
       --load_checkpoint \
       --format_weight 0.0 \
       --micro_train_batch_size 4 \
       --train_batch_size 1024 \
       --micro_rollout_batch_size 64 \
       --rollout_batch_size 128 \
       --temperature 1.0 \
       --n_caps_per_prompt 8 \
       --n_samples_per_prompt 8 \
       --max_epochs 1 \
       --num_episodes 30 \
       --prompt_max_len 4096 \
       --max_samples 100000 \
       --generate_max_len 4096 \
       --advantage_estimator rloo \
       --zero_stage 0 \
       --bf16 \
       --actor_learning_rate 1e-6 \
       --lr_warmup_steps 20 \
       --init_kl_coef 0.0 \
       --prompt_data $DATASET \
       --input_key message \
       --normalize_reward \
       --gradient_checkpointing \
       --save_steps 50 \
       --ckpt_path $SAVE_PATH/$MODEL_CPK_NAME/ckpt \
       --save_hf_ckpt \
       --train_vlm \
       --attn_implementation flash_attention_2 \
       --use_wandb $WANDB_API_KEY \
       --wandb_project $WANDB_PROJECT \
       --wandb_run_name $MODEL_CPK_NAME
else
    ray start --address="${MASTER_ADDR}:6379"
    sleep 60

    set +x
    while true; do
        ACTIVE_STATUS=$(ray status | grep Autoscaler | wc -l)
        if [ "$ACTIVE_STATUS" -lt 1 ]; then
            echo "No active Ray clusters. Stopping worker..."
            exit 0
        fi
        sleep 60
    done
fi
