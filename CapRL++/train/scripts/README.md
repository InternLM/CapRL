# CapRL++ Training Scripts

This directory contains the launch scripts for CapRL++ training with the bundled
verl source tree.

## Files

- `start_reward_serve_rm.sh`: starts the reward service and exposes
  `http://<host>:<port>/get_reward`.
- `train_caprl.sh`: launches single-node CapRL++ training with verl.

Both scripts infer `VERL_ROOT` as `CapRL++/train/verl`, so run them from
`CapRL++/train` unless you explicitly set `VERL_ROOT`.

## Prerequisites

Prepare a Python environment with the dependencies required by verl, vLLM, Ray,
PyTorch, and the model family you use. If you want the scripts to activate a
conda environment automatically, set `CONDA_ENV` before running them.

One possible setup is:

```bash
cd CapRL++/train

conda create -n caprl python=3.10 -y
conda activate caprl

pip install -r scripts/requirements.txt
pip install -e ./verl
```

You also need:

- a caption model, for example a local path or Hugging Face id for
  `Qwen3-VL-4B-Instruct`;
- a reward model, for example a local path or Hugging Face id for
  `Qwen3-4B-Instruct` when using `REWARD_SCORE_MODE=qa`;
- a training JSONL file compatible with the CapRL++ data loader (our training data **[CapRL-Video-QA-20K](https://huggingface.co/datasets/internlm/CapRL-Video-QA-20K)** has been released);
- writable output directories for checkpoints and optional W&B logs.

## 1. Start the reward service

Run this first. It can run on the same machine as training or on a separate
reward node.

```bash
cd CapRL++/train

REWARD_MODEL=/path/to/Qwen3-4B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
REWARD_PORT=18889 \
REWARD_NUM_WORKERS=1 \
bash scripts/start_reward_serve_rm.sh
```

Important reward variables:

- `REWARD_MODEL`: required. Reward model path or Hugging Face model id.
- `REWARD_PORT`: master service port. Defaults to `18889`.
- `REWARD_WORKER_BASE`: first worker port. Defaults to `REWARD_PORT + 10`.
- `CUDA_VISIBLE_DEVICES`: GPUs used by the reward service. Defaults to `0`.
- `REWARD_NUM_WORKERS`: number of reward workers. Defaults to `1` in the wrapper
  script.
- `REWARD_SCORE_MODE`: defaults to `qa`. Use `vl_judge` for direct VLM judge
  scoring.
- `REWARD_TASK`: defaults to `video`; set to `image` for image caption training.
- `REWARD_QA_NUM`: sampling rounds in `qa` mode. Defaults
  to `8`.
- `FORMAT_REWARD_WEIGHT`: video timestamp format reward weight. Defaults to
  `0.2`. For video captions, the unweighted format reward is
  `0.5 * N_valid / max(N_all, 1) + 0.5 * I_chrono`, where `N_all` is the
  number of timestamp-like brackets matched by the regex, `N_valid` is the
  number that satisfy logical constraints such as valid seconds and
  `t_end >= t_start`, and `I_chrono` is `1` only when valid timestamp start
  times are monotonically non-decreasing.

For `REWARD_SCORE_MODE=qa`, the reward model can be a text LLM. For
`REWARD_SCORE_MODE=vl_judge`, use a multimodal VLM.

## 2. Start video caption RL training

Run this after the reward service is ready.

```bash
cd CapRL++/train

CAPTION_MODEL=/path/to/Qwen3-VL-4B-Instruct \
DATASET=/path/to/video_train.jsonl \
SAVE_DIR=/path/to/output/checkpoints \
REWARD_NODE_IP=127.0.0.1 \
REWARD_PORT=18889 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_caprl.sh
```

If the reward service runs on another node, set `REWARD_NODE_IP` to that node's
reachable IP address. You can also bypass `REWARD_NODE_IP` and `REWARD_PORT` by
setting the full URL directly:

```bash
REWARD_REMOTE_URL=http://reward-node.example.com:18889/get_reward \
bash scripts/train_caprl.sh
```

Required training variables:

- `CAPTION_MODEL`: initial caption model path or Hugging Face model id.
- `DATASET`: training JSONL file.
- `SAVE_DIR`: checkpoint output directory.

Common training variables:

- `BATCH_SIZE`: generation and PPO mini-batch size. Defaults to `128`.
- `ROLLOUT_N`: number of responses sampled per prompt. Defaults to `8`.
- `TOTAL_EPOCHS`: total training epochs. Defaults to `3`.
- `SAVE_FREQ`: checkpoint interval in training steps. Defaults to `50`.
- `ACTOR_LR`: actor learning rate. Defaults to `1e-5`.
- `MAX_PROMPT_LENGTH`: data prompt length. Defaults to `4096`.
- `MAX_RESPONSE_LENGTH`: data response length. Defaults to `4096`.
- `ROLLOUT_PROMPT_LENGTH`: vLLM rollout prompt length. Defaults to `13000`.
- `ROLLOUT_RESPONSE_LENGTH`: vLLM rollout response length. Defaults to `4096`.
- `ROLLOUT_MAX_MODEL_LEN`: vLLM max model length. Defaults to `18000`.
- `ROLLOUT_GPU_MEMORY_UTILIZATION`: vLLM GPU memory fraction. Defaults to
  `0.88`.
- `ROLLOUT_AGENT_NUM_WORKERS`: async rollout workers. Defaults to `8`.
- `SAVE_HF_MODEL`: set to `False` to skip saving Hugging Face model weights.
- `WANDB_MODE`: defaults to `offline`.
- `WANDB_PROJECT`: defaults to `CapRL_video`.
- `WANDB_DIR`: defaults to `CapRL++/train/logs/wandb`.
- `RUN_NAME`: defaults to `qwen3_vl_4b_video`.

Length reward variables:

- `REWARD_LENGTH_TOKENIZER_PATH`: defaults to `CAPTION_MODEL`.
- `REWARD_LENGTH_L1`: defaults to `2048`.
- `REWARD_LENGTH_L2`: defaults to `3072`.
- `REWARD_LENGTH_WEIGHT`: defaults to `0.2`.

## Image caption training

The same reward service can be used for image caption training. Start it with
`REWARD_TASK=image`:

```bash
REWARD_TASK=image \
REWARD_MODEL=/path/to/Qwen3-4B-Instruct \
bash scripts/start_reward_serve_rm.sh
```

For training, `train_caprl.sh` currently documents the two image overrides in
comments:

```bash
data.input_type=image
data.prompt_key=prompt
```
