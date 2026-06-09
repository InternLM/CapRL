# CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning

[[Paper](https://arxiv.org/abs/2606.09393)] [[Project](https://github.com/InternLM/CapRL)] [[Model](https://huggingface.co/internlm/CapRL-Video-4B)] [[Dataset](https://huggingface.co/datasets/internlm/CapRL-Video-178K)]

**Authors:** Penghui Yang*, Long Xing*, Xiaoyi Dong, Yuhang Zang, Yuhang Cao,
Yibin Wang, Yujie Zhou, Jiazi Bu, Jianze Liang, Qidong Huang, Jiaqi Wang,
Feng Wu, Dahua Lin.

CapRL++ is a unified reinforcement learning framework for dense image and video
captioning with verifiable rewards. While the original CapRL release focuses on
dense image captioning, CapRL++ keeps the same central idea: train a caption
model with reward signals that measure whether the generated caption preserves
enough visual information for downstream question answering.

This folder contains the training and evaluation code used for the CapRL++
extension.

## Overview

CapRL++ adds three practical components on top of CapRL:

- unified image and video caption RL training based on the bundled verl
  framework;
- a remote reward service for QA-based or VLM-judge-based verifiable reward
  scoring;
- a video Prism evaluation pipeline that measures caption usefulness through
  downstream benchmark QA.

The training code is self-contained under `train/`. The evaluation code is
self-contained under `eval/`.

## Repository Layout

```text
CapRL++/
├── train/
│   ├── scripts/
│   │   ├── train_caprl.sh
│   │   ├── start_reward_serve_rm.sh
│   │   ├── requirements.txt
│   │   └── README.md
│   └── verl/
│       └── recipe/video_captionrl/
└── eval/
    ├── scripts/
    ├── tools/
    ├── requirements.txt
    └── README.md
```

## Training

The training implementation uses the bundled `train/verl` source tree. Start the
reward service first, then launch RL training.

```bash
cd CapRL++/train

conda create -n caprl python=3.10 -y
conda activate caprl

pip install -r scripts/requirements.txt
pip install -e ./verl
```

Start the reward service:

```bash
REWARD_MODEL=/path/to/Qwen3-4B-Instruct \
CUDA_VISIBLE_DEVICES=0 \
REWARD_PORT=18889 \
bash scripts/start_reward_serve_rm.sh
```

Launch training:

```bash
CAPTION_MODEL=/path/to/Qwen3-VL-4B-Instruct \
DATASET=/path/to/video_train.jsonl \
SAVE_DIR=/path/to/output/checkpoints \
REWARD_NODE_IP=127.0.0.1 \
REWARD_PORT=18889 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/train_caprl.sh
```

See `train/scripts/README.md` for the full list of environment variables and
runtime options.

## Evaluation

The evaluation code implements a two-stage Prism-style pipeline:

1. generate captions for videos with a caption model;
2. answer benchmark questions using only the generated captions;
3. score the downstream answers against benchmark annotations.

```bash
cd CapRL++/eval
pip install -r requirements.txt

source examples/env.example
bash scripts/run_vllm_prism.sh
```

Supported benchmarks include Video-MME, MVBench, MMVU, MMBench-Video, TOMATO,
and TimeLens-Bench. See `eval/README.md` for benchmark-specific paths and
evaluation options.

## Relation to CapRL

CapRL++ follows the CapRL philosophy of optimizing caption models through
question-answering feedback, but extends the workflow to video captioning and
uses verl as the RL training backend. For image caption models, datasets, and
the original CapRL training and evaluation pipeline, refer to the main
repository README.

## Citation

```bibtex
@article{yang2026caprlplusplus,
  title={CapRL++: Unified Reinforcement Learning with Verifiable Rewards for Dense Image and Video Captioning},
  author={Yang, Penghui and Xing, Long and Dong, Xiaoyi and Zang, Yuhang and Cao, Yuhang and Wang, Yibin and Zhou, Yujie and Bu, Jiazi and Liang, Jianze and Huang, Qidong and Wang, Jiaqi and Wu, Feng and Lin, Dahua},
  journal={arXiv preprint arXiv:2606.09393},
  year={2026}
}
```
