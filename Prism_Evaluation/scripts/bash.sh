#!/bin/bash

bench_idx=$1

cd CapRL/Prism_Evaluation

export NODE_RANK=${NODE_RANK}
export NNODES=${NODE_COUNT:-2}
export PROC_PER_NODE=${PROC_PER_NODE:-1}

export MODEL="PATH/CapRL/CapRL-3B"

# (100 200 300 400 500 600 700 800 900 1000 1100 1200)
export STEPS=(100 200 300 400 500 600 700 800 900 1000 1100 1200)
export STEP=${STEPS[$NODE_RANK]}

echo "===================================="
echo "NNODES=${NNODES}"
echo "Current (NODE_RANK): $NODE_RANK"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "All (STEPS): ${STEPS[@]}"
echo "Current (STEP): ${STEP}"
echo "===================================="

DATA_LIST=(
    "PATH/lmm_eval_chartqa.json"
    "PATH/lmm_eval_infovqa.json"
    "PATH/lmm_eval_MMStar.json"
    "PATH/lmm_eval_mmmu.json"
    "PATH/lmm_eval_CharXiv.json"
    "PATH/lmm_eval_ChartQA_Pro.json"
    "PATH/lmm_eval_MMMU_Pro.json"
    "PATH/lmm_eval_seed2k.json"
    "PATH/lmm_eval_MathVerse_MINIVInt.json"
    "PATH/lmm_eval_MathVision.json"
    "PATH/lmm_eval_WeMath.json"
    "PATH/lmm_eval_Visu.json"
)

TAG_LIST=(
    "chartqa"
    "infovqa"
    "MMStar"
    "mmmu"
    "CharXiv"
    "ChartQA_Pro"
    "MMMU_Pro"
    "seed2k"
    "MathVerse_MINIVInt"
    "MathVision"
    "WeMath"
    "VisuLogic"
)

echo "bench_idx : $bench_idx"

export DATA="${DATA_LIST[${bench_idx}]}"
export TAG="${TAG_LIST[${bench_idx}]}"

echo "===================================="
echo "current bench: $DATA"
echo "current tag: $TAG"
echo "===================================="


if [ ${#DATA_LIST[@]} -ne ${#TAG_LIST[@]} ]; then
    echo "error not identical"
    exit 1
fi


export GEN_NUM=4 

CUDA_VISIBLE_DEVICES=0 python -u -m Eval_CapRL \
    --model-path ${MODEL} \
    --data-path ${DATA} \
    --step ${STEP} \
    --tag ${TAG} \
    --gen_num ${GEN_NUM} \
    --eval_bs 1 \
    --stage_num 1

CUDA_VISIBLE_DEVICES=0 python -u -m Eval_CapRL \
    --model-path ${MODEL} \
    --data-path ${DATA} \
    --step ${STEP} \
    --tag ${TAG} \
    --gen_num ${GEN_NUM} \
    --eval_bs 1 \
    --stage_num 2
