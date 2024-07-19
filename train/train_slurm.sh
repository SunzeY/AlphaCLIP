#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

PARTITION=llm3
JOB_NAME=a_clip
GPUS=128 # 128 for ViT-L/14@336px, 32 for ViT-L/14, 8 for ViT-B/16.
GPUS_PER_NODE=8
CPUS_PER_TASK=12
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=spot \
    --async \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train/train_grit_1m.py --lr 1e-4 \
    --para_gamma 0.01 \
    --weight_decay 2e-2 \
    --warmup_length 800 \
    --log_scale 4.6052 \
    --lora_rank -1 \
    --common_pair 0.1 \
    --resume \
    --amp \
    --epoch_num 10 \
    --subnum 1e7
