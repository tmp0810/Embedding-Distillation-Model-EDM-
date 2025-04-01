#!/bin/bash
GPUS=(0 1 2 3)
WORK_DIR=path_to_dskd_project
MASTER_PORT=66$(($RANDOM%90+10))
DEVICE=$(IFS=,; echo "${GPUS[*]}")

CKPT_PATH=${1}
BATCH_SIZE=${2-32}

for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} dolly ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} self-inst ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} vicuna ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} sinst/11_ ${BATCH_SIZE} $seed
done
for seed in 10 20 30 40 50
do
    bash ${WORK_DIR}/scripts/eval/eval_main.sh ${DEVICE} ${MASTER_PORT} ${#GPUS[@]} ${WORK_DIR} ${CKPT_PATH} uinst/11_ ${BATCH_SIZE} $seed 10000
done
