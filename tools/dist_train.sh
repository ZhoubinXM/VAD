#!/usr/bin/env bash

CONFIG=projects/configs/VAD/VAD_tiny_stage_1.py
GPUS=8
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic \
    --work-dir outputs/tiny_stage_1
