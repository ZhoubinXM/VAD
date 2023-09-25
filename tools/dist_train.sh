#!/usr/bin/env bash

# CONFIG=projects/configs/VAD/VAD_tiny_stage_1.py
CONFIG=projects/configs/ADMS/adms_tiny_stage_1.py
GPUS=7
PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic \
    --work-dir outputs/adms_tiny_stage_1 \
     --cfg-options \
     data.train.ann_file='data/nuscenes/trainval/infos/vad_nuscenes_2_3_infos_temporal_train.pkl'\
     data.val.ann_file='data/nuscenes/trainval/infos/vad_nuscenes_2_3_infos_temporal_val.pkl'\
     data.test.ann_file='data/nuscenes/trainval/infos/vad_nuscenes_2_3_infos_temporal_val.pkl'\
    # --work-dir outputs/tiny_stage_1
