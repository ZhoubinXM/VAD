#!/usr/bin/env bash

python tools/data_converter/vad_nuscenes_converter.py \
 nuscenes \
 --root-path ./data/nuscenes/trainval \
 --out-dir ./data/nuscenes/trainval/infos \
 --extra-tag vad_nuscenes_3_6 \
 --version v1.0 \
 --canbus ./data/nuscenes/trainval