#!/usr/bin/env sh

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "${REPO_ROOT}"

# Adapter 为单卡训练（实测 num_workers=32 / batch=64 下 2.1 min/epoch，100 epoch 约 3.5 小时）。
# 共享服务器上按空闲情况选卡：CUDA_VISIBLE_DEVICES=3 bash run/train_adapter.sh
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

python scripts/train_adapter.py --device cuda:0 "$@"
