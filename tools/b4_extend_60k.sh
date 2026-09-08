#!/usr/bin/env bash
# B4 从 40000 续训到 60001。
# 60000 是 B3 的 val 峰值点，两臂在同一步数上可严格对比；也堵住
# 「没训到收敛」的质疑（Exp_Design §E2 要求给学习曲线）。
#
# 🔴 不能用 set -u：run/env.sh 的 PYTHONPATH 未定义时会把脚本直接打死。
set -o pipefail
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /data0/xiexiao/VQAP/run/env.sh

# 等 B2 的评测收尾，避免和 12 个分片抢 CPU（负载 527 那次的教训）
while pgrep -u "$(id -u)" -f "stage3_eval.py run --arm B2 --ckpt 40000" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] B2 评测已结束，准备续训 B4"

cd /data0/xiexiao/VQAP
python tools/gpu_reserver.py hold --gpus 3,4        # 把 3,4 让给训练
sleep 5
cd /data0/xiexiao/VQAP/source/peract
CUDA_VISIBLE_DEVICES=3,4 python train.py --config-name=stage3 \
    stage3.arm=B4 ddp.master_port=29560 \
    framework.training_iterations=60001 \
    framework.num_weights_to_keep=30
echo "[$(date +%H:%M:%S)] B4 续训退出码 $?"
