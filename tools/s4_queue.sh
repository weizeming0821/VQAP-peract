#!/usr/bin/env bash
# S4：B4 val 曲线，**串行**跑完剩余 ckpt。
#
# 为什么串行：实测 3 组并发（36 分片）聚合吞吐 3.4 局/min，
# 而单组 12 分片是 11.2 局/min —— 并发反而慢 3.3 倍（严重超订）。
# 评测是 CPU 瓶颈（GPU 利用率近 0%），多开只会互相抢核。
set -o pipefail
# 🔴 不能用 set -u：run/env.sh:36 是 `export PYTHONPATH="${VQAP_ROOT}:${PYTHONPATH}"`，
#    PYTHONPATH 未定义时在 set -u 下直接把脚本打死（第一次起队列就是这么挂的，
#    而且只在日志里留一行 "unbound variable"，看起来像启动失败而非逻辑错）。
#    先给这些变量兜底再 source。
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /data0/xiexiao/VQAP/run/env.sh
cd /data0/xiexiao/VQAP

# 等当前那一路（@10000）跑完
while pgrep -u "$(id -u)" -f "stage3_eval.py run --arm B4 --ckpt 10000 " >/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] B4@10000 已结束，开始串行队列"

for CK in 40000 2500 5000 20000 30000; do
    echo "[$(date +%H:%M:%S)] ===== B4@${CK} 开始 ====="
    python scripts/stage3_eval.py run --arm B4 --ckpt "$CK" --split val \
        --episodes 25 --shards 12 --gpu "$(python tools/gpu_reserver.py pick --opportunistic)" \
        --stagger 20 --display-base 130 \
        > "log/p5/S4_B4_val_${CK}.out" 2>&1
    echo "[$(date +%H:%M:%S)] ===== B4@${CK} 退出码 $? ====="
    tail -2 "log/p5/S4_B4_val_${CK}.out"
done
echo "[$(date +%H:%M:%S)] S4 队列全部完成"
