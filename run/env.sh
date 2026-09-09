#!/usr/bin/env bash
# AtomAction-VLA (AA VLA) 运行环境。用法： source run/env.sh
#
# 约定：
#   - conda env         : aavla  (python 3.10 / torch 2.4.1+cu121 / numpy 1.26)
#   - CoppeliaSim 4.1   : 复用 /data0/weizeming/CoppeliaSim（不重复下载）
#   - PyRep             : source/PyRep -> /data0/weizeming/PyRep (commit 8f420be)
#   - RLBench           : source/RLBench (MohitShridhar/RLBench, PerAct fork)
#   - YARR              : source/YARR   (MohitShridhar/YARR, peract 分支)
#   - PerAct            : source/peract

VQAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VQAP_ROOT
export PERACT_ROOT="${VQAP_ROOT}/source/peract"

# ---- CoppeliaSim / PyRep ----
export COPPELIASIM_ROOT=/data0/weizeming/CoppeliaSim
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

# ---- conda env ----
CONDA_BASE=/home/weizeming/weizeming/miniconda3
export AAVLA_PYTHON="${CONDA_BASE}/envs/aavla/bin/python"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate aavla
fi

# ---- 本地密钥（存在才加载，chmod 600，勿提交）----
if [ -f "${VQAP_ROOT}/run/env.local.sh" ]; then
    . "${VQAP_ROOT}/run/env.local.sh"
fi

# ---- 其它 ----
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${VQAP_ROOT}:${PYTHONPATH}"

# ---- HuggingFace 离线模式 ----
# 🔴 2026-09-09 两次事故：CLIP text tower（openai/clip-vit-base-patch16）
#    每次评测都去 huggingface.co 校验，网络抖动时硬失败：
#      "Failed to load CLIP text tower" -> 分片启动即死
#    v3.5 跑丢 close_jar 整个任务；v3.6-test 12 个分片死了 6 个（150/300 局）。
#    模型本地已缓存 1.2 GB（~/.cache/huggingface/hub/），根本不需要联网。
#    离线模式让它直接用缓存，把这个故障源彻底去掉。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
