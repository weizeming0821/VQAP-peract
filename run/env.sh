#!/usr/bin/env bash
# AtomAction-VLA (AA VLA) 运行环境。用法： source run/env.sh
#
# 约定：
#   - conda env         : /root/autodl-tmp/envs/aavla  (python 3.10 / torch 2.8.0+cu128 / numpy 1.26.4)
#                         🔴 本机是 RTX 5090 (sm_120)，必须 torch>=2.7+cu128；
#                            旧机的 torch 2.4.1+cu121 不支持该架构。
#   - CoppeliaSim 4.1.0 : /root/autodl-tmp/CoppeliaSim (Edu V4_1_0_Ubuntu20_04)
#                         自带全部 Qt5/lua 库，靠 LD_LIBRARY_PATH 解析。
#   - PyRep             : source/PyRep (stepjam/PyRep, commit 8f420be)
#   - RLBench           : source/RLBench (MohitShridhar/RLBench, PerAct fork)
#   - YARR              : source/YARR   (MohitShridhar/YARR, peract 分支)
#   - PerAct            : source/peract
#
# 迁移备注（2026-09）：整仓从旧机 /data0/xiexiao/VQAP 迁到 /root/autodl-tmp/VQAP，
#   环境按上述新版本重建。数据等价性由 Step 1a 的逐字段比对验证。

VQAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export VQAP_ROOT
export PERACT_ROOT="${VQAP_ROOT}/source/peract"

# ---- CoppeliaSim / PyRep ----
export COPPELIASIM_ROOT=/root/autodl-tmp/CoppeliaSim
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"

# ---- conda env（prefix 式：env 建在数据盘，根分区只有 27G）----
CONDA_BASE=/root/miniconda3
export AAVLA_ENV=/root/autodl-tmp/envs/aavla
export AAVLA_PYTHON="${AAVLA_ENV}/bin/python"
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${AAVLA_ENV}"
fi

# ---- 本地密钥（存在才加载，chmod 600，勿提交）----
if [ -f "${VQAP_ROOT}/run/env.local.sh" ]; then
    . "${VQAP_ROOT}/run/env.local.sh"
fi

# ---- 其它 ----
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${VQAP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- HuggingFace 离线模式 ----
# 🔴 2026-09-09 两次事故：CLIP text tower（openai/clip-vit-base-patch16）
#    每次评测都去 huggingface.co 校验，网络抖动时硬失败：
#      "Failed to load CLIP text tower" -> 分片启动即死
#    v3.5 跑丢 close_jar 整个任务；v3.6-test 12 个分片死了 6 个（150/300 局）。
#    模型本地已缓存 1.2 GB（~/.cache/huggingface/hub/），根本不需要联网。
#    离线模式让它直接用缓存，把这个故障源彻底去掉。
#    ⚠️ 迁移到新机后需先确认缓存存在（run/env.sh 不做检查）。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
