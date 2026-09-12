"""按原子取码向量的**质心** —— 把码里的任务身份洗掉，只留动作语义。

# 为什么（实测诊断）

直接在 `data/atomaction_codebook_index.json` 的 56,496 条记录上算互信息：

    I(k_global; task  ) = 1.93 bit      ← 码更像「任务指纹」
    I(k_global; action) = 1.20 bit
    I(k_global; action | task) = 1.10 bit   而 H(action) = 2.97 bit

也就是说**码携带的任务身份多于动作语义**。而任务身份正是 PerAct 从图像和
整任务指令里**已经拿到**的信息：

  · 在 Seen 上 → 冗余，只会加噪（实测 B4 的训练 loss 高于无码基线）
  · 在 UnSeen 上 → 一个没见过的任务的指纹，本身就没有意义

# 做法

对每个原子 `a`，把语料里与它共现的全部 `k_global` 按出现频次加权平均：

    centroid[a] = Σ_k  p(k | a) · codebook.global[k]

注入时用 `centroid[subtask_action]` 代替 `codebook.global[k_global]`。
跨任务平均把任务分量抵消掉，留下的是该原子的共性方向 —— **任务无关，
因此能迁移到 UnSeen**（那里任务指纹无意义，但「这是一次 grasp」仍然成立）。

# 这是一个判决性实验

  · 若 centroid ≥ 原码 ⇒ 码的价值在**动作语义**，任务指纹是噪声；
  · 若 centroid < 原码 ⇒ 码里还有超出动作类别的实例级信息（那正是细节码
    本该承载、却 99.7% 退化掉的那一层），值得回头修 Stage 0/1。

两种结果都有用，所以值得跑。

# 代价

零。不重训码本、不重建 replay、不增加任何可训参数 —— 只是换一张查找表。
`subtask_action` 在 replay 里 100% 可用，在线 planner 也逐段输出动作名。
"""

from __future__ import annotations

import collections
import functools
import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "data" / "atomaction_codebook_index.json"


@functools.lru_cache(maxsize=1)
def action_code_hist() -> dict[str, dict[int, int]]:
    """→ {原子: {k_global: 出现次数}}，取自码本预训练语料。"""
    recs = json.loads(INDEX.read_text())["records"]
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        out[r["action"]][int(r["k_global"])] += 1
    return {a: dict(c) for a, c in out.items()}


def build_centroids(global_codebook: torch.Tensor,
                    match_norm: bool = True) -> tuple[torch.Tensor, dict[str, int]]:
    """→ (centroids [A, D], {原子: 行号})。

    `global_codebook` 是 `load_codebook()` 装出来的 `quantizer.codebooks`，
    形状 [36, 512]。⚠️ 不要错拿 `projection.weight`（也是 2 维、512×512）——
    那是投影矩阵不是码本，我第一版就拿错了。

    未知原子（planner 偶尔会给白名单外的动作）映射到最后一行 —— **全零**，
    与 `code_mask=0` 的效果一致，这样「没见过的动作」不会拿到某个别的原子的质心。

    `match_norm`：把质心的范数缩放到原码的平均范数。跨码平均会让范数缩水
    （实测 0.39 vs 0.69），不校正的话注入幅度就同时变了 —— 那就分不清
    「质心更好/更差」是因为**内容**还是因为**强度**。这个实验要隔离的是内容。
    """
    hist = action_code_hist()
    acts = sorted(hist)
    D = global_codebook.shape[1]
    cent = torch.zeros(len(acts) + 1, D, dtype=global_codebook.dtype)
    for i, a in enumerate(acts):
        ks = torch.tensor(sorted(hist[a]), dtype=torch.long)
        w = torch.tensor([hist[a][int(k)] for k in ks], dtype=global_codebook.dtype)
        w = w / w.sum()
        cent[i] = (global_codebook[ks] * w.unsqueeze(1)).sum(0)
    if match_norm:
        tgt = global_codebook.norm(dim=1).mean()
        n = cent[:-1].norm(dim=1, keepdim=True).clamp_min(1e-6)
        cent[:-1] = cent[:-1] / n * tgt
    index = {a: i for i, a in enumerate(acts)}
    index["<unk>"] = len(acts)                      # 全零行
    return cent, index


def summary() -> str:
    hist = action_code_hist()
    parts = [f"{a}({len(ks)}码/{sum(ks.values())}段)"
             for a, ks in sorted(hist.items(), key=lambda kv: -sum(kv[1].values()))]
    return f"action_centroid: {len(hist)} 个原子 -> " + " ".join(parts[:6]) + " …"
