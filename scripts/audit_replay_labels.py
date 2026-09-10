#!/usr/bin/env python
"""扫描 replay 工件里**标签本身不可表示**的样本，并把它们标为不可采样。

# 为什么需要这个

PerAct 把动作离散成 100³ 体素网格的索引。`helpers/utils.point_to_voxel_index`
对**上界**做了 clamp（`np.minimum(..., dims_m_one)`），对**下界没有**：

    动作低于 bounds 下界  ->  索引为负  ->  one-hot 时静默绕回成另一个体素
    动作高于 bounds 上界  ->  索引被 clamp 成 99  ->  静默落在网格边缘

两种都是**错误的训练目标，且都不报错**。下界那种在 SE3 增广开着时会以
`Failing to perturb action and keep it within bounds` 的形式炸出来（增广要求
扰动后仍在界内，而本来就在界外的样本永远做不到）—— 那反而是运气好，
因为它至少响了。上界那种连响都不响。

# 2026-09-10 实测

Seen18 全量重建后，`turn_tap` 有 8 个样本的 y = −0.657，比下界 −0.5 低 0.157 m，
`trans_action_indicies = [22, **−16**, 52]`。它们来自同一个关键帧
（`demo_augmentation_every_n=10` 让同一目标重复 8 次）。B1 与 B4 训练分别在
step ~1500 / ~500 被它们打死 —— `task_uniform` 下每步抽中概率约 1.2e-3，
1500 步内命中概率 84%，与实测吻合。

旧机只训 Seen12 从未碰到：`turn_tap` 是 Seen18 新增的 6 个任务之一。

# 处置

把这些样本的 `terminal` 置 **−1**。这不是新机制 —— `add_final` 的填充帧用的
就是它：`uniform_replay_buffer.is_valid_transition` 里
`if term_stack[-1] == -1: return False`，采样器遇到无效索引会自动重试
（`task_uniform_replay_buffer.sample_index_batch`）。于是这些样本**永久不被采样**，
而不需要重建 292 GB 的工件、也不会在 cursor 序列里留洞。

    python scripts/audit_replay_labels.py                 # 只扫描并报告
    python scripts/audit_replay_labels.py --apply         # 扫描并标记
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPLAY = (REPO_ROOT / "aavla_data" / "replay" / "seen18" /
                  "multi" / "PERACT_BC" / "seed0")
#: 与 conf/stage3.yaml 的 rlbench.scene_bounds 一致。
DEFAULT_BOUNDS = (-0.3, -0.5, 0.6, 0.7, 0.5, 1.6)
VOXEL_SIZE = 100


def _scan_one(path_str: str) -> dict | None:
    """→ 该样本的缺陷描述；无缺陷返回 None。"""
    p = Path(path_str)
    try:
        with p.open("rb") as f:
            d = pickle.load(f)
    except Exception as exc:                                  # noqa: BLE001
        return {"cursor": int(p.stem), "task": "?", "why": f"读不出: {exc}"}
    idx = d.get("trans_action_indicies")
    if idx is None:
        return None
    idx = np.asarray(idx)
    pose = np.asarray(d.get("gripper_pose", [np.nan] * 7), dtype=float)
    lo = np.array(DEFAULT_BOUNDS[:3])
    hi = np.array(DEFAULT_BOUNDS[3:])
    why = []
    if (idx < 0).any():
        why.append("索引为负（动作低于 bounds 下界，one-hot 会绕回）")
    if np.isfinite(pose[:3]).all():
        if (pose[:3] < lo).any():
            why.append("位姿低于下界")
        if (pose[:3] > hi).any():
            # 上界越界被 point_to_voxel_index 静默 clamp 成 99，比下界更隐蔽
            why.append("位姿高于上界（索引被静默 clamp 成 99）")
    if not why:
        return None
    return {"cursor": int(p.stem), "task": d.get("task", "?"),
            "pose": [round(float(x), 4) for x in pose[:3]],
            "idx": idx.tolist(), "terminal": int(d.get("terminal", 0)),
            "why": " / ".join(why)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=str(DEFAULT_REPLAY))
    ap.add_argument("--apply", action="store_true",
                    help="把查出的样本 terminal 置 -1（永久不可采样）")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    d = Path(a.replay)
    files = sorted((p for p in d.iterdir() if p.suffix == ".replay"),
                   key=lambda p: int(p.stem))
    print(f"扫描 {len(files):,} 个样本 …  {d}")
    with mp.Pool(a.workers) as pool:
        got = pool.map(_scan_one, [str(p) for p in files], chunksize=512)
    bad = [g for g in got if g]

    if not bad:
        print("✅ 未发现标签不可表示的样本")
        return 0

    by_task: dict[str, list] = {}
    for g in bad:
        by_task.setdefault(g["task"], []).append(g)
    print(f"\n🔴 发现 {len(bad)} 个缺陷样本，分布在 {len(by_task)} 个任务：")
    for t, gs in sorted(by_task.items(), key=lambda kv: -len(kv[1])):
        print(f"  {t:30s} {len(gs):5d} 个")
        for g in gs[:3]:
            print(f"      cursor={g['cursor']:<7d} pose={g['pose']} "
                  f"idx={g['idx']}  {g['why']}")
        if len(gs) > 3:
            print(f"      …另有 {len(gs) - 3} 个")

    already = [g for g in bad if g["terminal"] == -1]
    if already:
        print(f"\n  其中 {len(already)} 个的 terminal 已经是 -1（本就不可采样）")

    if not a.apply:
        print("\n（只扫描未改动。加 --apply 把它们标为不可采样）")
        return 1

    # ---- 标记：_INDEX.pkl 与 .replay 双写，两处都改才算数 ----
    idx_path = d / "_INDEX.pkl"
    index = pickle.loads(idx_path.read_bytes())
    term = np.asarray(index["terminal"])
    n_idx = 0
    for g in bad:
        if term[g["cursor"]] != -1:
            term[g["cursor"]] = -1
            n_idx += 1
    index["terminal"] = term
    idx_path.write_bytes(pickle.dumps(index))
    print(f"\n  ✅ _INDEX.pkl：{n_idx} 个样本的 terminal 置 -1")

    n_file = 0
    for g in bad:
        p = d / f"{g['cursor']}.replay"
        with p.open("rb") as f:
            rec = pickle.load(f)
        if int(rec.get("terminal", 0)) != -1:
            rec["terminal"] = np.int8(-1)
            with p.open("wb") as f:
                pickle.dump(rec, f)
            n_file += 1
    print(f"  ✅ .replay 文件：{n_file} 个的 terminal 置 -1")

    # ---- 记进 manifest，留下可审计的痕迹 ----
    man_path = d / "_MANIFEST.json"
    man = json.loads(man_path.read_text())
    man.setdefault("label_audits", []).append({
        "date": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "tool": "scripts/audit_replay_labels.py",
        "bounds": list(DEFAULT_BOUNDS),
        "n_marked": len(bad),
        "per_task": {t: len(gs) for t, gs in by_task.items()},
        "cursors": [g["cursor"] for g in bad],
        "why": "动作超出 scene_bounds ⇒ 体素索引不可表示（负索引会绕回，"
               "上界越界被静默 clamp）。terminal=-1 使其永久不被采样。",
    })
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"  ✅ _MANIFEST.json 已记录本次审计")
    print(f"\n有效样本数：{man['n_samples']:,} − {len(bad)} = "
          f"{man['n_samples'] - len(bad):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
