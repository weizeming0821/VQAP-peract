#!/usr/bin/env python
"""关键帧统计与 replay 体积精算。

对 RLBench demo 跑 PerAct 自己的 `keypoint_discovery`，产出三类信息：

1. **关键帧粒度**（VLA_Design §7 T3）：每 (task, variation) 的关键帧数分布，
   与 `Phase_Action_Label.csv` 人工先验的段数对照。若关键帧数 < 先验段数，
   「子任务 = 连续关键帧区间」这一地基假设对该任务不成立，先验必须合并。

2. **replay 样本数精算**：PerAct 的 `fill_replay` 对每条 demo，按
   `demo_augmentation_every_n` 取起始帧，再对每个「晚于该起始帧的关键帧」生成一个样本。
   本脚本解析地复现这个计数，无需真的填 replay，即可给出精确的样本数与体积。

3. **夹爪翻转模式**：关键帧上 `gripper_open` 的变化序列，是离线分段与在线 Planner
   触发规则的共同基础信号。

用法：
    source run/env.sh
    python scripts/analyze_keyframes.py --splits train --out result/p1_data/keyframe_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

PERACT_18 = [
    "open_drawer", "slide_block_to_color_target", "sweep_to_dustpan_of_size",
    "meat_off_grill", "turn_tap", "put_item_in_drawer", "close_jar",
    "reach_and_drag", "stack_blocks", "light_bulb_in", "put_money_in_safe",
    "place_wine_at_rack_location", "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter", "push_buttons", "insert_onto_square_peg",
    "stack_cups", "place_cups",
]
# 🔴 Seen12 划分已于 2026-09 作废（现在是 Seen18 全集，见 stage3/tasks.py）。
# 本脚本是 replay 体积估算工具，输出里的 `seen12_*` 字段只对旧工件有意义；
# 保留字段名是为了旧的 keyframe_stats.json 仍能读，**不要拿它做新的口径判断**。
from stage3.tasks import LEGACY_SEEN12                                # noqa: E402
SEEN12 = set(LEGACY_SEEN12)
# fill_replay 的默认值，三臂一致（VLA_Design §4.3）
DEMO_AUG_EVERY_N = 10
# P0 实测：open_drawer 10 demo -> 373 个 .replay 文件 / 410 MB
MB_PER_SAMPLE = 410.0 / 373.0


def replay_samples(demo_len: int, keypoints: list[int]) -> int:
    """解析复现 `_add_keypoints_to_replay` 的样本计数。

    fill_replay 对 i in range(0, len(demo)-1, N) 取起始帧 obs=demo[i]，
    再对每个 keypoint > i 生成一个样本（obs 随之前进到该关键帧）。
    """
    n = 0
    for i in range(0, demo_len - 1, DEMO_AUG_EVERY_N):
        n += sum(1 for k in keypoints if k > i)
    return n


def process_episode(args: tuple[str, str, str]) -> dict | None:
    split, task, ep_dir = args
    try:
        from helpers.demo_loading_utils import keypoint_discovery
        from rlbench.demo import Demo

        with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "rb") as f:
            obs = pickle.load(f)
        demo = obs if isinstance(obs, Demo) else Demo(obs)
        kps = keypoint_discovery(demo)
        with open(os.path.join(ep_dir, "variation_number.pkl"), "rb") as f:
            var = int(pickle.load(f))
        grip = [int(round(float(demo[k].gripper_open))) for k in kps]
        return {
            "split": split, "task": task, "variation": var,
            "episode": os.path.basename(ep_dir),
            "demo_len": len(demo), "n_keypoints": len(kps), "keypoints": kps,
            "gripper_at_keypoints": grip,
            "replay_samples": replay_samples(len(demo), kps),
        }
    except Exception as exc:  # 单条 episode 失败不应中断全局统计
        return {"split": split, "task": task, "episode": os.path.basename(ep_dir),
                "error": f"{type(exc).__name__}: {exc}"}


def load_prior(csv_path: Path) -> dict[str, list[str]]:
    """Phase_Action_Label.csv -> {task: 该 task 第一条 variation 的动作序列}。"""
    prior: dict[str, list[str]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t = row["Task"]
            if t in prior:
                continue
            prior[t] = [row[f"Phase{i}"] for i in range(34)
                        if row.get(f"Phase{i}")]
    return prior


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "aavla_data/rlbench"))
    ap.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "val", "test"])
    ap.add_argument("--tasks", nargs="+", default=PERACT_18)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=str(REPO_ROOT / "result" / "p1_data" / "keyframe_stats.json"))
    args = ap.parse_args()

    root = Path(args.root)
    jobs = []
    for split in args.splits:
        for task in args.tasks:
            d = root / split / task / "all_variations" / "episodes"
            if not d.is_dir():
                continue
            for ep in sorted(d.iterdir()):
                if (ep / "low_dim_obs.pkl").is_file():
                    jobs.append((split, task, str(ep)))
    print(f"待分析 {len(jobs)} 条 episode，{args.workers} 进程")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(process_episode, jobs, chunksize=8))

    errs = [r for r in records if r and "error" in r]
    records = [r for r in records if r and "error" not in r]
    if errs:
        print(f"⚠️ {len(errs)} 条 episode 分析失败，例：{errs[0]}")

    prior = load_prior(REPO_ROOT / "Phase_Action_Label.csv")

    # ---- 逐 task 汇总 ----
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)

    print()
    print("=" * 104)
    print("关键帧粒度 vs Phase_Action_Label.csv 先验段数")
    print("=" * 104)
    hdr = (f"{'task':32s} {'ep':>5} {'demo长':>12} {'关键帧':>14} "
           f"{'先验段':>6} {'差':>5} {'样本/demo':>9}")
    print(hdr)
    summary = {}
    red = []
    for task in PERACT_18:
        rs = by_task.get(task, [])
        if not rs:
            continue
        kp = sorted(r["n_keypoints"] for r in rs)
        dl = sorted(r["demo_len"] for r in rs)
        sm = [r["replay_samples"] for r in rs]
        med_kp = kp[len(kp) // 2]
        n_prior = len(prior.get(task, []))
        gap = med_kp - n_prior if n_prior else None
        flag = ""
        if n_prior and med_kp < n_prior:
            flag = "  🔴 关键帧少于先验段"
            red.append(task)
        print(f"{task:32s} {len(rs):>5} "
              f"{f'{dl[0]}~{dl[-1]}':>12} "
              f"{f'{kp[0]}~{kp[-1]} (中{med_kp})':>14} "
              f"{n_prior if n_prior else '-':>6} "
              f"{gap if gap is not None else '-':>5} "
              f"{sum(sm)/len(sm):>9.1f}{flag}")
        summary[task] = {
            "n_episodes": len(rs),
            "demo_len": {"min": dl[0], "median": dl[len(dl) // 2], "max": dl[-1]},
            "n_keypoints": {"min": kp[0], "median": med_kp, "max": kp[-1],
                            "hist": dict(sorted(Counter(kp).items()))},
            "n_prior_segments": n_prior,
            "prior_sequence": prior.get(task, []),
            "replay_samples_per_demo": round(sum(sm) / len(sm), 2),
            "replay_samples_total": sum(sm),
            "seen12": task in SEEN12,
        }

    # ---- replay 体积 ----
    print()
    print("=" * 104)
    print("Replay 体积精算（单样本 %.2f MB，P0 实测）" % MB_PER_SAMPLE)
    print("=" * 104)
    tot_all = sum(v["replay_samples_total"] for v in summary.values())
    tot_s12 = sum(v["replay_samples_total"] for v in summary.values() if v["seen12"])
    for name, n in (("Seen12（Stage 3 主线训练集）", tot_s12), ("全部 18 任务", tot_all)):
        print(f"  {name:32s} {n:>9,} 样本  →  {n * MB_PER_SAMPLE / 1024:>7.1f} GB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "config": {"splits": args.splits, "demo_augmentation_every_n": DEMO_AUG_EVERY_N,
                       "mb_per_replay_sample": round(MB_PER_SAMPLE, 4),
                       "keypoint_method": "heuristic"},
            "per_task": summary,
            "replay_estimate": {
                "seen12_samples": tot_s12,
                "seen12_gb": round(tot_s12 * MB_PER_SAMPLE / 1024, 1),
                "all18_samples": tot_all,
                "all18_gb": round(tot_all * MB_PER_SAMPLE / 1024, 1),
            },
            "tasks_with_fewer_keypoints_than_prior": red,
            "episodes": records,
        }, f, indent=1)
    print(f"\n明细写入 {out}")
    if red:
        print(f"🔴 {len(red)} 个任务的关键帧数中位数少于 CSV 先验段数：{red}")
        print("   → 这些任务的先验无法 1:1 落到关键帧上，Planner 必须合并先验段")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
