#!/usr/bin/env python
"""RLBench 数据集完整性与纯净性审计。

按 `VLA_Design §2.5` 的「防泄漏硬约束」精神，任何「多套数据共存、按配置选用」的结构
都必须有硬校验。本脚本对生成好的 `data_rlbench/` 做六项检查，任一项失败即报错退出。

用法：
    source run/env.sh
    python scripts/audit_dataset.py
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "RLBench")):
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
SPLIT_EPISODES = {"train": 100, "val": 25, "test": 25}
CAMERAS = ["front", "left_shoulder", "right_shoulder", "overhead", "wrist"]
MODALITIES = ["rgb", "depth", "mask"]


def audit(root: Path, splits: list[str], tasks: list[str]) -> tuple[list[str], dict]:
    errors: list[str] = []
    stats: dict = {}
    # 每个任务的指令词表，用于跨任务污染检测
    task_vocab: dict[str, set[str]] = collections.defaultdict(set)
    desc_by_task_var: dict[tuple[str, int], set[str]] = collections.defaultdict(set)

    for split in splits:
        for task in tasks:
            d = root / split / task / "all_variations" / "episodes"
            key = f"{split}/{task}"
            if not d.is_dir():
                errors.append(f"[缺目录] {key}: {d} 不存在")
                continue

            # 按 episode 编号的数值排序，不能用字典序（episode10 会排在 episode2 前）
            names = sorted((p.name for p in d.iterdir() if p.is_dir()),
                           key=lambda n: int(n.removeprefix("episode")) if
                           n.removeprefix("episode").isdigit() else 10 ** 9)
            want = SPLIT_EPISODES[split]

            # ① episode 目录必须恰好是 episode0..episode{N-1}，无空缺无多余
            expect = {f"episode{i}" for i in range(want)}
            if set(names) != expect:
                extra = set(names) - expect
                missing = expect - set(names)
                errors.append(f"[编号不连续] {key}: 期望 episode0..{want-1}，"
                              f"多出 {sorted(extra)[:5]}，缺失 {sorted(missing)[:5]}")

            n_ok = 0
            for name in names:
                ep = d / name
                # ② 三个必需 pickle 齐全
                miss = [f for f in ("low_dim_obs.pkl", "variation_number.pkl",
                                    "variation_descriptions.pkl")
                        if not (ep / f).is_file()]
                if miss:
                    errors.append(f"[缺文件] {key}/{name}: 缺 {miss}")
                    continue

                with (ep / "low_dim_obs.pkl").open("rb") as f:
                    demo = pickle.load(f)
                n_frames = len(demo)
                with (ep / "variation_number.pkl").open("rb") as f:
                    var = int(pickle.load(f))
                with (ep / "variation_descriptions.pkl").open("rb") as f:
                    descs = pickle.load(f)

                # ③ 五相机 × 三模态的帧数必须与 demo 长度一致
                bad_cam = []
                for cam in CAMERAS:
                    for mod in MODALITIES:
                        sub = ep / f"{cam}_{mod}"
                        if not sub.is_dir():
                            bad_cam.append(f"{cam}_{mod}=缺目录")
                            continue
                        n = sum(1 for _ in sub.iterdir())
                        if n != n_frames:
                            bad_cam.append(f"{cam}_{mod}={n}")
                if bad_cam:
                    errors.append(f"[帧数不符] {key}/{name}: demo={n_frames} 但 "
                                  f"{bad_cam[:4]}")

                # ④ demo 长度合理（至少能跑出关键帧）
                if n_frames < 10:
                    errors.append(f"[退化 demo] {key}/{name}: 仅 {n_frames} 帧")

                # ⑤ variation 号非负
                if var < 0:
                    errors.append(f"[非法 variation] {key}/{name}: {var}")

                for s in descs:
                    desc_by_task_var[(task, var)].add(s)
                    task_vocab[task] |= set(s.lower().split())
                n_ok += 1

            stats[key] = {"episodes": n_ok, "expected": want}
            if n_ok != want:
                errors.append(f"[数量不符] {key}: {n_ok}/{want}")

    # ⑥ 跨任务污染：一个 (task, variation) 下的指令集合必须自洽；
    #    且任务之间的指令不得互相出现（用整句而非词，词会天然重叠）
    all_desc: dict[str, set[str]] = collections.defaultdict(set)
    for (task, var), ds in desc_by_task_var.items():
        all_desc[task] |= ds
    cross = []
    tasks_present = [t for t in tasks if t in all_desc]
    for i, a in enumerate(tasks_present):
        for b in tasks_present[i + 1:]:
            shared = all_desc[a] & all_desc[b]
            if shared:
                cross.append((a, b, sorted(shared)[:3]))
    for a, b, sh in cross:
        errors.append(f"[跨任务污染] {a} 与 {b} 共享指令 {sh}")

    stats["_desc_counts"] = {t: len(all_desc[t]) for t in tasks_present}
    return errors, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "data_rlbench"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--tasks", nargs="+", default=PERACT_18)
    ap.add_argument("--out", default=str(REPO_ROOT / "data_rlbench" / "audit_report.json"))
    args = ap.parse_args()

    errors, stats = audit(Path(args.root), args.splits, args.tasks)

    print("=" * 80)
    print("数据集审计")
    print("=" * 80)
    total = sum(v["episodes"] for k, v in stats.items() if not k.startswith("_"))
    want = sum(v["expected"] for k, v in stats.items() if not k.startswith("_"))
    print(f"作业数 {len([k for k in stats if not k.startswith('_')])}  "
          f"episode {total}/{want}")
    print(f"每任务的唯一指令数：")
    for t, n in sorted(stats["_desc_counts"].items(), key=lambda x: -x[1]):
        print(f"   {t:32s} {n:>4}")
    print()
    if errors:
        print(f"❌ {len(errors)} 项检查失败：")
        for e in errors[:40]:
            print(f"   {e}")
        if len(errors) > 40:
            print(f"   ... 另 {len(errors)-40} 项")
    else:
        print("✅ 六项检查全部通过：")
        print("   ① episode 编号 0..N-1 连续无空缺无多余")
        print("   ② 每条 episode 的三个必需 pickle 齐全")
        print("   ③ 5 相机 × 3 模态的帧数与 demo 长度逐一一致")
        print("   ④ 无退化 demo（帧数 < 10）")
        print("   ⑤ variation 号合法")
        print("   ⑥ 任务之间无共享指令（无跨任务数据混入）")

    Path(args.out).write_text(json.dumps(
        {"errors": errors, "stats": stats}, indent=1, ensure_ascii=False))
    print(f"\n报告写入 {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
