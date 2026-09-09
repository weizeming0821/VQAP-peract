#!/usr/bin/env python
"""replay 工件的完整性验收 —— Step 1f 的 G3 关卡。

覆盖 `build_replay.py` 内建校验与 `replay_dataset.attach()` 六道校验**之外**的项：

    1. 磁盘上的 .replay 文件数 == manifest.n_samples（attach 不查这条）
    2. cursor 连续无洞（0 … n-1）
    3. terminal 语义配平：count(-1) == count(1)，即每个 episode 恰好一个 add_final 帧
    4. 逐任务样本数 == `expected_per_task` 独立复算值
       （该基准来自 keyframe_stats.json + planner cache，与 fill 路径完全独立）
    5. 随机抽验 N 个样本文件可反序列化，且 `task` 字段与索引一致

用法：
    python scripts/verify_replay_artifact.py --replay <dir> [--sample 5000] [--full]
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from stage3 import replay_dataset  # noqa: E402


def _check_one(args):
    d, cursor, want_task = args
    try:
        with open(f"{d}/{cursor}.replay", "rb") as f:
            got = pickle.load(f).get("task")
    except FileNotFoundError:
        return cursor, "文件不存在"
    except Exception as exc:
        return cursor, f"反序列化失败 {type(exc).__name__}"
    if got != want_task:
        return cursor, f"task='{got}' 索引却记为 '{want_task}'"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--sample", type=int, default=5000, help="抽验样本文件数")
    ap.add_argument("--full", action="store_true", help="全量遍历（240k 文件约需十几分钟）")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()

    d = Path(a.replay)
    man = replay_dataset.read_manifest(d)
    with open(d / "_INDEX.pkl", "rb") as f:
        idx = pickle.load(f)

    fails: list[str] = []

    def chk(name, ok, detail=""):
        print(f"  {'✅' if ok else '❌'} {name}{('  ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    print(f"工件 {d}")
    print(f"  manifest: {man['n_samples']:,} 样本 / {man.get('size_gb')} GB / "
          f"{len(man['per_task'])} 任务 / 建于 {man['built_at']}\n")

    # ---- 1. 文件数 ----
    n_files = sum(1 for p in d.iterdir() if p.suffix == ".replay")
    n = int(man["n_samples"])
    chk("① 磁盘文件数 == manifest.n_samples", n_files == n, f"{n_files:,} vs {n:,}")

    # ---- 2. 索引自洽 + cursor 连续 ----
    ti = {k: sorted(int(c) for c in v) for k, v in idx["task_idxs"].items()}
    all_c = sorted(c for v in ti.values() for c in v)
    chk("② add_count == 索引条数 == manifest",
        int(idx["add_count"]) == len(all_c) == n,
        f"{int(idx['add_count']):,} / {len(all_c):,} / {n:,}")
    chk("③ cursor 连续无洞", all_c == list(range(n)),
        f"0…{all_c[-1]:,}" if all_c else "空")

    # ---- 3. terminal 配平 ----
    term = np.asarray(idx["terminal"])[:n]
    cnt = Counter(term.tolist())
    chk("④ terminal 配平 count(-1)==count(1)", cnt.get(-1, 0) == cnt.get(1, 0),
        f"-1:{cnt.get(-1,0):,}  1:{cnt.get(1,0):,}  0:{cnt.get(0,0):,}")

    # ---- 4. 逐任务 vs 独立复算 ----
    exp = man.get("expected_per_task") or {}
    if exp:
        bad = [(t, exp[t], len(ti.get(t, []))) for t in exp
               if len(ti.get(t, [])) != exp[t]]
        chk("⑤ 逐任务样本数 == 独立复算值", not bad,
            "全部相符" if not bad else f"{bad[:3]}")
    else:
        print("  ⚠️ manifest 无 expected_per_task，跳过 ⑤")

    # ---- 5. 抽验文件 ----
    cursor_to_task = {c: t for t, v in ti.items() for c in v}
    picks = (sorted(cursor_to_task) if a.full
             else random.Random(0).sample(sorted(cursor_to_task),
                                          min(a.sample, len(cursor_to_task))))
    print(f"\n  抽验 {len(picks):,} 个样本文件（{a.workers} 进程并行）…")
    bad_files = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_check_one,
                        ((str(d), c, cursor_to_task[c]) for c in picks),
                        chunksize=64):
            if r is not None:
                bad_files.append(r)
    chk("⑥ 样本文件可反序列化且 task 字段与索引一致", not bad_files,
        f"{len(picks):,} 个全部通过" if not bad_files else f"{bad_files[:3]}")

    print()
    if fails:
        print(f"❌ 工件完整性校验未通过：{fails}", file=sys.stderr)
        return 1
    print("✅ 工件完整性校验全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
