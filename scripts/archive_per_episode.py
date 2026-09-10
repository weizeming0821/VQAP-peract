#!/usr/bin/env python
"""从评测日志里抽出**逐局分数**并归档。

# 为什么这是必须的

`eval_data.csv` 与 `snapshot()` 只留逐任务均值。而判断一个改动是否真的有效，
**必须做配对 McNemar**（同 episode 索引 ⇒ 同物体摆放），比较均值做不到 ——
评测噪声地板实测：整体 300 局单次 SD 1.49 pp、两次之差 SD 2.11 pp，
单任务 25 局的中位 |Δ| 就有 4.0 pp。均值层面的几个百分点全在噪声里。

而逐局分数**只存在于评测日志里**（`Evaluating <task> | Episode <n> | Score: <s>`），
下一轮运行就会把日志覆盖掉。旧机正是因此丢掉了 v3.2c 的全部逐局数据，
导致无法归因；迁移时 `result/p7/versions/` 整个没带过来，之前那张 test 主表
现在已经无法复核。

**所以：每次评测跑完立刻归档，不要等。**

    python scripts/archive_per_episode.py --log log/p5/B1.test28.out \\
        --out result/p7/B1_test_ep25_20000.per_episode.json

输出 `{"task|episode": {"score": float}}`，与旧机 `per_episode.json` 同格式，
另附 meta（来源日志、任务数、局数、逐任务均值）便于核对。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: eval.py 每局结束时打印的行。分片并发写同一个文件，但每行是完整的一条。
LINE = re.compile(
    r"Evaluating\s+(?P<task>[a-z_0-9]+)\s*\|\s*Episode\s+(?P<ep>\d+)\s*\|"
    r"\s*Score:\s*(?P<score>[-\d.]+)")


def parse(log: Path) -> tuple[dict, list[str]]:
    per_ep: dict[str, float] = {}
    dup: list[str] = []
    for line in log.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        key = f"{m['task']}|{m['ep']}"
        score = float(m["score"])
        if key in per_ep and per_ep[key] != score:
            # 同一局出现两个不同分数 —— 多半是日志里混进了上一轮的内容，
            # 或同臂并发把两次运行写进了同一个文件。必须报出来，不能默默取后者。
            dup.append(f"{key}: {per_ep[key]} vs {score}")
        per_ep[key] = score
    return per_ep, dup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="评测日志（--foreground 的输出）")
    ap.add_argument("--out", required=True, help="归档路径 result/p7/<名字>.json")
    ap.add_argument("--expect-tasks", type=int, default=None)
    ap.add_argument("--expect-episodes", type=int, default=None,
                    help="每任务应有多少局；给了就做完整性硬校验")
    a = ap.parse_args()

    log = Path(a.log)
    if not log.is_file():
        print(f"❌ 找不到日志 {log}", file=sys.stderr)
        return 2
    per_ep, dup = parse(log)
    if not per_ep:
        print(f"❌ {log} 里没有任何逐局记录", file=sys.stderr)
        return 2

    by_task: dict[str, list[float]] = defaultdict(list)
    for k, v in per_ep.items():
        by_task[k.split("|")[0]].append(v)

    ok = True
    if dup:
        ok = False
        print(f"🔴 {len(dup)} 局出现互相矛盾的分数（日志被混写？）：", file=sys.stderr)
        for d in dup[:5]:
            print(f"     {d}", file=sys.stderr)
    if a.expect_tasks is not None and len(by_task) != a.expect_tasks:
        ok = False
        print(f"🔴 任务数 {len(by_task)} != 预期 {a.expect_tasks}", file=sys.stderr)
    if a.expect_episodes is not None:
        short = {t: len(v) for t, v in by_task.items() if len(v) != a.expect_episodes}
        if short:
            ok = False
            print(f"🔴 局数不足的任务：{short}", file=sys.stderr)

    means = {t: sum(v) / len(v) for t, v in sorted(by_task.items())}
    doc = {
        "meta": {
            "source_log": str(log),
            "archived_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_tasks": len(by_task),
            "n_episodes": len(per_ep),
            "complete": ok,
            "per_task_mean": {t: round(m, 2) for t, m in means.items()},
            "overall_mean": round(sum(means.values()) / len(means), 4),
        },
        "per_episode": {k: {"score": v} for k, v in sorted(per_ep.items())},
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"  ✅ {len(per_ep)} 局 / {len(by_task)} 任务 -> {out}")
    print(f"     总均值 {doc['meta']['overall_mean']:.2f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
