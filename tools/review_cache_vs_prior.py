#!/usr/bin/env python
"""对照 CSV 先验人工检查 planner cache。

CSV 是**软先验**（VLA_Design §4.9），偏离本身不算错——真正要判读的是：
偏离是否**成系统**、方向是否**一致**、理由是否**站得住**。所以本工具不给
一个笼统的编辑距离，而是把分歧拆成可逐条判读的形态：

  1. 逐 (task, variation) 的动作序列**众数**，与先验并排对照
  2. 序列层面的**对齐 diff**（哪一步被替换成了什么、哪一步被删/增）
  3. 每类替换的**全局频次**（例如 pose-adjust→transfer 出现 N 次）
  4. 组内**稳定性**：同组内动作序列有几种取值——不稳定比偏离先验严重得多
  5. 指令**抽样**，供肉眼核对措辞与区分性信息

用法：
    source run/env.sh
    python tools/review_cache_vs_prior.py --cache-dir aavla_data/planner_cache/train \
        --out result/p3_planner_cache/prior_review.json
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.gates import load_cache          # noqa: E402
from planner.offline import load_priors        # noqa: E402
from planner.contract import n_repeat_prior    # noqa: E402


def seq_diff(prior: list[str], got: list[str]) -> list[str]:
    """把 prior→got 的差异写成人类可读的操作串。"""
    ops = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=prior, b=got).get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            ops.append(f"{'+'.join(prior[i1:i2])}→{'+'.join(got[j1:j2])}")
        elif tag == "delete":
            ops.append(f"-{'+'.join(prior[i1:i2])}")
        elif tag == "insert":
            ops.append(f"+{'+'.join(got[j1:j2])}")
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "aavla_data" / "planner_cache" / "train"))
    ap.add_argument("--out", default=str(REPO_ROOT / "result" / "p3_planner_cache" / "prior_review.json"))
    ap.add_argument("--samples", type=int, default=2, help="每任务打印几条指令样例")
    args = ap.parse_args()

    priors = load_priors()
    _, eps = load_cache(args.cache_dir)
    ok = [e for e in eps if not e.get("errors")]

    by_tv = collections.defaultdict(list)
    for e in ok:
        by_tv[(e["task"], e["variation"])].append(e)

    replacements = collections.Counter()
    per_task: dict[str, dict] = {}
    unstable_groups = []

    for task in sorted({t for t, _ in by_tv}):
        base_prior = priors.get(task, [])
        rows = []
        seqs_all = collections.Counter()
        for (t, v), group in sorted(by_tv.items()):
            if t != task:
                continue
            seqs = collections.Counter(tuple(s["action"] for s in e["segments"])
                                       for e in group)
            for s, n in seqs.items():
                seqs_all[s] += n
            mode, n_mode = seqs.most_common(1)[0]
            n_rep = n_repeat_prior(task, v)
            prior = base_prior * n_rep if (n_rep and n_rep > 1 and base_prior) else base_prior
            ops = seq_diff(prior, list(mode))
            for o in ops:
                replacements[o] += 1
            stability = n_mode / sum(seqs.values())
            if len(seqs) > 3 and stability < 0.5:
                unstable_groups.append(
                    {"task": t, "variation": v, "n_distinct": len(seqs),
                     "mode_share": round(stability, 3), "n": sum(seqs.values())})
            rows.append({"variation": v, "n_episodes": sum(seqs.values()),
                         "prior": prior, "mode_sequence": list(mode),
                         "mode_share": round(stability, 3),
                         "n_distinct_sequences": len(seqs),
                         "diff_vs_prior": ops})
        per_task[task] = {
            "prior": base_prior,
            "n_variations": len(rows),
            "global_mode_sequence": list(seqs_all.most_common(1)[0][0]) if seqs_all else [],
            "variations": rows,
        }

    # 指令抽样
    samples = {}
    for task in sorted(per_task):
        got = []
        for e in ok:
            if e["task"] != task:
                continue
            got.append({"variation": e["variation"], "episode": e["episode"],
                        "task_instruction": e["task_instruction"],
                        "segments": [f"{s['action']}: {s['instruction']}"
                                     for s in e["segments"]]})
            if len(got) >= args.samples:
                break
        samples[task] = got

    # ---- 打印 ----
    print("=" * 108)
    print("Cache vs CSV 先验：逐 (task, variation) 众数序列对照")
    print("=" * 108)
    for task, d in per_task.items():
        print(f"\n### {task}   先验: {' → '.join(d['prior']) if d['prior'] else '(无)'}")
        for r in d["variations"]:
            flag = "" if not r["diff_vs_prior"] else "   diff: " + ", ".join(r["diff_vs_prior"])
            warn = "  ⚠️不稳定" if r["mode_share"] < 0.5 else ""
            print(f"  var{r['variation']:<3} n={r['n_episodes']:<4} "
                  f"众数占比 {r['mode_share']:.2f} 序列种数 {r['n_distinct_sequences']:<3} "
                  f"{' → '.join(r['mode_sequence'])}{flag}{warn}")

    print("\n" + "=" * 108)
    print("全局：相对先验的替换/增删 频次（判断偏离是否成系统）")
    print("=" * 108)
    for op, n in replacements.most_common(25):
        print(f"  {op:<46s} {n}")

    if unstable_groups:
        print(f"\n⚠️ 组内序列不稳定的 (task,variation) 共 {len(unstable_groups)} 组：")
        for g in unstable_groups[:15]:
            print(f"  {g['task']}/var{g['variation']}: {g['n_distinct']} 种序列，"
                  f"众数仅占 {g['mode_share']:.2f}（n={g['n']}）")

    print("\n" + "=" * 108)
    print("指令抽样")
    print("=" * 108)
    for task, got in samples.items():
        for g in got[:1]:
            print(f"\n{task} var{g['variation']} ep{g['episode']}  \"{g['task_instruction']}\"")
            for s in g["segments"]:
                print(f"    {s}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_task": per_task,
                               "replacement_ops": dict(replacements.most_common()),
                               "unstable_groups": unstable_groups,
                               "instruction_samples": samples},
                              ensure_ascii=False, indent=1))
    print(f"\n明细写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
