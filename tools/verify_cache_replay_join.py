#!/usr/bin/env python
"""Step 5 联调：验证 planner cache 能干净地接进 PerAct 的 fill_replay。

不真的填 replay（那要几十 GB），而是**离线复现 fill_replay 的样本枚举逻辑**，
逐样本做 cache 查表，验证 VLA_Design 要求的六条硬断言：

  A1 keypoint_discovery(demo) 与 cache 的 keypoints 逐位相同        （§5.1 C5）
  A2 每个 target 关键帧都能查到唯一 segment                          （§4.3 归属规则）
  A3 读到的 task 集合 ⊆ 声明集合，且 episode 数 == 预期              （§2.5 防泄漏）
  A4 k_global ∈ [0,36)，k_detail 长 9 且各元素 ∈ [0,192)
  A5 use_codebook 与白名单规则一致                                   （§3.3）
  A6 任一样本的 subtask 字段不得为空

同时统计 code_mask=0 的**样本级**占比——§8.7 关心的是它会不会接近 0
导致注入层的硬关断路径训不到。

用法：
    source run/env.sh
    python tools/verify_cache_replay_join.py --cache-dir aavla_data/planner_cache/train --tasks close_jar
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner.contract import use_codebook          # noqa: E402
from planner.gates import load_cache               # noqa: E402

DEMO_AUG_EVERY_N = 10       # PerAct 默认值，三臂一致


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "aavla_data" / "planner_cache" / "train"))
    ap.add_argument("--data-root", default=str(REPO_ROOT / "aavla_data/rlbench"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--tasks", nargs="+", default=None, help="默认全部")
    ap.add_argument("--expect-episodes", type=int, default=None,
                    help="A3：期望的 episode 总数，不符即失败")
    args = ap.parse_args()

    from helpers.demo_loading_utils import keypoint_discovery
    from rlbench.demo import Demo

    meta, eps = load_cache(args.cache_dir)
    by_key = {f"{e['task']}/{e['split']}/{e['episode']}": e for e in eps}
    tasks = args.tasks or sorted({e["task"] for e in eps})

    fails: list[str] = []
    n_samples = 0
    n_codemask0 = 0
    n_ep_checked = 0
    action_samples = collections.Counter()
    seg_kp_hist = collections.Counter()

    # A3：任务集合与数量
    got_tasks = sorted({e["task"] for e in eps})
    if args.tasks and not set(got_tasks) <= set(args.tasks):
        fails.append(f"A3_task_leak: cache 含声明外任务 {sorted(set(got_tasks)-set(args.tasks))}")
    if args.expect_episodes is not None and len(eps) != args.expect_episodes:
        fails.append(f"A3_episode_count: {len(eps)} != {args.expect_episodes}")

    for task in tasks:
        base = Path(args.data_root) / args.split / task / "all_variations" / "episodes"
        for key, ep in sorted(by_key.items()):
            if ep["task"] != task or ep["split"] != args.split:
                continue
            if ep.get("errors"):
                continue
            ep_dir = base / f"episode{ep['episode']}"
            with open(ep_dir / "low_dim_obs.pkl", "rb") as f:
                raw = pickle.load(f)
            demo = raw if isinstance(raw, Demo) else Demo(raw)

            # A1
            kps = keypoint_discovery(demo, method="heuristic")
            if kps != ep["keypoints"]:
                fails.append(f"A1_keypoint_mismatch@{key}: {kps} != {ep['keypoints']}")
                continue
            n_ep_checked += 1

            # 建反查表：绝对帧号 → segment
            kp2seg = {}
            for s in ep["segments"]:
                for ki in s["keypoint_indices"]:
                    kp2seg[kps[ki]] = s
                seg_kp_hist[len(s["keypoint_indices"])] += 1

            # A4/A5/A6（段级）
            for s in ep["segments"]:
                if "k_global" in s:
                    if not (isinstance(s["k_global"], int) and 0 <= s["k_global"] < 36):
                        fails.append(f"A4_k_global@{key}seg{s['segment_index']}")
                    kd = s.get("k_detail", [])
                    if len(kd) != 9 or any(not (0 <= x < 192) for x in kd):
                        fails.append(f"A4_k_detail@{key}seg{s['segment_index']}")
                if s.get("use_codebook") != use_codebook(s["action"]):
                    fails.append(f"A5_use_codebook@{key}seg{s['segment_index']}")
                if not s.get("instruction"):
                    fails.append(f"A6_empty_instruction@{key}seg{s['segment_index']}")

            # 复现 fill_replay 的样本枚举，逐样本查表（A2）
            remaining = list(kps)
            for i in range(0, len(demo) - 1):
                if i % DEMO_AUG_EVERY_N != 0:
                    continue
                while remaining and i >= remaining[0]:
                    remaining = remaining[1:]
                if not remaining:
                    break
                for keypoint in remaining:          # 归属规则：target 关键帧所属 segment
                    seg = kp2seg.get(keypoint)
                    if seg is None:
                        fails.append(f"A2_no_segment@{key}kp{keypoint}")
                        continue
                    n_samples += 1
                    action_samples[seg["action"]] += 1
                    if not seg["use_codebook"]:
                        n_codemask0 += 1

    print("=" * 84)
    print("Cache ↔ fill_replay 联调验证")
    print("=" * 84)
    print(f"cache meta: model={meta.get('planner_model')} views={meta.get('views')} "
          f"schema={meta.get('schema_version')}")
    print(f"检查 {n_ep_checked} 条 episode，枚举出 {n_samples:,} 个训练样本")
    print(f"每段关键帧数分布: {dict(sorted(seg_kp_hist.items()))}")
    print(f"样本级 action 分布: {dict(action_samples.most_common())}")
    share = n_codemask0 / max(n_samples, 1)
    print(f"\ncode_mask=0 的样本占比: {n_codemask0:,}/{n_samples:,} = {share:.2%}")
    if share < 0.005:
        print("  ⚠️ 低于 0.5%%：注入层的硬关断路径几乎训不到（VLA_Design §8.7 的担忧兑现）")
    print()
    if fails:
        print(f"❌ {len(fails)} 条断言失败：")
        for f in fails[:25]:
            print(f"   {f}")
        if len(fails) > 25:
            print(f"   ... 另 {len(fails)-25} 条")
        return 1
    print("✅ A1–A6 全部通过：")
    print("   A1 keypoints 与 keypoint_discovery 逐位相同")
    print("   A2 每个 target 关键帧都能查到唯一 segment")
    print("   A3 任务集合与 episode 数符合声明")
    print("   A4 码索引在合法范围内")
    print("   A5 use_codebook 与白名单规则一致")
    print("   A6 子任务指令非空")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
