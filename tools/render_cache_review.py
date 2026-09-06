#!/usr/bin/env python
"""门禁③人工抽检：把 cache 的一条 episode 渲染成一张对照图。

VLA_Design §5.3：每条 episode 渲染为关键帧图像横排 + 下方标注 Planner 给出的
segment 归组与指令，供人工快速判读。

用法：
    source run/env.sh
    python tools/render_cache_review.py --cache-dir aavla_data/planner_cache/train \
        --out-dir result/p3_planner_cache/review --limit 100
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.gates import gate3_sampling, load_cache        # noqa: E402

TILE = 160
PAD = 8
HEADER = 46
FOOTER = 78
# 段落配色，相邻段一眼可分
COLORS = [(70, 130, 200), (220, 130, 60), (90, 170, 90), (190, 90, 170),
          (200, 170, 60), (110, 110, 200), (200, 90, 90), (80, 170, 170)]


def render(ep: dict, data_root: Path, out_path: Path, views=("front", "wrist")) -> None:
    kps = ep["keypoints"]
    n = len(kps)
    ep_dir = (data_root / ep["split"] / ep["task"] / "all_variations" /
              "episodes" / f"episode{ep['episode']}")
    W = PAD + n * (TILE + PAD)
    H = HEADER + len(views) * (TILE + PAD) + FOOTER
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)

    d.text((PAD, 6), f"{ep['task']}  ep{ep['episode']}  var{ep['variation']}"
                     f"   \"{ep['task_instruction']}\"", fill=(20, 20, 20))
    d.text((PAD, 24), f"prior: {' -> '.join(ep.get('prior_sequence') or []) or '(none)'}"
                      f"   align={ep.get('prior_alignment')}"
                      f"   quality={ep.get('episode_quality')}", fill=(90, 90, 90))

    kp2seg = {}
    for s in ep["segments"]:
        for ki in s["keypoint_indices"]:
            kp2seg[ki] = s

    for j, k in enumerate(kps):
        x = PAD + j * (TILE + PAD)
        seg = kp2seg.get(j)
        col = COLORS[(seg["segment_index"] if seg else 0) % len(COLORS)]
        for vi, v in enumerate(views):
            y = HEADER + vi * (TILE + PAD)
            p = ep_dir / f"{v}_rgb" / f"{k}.png"
            if p.is_file():
                canvas.paste(Image.open(p).convert("RGB").resize((TILE, TILE)), (x, y))
            d.rectangle([x, y, x + TILE, y + TILE], outline=col, width=3)
        yb = HEADER + len(views) * (TILE + PAD)
        g = ep["gripper_at_keypoints"][j]
        d.text((x + 2, yb), f"[{j}] f={k} {'OPEN' if g else 'CLOSED'}", fill=(40, 40, 40))
        if seg:
            d.text((x + 2, yb + 14), f"seg{seg['segment_index']} {seg['action']}", fill=col)
            ins = seg["instruction"]
            d.text((x + 2, yb + 28), ins[:22], fill=(60, 60, 60))
            if len(ins) > 22:
                d.text((x + 2, yb + 40), ins[22:44], fill=(60, 60, 60))
            if seg.get("use_codebook") is False:
                d.text((x + 2, yb + 54), "use_codebook=FALSE", fill=(200, 60, 60))
            elif "k_global" in seg:
                d.text((x + 2, yb + 54), f"k_g={seg['k_global']}", fill=(60, 60, 60))
    if ep.get("warnings"):
        d.text((PAD, H - 14), "WARN: " + "; ".join(ep["warnings"])[:180], fill=(200, 60, 60))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "aavla_data" / "planner_cache" / "train"))
    ap.add_argument("--data-root", default=str(REPO_ROOT / "aavla_data/rlbench"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "result" / "p3_planner_cache" / "review"))
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--per-task", type=int, default=3)
    args = ap.parse_args()

    _, eps = load_cache(args.cache_dir)
    # 按任务分配固定配额，保证 18 个任务都被覆盖。
    # 直接截断 gate3_sampling 的结果会让 modified 多的任务挤占其余任务的名额
    # （实测 light_bulb_in 一个任务就占了 62/200）。
    ranked = gate3_sampling(eps, per_task=args.per_task)
    by_task: dict[str, list] = {}
    for e in ranked:
        # 每个任务内部优先排 modified / 有告警的，再排普通分层样本
        by_task.setdefault(e["task"], []).append(e)
    for t in by_task:
        by_task[t].sort(key=lambda e: (e.get("prior_alignment") != "modified",
                                       not e.get("warnings"), e["episode"]))
    pool = []
    for t in sorted(by_task):
        pool.extend(by_task[t][:args.per_task])
    pool = pool[:args.limit]
    out_dir = Path(args.out_dir)
    for ep in pool:
        render(ep, Path(args.data_root),
               out_dir / f"{ep['task']}__ep{ep['episode']}.png")
    idx = [{"task": e["task"], "episode": e["episode"], "variation": e["variation"],
            "align": e.get("prior_alignment"), "quality": e.get("episode_quality"),
            "warnings": e.get("warnings", []),
            "png": f"{e['task']}__ep{e['episode']}.png"} for e in pool]
    (out_dir / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    print(f"渲染 {len(pool)} 条抽检对照图 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
