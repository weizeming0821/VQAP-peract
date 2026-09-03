#!/usr/bin/env python
"""独立构建共享 replay buffer（Stage 3 三臂共用）。

为什么要从训练里剥离出来
  train.py 用 mp.spawn 起 ddp.num_devices 个 rank，而 replay_path 里**不含 rank**
  （run_seed_fn.py:41），于是 6 个 rank 会各自把同一份 replay 完整填一遍 ——
  同样的内容写 6 遍，纯浪费。剥离之后：
    * 填一次，三臂共享，样本逐字节相同（这是公平性对比的前提）；
    * 训练启动时 replay 已存在，各 rank 直接打开；
    * 断点恢复不再触发重填。

用法
    source run/env.sh
    python scripts/build_replay.py                       # Seen12 全量
    python scripts/build_replay.py --tasks close_jar --demos 5 --out /tmp/try   # 小规模验证
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# 与 train.py 一致：fill_multi_task_replay 的子进程要用 CUDA，
# fork 模式下会报 "Cannot re-initialize CUDA in forked subprocess"。
try:
    from torch.multiprocessing import get_start_method, set_start_method
    if get_start_method(allow_none=True) != "spawn":
        set_start_method("spawn", force=True)
except RuntimeError:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MARKER = "_BUILD_COMPLETE.json"


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PERACT_ROOT / "conf" / "stage3.yaml"))
    ap.add_argument("--out", default=None, help="默认取配置里的 replay.path")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--demos", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="已完成也重建")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    # stage3.yaml 用了 hydra 的 defaults 机制，独立加载时需要手动并入 method
    if "method" not in cfg or "voxel_sizes" not in cfg.get("method", {}):
        base = OmegaConf.load(PERACT_ROOT / "conf" / "method" / "PERACT_BC.yaml")
        cfg.method = OmegaConf.merge(base, cfg.get("method", {}))
    tasks = args.tasks or list(cfg.rlbench.tasks)
    demos = args.demos if args.demos is not None else int(cfg.rlbench.demos)
    out = Path(args.out or cfg.replay.path)
    marker = out / MARKER

    if marker.is_file() and not args.force:
        info = json.loads(marker.read_text())
        print(f"replay 已完整（{info['n_samples']:,} 样本 / {info['size_gb']:.1f} GB，"
              f"建于 {info['built_at']}），跳过。--force 可重建。")
        return 0
    if out.exists() and args.force:
        print(f"--force：清空 {out}")
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    # 磁盘预检：Seen12 约 141,300 样本 × 1.25 MB -> 176 GB
    free_gb = shutil.disk_usage(out).free / 1024 ** 3
    # 实测单样本 1.25 MB（含 subtask 字段；不含时 1.10 MB）
    need_gb = 176.0 * (len(tasks) / 12) * (demos / 100)
    print(f"磁盘可用 {free_gb:.0f} GB，预计需要 {need_gb:.0f} GB")
    if free_gb < need_gb * 1.2:
        print(f"❌ 磁盘不足（需要 {need_gb*1.2:.0f} GB 含 20% 余量）", file=sys.stderr)
        return 2

    # YARR 的 replay 依赖分布式进程组
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29520")
        dist.init_process_group("gloo", rank=0, world_size=1)

    from agents.peract_bc.launch_utils import create_replay, fill_multi_task_replay
    from helpers import utils

    s3 = cfg.get("stage3", None)
    cache_dir = s3.get("planner_cache_dir", None) if s3 else None
    split = s3.get("split", "train") if s3 else "train"
    print(f"任务 {len(tasks)} 个 × {demos} demo；cache={cache_dir}；split={split}")

    obs_config = utils.create_obs_config(
        list(cfg.rlbench.cameras), list(cfg.rlbench.camera_resolution), "PERACT_BC")

    replay = create_replay(
        int(cfg.replay.batch_size), int(cfg.replay.timesteps),
        bool(cfg.replay.prioritisation), bool(cfg.replay.task_uniform),
        str(out), list(cfg.rlbench.cameras), list(cfg.method.voxel_sizes),
        list(cfg.rlbench.camera_resolution),
        stage3_subtask_fields=cache_dir is not None)

    t0 = time.time()
    fill_multi_task_replay(
        cfg, obs_config, 0, replay, tasks, demos,
        bool(cfg.method.demo_augmentation), int(cfg.method.demo_augmentation_every_n),
        list(cfg.rlbench.cameras), list(cfg.rlbench.scene_bounds),
        list(cfg.method.voxel_sizes), list(cfg.method.bounds_offset),
        int(cfg.method.rotation_resolution), bool(cfg.method.crop_augmentation),
        keypoint_method=str(cfg.method.keypoint_method),
        planner_cache_dir=cache_dir, data_split=split)
    dt = time.time() - t0

    n = int(replay.add_count)
    size = dir_size_gb(out)

    # 🔴 硬校验：fill_multi_task_replay 用 Process 逐任务 spawn，**子进程崩溃不会
    # 传播到父进程** —— 它只在自己的 stderr 打 traceback，父进程照常 join 并继续。
    # 不做这道检查，就会出现「子进程全崩、脚本却报 ✅ 完成并写下完成标记」的
    # 静默成功，下游训练拿到空 replay 才发现问题。
    # 期望样本数：P1 用关键帧解析算出 Seen12 全量 141,300 条，按比例折算。
    expect = 141300 * (len(tasks) / 12) * (demos / 100)
    if n < expect * 0.5:
        print(f"\n❌ 样本数 {n:,} 远低于预期 {expect:,.0f}（不足一半）。"
              f"极可能是 fill_replay 子进程崩溃 —— 请检查上方是否有 Traceback。"
              f"未写入完成标记。", file=sys.stderr)
        return 3
    if n == 0:
        print("\n❌ 零样本，未写入完成标记。", file=sys.stderr)
        return 3

    marker.write_text(json.dumps({
        "n_samples": n, "size_gb": round(size, 2), "tasks": tasks, "demos": demos,
        "split": split, "planner_cache_dir": cache_dir,
        "subtask_fields": cache_dir is not None,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(dt, 1),
    }, ensure_ascii=False, indent=1))

    print(f"\n✅ 完成：{n:,} 样本 / {size:.1f} GB / {dt/60:.1f} 分钟")
    print(f"   单样本 {size*1024/max(n,1):.2f} MB")
    print(f"   标记文件 {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
