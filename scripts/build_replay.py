#!/usr/bin/env python
"""构建共享 replay 工件（Stage 3 三臂共用）。

# 为什么要从训练里剥离出来

`train.py` 用 `mp.spawn` 起 `ddp.num_devices` 个 rank，而 replay 路径里不含 rank
（run_seed_fn.py:57），上游又在每个 rank 里无条件调 `fill_multi_task_replay`。
于是 N 个 rank 会各自把同一份 replay 完整填一遍，写的还是同一批 `<cursor>.replay`
文件名 —— 不只是浪费，同名文件被两个进程同时 `open('wb')` 会写出半截 pickle，
训练跑到几小时后才以 `UnpicklingError` 爆出来。

剥离 + 工件化之后：
  * 填一次，三臂共享，样本逐字节相同（这是公平性对比的前提）；
  * 训练启动时 `replay_dataset.attach()` 毫秒级挂载，不再重填；
  * 中途换卡续训不再付 176 GB 的重建代价。

# 用法

    source run/env.sh
    python scripts/build_replay.py                                  # Seen12 全量
    python scripts/build_replay.py --tasks close_jar --demos 5 --out /tmp/try
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

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

from stage3 import replay_dataset  # noqa: E402


def dir_size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3


def _n_stored(demo_len: int, keypoints: list[int], every_n: int) -> int:
    """逐位复刻 `fill_replay` 的双层循环，算出一个 episode 会落盘多少个文件。

    ⚠️ 这个数**不等于** P1 的 `replay_samples`。P1 只数了 `replay.add` 的调用，
    而 `_add_keypoints_to_replay` 每轮末尾还会调一次 `replay.add_final`，
    它同样走 `_add()`、同样占一个 cursor、同样落一个 .replay 文件
    （uniform_replay_buffer.py:292）。差值实测 15.2%。

    两个数各自都对，用途不同：
        P1 口径（不含 final）  = **可训练**样本数。final 帧的 terminal 被置 -1，
                                `is_valid_transition` 直接判否，永远不会被采到，
                                它只是前一条样本的 _tp1。
        本函数（含 final）     = **落盘文件数** = replay.add_count。
                                校验与磁盘估算必须用这个。

    公式已在 close_jar / open_drawer 上与实测逐位核对（244/244、119/119）。
    """
    kps = list(keypoints)
    total = 0
    for i in range(demo_len - 1):
        if i % every_n != 0:
            continue
        while kps and i >= kps[0]:
            kps = kps[1:]
        if not kps:
            break
        total += len(kps) + 1        # len(kps) 次 add + 1 次 add_final
    return total


def expected_per_task(tasks: list[str], demos: int, split: str,
                      cache_dir: str | None, every_n: int) -> dict[str, int] | None:
    """逐任务的预期落盘文件数。

    来源是 P1 独立解析出的 `aavla_data/rlbench/keyframe_stats.json`（每个 episode 记录了
    `demo_len` 与 `keypoints`），扣掉 planner cache 判为不合格、fill 时会 `continue`
    掉的 episode。两条路径完全独立 —— 一边是关键帧解析，一边是 VLM 分段 —— 所以
    这个数能当作交叉验证的基准。

    🔴 必须**逐任务**校验：`fill_multi_task_replay` 用 `Process` 逐任务 spawn，
    子进程崩溃不会传播到父进程（只在自己的 stderr 打 traceback，父进程照常 join）。
    12 个任务崩 1 个只丢约 8%，任何「总数低于预期一半」之类的宽松阈值都查不出来。

    统计文件不可用时返回 None，调用方退回宽松校验。
    """
    stats_p = REPO_ROOT / "aavla_data/rlbench" / "keyframe_stats.json"
    if not stats_p.is_file():
        return None
    try:
        eps = json.loads(stats_p.read_text())["episodes"]
    except Exception:
        return None

    per_ep = {(e["task"], e["split"],
               int(str(e["episode"]).replace("episode", ""))): e for e in eps}

    cache = None
    if cache_dir is not None:
        from stage3.cache_join import PlannerCache
        cache = PlannerCache(cache_dir)

    out: dict[str, int] = {}
    for t in tasks:
        total = 0
        for i in range(demos):
            e = per_ep.get((t, split, i))
            if e is None:
                return None                      # 统计不全，不做强校验
            if cache is not None and cache.get(t, split, i) is None:
                continue                         # fill_replay 会跳过这条 episode
            total += _n_stored(int(e["demo_len"]), list(e["keypoints"]), every_n)
        out[t] = total
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PERACT_ROOT / "conf" / "stage3.yaml"))
    ap.add_argument("--out", default=None,
                    help="默认由 replay_dataset.replay_dir() 从配置推出（与训练一致）")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--demos", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
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

    # 🔴 路径必须与 run_seed_fn 的计算逐字符一致，否则预建的 replay 训练时用不上
    #    （这正是修复前的 P0-1）。两边共用 replay_dataset.replay_dir()。
    out = (Path(args.out) if args.out else
           replay_dataset.replay_dir(cfg.replay.path, tasks,
                                     str(cfg.method.name), args.seed))

    if replay_dataset.is_built(out) and not args.force:
        man = replay_dataset.read_manifest(out)
        print(f"replay 工件已存在（{man['n_samples']:,} 样本，建于 {man['built_at']}）"
              f"\n  {out}\n跳过。--force 可重建。")
        return 0
    if out.exists() and args.force:
        print(f"--force：清空 {out}")
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    s3 = cfg.get("stage3", None)
    cache_dir = s3.get("planner_cache_dir", None) if s3 else None
    split = s3.get("split", "train") if s3 else "train"

    # ---- 逐任务预期样本数（P1 独立解析，交叉验证用）----
    exp = expected_per_task(tasks, demos, split, cache_dir,
                            int(cfg.method.demo_augmentation_every_n))
    exp_total = sum(exp.values()) if exp else None

    # ---- 磁盘：只报告，不做看门狗。不够就直接停，并把数字说清楚 ----
    free_gb = shutil.disk_usage(out).free / 1024 ** 3
    # 实测 1.25 MB/文件（含 subtask 字段，2 任务 × 3 demo 实测）。
    # 🔴 注意别拿「P1 的 141,300 × 1.25 MB = 176 GB」去算 —— 那是把两个口径的数
    #    混用了：141,300 是**可训练样本数**（不含 add_final），落盘文件数是 165,773。
    #    Seen12 全量的真实需求是 165,773 × 1.25 MB ≈ 202 GB。
    need_gb = (exp_total if exp_total else 165773 * (len(tasks) / 12) * (demos / 100)) \
        * 1.25 / 1024
    print(f"磁盘可用 {free_gb:.0f} GB，预计需要 {need_gb:.0f} GB")
    if free_gb < need_gb * 1.1:
        print(f"\n❌ 磁盘不足：需要约 {need_gb:.0f} GB（含 10% 余量共 "
              f"{need_gb*1.1:.0f} GB），当前只有 {free_gb:.0f} GB。未开始构建。",
              file=sys.stderr)
        return 2

    # YARR 的 replay 依赖分布式进程组
    if not dist.is_initialized():
        import os
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29520")
        dist.init_process_group("gloo", rank=0, world_size=1)

    from agents.peract_bc.launch_utils import create_replay, fill_multi_task_replay
    from helpers import utils

    print(f"任务 {len(tasks)} 个 × {demos} demo；cache={cache_dir}；split={split}")
    print(f"输出 {out}")
    if exp:
        print(f"预期样本数 {exp_total:,}（逐任务校验已启用）")

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
    got = {t: len(v) for t, v in replay._task_idxs.items()}

    # ---- 硬校验：逐任务 ----
    if n == 0:
        print("\n❌ 零样本，未写入工件。", file=sys.stderr)
        return 3
    if exp:
        bad = []
        for t in tasks:
            want, have = exp[t], got.get(t, 0)
            # 1% 容差：demo 解析的边界条件可能有个位数差异；成建制的缺失一定超过它
            if abs(have - want) > max(1, int(want * 0.01)):
                bad.append((t, want, have))
        if bad:
            print("\n❌ 逐任务样本数与 P1 独立解析对不上 —— 极可能是该任务的 "
                  "fill 子进程崩溃（子进程异常不会传播到父进程，"
                  "请翻上方 stderr 找 Traceback）。未写入工件：", file=sys.stderr)
            for t, want, have in bad:
                print(f"    {t:36s} 预期 {want:>7,}  实际 {have:>7,}", file=sys.stderr)
            return 3
        missing = [t for t in tasks if t not in got]
        if missing:
            print(f"\n❌ 这些任务一条样本都没有：{missing}。未写入工件。", file=sys.stderr)
            return 3
    elif n < 1000:
        print(f"\n❌ 只有 {n:,} 条样本，明显异常。未写入工件。", file=sys.stderr)
        return 3

    size = dir_size_gb(out)
    man = replay_dataset.save(replay, out, extra={
        "tasks": list(tasks), "demos": demos, "split": split,
        "planner_cache_dir": cache_dir,
        "subtask_fields": cache_dir is not None,
        "demo_path": str(cfg.rlbench.demo_path),
        "keypoint_method": str(cfg.method.keypoint_method),
        "demo_augmentation_every_n": int(cfg.method.demo_augmentation_every_n),
        "expected_per_task": exp,
        "size_gb": round(size, 2),
        "build_seconds": round(dt, 1),
        "built_by": "scripts/build_replay.py",
    })

    print(f"\n✅ 完成：{man['n_samples']:,} 样本 / {size:.1f} GB / {dt/60:.1f} 分钟")
    print(f"   单样本 {size*1024/max(n,1):.2f} MB")
    print(f"   工件 {out}/{replay_dataset.MANIFEST}")
    free_after = shutil.disk_usage(out).free / 1024 ** 3
    print(f"   构建后磁盘可用 {free_after:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
