#!/usr/bin/env python
"""并行生成 PerAct 官方 18 任务的 RLBench demo 数据（train / val / test）。

对每个 (split, task) 起一个 `RLBench/tools/dataset_generator.py` 子进程（xvfb-run 包裹，
每个子进程内部再按 --processes 拆分 episode）。已完成的 (split, task) 会被跳过，可断点续跑。

用法：
    source run/env.sh
    python scripts/gen_peract_data.py --splits train val test
    python scripts/gen_peract_data.py --splits train --tasks open_drawer close_jar --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPO_ROOT / "source" / "RLBench" / "tools" / "dataset_generator.py"

# PerAct 论文 18 任务（顺序与论文 Table 1 一致）
PERACT_18 = [
    "open_drawer", "slide_block_to_color_target", "sweep_to_dustpan_of_size",
    "meat_off_grill", "turn_tap", "put_item_in_drawer", "close_jar",
    "reach_and_drag", "stack_blocks", "light_bulb_in", "put_money_in_safe",
    "place_wine_at_rack_location", "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter", "push_buttons", "insert_onto_square_peg",
    "stack_cups", "place_cups",
]

SPLIT_EPISODES = {"train": 100, "val": 25, "test": 25}


def episodes_dir(root: Path, split: str, task: str) -> Path:
    return root / split / task / "all_variations" / "episodes"


def count_done(root: Path, split: str, task: str) -> int:
    d = episodes_dir(root, split, task)
    if not d.is_dir():
        return 0
    n = 0
    for ep in d.iterdir():
        # 只认写完整的 episode：low_dim_obs.pkl 是 save_demo 的最后一步产物
        if ep.is_dir() and (ep / "low_dim_obs.pkl").is_file():
            n += 1
    return n


def child_env() -> dict[str, str]:
    """显式构造子进程环境，不依赖调用者的 shell。

    踩过的坑：`~/.bashrc` 里 `QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT` 那一行少了换行符，
    被解析成未定义变量 `$COPPELIASIM_ROOTexport` → 空字符串 → CoppeliaSim 起不来，
    报 `Could not find the Qt platform plugin "xcb" in ""` 并 core dump。
    因此这里从 COPPELIASIM_ROOT 现算三个变量，覆盖掉继承来的任何值。
    """
    env = dict(os.environ)
    root = env["COPPELIASIM_ROOT"]
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = root
    ld = env.get("LD_LIBRARY_PATH", "")
    if root not in ld.split(":"):
        env["LD_LIBRARY_PATH"] = f"{ld}:{root}" if ld else root
    return env


def preflight() -> str | None:
    """返回错误信息；None 表示检查通过。"""
    root = os.environ.get("COPPELIASIM_ROOT")
    if not root:
        return "COPPELIASIM_ROOT 未设置，请先 `source run/env.sh`"
    rootp = Path(root)
    if not rootp.is_dir():
        return f"COPPELIASIM_ROOT 指向的目录不存在: {root}"
    if not (rootp / "platforms" / "libqxcb.so").is_file():
        return f"缺少 Qt xcb 插件: {rootp / 'platforms' / 'libqxcb.so'}"
    if shutil.which("xvfb-run") is None:
        return "xvfb-run 不在 PATH 上（无头渲染必需）"
    if not GENERATOR.is_file():
        return f"找不到 dataset_generator.py: {GENERATOR}"
    return None


def build_cmd(root: Path, split: str, task: str, processes: int, image_size: str, renderer: str,
              display: int) -> list[str]:
    # 用 `-n <display>` 指定固定的 X display 号，不用 `-a`（auto-servernum）。
    # `-a` 在并发启动时有竞态：多个 xvfb-run 会挑到同一个空闲号，一个的 Xvfb 被顶掉，
    # 另一个报 `qt.qpa.xcb: could not connect to display :NNN` 并 core dump。
    # 每个 worker 槽位分配一个互不相同的号即可根除。
    return [
        "xvfb-run", "-n", str(display), sys.executable, str(GENERATOR),
        f"--tasks={task}",
        f"--save_path={root / split}",
        f"--image_size={image_size}",
        f"--renderer={renderer}",
        f"--episodes_per_task={SPLIT_EPISODES[split]}",
        f"--processes={processes}",
        "--all_variations=True",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "data_rlbench"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLIT_EPISODES))
    ap.add_argument("--tasks", nargs="+", default=PERACT_18)
    ap.add_argument("--concurrent-tasks", type=int, default=30,
                    help="同时进行的 (split,task) 数。这是唯一有效的并行开关——"
                         "实测每个任务约占 3.1 核，208 核机器上 30 并发约用 93 核")
    ap.add_argument("--processes", type=int, default=1,
                    help="传给 dataset_generator 的 --processes。"
                         "⚠️ all_variations 模式下上游根本不支持多进程（见 source/RLBench/PATCHES.md），"
                         "此值不会带来任何并行，保持 1")
    ap.add_argument("--image-size", default="128,128")
    ap.add_argument("--renderer", default="opengl")
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "log" / "gen_peract_data"))
    ap.add_argument("--force", action="store_true", help="即使已完成也重跑")
    ap.add_argument("--display-base", type=int, default=100,
                    help="X display 号起点。每个 worker 槽位用 display-base+wid，避免 "
                         "`xvfb-run -a` 的并发抢号竞态。**同时跑多个本脚本实例时必须各用不同起点**")
    ap.add_argument("--retries", type=int, default=2, help="零产出任务的自动重试次数")
    ap.add_argument("--fail-fast-after", type=int, default=3,
                    help="前 N 个完成的任务若全部失败则中止（判定为环境类问题）")
    ap.add_argument("--no-fail-fast", action="store_true", help="关闭快速失败")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    err = preflight()
    if err is not None:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    root = Path(args.root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str]] = []
    skipped: list[str] = []
    for split in args.splits:
        for task in args.tasks:
            done = count_done(root, split, task)
            want = SPLIT_EPISODES[split]
            if done >= want and not args.force:
                skipped.append(f"{split}/{task} ({done}/{want})")
                continue
            jobs.append((split, task))

    if args.processes != 1:
        print(f"⚠️ --processes={args.processes} 在 all_variations 模式下无效，上游只会跑 1 个进程；"
              f"要提高并行度请调 --concurrent-tasks", flush=True)
    print(f"待生成 {len(jobs)} 个 (split,task)，跳过 {len(skipped)} 个已完成；"
          f"并发 {args.concurrent_tasks}", flush=True)
    for s in skipped:
        print(f"  skip  {s}")
    for split, task in jobs:
        print(f"  todo  {split}/{task}  ({count_done(root, split, task)}/{SPLIT_EPISODES[split]})")
    print(flush=True)
    if args.dry_run:
        return 0

    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    results: list[dict] = []
    lock = threading.Lock()
    t_start = time.time()
    env = child_env()
    abort = threading.Event()

    def worker(wid: int) -> None:
        while not abort.is_set():
            try:
                split, task = q.get_nowait()
            except queue.Empty:
                return
            # 断点续跑：dataset_generator 会跳过已存在的 episode 目录，但为了干净，
            # 未完成的任务目录直接重跑（episode 数不足说明上次中断）。
            log_path = log_dir / f"{split}__{task}.log"
            want = SPLIT_EPISODES[split]
            t0 = time.time()
            # 零产出（环境/竞态类失败）自动重试；跑到一半的失败不重试，交给续跑。
            for attempt in range(args.retries + 1):
                display = args.display_base + wid + attempt * args.concurrent_tasks
                cmd = build_cmd(root, split, task, args.processes, args.image_size,
                                args.renderer, display)
                mode = "w" if attempt == 0 else "a"
                with open(log_path, mode) as lf:
                    if attempt:
                        lf.write(f"\n===== retry {attempt} (display :{display}) =====\n")
                        lf.flush()
                    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                         cwd=str(GENERATOR.parent), env=env)
                got = count_done(root, split, task)
                if got > 0:
                    break
                if attempt < args.retries:
                    print(f"[RETRY] {split}/{task} 零产出（display :{display}），重试 "
                          f"{attempt + 1}/{args.retries}", flush=True)
                    time.sleep(5 + 5 * wid % 20)
            dt = time.time() - t0
            # 判据是「落盘的完整 episode 数」，不是退出码。上游 dataset_generator 在数据采集
            # 全部成功之后仍可能因打印环节的 bug 非零退出（见 source/RLBench/PATCHES.md），
            # 用退出码判断会把成功的任务误判为失败。退出码单独记录在 rc 字段里备查。
            status = "OK" if got >= want else "FAIL"
            with lock:
                results.append({"split": split, "task": task, "rc": rc, "episodes": got,
                                "expected": want, "seconds": round(dt, 1), "status": status})
                warn = "" if rc == 0 else f"  ⚠️ rc={rc}（数据完整，退出码非零）" if status == "OK" else ""
                print(f"[{status}] {split}/{task}  {got}/{want} ep  {dt/60:.1f} min  "
                      f"(剩余 {q.qsize()}，已用 {(time.time()-t_start)/60:.1f} min){warn}", flush=True)
                # 快速失败只针对环境类错误：其特征是「一个 episode 都没产出」。
                #
                # ⚠️ 阈值必须 >= 并发数。踩过的坑：瞬时失败总是最先返回结果，
                # 曾出现 24 并发下最先返回的 3 条恰好都是竞态失败，导致
                # 「已完成结果全是零产出」成立、误判为环境问题，把队列里剩下的
                # 10 个作业静默跳过。要求「整整一波并发全军覆没」才判定，
                # 才能把真环境故障与零星竞态区分开。
                zero_output = [r for r in results if r["episodes"] == 0]
                threshold = max(args.fail_fast_after, args.concurrent_tasks)
                if (not args.no_fail_fast and len(zero_output) >= threshold
                        and len(zero_output) == len(results)):
                    abort.set()
                    print(f"\n❌ 前 {len(results)} 个任务均为零产出，判定为环境类问题，已中止领取剩余 "
                          f"{q.qsize()} 个任务（正在跑的任务会跑完）。请看 {log_path}", flush=True)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrent_tasks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = log_dir / "summary.json"
    summary.write_text(json.dumps(sorted(results, key=lambda r: (r["split"], r["task"])), indent=1))
    bad = [r for r in results if r["status"] != "OK"]
    print(f"\n完成，用时 {(time.time()-t_start)/60:.1f} min。汇总 -> {summary}")
    if bad:
        print(f"❌ {len(bad)} 个失败：")
        for r in bad:
            print(f"   {r['split']}/{r['task']}  rc={r['rc']}  {r['episodes']}/{r['expected']}  日志 {log_dir}/{r['split']}__{r['task']}.log")
        return 1
    print("✅ 全部成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
