#!/usr/bin/env python
"""多任务并行监督：同时盯 S4 串行队列与 X3 探路，算速率、报异常。

# 为什么需要

今天已经踩过两次：
  * 3 组评测并发（36 分片）聚合吞吐反而慢 3.3 倍，而当时**没有任何指标**
    告诉我这件事，是事后手算才发现的；
  * 队列脚本因 `set -u` 撞上未定义的 PYTHONPATH，启动即死，日志里只留一行
    "unbound variable"，看起来像正常等待。

所以这里盯的不只是「活着没有」，而是**速率**和**是否真的在推进**。

# 只报告，不干预

绝不 kill 任何进程（本项目在 kill 上踩过两个坑：pkill -f 会匹配到自己这条
命令行，已自杀三次；DDP/评测的真实工作进程 cmdline 里不含入口脚本名）。
要不要停由人决定。

# 用法

    python tools/job_monitor.py --interval 300
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/data0/xiexiao/VQAP")
LOG = ROOT / "log" / "p5"

#: 单组 12 分片的历史基准吞吐（局/min）。低于它的一半就告警。
BASELINE_RATE = 11.2


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def count_episodes(pattern: str, since: float | None = None) -> int:
    """匹配到的分片日志里 'Episode N' 的总行数。

    since 给定时只数比它新的文件 —— 分片日志会被下一轮同名覆盖，但**分片数
    变少时旧文件会留下**（12 分片的 sh6~sh11 在 6 分片那轮不会被覆盖），
    不过滤就会把上一轮的成绩混进来。
    """
    n = 0
    for f in LOG.glob(pattern):
        if since is not None and f.stat().st_mtime < since:
            continue
        try:
            n += sum(1 for line in f.read_text(errors="ignore").splitlines()
                     if "| Episode " in line)
        except OSError:
            pass
    return n


def alive(pat: str) -> bool:
    out = sh(f"pgrep -u $(id -u) -f {pat!r}")
    return bool(out.strip())


def gpu_line() -> str:
    raw = sh("nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
             "--format=csv,noheader,nounits")
    parts = []
    for line in raw.splitlines():
        i, used, util = (x.strip() for x in line.split(","))
        parts.append(f"{i}:{int(used)//1024}G/{util}%")
    return " ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    prev: dict[str, tuple[float, int]] = {}
    start = time.time() - 60          # 只数本次启动之后写的分片日志

    print(f"[{time.strftime('%H:%M:%S')}] 监督启动 interval={a.interval}s；"
          f"基准吞吐 {BASELINE_RATE} 局/min（单组 12 分片）", flush=True)
    while True:
        now = time.time()
        rows = []

        # ---- S4：串行队列，当前在跑哪个 ckpt ----
        q = sh(f"grep -oE '===== B4@[0-9]+ 开始' {LOG/'S4_queue.out'} | tail -1")
        ck = re.search(r"B4@(\d+)", q)
        s4_ck = ck.group(1) if ck else "?"
        s4_n = count_episodes(f"B4.eval.{s4_ck}_template.sh*.out")
        s4_up = alive(f"stage3_eval.py run --arm B4")
        rows.append(("S4/B4@" + s4_ck, s4_n, 300, s4_up))

        # ---- X3 探路 ----
        x3_n = count_episodes("B3.eval.40000_vlm-plan+adapter.sh*.out", since=start)
        x3_up = alive("stage3_eval.py run --arm B3")
        rows.append(("X3/v3.2a", x3_n, 120, x3_up))

        la = Path("/proc/loadavg").read_text().split()[0]
        line = [f"[{time.strftime('%H:%M:%S')}] load={la}"]
        for name, n, tot, up in rows:
            key = name.split("@")[0]
            r = ""
            if key in prev:
                pt, pn = prev[key]
                dt = (now - pt) / 60
                if dt > 0 and n >= pn:
                    rate = (n - pn) / dt
                    eta = (tot - n) / rate if rate > 0.05 else 0
                    r = f" {rate:.1f}局/min" + (f" ETA{eta:.0f}min" if eta else "")
                    if up and rate < BASELINE_RATE / 2 and n < tot:
                        r += "  ⚠️吞吐低于基准一半"
            prev[key] = (now, n)
            line.append(f"| {name} {n}/{tot}{' ✅完' if n >= tot else ''}"
                        f"{'' if up else ' ⛔已停'}{r}")
        line.append("| GPU " + gpu_line())
        print("  ".join(line), flush=True)

        if not any(up for _, _, _, up in rows):
            print(f"[{time.strftime('%H:%M:%S')}] 两个任务都已结束，监督退出。", flush=True)
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
