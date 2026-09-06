#!/usr/bin/env python
"""训练进度看门狗 —— 只监控、只告警，**绝不 kill 任何进程**。

# 为什么需要

B2 连续两次死在 step 400，两次都是同一个形态（HANDOFF 坑 #11）：

    Train Step 000300 | Sample time: 0.000417 | Step time: 0.8490
    Train Step 000400 | Sample time: 0.145371 | Step time: 5.8817   ← 然后再无输出

共享机器被打穿 -> DataLoader 饿死 -> 两个 rank 都停在 futex_wait_queue ->
**完全卡死且不报错、不退出**。父进程还在、GPU 显存还占着，从外面看一切正常。
两次事故都发生在无人值守时，而且事后现场（当时的 load / IO）已经没有了。

本脚本做三件事：

  1. **停滞检测**：`Train Step` 超过 --stall-min 分钟不推进就持续告警。
  2. **早期预警**：`Sample time` 是卡死的先行指标（正常 ~0.0004，饿死时 >0.1）。
     超过 --sample-warn 就提前喊，通常比彻底卡死早几分钟。
  3. **留现场**：每轮记 load average 与 sda 队列深度。下次再死，至少知道
     当时机器在干什么 —— 前两次就是因为没记，事后查不出所以然。

# 为什么不自动重启（用户 2026-09-07 决定）

自动重启要 kill 进程，而本项目在 kill 上踩过两个坑：`pkill -f` 会匹配到自己
这条命令行（已自杀三次），DDP 真正占卡的是 `mp.spawn` 子进程、cmdline 里
不含 train.py。风险大于收益，所以只报警，由人决定要不要重启
（`save_freq=2500` + `load_existing_weights=True`，最多损失 2500 步）。

# 用法

    python tools/train_watchdog.py --log log/p5/B4.train1.out --label B4 \
        --target 40000 --stall-min 20 --interval 180

输出建议重定向到 log/p5/<label>.watchdog.out。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

# YARR 每 log_freq 步打一行，注意**行尾有个句点**：
#   Train Step 000100 | Loss: 14.63948 | Sample time: 0.000358 | Step time: 0.8621.
# 所以最后一个数不能用 [\d.]+ 去匹配（会把句点一起吃进去，float() 直接抛
# ValueError 让看门狗自己挂掉 —— 那就等于没有看门狗）。用 \d+\.\d+ 精确匹配。
STEP_RE = re.compile(
    r"Train Step (\d+) \| Loss: (\d+\.\d+) \| Sample time: (\d+\.\d+) \| "
    r"Step time: (\d+\.\d+)")


def tail_bytes(p: Path, n: int = 200_000) -> str:
    """只读日志尾部 —— 训练日志会长到几十 MB，每轮全读太浪费。"""
    with p.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - n))
        return f.read().decode("utf-8", errors="ignore")


def last_step(p: Path):
    """返回 (step, loss, sample_time, step_time)；一条都没有则 None。"""
    m = STEP_RE.findall(tail_bytes(p))
    if not m:
        return None
    s, loss, samp, stept = m[-1]
    return int(s), float(loss), float(samp), float(stept)


def loadavg() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def sda_queue() -> float:
    """sda 的平均队列深度（aqu-sz）。取不到就返回 -1，不让看门狗自己挂掉。"""
    try:
        out = subprocess.run(["iostat", "-x", "1", "2"], capture_output=True,
                             text=True, timeout=15).stdout
        rows = [l.split() for l in out.splitlines() if l.startswith("sda ")]
        return float(rows[-1][-2]) if rows else -1.0
    except Exception:
        return -1.0


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="训练的 nohup 日志")
    ap.add_argument("--label", default="train")
    ap.add_argument("--target", type=int, default=0,
                    help="训到这一步就算完成并退出；0 = 一直盯着")
    ap.add_argument("--interval", type=int, default=180, help="检查间隔（秒）")
    ap.add_argument("--stall-min", type=float, default=20.0,
                    help="多少分钟不推进算停滞")
    ap.add_argument("--sample-warn", type=float, default=0.05,
                    help="Sample time 超过它就早期预警（正常约 0.0004）")
    a = ap.parse_args()

    log = Path(a.log)
    prev_step, prev_at, stalled = None, time.time(), False
    print(f"[{stamp()}] watchdog 启动: {a.label} <- {log}  "
          f"(interval={a.interval}s, stall={a.stall_min}min, target={a.target})",
          flush=True)

    while True:
        now = time.time()
        if not log.is_file():
            print(f"[{stamp()}] [WARN] {a.label}: 日志不存在 {log}", flush=True)
            time.sleep(a.interval)
            continue

        try:
            cur = last_step(log)
        except Exception as exc:            # 解析出任何问题都只告警，不退出
            print(f"[{stamp()}] [WARN] {a.label}: 解析日志失败 {exc!r}", flush=True)
            time.sleep(a.interval)
            continue
        la, q = loadavg(), sda_queue()
        if cur is None:
            print(f"[{stamp()}] [WAIT] {a.label}: 还没有 Train Step 行  "
                  f"load={la:.1f} sda_q={q:.2f}", flush=True)
            time.sleep(a.interval)
            continue

        step, loss, samp, stept = cur
        if prev_step is None or step > prev_step:
            rate = ((step - prev_step) / (now - prev_at) * 60
                    if prev_step is not None else 0.0)
            eta = ((a.target - step) / rate / 60 if a.target and rate > 0 else 0.0)
            print(f"[{stamp()}] [OK]   {a.label}: step {step} "
                  f"(+{0 if prev_step is None else step - prev_step}) "
                  f"loss={loss:.3f} sample={samp:.4f} step_t={stept:.3f} "
                  f"| {rate:.0f} step/min"
                  + (f" ETA {eta:.1f}h" if eta else "")
                  + f" | load={la:.1f} sda_q={q:.2f}", flush=True)
            prev_step, prev_at, stalled = step, now, False
        else:
            mins = (now - prev_at) / 60
            if mins >= a.stall_min:
                stalled = True
                print(f"[{stamp()}] [STALL] {a.label}: step {step} 已 {mins:.1f} "
                      f"分钟未推进！最后一行 sample={samp:.4f} step_t={stept:.3f} "
                      f"| load={la:.1f} sda_q={q:.2f} "
                      f"—— 疑似 DataLoader 饿死（坑 #11）。看门狗不会自动处理，"
                      f"需人工判断是否从最近 ckpt 续训。", flush=True)
            else:
                print(f"[{stamp()}] [..]   {a.label}: step {step} 停 {mins:.1f} min "
                      f"| load={la:.1f} sda_q={q:.2f}", flush=True)

        # 早期预警：Sample time 是卡死的先行指标
        if not stalled and samp > a.sample_warn:
            print(f"[{stamp()}] [WARN] {a.label}: Sample time {samp:.4f} "
                  f"> {a.sample_warn}（正常约 0.0004）—— DataLoader 开始饿了，"
                  f"load={la:.1f} sda_q={q:.2f}", flush=True)

        if a.target and step >= a.target:
            print(f"[{stamp()}] [DONE] {a.label}: 已到 step {step} >= {a.target}，"
                  f"看门狗退出。", flush=True)
            return 0

        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
