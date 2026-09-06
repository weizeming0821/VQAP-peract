#!/usr/bin/env python
"""训练→占位的零间隙交接：作业一结束就立刻把它的卡占回来。

# 为什么需要

`gpu_reserver.py watch` 会把可用卡补齐到 4 张，但它**跳过 hold 名单里的卡**
（hold = 「这几张留给作业，看门程序别碰」）。于是训练跑完时是这样的：

    训练退出 → 卡空出来 → hold 还在 → 看门程序永远不会去占 → 别人几秒内抢走

实测发生过：B1 训练跑完释放 GPU 6/7，另一个用户的作业立刻占满，导致随后的
评测 4/6 个分片 CUDA OOM 死掉。就算把 interval 调到最短，watch 也有几十秒空档。

本脚本盯着**训练进程的 PID**，进程一没就立刻在同一批卡上起占位，并把这些卡
从 hold 名单里摘掉交还给看门程序。盘子数不变（本来就是我们的 4 张里的卡，
只是从「训练在用」变成「占位占着」），要用时照常 `gpu_reserver.py hold`。

# 安全边界

  * **绝不 kill 任何进程**（本项目在 kill 上踩过两个坑，见 HANDOFF #7 #9）
  * 占位前重新数盘子，超过 MAX_CARDS 就不占
  * 卡已经被别人占走（空闲不足）就如实报告「已失守」，不去挤别人

# 用法

    python tools/gpu_handoff.py --job B4:1485097:3,4 --job B2:1488617:1,2

    每个 --job 是 <标签>:<训练进程PID>:<该作业的GPU号，逗号分隔>。
    PID 用 `pgrep -u $(id -u) -af train.py` 里那个真正的 python 进程，
    不是启动它的 bash。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.gpu_reserver import (            # noqa: E402  复用，不重复实现
    KEEP_FREE_MIB, MAX_CARDS, alive, gpu_state, is_ours, load_holds,
    load_state, others_on, prune, reserve, save_holds, save_state,
)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def plate_size() -> int:
    """当前盘子：占位的卡 + 自己正在跑作业的卡 + hold 名单。"""
    state = prune(load_state())
    holds = load_holds()
    busy = {r["index"] for r in gpu_state()
            if r["index"] not in {int(g) for g in state} and is_ours(r["index"])}
    return len(state) + len(busy | {int(g) for g in holds})


def take_back(label: str, gpus: list[int]) -> None:
    """作业已退出，把它的卡占回来并解除 hold。"""
    state = prune(load_state())
    for g in gpus:
        # 训练的 mp.spawn 子进程可能比父进程晚几秒才释放显存，等一小会儿
        for _ in range(20):
            if not is_ours(g):
                break
            time.sleep(3)

        row = next((r for r in gpu_state() if r["index"] == g), None)
        if row is None:
            print(f"[{stamp()}] [WARN] {label}: 读不到 GPU{g} 状态", flush=True)
            continue
        mib = row["free"] - KEEP_FREE_MIB
        if mib <= 0:
            print(f"[{stamp()}] [LOST] {label}: GPU{g} 已被占满"
                  f"（空闲 {row['free']} MiB）—— 没抢回来，不去挤别人。", flush=True)
        elif plate_size() >= MAX_CARDS and str(g) not in load_holds():
            print(f"[{stamp()}] [SKIP] {label}: 盘子已满 {MAX_CARDS} 张，"
                  f"GPU{g} 不再占位。", flush=True)
        else:
            pid = reserve(g, mib)
            if pid:
                state[str(g)] = {"pid": pid, "mib": mib, "at": stamp()}
                save_state(state)
                print(f"[{stamp()}] [HELD] {label}: GPU{g} 已占回 {mib} MiB "
                      f"(pid={pid})"
                      + ("  ⚠️ 该卡上有别人的进程" if others_on(g) else ""),
                      flush=True)
            else:
                print(f"[{stamp()}] [FAIL] {label}: GPU{g} 占位进程没起来",
                      flush=True)

        # 无论占位成没占上，都把 hold 摘掉 —— 否则看门程序永远绕开这张卡，
        # 下一轮想补都补不回来。
        h = load_holds() - {str(g)}
        save_holds(h)
    print(f"[{stamp()}] [DONE] {label}: 交接完毕，剩余 hold {sorted(load_holds()) or '无'}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", action="append", required=True,
                    help="<标签>:<训练PID>:<GPU号,逗号分隔>，可重复")
    ap.add_argument("--interval", type=int, default=15)
    a = ap.parse_args()

    jobs = []
    for spec in a.job:
        label, pid, gpus = spec.split(":")
        jobs.append({"label": label, "pid": int(pid),
                     "gpus": [int(g) for g in gpus.split(",")]})

    for j in jobs:
        if not alive(j["pid"]):
            print(f"[{stamp()}] [WARN] {j['label']}: PID {j['pid']} 一开始就不在，"
                  f"确认 PID 是否写错（要的是 python train.py 那个进程）", flush=True)
    print(f"[{stamp()}] 交接守卫启动：" +
          "，".join(f"{j['label']}(pid={j['pid']}) -> GPU{j['gpus']}" for j in jobs),
          flush=True)

    pending = list(jobs)
    while pending:
        still = []
        for j in pending:
            if alive(j["pid"]):
                still.append(j)
            else:
                print(f"[{stamp()}] {j['label']}: 训练进程 {j['pid']} 已退出，"
                      f"立刻接管 GPU{j['gpus']}", flush=True)
                take_back(j["label"], j["gpus"])
        pending = still
        if pending:
            time.sleep(a.interval)
    print(f"[{stamp()}] 全部作业已交接，守卫退出。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
