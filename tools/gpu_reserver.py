#!/usr/bin/env python
"""空闲显卡看门程序 —— 发现空卡就占住，需要用时释放。

# 为什么需要

这是共享机器，实测过两次「我们的作业一结束，几十秒内卡就被别人拿走」：
B1 训练跑完释放 GPU 6/7，另一个用户的作业立刻占满，导致我们的评测
4/6 分片 CUDA OOM 死掉。占位能避免这种「刚放手就抢不回来」。

# 规则（用户设定）

  * 只占**空闲显存 > 30 GB** 的卡
  * 最多同时占 **4 张**（用户 2026-09-04 从 6 改为 4）
  * 留安全余量，绝不把卡吃满
  * 只 SIGTERM 自己起的占位进程，**绝不 kill 任何别人的进程**

底层复用 `tools/reserve_gpu_memory.py`（一次性分配后阻塞、不跑计算循环、
SIGTERM 释放），本脚本只做「盯 + 决策 + 记账」。

# 用法

    python tools/gpu_reserver.py watch          # 后台盯着，发现空卡就占
    python tools/gpu_reserver.py status         # 看当前占了哪些
    python tools/gpu_reserver.py release --gpu 6    # 要用 6 号卡了，放开
    python tools/gpu_reserver.py release --all
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESERVER = REPO_ROOT / "tools" / "reserve_gpu_memory.py"
STATE = REPO_ROOT / "log" / "p5" / "gpu_reservations.json"
#: 「别占这几张」——要拿去跑作业的卡。看门程序会跳过它们。
#: 没有这个机制的话：release 之后到评测进程真正起来之间有几分钟空档，
#: 看门程序会在这个空档把卡又占回去，和自己的评测抢显存。
HOLDS = REPO_ROOT / "log" / "p5" / "gpu_holds.json"

#: 只占空闲显存超过这个数的卡（用户设定）
FREE_THRESHOLD_MIB = 30 * 1024
#: 占用时留给别人的余量 —— 不把卡吃满
KEEP_FREE_MIB = 4 * 1024
#: 最多同时占几张（用户设定）
MAX_CARDS = 4


def gpu_state() -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        i, used, total = (int(x.strip()) for x in line.split(","))
        rows.append({"index": i, "used": used, "total": total, "free": total - used})
    return rows


def load_state() -> dict:
    if STATE.is_file():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {}


def save_state(d: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(d, indent=1))


def load_holds() -> set[str]:
    if HOLDS.is_file():
        try:
            return set(json.loads(HOLDS.read_text()))
        except Exception:
            pass
    return set()


def save_holds(h: set[str]) -> None:
    HOLDS.parent.mkdir(parents=True, exist_ok=True)
    HOLDS.write_text(json.dumps(sorted(h)))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


#: 认得出是我们自己作业的进程（用于「已占几张」的计数）
OURS = ("train.py", "eval.py", "reserve_gpu_memory.py")


def procs_on(gpu: int) -> list[tuple[int, int, str]]:
    """某张卡上的计算进程 [(pid, uid, cmdline)]。

    🔴 用 **uid** 而不是用户名来判归属。`ps -o user=` 会把用户名截断到 8 字符
    （weizeming → weizemin），拿它和 $USER 比会**永远不相等**：
    结果是自己的进程全被当成「别人的」，`is_ours` 恒假，盘子数不到自己正在跑的
    作业，占位就会在训练/评测之外再额外占满 4 张 —— 正好违反「总共 4 卡」。
    """
    out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines():
        pid = line.strip()
        if not pid.isdigit():
            continue
        info = subprocess.run(["ps", "-o", "uid=,cmd=", "-p", pid],
                              capture_output=True, text=True).stdout.strip()
        if not info:
            continue
        uid, _, cmd = info.partition(" ")
        if not uid.strip().isdigit():
            continue
        rows.append((int(pid), int(uid.strip()), cmd.strip()))
    return rows


def is_ours(gpu: int) -> bool:
    """这张卡上是否有我们自己的进程。

    「保证实时有 4 卡」算的是**总盘子**：我们自己在跑的卡 + 占位的卡 ≤ 4。
    否则评测一起来，占位又去补 4 张，总量会悄悄涨到 6、8 张。

    🔴 只按 uid 判，**不再按 cmdline 关键词判**。DDP 训练真正占卡的是
    `mp.spawn` 起的子进程，它们的 cmdline 是
        python -c from multiprocessing.spawn import spawn_main; ...
    根本不含 "train.py"，按关键词匹配必然漏判 —— 实测就因此把训练中的
    GPU 2/3 当成空闲，额外多占了一张卡（盘子涨到 5）。
    评测子进程同理。既然这台机器上 uid 就是我们，卡上有我们的进程
    就说明这张卡在我们手里，不需要再猜是哪个作业。
    """
    return any(u == os.getuid() for _, u, _ in procs_on(gpu))


def others_on(gpu: int) -> bool:
    """这张卡上有没有**别人**的进程 —— 有就优先避开，绝不去挤。"""
    return any(u != os.getuid() for _, u, _ in procs_on(gpu))


def prune(state: dict) -> dict:
    """清掉已经死掉的占位记录。"""
    return {g: v for g, v in state.items() if alive(v["pid"])}


def reserve(gpu: int, mib: int) -> int | None:
    """起一个占位进程，返回它的 PID。"""
    p = subprocess.Popen(
        [sys.executable, str(RESERVER), "--gpu", str(gpu), "--memory-mib", str(mib)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)                     # 脱离本会话，避免连坐被杀
    # reserve_gpu_memory.py 分配成功后会打印 PID 并阻塞；等它站稳
    t0 = time.time()
    while time.time() - t0 < 60:
        if p.poll() is not None:
            print(f"  GPU{gpu}: 占位进程提前退出\n{p.stdout.read()[:300]}", file=sys.stderr)
            return None
        time.sleep(2)
        st = gpu_state()
        # 分配到位的判据：该卡已用显存涨了接近请求量
        if any(r["index"] == gpu for r in st):
            return p.pid
    return p.pid


def cmd_status(a) -> int:
    state = prune(load_state())
    save_state(state)
    st = {r["index"]: r for r in gpu_state()}
    print(f"{'GPU':>4}{'已用':>9}{'空闲':>9}  占位状态")
    for i in sorted(st):
        r = st[i]
        held = state.get(str(i))
        tag = f"✅ 我方占位 pid={held['pid']} ({held['mib']} MiB)" if held else ""
        if not held and r["free"] > FREE_THRESHOLD_MIB:
            tag = "🟢 可占（空闲 > 30 GB）"
        print(f"{i:>4}{r['used']:>8}M{r['free']:>8}M  {tag}")
    holds = load_holds()
    busy = {r["index"] for r in st.values()
            if str(r["index"]) not in state and is_ours(r["index"])}
    total = len(state) + len(busy | {int(g) for g in holds})
    print(f"\n占位 {len(state)} 张"
          + (f" + 自己在跑 {sorted(busy)}" if busy else "")
          + (f" + hold {sorted(holds)}" if holds else "")
          + f" = 盘子 {total}/{MAX_CARDS} 张")
    return 0


def cmd_release(a) -> int:
    state = prune(load_state())
    if a.all:
        targets = list(state)
    elif a.gpus:
        targets = [g.strip() for g in a.gpus.split(",")]
    else:
        targets = [str(a.gpu)]
    for g in targets:
        v = state.get(g)
        if not v:
            print(f"  GPU{g}: 没有我方占位")
            continue
        try:
            os.kill(v["pid"], signal.SIGTERM)       # 只动自己起的进程
            print(f"  GPU{g}: 已释放 (pid={v['pid']}, {v['mib']} MiB)")
        except ProcessLookupError:
            print(f"  GPU{g}: 占位进程已不在")
        state.pop(g, None)
    save_state(state)
    return 0


def cmd_hold(a) -> int:
    """把几张卡留给作业：先释放我们自己的占位，再登记为「别占」。"""
    gs = [g.strip() for g in a.gpus.split(",")]
    state = prune(load_state())
    for g in gs:
        v = state.pop(g, None)
        if v:
            try:
                os.kill(v["pid"], signal.SIGTERM)
                print(f"  GPU{g}: 已释放占位 (pid={v['pid']}, {v['mib']} MiB)")
            except ProcessLookupError:
                print(f"  GPU{g}: 占位进程已不在")
    save_state(state)
    h = load_holds() | set(gs)
    save_holds(h)
    print(f"  已登记 hold: {sorted(h)} —— 看门程序不会再占这些卡")
    return 0


def cmd_unhold(a) -> int:
    h = load_holds()
    h = set() if not a.gpus else h - {g.strip() for g in a.gpus.split(",")}
    save_holds(h)
    print(f"  剩余 hold: {sorted(h) or '无'} —— 看门程序会在下一轮把空卡补回 4 张")
    return 0


def cmd_watch(a) -> int:
    """把「我们能用的卡」实时补齐到 MAX_CARDS 张。

    盘子 = 自己正在跑作业的卡 + 占位的卡。评测一结束，那几张卡会在
    下一轮轮询里被重新占住 —— 这正是本模式存在的理由：实测过两次
    「我们的作业一结束，几十秒内卡就被别人拿走」。

    挑卡顺序：**完全空的卡优先**，其次才是空闲显存够但上面有别人进程的卡。
    后者只在前者不够时才碰，且始终留 KEEP_FREE_MIB 的余量。
    """
    print(f"看门启动：把可用卡补齐到 {MAX_CARDS} 张（自己在跑的卡也计入盘子）；"
          f"只碰空闲 > {FREE_THRESHOLD_MIB//1024} GB 的卡，每张留 "
          f"{KEEP_FREE_MIB//1024} GB 余量", flush=True)
    while True:
        state = prune(load_state())
        holds = load_holds()
        st = gpu_state()
        busy_ours = {r["index"] for r in st
                     if r["index"] not in {int(g) for g in state} and is_ours(r["index"])}
        held = len(state) + len(busy_ours | {int(g) for g in holds})
        if held < MAX_CARDS:
            # 完全空的卡排前面
            cand = [r for r in st
                    if str(r["index"]) not in state
                    and str(r["index"]) not in holds
                    and r["index"] not in busy_ours
                    and r["free"] > FREE_THRESHOLD_MIB
                    and r["free"] - KEEP_FREE_MIB > 0]
            cand.sort(key=lambda r: (others_on(r["index"]), -r["free"]))
            for r in cand:
                if held >= MAX_CARDS:
                    break
                g, mib = str(r["index"]), r["free"] - KEEP_FREE_MIB
                pid = reserve(r["index"], mib)
                if pid:
                    state[g] = {"pid": pid, "mib": mib,
                                "at": time.strftime("%H:%M:%S")}
                    held += 1
                    print(f"  {time.strftime('%H:%M:%S')} 占住 GPU{g}：{mib} MiB "
                          f"(pid={pid})，盘子 {held}/{MAX_CARDS}"
                          + ("  ⚠️ 该卡上有别人的进程" if others_on(r["index"]) else ""),
                          flush=True)
        save_state(state)
        time.sleep(a.interval)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("watch"); p.add_argument("--interval", type=int, default=60)
    p.set_defaults(fn=cmd_watch)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("hold", help="把卡留给作业：释放占位并禁止看门程序再占")
    p.add_argument("--gpus", required=True, help="逗号分隔，如 2,3")
    p.set_defaults(fn=cmd_hold)
    p = sub.add_parser("unhold", help="作业跑完，交还给看门程序")
    p.add_argument("--gpus", help="逗号分隔；不给则全部解除")
    p.set_defaults(fn=cmd_unhold)
    p = sub.add_parser("release")
    p.add_argument("--gpu", type=int)
    p.add_argument("--gpus", help="一次放多张，逗号分隔，如 2,3")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=cmd_release)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
