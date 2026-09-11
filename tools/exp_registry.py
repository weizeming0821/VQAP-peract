#!/usr/bin/env python
"""改进方案的版本登记 —— 让每个尝试可追溯、可回退、可对照。

# 为什么需要

改进是一轮一轮试出来的，而本项目已经吃过两次亏：
  · 旧机 X3 迭代了六个版本，每次「机制精确生效、成绩纹丝不动」，
    但因为没有统一登记，事后无法回答「第 N 版到底改了什么、相对谁在比」；
  · `result/p7/versions/` 的逐局分数没随迁移带过来，旧 test 主表再也无法复核。

所以每个方案在**开训之前**登记一条：改了什么、基于哪个 commit、怎么回退。
跑完把结果写回同一条。失效就按 `rollback` 字段退回去。

# 一个方案 = 一个臂名

`stage3/arms.py` 的注释写得很清楚：臂名是归档文件名、`--arm` 参数、结果汇总表
的唯一区分依据，**混用一定出乱子**。所以每个改进变体都必须有自己的臂名，
权重落在自己的 `checkpoints/stage3_main/<arm>/`，不覆盖任何已有结果。

    python tools/exp_registry.py add  --arm B4L --base B4 \
        --change "lang=task(保留整任务指令)" --change "use_detail=False" \
        --why "flat 实测 +3.78pp；细节码 9 槽恒等占 99.7%" \
        --rollback "删 checkpoints/stage3_main/B4L，代码改动都在开关后面"
    python tools/exp_registry.py result --arm B4L --metric seen18_template 40.1 \
        --note "450 局 test"
    python tools/exp_registry.py list
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REG = REPO_ROOT / "result" / "p5_train" / "experiments.json"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    except Exception:
        return "?"


def load() -> dict:
    if REG.is_file():
        return json.loads(REG.read_text())
    return {"schema": "aavla_exp_registry_v1", "experiments": []}


def save(doc: dict) -> None:
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(doc, ensure_ascii=False, indent=1))


def cmd_add(a) -> int:
    doc = load()
    if any(e["arm"] == a.arm for e in doc["experiments"]):
        print(f"❌ 臂名 {a.arm} 已登记。换一个名字 —— 臂名复用会让归档、"
              f"--arm 参数、汇总表全部对不上。", file=sys.stderr)
        return 2
    # 🔴 只拦**步数 > 0** 的权重。weights/0 是 init_from_official.py 装的官方起点，
    #    是合法的、必须存在的；拦它会让正常流程走不通。要拦的是「上一轮训练留下的
    #    中间 ckpt」—— offline_train_runner 按最大步数恢复，会从那里静默续训。
    wd = REPO_ROOT / "checkpoints" / "stage3_main" / a.arm / "seed0" / "weights"
    stale = sorted(d.name for d in wd.glob("*")
                   if d.is_dir() and d.name.isdigit() and int(d.name) > 0) if wd.is_dir() else []
    if not a.already_run and stale:
        print(f"❌ {wd} 下已有步数 >0 的权重 {stale[:5]}。先清理或换臂名，"
              f"否则会从旧权重静默续训（offline_train_runner 取最大步数恢复，不报错）。",
              file=sys.stderr)
        return 2
    doc["experiments"].append({
        "arm": a.arm,
        "base": a.base,
        "status": "done" if a.already_run else "planned",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "changes": a.change or [],
        "why": a.why or "",
        "rollback": a.rollback or f"删除 checkpoints/stage3_main/{a.arm}/",
        "results": {},
        "verdict": "",
    })
    save(doc)
    print(f"  ✅ 已登记 {a.arm}（基于 {a.base}，commit {doc['experiments'][-1]['commit']}）")
    for c in (a.change or []):
        print(f"       改动: {c}")
    return 0


def cmd_result(a) -> int:
    doc = load()
    e = next((x for x in doc["experiments"] if x["arm"] == a.arm), None)
    if e is None:
        print(f"❌ {a.arm} 未登记，先 add", file=sys.stderr)
        return 2
    name, value = a.metric
    e["results"][name] = {"value": float(value), "note": a.note or "",
                          "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    e["status"] = "measured"
    if a.verdict:
        e["verdict"] = a.verdict
    save(doc)
    print(f"  ✅ {a.arm}.{name} = {value}" + (f"   {a.note}" if a.note else ""))
    return 0


def cmd_list(a) -> int:
    doc = load()
    if not doc["experiments"]:
        print("（还没有登记任何方案）")
        return 0
    for e in doc["experiments"]:
        flag = {"planned": "⬜", "measured": "📊"}.get(e["status"], "  ")
        print(f"\n{flag} {e['arm']}  ← 基于 {e['base']}   "
              f"commit {e['commit']}{'(dirty)' if e.get('dirty') else ''}   {e['created_at']}")
        for c in e["changes"]:
            print(f"     · {c}")
        if e["why"]:
            print(f"     依据: {e['why']}")
        for k, v in e["results"].items():
            print(f"     {k} = {v['value']}" + (f"   （{v['note']}）" if v["note"] else ""))
        if e["verdict"]:
            print(f"     判定: {e['verdict']}")
        print(f"     回退: {e['rollback']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add", help="开训之前登记一个方案")
    p.add_argument("--arm", required=True, help="新臂名，不得与已有臂重名")
    p.add_argument("--base", required=True, help="基于哪个臂")
    p.add_argument("--change", action="append", help="可重复；一条改动一个")
    p.add_argument("--why", help="依据（最好带实测数字）")
    p.add_argument("--rollback", help="怎么退回去")
    p.add_argument("--already-run", action="store_true",
                   help="补登记已经跑完的臂 —— 跳过「权重目录必须为空」的守卫。"
                        "新方案绝不要用这个：那条守卫防的是从旧权重静默续训。")
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("result", help="把测量结果写回")
    p.add_argument("--arm", required=True)
    p.add_argument("--metric", nargs=2, metavar=("NAME", "VALUE"), required=True)
    p.add_argument("--note")
    p.add_argument("--verdict", help="采纳 / 放弃 / 待定")
    p.set_defaults(fn=cmd_result)
    p = sub.add_parser("list")
    p.set_defaults(fn=cmd_list)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
