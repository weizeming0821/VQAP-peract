"""原子在码本语料里的**支撑度**，以及据此做的码注入门控 —— 单一真源。

# 依据

码本与 Adapter 在 `AtomAction_Dataset` 上预训练，但各原子的样本量差两个数量级
（实测，`data/atomaction_codebook_index.json`）：

    grasp 13074  approach 11998  lift 9104  transfer 8340  place 6265
    rotate 2132  press 1200 │ pull 800  push 800  revolve-out 600  slide 600
    flip-close 400  wipe 400  insert 283  flip-open 200  revolve-in 200
    hang 100（且只来自 1 个任务）

支撑度低的原子，其码向量是从极少样本里学出来的，携带的动作语义不可靠。

# 为什么按「原子」而不是按「任务」门控

旧机实测过一份**任务级白名单**（把码本没覆盖的任务整体关掉码注入），
未覆盖组涨了 +9 pp。但任务白名单有两个问题：
  · 粒度太粗 —— 同一局里 grasp 段的码明明可靠，却被一起关掉；
  · **UnSeen 上没法构造** —— 它需要事先知道任务在不在码本的 69 个任务里，
    而 Tier-B 按定义就不在，白名单在那里等于全关。

改成按段的原子支撑度之后，2026-09-11 实测：阈值 <1000、段占比 >15% 触发时，
与那份实测有效的任务白名单**重合 4/4**（`place_cups` / `place_wine_at_rack_location` /
`slide_block_to_color_target` / `sweep_to_dustpan_of_size` 全部命中），
而它只依赖 planner 输出的动作标签，**UnSeen 上照样可用**。

# 阈值是事先冻结的

`SUPPORT_MIN = 1000`。这个值来自码本语料自身的统计（段数在 800 与 1200 之间
有一个明显的量级断层），**不是看着测试集结果调出来的**。改它必须在论文里说明。
"""

from __future__ import annotations

import collections
import functools
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INDEX = REPO_ROOT / "data" / "atomaction_codebook_index.json"

#: 支撑度低于此值的原子，其码不注入。事先冻结，见模块 docstring。
SUPPORT_MIN = int(os.environ.get("AAVLA_ATOM_SUPPORT_MIN", "1000"))

#: 是否对该臂启用门控 —— **真源是臂**（stage3/arms.py 的 Arm.atom_gate）。
#: 环境变量只作强制覆盖，用于消融；正常路径一律走臂。
def enabled_for(arm: str | None) -> bool:
    force = os.environ.get("AAVLA_ATOM_GATE")
    if force is not None:
        return force == "1"
    if not arm:
        return False
    from stage3.arms import ARMS
    return bool(ARMS[arm].atom_gate) if arm in ARMS else False


@functools.lru_cache(maxsize=1)
def support() -> dict[str, int]:
    """→ {原子: 该原子在码本语料里的段数}。读不到就返回空字典（等于不门控）。"""
    try:
        recs = json.loads(INDEX.read_text())["records"]
    except Exception:
        return {}
    return dict(collections.Counter(r["action"] for r in recs))


@functools.lru_cache(maxsize=1)
def low_support_atoms() -> frozenset[str]:
    s = support()
    if not s:
        return frozenset()
    return frozenset(a for a, n in s.items() if n < SUPPORT_MIN)


def gated(action: str | None, arm: str | None = None) -> bool:
    """该段是否应当**关闭**码注入。

    未知动作按「关闭」处理：码本没见过的动作，它的码没有意义。
    """
    if not enabled_for(arm):
        return False
    if not action:
        return True
    return action in low_support_atoms() or action not in support()


def summary() -> str:
    s = support()
    low = sorted(low_support_atoms(), key=lambda a: s.get(a, 0))
    return (f"atom_support: 阈值 <{SUPPORT_MIN} 段；"
            f"{len(low)}/{len(s)} 个原子被门控 -> "
            + ", ".join(f"{a}({s.get(a,0)})" for a in low))
