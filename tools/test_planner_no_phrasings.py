#!/usr/bin/env python
"""没有候选措辞清单时，在线 planner 必须仍然能跑完 —— 门禁。

# 事故（2026-09-10，B4 探针 280 局）

候选清单取自 **train cache**，UnSeen 任务天然没有。但 PLAN 的 system prompt
无条件带着三层措辞协议，于是 VLM 照办返回 `{"action":"approach","phrasing_id":0}`，
而下游查不到候选 —— `_resolve_phrasing` 直接 `raise PlannerError`，**打死整个分片**。

分片是交错切分的（`tasks[i::n]`），一个 UnSeen 任务能把同分片的 Seen 任务一起拖死：
**9 个分片死亡，28 个任务只剩 7 个出结果**，一次运行报废。

# 两道防线，缺一不可

  1. prompt 侧：没有清单就不给三层协议，明确要求写 instruction
  2. 代码侧：即使 VLM 仍然只给 phrasing_id，也**降级兜底**而不是 raise
     —— 措辞拿不到只是「这一段的话不好听」，不是「计划不可执行」

    source run/env.sh && python tools/test_planner_no_phrasings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planner.prompts import build_plan_system_prompt          # noqa: E402
from stage3.vlm_planner import _resolve_phrasing              # noqa: E402
from stage3.online_planner import PlannerError                # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   {extra}"))
    if not cond:
        FAILED.append(name)


def main() -> int:
    print("① prompt 侧：没有清单就不给三层协议")
    with_list = build_plan_system_prompt(has_phrasings=True)
    no_list = build_plan_system_prompt(has_phrasings=False)
    check("有清单时保留三层协议", "HARD PROTOCOL" in with_list)
    check("有清单时示例含 phrasing_id", '"phrasing_id": 3' in with_list)
    check("无清单时不出现三层协议", "HARD PROTOCOL" not in no_list)
    check("无清单时明确禁止 phrasing_id",
          'Do NOT emit "phrasing_id"' in no_list)
    check("无清单时的输出示例只用 instruction",
          "phrasing_id" not in no_list.split("OUTPUT FORMAT")[1])

    print("\n② 代码侧：拿不到措辞也不能杀分片")
    cases = [
        ("清单为空 + 只有 phrasing_id", {"action": "approach", "phrasing_id": 0}, []),
        ("phrasing_id 越界", {"action": "grasp", "phrasing_id": 99},
         ["grasp: grasp the red jar lid"]),
        ("phrasing_id 非法类型", {"action": "lift", "phrasing_id": "x"}, []),
    ]
    for name, item, phr in cases:
        try:
            ins, layer = _resolve_phrasing(item, phr)
            ok = bool(ins) and layer == "D"
            check(f"{name} → 降级兜底而非抛异常", ok,
                  f"得到 ins={ins!r} layer={layer!r}")
            print(f"       兜底指令: {ins!r}")
        except PlannerError as exc:
            check(f"{name} → 降级兜底而非抛异常", False, f"抛了 {exc}")

    print("\n③ 正常路径不受影响")
    ins, layer = _resolve_phrasing({"action": "grasp", "phrasing_id": 0},
                                   ["grasp: grasp the red jar lid"])
    check("A 层（直接复用）仍然工作", (ins, layer) == ("grasp the red jar lid", "A"),
          f"得到 {(ins, layer)}")
    ins, layer = _resolve_phrasing(
        {"action": "grasp", "phrasing_id": 0, "substitute": {"red": "black"}},
        ["grasp: grasp the red jar lid"])
    check("B 层（整词替换）仍然工作",
          (ins, layer) == ("grasp the black jar lid", "B"), f"得到 {(ins, layer)}")
    ins, layer = _resolve_phrasing({"action": "wipe", "instruction": "sweep the dirt"}, [])
    check("C 层（自由生成）仍然工作", (ins, layer) == ("sweep the dirt", "C"),
          f"得到 {(ins, layer)}")

    print("\n④ 连 action 都没有时仍应硬失败（那是真的不可执行）")
    try:
        _resolve_phrasing({"phrasing_id": 0}, [])
        check("既无 action 又无 instruction → 抛异常", False, "没有抛")
    except PlannerError:
        check("既无 action 又无 instruction → 抛异常", True)

    print("\n" + ("通过" if not FAILED else f"失败 {len(FAILED)} 项: {FAILED}"))
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
