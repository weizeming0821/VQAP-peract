#!/usr/bin/env python
"""v3.2 三项改动的单测：指令三层协议 / 执行记忆 / 先验变体。

改动的动机全部来自 v3.1 的实测（B3@40000 · val · 120 局 · 29.17%）：

  ① 自由生成的指令只有 **45%** 落在训练分布内（模板法按定义 100%），
     而分布内比例最低的三个任务正是掉分最多的（−34 ~ −40 pp）。
     → 三层协议：优先按编号复用训练原文，其次替换 variation 词，
       都不行才模仿句式自由生成。

  ② MONITOR 的 prompt 里没有「已经尝试过什么」，RETRY 回退后只显示
     CURRENT=0，看不出「我已经试过第 1~4 步并退回来了」。实测出现过
     连判 10 次同一句 "grasp failed" —— 判断次次正确却没有依据改变做法。
     → 执行记忆进 prompt，措辞是 ATTEMPTED 不是 COMPLETED。

  ③ slide_block 的先验是 5 段且动作词写成 press，而 train 里最常见的是
     单个 push（6/10）。在线 planner 忠实照它生成 5 段，成绩 64 → 30。
     → 多变体先验，走不通时换一种分解。
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner.offline import PRIOR_VARIANTS, load_priors, load_prior_variants   # noqa: E402
from planner.prompts import (build_monitor_user_content,                        # noqa: E402
                             build_plan_system_prompt, build_plan_user_content,
                             render_attempt_log)
from stage3.vlm_planner import (OnlineVLMPlanner, PlannerError,                 # noqa: E402
                                _resolve_phrasing)

OK = True


def check(name, cond, extra=""):
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


BANK = ["grasp: grasp the red jar lid",
        "lift: lift the red jar lid",
        "transfer: move the lid over the red jar"]

print("=== 1. 指令三层协议 ===")
ins, layer = _resolve_phrasing({"action": "grasp", "phrasing_id": 0}, BANK)
check("A 复用：切掉 'action: ' 前缀", ins == "grasp the red jar lid", ins)
check("A 层标记正确", layer == "A", layer)

ins, layer = _resolve_phrasing(
    {"action": "grasp", "phrasing_id": 0, "substitute": {"red": "black"}}, BANK)
check("B 替换：variation 词被换掉", ins == "grasp the black jar lid", ins)
check("B 层标记正确", layer == "B", layer)

ins, layer = _resolve_phrasing({"action": "wipe",
                                "instruction": "sweep dirt into the short dustpan"}, BANK)
check("C 自由生成原样保留", ins == "sweep dirt into the short dustpan", ins)
check("C 层标记正确", layer == "C", layer)

# 整词替换：不能命中子串
ins, _ = _resolve_phrasing({"phrasing_id": 0, "substitute": {"red": "blue"}},
                           ["grasp: grasp the predicted red lid"])
check("整词替换：'red' 不该命中 'predicted' 里的 red",
      ins == "grasp the predicted blue lid", ins)

# 越界的 phrasing_id 要能退到 instruction
ins, layer = _resolve_phrasing({"phrasing_id": 99, "instruction": "grasp the lid"}, BANK)
check("phrasing_id 越界 -> 退回 instruction", ins == "grasp the lid" and layer == "C",
      f"{ins} / {layer}")
try:
    _resolve_phrasing({"action": "grasp"}, BANK)
    check("两者都没有必须 raise", False)
except PlannerError:
    check("两者都没有必须 raise", True)

print("=== 2. prompt 里的协议与编号 ===")
sysp = build_plan_system_prompt()
check("系统 prompt 含三层协议", "HARD PROTOCOL" in sysp)
for tag in ('"phrasing_id"', '"substitute"', "REUSE", "SUBSTITUTE", "IMITATE"):
    check(f"系统 prompt 含 {tag}", tag in sysp)
txt = build_plan_user_content("close_jar", "close the red jar", [],
                              phrasings=BANK)[0]["text"]
check("候选清单带编号 [0]/[1]/[2]",
      all(f"[{i}]" in txt for i in range(len(BANK))), txt[-200:])

print("=== 3. 执行记忆 ===")
H = ([{"subtask_index": 0, "action": "approach", "instruction": "move toward the red block"}] * 3
     + [{"subtask_index": 1, "action": "push", "instruction": "push the red block"}] * 6
     + [{"subtask_index": 0, "action": "approach", "instruction": "move toward the red block"}] * 2)
D = [{"decision": "NEXT", "idx": 0, "reason": "in position"},
     {"decision": "RETRY", "idx": 1, "reason": "block not on target"}]
log = "\n".join(render_attempt_log(H, D, budget=26))
check("措辞是 ATTEMPTED 不是 completed",
      "ATTEMPTED" in log and "completed" not in log.replace("NOT necessarily completed", ""))
check("逐帧记录被压成段+帧数", "3 keyframe(s)" in log and "6 keyframe(s)" in log, log)
check("标出 CURRENT", "<- CURRENT" in log)
check("检测到回退并给出警告", "rolled back" in log, log)
check("累计帧数与预算", "total keyframes used: 11 of about 26" in log, log)
check("决策历史在内", "RETRY" in log and "block not on target" in log)
check("空历史返回空列表", render_attempt_log([]) == [])

mon = build_monitor_user_content("close_jar", "close the red jar",
                                 [{"action": "grasp", "instruction": "grasp the red jar lid"}],
                                 0, 3, 1.0, [], history=H, decisions=D, budget=26)[0]["text"]
check("MONITOR 里有最终目标（整任务指令）", "task instruction: close the red jar" in mon)
check("MONITOR 里有执行记忆", "ATTEMPTED" in mon)

print("=== 4. 先验变体 ===")
t = "slide_block_to_color_target"
check("slide_block 登记了 2 个变体", len(PRIOR_VARIANTS[t]) == 2)
check("变体 0 是训练里更常见的单段 push", PRIOR_VARIANTS[t][0] == ["push"],
      str(PRIOR_VARIANTS[t][0]))
check("变体 1 是 5 段版", len(PRIOR_VARIANTS[t][1]) == 5)
check("动作词是 push 不是 press（原 override 写错了）",
      "press" not in PRIOR_VARIANTS[t][1], str(PRIOR_VARIANTS[t][1]))
# 🔴 离线与在线**有意不同**：load_priors() 必须与建离线 cache 时逐位一致
#    （scripts/planner_cache.py 依赖它，而那份 cache 是已冻结的训练工件）；
#    修正只作用于在线 planner。这条断言就是防止有人"顺手统一"两者。
check("离线 load_priors 保持原样（可复现性）",
      load_priors()[t] == ["approach", "press", "pose-adjust", "approach", "press"],
      str(load_priors()[t]))
check("在线变体 0 才是修正后的 push", load_prior_variants()[t][0] == ["push"],
      str(load_prior_variants()[t][0]))
var = load_prior_variants()
check("只有 slide_block 是多变体",
      [k for k, v in var.items() if len(v) > 1] == [t])
check("其余任务仍是单变体列表", len(var["close_jar"]) == 1)

pl = OnlineVLMPlanner(t, "slide the block", None, prior=PRIOR_VARIANTS[t])
check("开局用变体 0", pl.prior == ["push"], str(pl.prior))
check("切换成功", pl._next_variant() is True)
check("切换后是变体 1", pl.prior == PRIOR_VARIANTS[t][1], str(pl.prior))
check("没有更多变体时返回 False", pl._next_variant() is False)
check("切换次数被记账", pl.stats.get("prior_variant_switch") == 1,
      str(pl.stats.get("prior_variant_switch")))

pl2 = OnlineVLMPlanner("close_jar", "close the red jar", None, prior=["grasp", "lift"])
check("旧的 list[str] 先验仍兼容", pl2.prior == ["grasp", "lift"], str(pl2.prior))
check("单变体不切换", pl2._next_variant() is False)

print("\nv3.2 单测: " + ("PASS" if OK else "FAIL"))
raise SystemExit(0 if OK else 1)
