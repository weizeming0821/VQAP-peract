#!/usr/bin/env python
"""在线 planner 的断线韧性（2026-09-09 事故的回归门禁）。

一次 APIConnectionError 曾让 12 个分片分别在第 7~24 局被打死，
300 局只跑出 126 局。这里钉死三件事：
  1. PLAN 失败要回落到模板计划，而不是抛异常杀掉整个分片
  2. MONITOR 失败要退化成 CONTINUE，局照常跑完
  3. 退化过的局必须**可识别**（plan_source / stats），否则成绩单说不清
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import stage3.vlm_planner as V
from stage3.online_planner import PlannerError

ok = True
def t(name, cond, extra=""):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else "  " + str(extra)))
    ok = ok and cond


class DeadClient:
    """永远连不上的客户端 —— 模拟 APIConnectionError。"""
    def __init__(self): self.calls = 0
    def chat(self, msgs, model=None):
        self.calls += 1
        raise RuntimeError("调用失败（重试 4 次）: APIConnectionError: Connection error.")


FALLBACK = [
    {"action": "grasp", "instruction": "grasp the red block",
     "use_codebook": True, "k_global": 3, "k_detail": [1]*9, "n_keyframes": 2},
    {"action": "lift", "instruction": "lift the red block",
     "use_codebook": True, "k_global": 4, "k_detail": [2]*9, "n_keyframes": 1},
]
FRAME = {"front": np.zeros((3, 64, 64), np.uint8),
         "wrist": np.zeros((3, 64, 64), np.uint8)}


def make(fallback):
    c = DeadClient()
    p = V.OnlineVLMPlanner("close_jar", "close the red jar", c,
                           prior=[["grasp", "lift"]], phrasings=["grasp the red block"],
                           fallback=fallback)
    return p, c


print("=== 默认开关 = X3 最终版配置 ===")
t("USE_HISTORY 默认关（v3.2c 起的最优配置）", V.USE_HISTORY is False, V.USE_HISTORY)
t("PHRASING_MODE 默认 select", V.PHRASING_MODE == "select")
t("USE_PRIOR_VARIANTS 默认开", V.USE_PRIOR_VARIANTS is True)
t("USE_GRIPPER_SEMANTICS 默认开", V.USE_GRIPPER_SEMANTICS is True)

print("=== PLAN 失败 → 回落模板计划，不抛异常 ===")
p, c = make(FALLBACK)
try:
    p.prime(1.0, FRAME)
    raised = False
except PlannerError:
    raised = True
t("prime 没有抛 PlannerError", not raised)
t("用上了模板计划", [s["instruction"] for s in p.subtasks]
  == ["grasp the red block", "lift the red block"], p.subtasks)
t("plan_source 标记为 template_fallback", p.plan_source == "template_fallback", p.plan_source)
t("stats 记了 plan_fallback", p.stats.get("plan_fallback") == 1, p.stats)
t("判定 API 已断", p._api_down is True)

print("=== 断线后本局不再发起调用 ===")
before = c.calls
for _ in range(5):
    p.observe(1.0, FRAME, robot={"gripper_pose": np.zeros(7, np.float32),
                                 "joint_velocities": np.zeros(7, np.float32)})
t("observe 期间零调用", c.calls == before, f"{before} → {c.calls}")
t("记了 skipped_while_down", p.stats.get("skipped_while_down", 0) >= 1, p.stats)

print("=== 没有兜底计划时，仍然抛错（不静默） ===")
p2, _ = make(None)
try:
    p2.prime(1.0, FRAME); raised = False
except PlannerError:
    raised = True
t("无兜底 → 抛 PlannerError", raised)

print("=== MONITOR 失败 → 退化成 CONTINUE，不抛 ===")
p3, c3 = make(FALLBACK)
p3.subtasks = [dict(s) for s in FALLBACK]
p3.plan_source = "vlm"; p3._api_down = False; p3._consec_fail = 0
d, tgt, reason = p3._monitor(1.0, FRAME)
t("返回 CONTINUE", d == "CONTINUE", (d, tgt, reason))
t("理由标了 api_error", reason.startswith("api_error"), reason)
t("记了 monitor_fallback", p3.stats.get("monitor_fallback") == 1, p3.stats)
t("单次失败还不算断线", p3._api_down is False)
p3._monitor(1.0, FRAME)
t("连续两次失败 → 判定断线", p3._api_down is True)

print("\n" + ("全部通过 ✅" if ok else "有失败 ❌"))
sys.exit(0 if ok else 1)
