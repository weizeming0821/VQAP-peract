#!/usr/bin/env python
"""v3.5：末段禁用 NEXT + EXTEND 续写计划。

由来（B4@40000 · val 300 局 · 143 个纯 v3.3 局）：
  末段决策 95% 是 NEXT（500/526），而末段 NEXT 被 clamp 成空操作 ——
  指令一字不变 → PerAct 输入不变 → 输出必然不变。
  末段死锁 ≥5 次的 40 局成功率 0%，其余局 67%。
"""
import sys, json
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


def seg(a, i):
    return {"action": a, "instruction": f"{a} the block {i}", "use_codebook": True,
            "k_global": -1, "k_detail": [-1]*9, "n_keyframes": 1}


class FakeClient:
    """PLAN 调用返回固定的两段续写；MONITOR 不经由它（测试直接调 _apply）。"""
    def __init__(self): self.calls = 0
    def chat(self, msgs, model=None):
        self.calls += 1
        return {"content": json.dumps({"plan": [
            {"action": "push", "instruction": "push the block again"},
            {"action": "pose-adjust", "instruction": "adjust arm pose"}]})}


def make(nseg=3):
    p = V.OnlineVLMPlanner("slide_block_to_color_target", "slide the block to green",
                           FakeClient(), prior=[["push"]], phrasings=[])
    p.subtasks = [seg("push", i) for i in range(nseg)]
    p.idx = nseg - 1                       # 停在末段
    return p


FRAME = {"front": np.zeros((3, 32, 32), np.uint8),
         "wrist": np.zeros((3, 32, 32), np.uint8)}

print("=== 决策白名单 ===")
t("EXTEND 已进白名单", "EXTEND" in V._DECISIONS, V._DECISIONS)
t("裸文本也能解析出 EXTEND", V._parse_decision("EXTEND")[0] == "EXTEND")
t("EXTEND 不被 CONTINUE 子串吞掉",
  V._parse_decision('{"decision":"EXTEND","reason":"plan ran out"}')[0] == "EXTEND")

print("=== 末段的 NEXT 判为 EXTEND（语义忠实的兜底）===")
V.ALLOW_EXTEND = True; V.MAX_EXTEND = 2; V.MAX_SEGMENTS = 10
# 走真实路径：从第 0 段用 NEXT 推到末段，done 沿途累积
p = make(3); p.idx = 0
p._apply("NEXT", FRAME); p._apply("NEXT", FRAME)
t("推到末段", p.idx == 2 and len(p.done) == 2, (p.idx, len(p.done)))
p._apply("NEXT", FRAME)                     # 末段再 NEXT -> 应判为 EXTEND
t("记了 next_at_last", p.stats.get("next_at_last") == 1, p.stats)
t("计划被续写到 5 段", len(p.subtasks) == 5, len(p.subtasks))
t("idx 落在新追加的第一段", p.idx == 3, p.idx)
t("已完成的段全部保留（3 段，对比 REPLAN 会清空）", len(p.done) == 3, len(p.done))
t("续写留痕", len(p.extends) == 1 and p.extends[0]["n"] == 1, p.extends)
t("code_epoch 递增（Adapter 重算码）", p.code_epoch >= 1, p.code_epoch)

print("=== EXTEND vs REPLAN：进度存留（核心区别）===")
pr = make(3); pr.idx = 0
pr._apply("NEXT", FRAME); pr._apply("NEXT", FRAME)
before = len(pr.done)
pr._apply("REPLAN", FRAME)
t("REPLAN 清空 done 并把 idx 归零", pr.done == [] and pr.idx == 0, (pr.done, pr.idx))
pe = make(3); pe.idx = 0
pe._apply("NEXT", FRAME); pe._apply("NEXT", FRAME)
pe._apply("EXTEND", FRAME)
t("EXTEND 保留 done 且 idx 不回零", len(pe.done) == 3 and pe.idx == 3, (len(pe.done), pe.idx))
t(f"两者对比：REPLAN done {before}→0，EXTEND done {before}→{len(pe.done)}", True)

print("=== 非末段的 NEXT 行为逐字不变 ===")
p2 = make(3); p2.idx = 0
p2._apply("NEXT", FRAME)
t("正常前进一段", p2.idx == 1, p2.idx)
t("没有触发续写", p2.extends == [] and p2.stats.get("next_at_last") is None)
t("计划长度不变", len(p2.subtasks) == 3)

print("=== 护栏 ===")
p3 = make(3)
for _ in range(4):
    p3._apply("EXTEND", FRAME)
t(f"最多续写 MAX_EXTEND={V.MAX_EXTEND} 次", p3.n_extend == 2, p3.n_extend)
t("超限后记 extend_blocked", p3.stats.get("extend_blocked", 0) >= 1, p3.stats)
t("超限后置 _plan_exhausted（本局停止再问）", p3._plan_exhausted is True)

V.MAX_SEGMENTS = 4
p4 = make(3)
p4._apply("EXTEND", FRAME)          # 3 -> 5 段
p4._apply("EXTEND", FRAME)          # 已超 MAX_SEGMENTS，应被挡
t("总段数上限生效", len(p4.subtasks) <= 5 and p4.n_extend == 1,
  (len(p4.subtasks), p4.n_extend))
V.MAX_SEGMENTS = 10

print("=== 关掉 EXTEND → 逐字退化成 v3.4 行为 ===")
V.ALLOW_EXTEND = False
p5 = make(3)
p5._apply("NEXT", FRAME)
t("末段 NEXT 不再续写", p5.extends == [] and len(p5.subtasks) == 3)
t("idx 停在末段（空操作，与旧版一致）", p5.idx == 2, p5.idx)
p6 = make(3)
p6._apply("EXTEND", FRAME)
t("显式 EXTEND 也被挡下", p6.n_extend == 0 and p6.stats.get("extend_blocked") == 1)
V.ALLOW_EXTEND = True

print("=== prompt：末段必须说明「没有下一步」===")
from planner.prompts import build_monitor_user_content, build_monitor_system_prompt
c = build_monitor_user_content("t", "ti", [seg("push", 0)], 0, 1, 1.0, [], at_last=True)
txt = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
t("末段有 LAST-STEP NOTE", "LAST-STEP NOTE" in txt)
t("明说 NEXT 不可用", "NEXT is NOT available" in txt)
t("给出 EXTEND 出路", "EXTEND" in txt)
c2 = build_monitor_user_content("t", "ti", [seg("push", 0)], 0, 1, 1.0, [], at_last=False)
txt2 = " ".join(x.get("text", "") for x in c2 if isinstance(x, dict))
t("非末段不出现该提示（不污染正常路径）", "LAST-STEP NOTE" not in txt2)
sysp = build_monitor_system_prompt()
t("system prompt 定义了 EXTEND", "EXTEND" in sysp)
t("说明 EXTEND 保留进度而 REPLAN 丢弃", "discarding all progress" in sysp and "KEPT" in sysp)

print("\n" + ("全部通过 ✅" if ok else "有失败 ❌"))
sys.exit(0 if ok else 1)
