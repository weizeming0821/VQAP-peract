#!/usr/bin/env python
"""v3.6：每段最短停留帧数（来自 train split 真实分段统计）。

由来：sweep 与 reach_and_drag 在 v3.4/v3.5 上都比模板法跌 18~36 pp，
跨版本重现，且不是 v3.4/v3.5 的机制造成的（EXTEND 在 sweep 上触发 0 次）。
逐局对照：成功与失败的差别精确地就是「多推进了一段」——
  sweep 成功局到达段位 2.2 / 失败局 3.0；drag 2.1 / 3.1。
模板计划带着每段真实关键帧数（transfer 要 2 帧），X3 生成的一律是 1。
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import stage3.vlm_planner as V

ok = True
def t(name, cond, extra=""):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else "  " + str(extra)))
    ok = ok and cond

seg = lambda a: {"action": a, "instruction": f"{a} it", "use_codebook": True,
                 "k_global": -1, "k_detail": [-1]*9, "n_keyframes": 1}
FRAME = {"front": np.zeros((3, 32, 32), np.uint8), "wrist": np.zeros((3, 32, 32), np.uint8)}

class C:
    def chat(self, m, model=None): return {"content": json.dumps({"plan":[{"action":"push","instruction":"push it"}]})}

print("=== 先验来自 train split，且只覆盖少数组合 ===")
D = V.load_dwell_prior()
t("载入成功", isinstance(D, dict) and len(D) > 0, len(D))
t("sweep 的 transfer 要待 2 帧", D.get("sweep_to_dustpan_of_size", {}).get("transfer") == 2, D.get("sweep_to_dustpan_of_size"))
t("reach_and_drag 的 transfer 要待 2 帧", D.get("reach_and_drag", {}).get("transfer") == 2, D.get("reach_and_drag"))
n_pairs = sum(len(v) for v in D.values())
t(f"只有 {n_pairs} 个 (task,action) 组合有约束（其余默认 1，规则不触发）", n_pairs <= 12, n_pairs)
t("open_drawer 无约束（不受影响）", "open_drawer" not in D, D.get("open_drawer"))

def make(dwell):
    p = V.OnlineVLMPlanner("sweep_to_dustpan_of_size", "sweep dirt", C(),
                           prior=[["push"]], phrasings=[], dwell=dwell)
    p.subtasks = [seg("approach"), seg("transfer"), seg("wipe")]
    return p

print("=== 表达方式 A：执行期硬拦截（需先关掉 B，两者互斥）===")
V.MIN_DWELL = True; V.EXPAND_DWELL = False
p = make({"transfer": 2}); p.idx = 1; p.used = 1        # transfer 只待了 1 帧
p._apply("NEXT", FRAME)
t("段位不动", p.idx == 1, p.idx)
t("记 next_too_early", p.stats.get("next_too_early") == 1, p.stats)

print("=== 待够了就正常推进 ===")
p = make({"transfer": 2}); p.idx = 1; p.used = 2
p._apply("NEXT", FRAME)
t("正常前进", p.idx == 2, p.idx)
t("没有误记", p.stats.get("next_too_early") is None)

print("=== 无约束的段逐字不变 ===")
p = make({"transfer": 2}); p.idx = 0; p.used = 1        # approach 无约束
p._apply("NEXT", FRAME)
t("approach 段 used=1 也能推进", p.idx == 1, p.idx)

print("=== 关掉开关 → 逐字退回 v3.5 ===")
V.MIN_DWELL = False
p = make({"transfer": 2}); p.idx = 1; p.used = 1
p._apply("NEXT", FRAME)
t("不再拦截", p.idx == 2 and p.stats.get("next_too_early") is None, (p.idx, p.stats))
V.MIN_DWELL = True

print("=== 没有先验时也不报错（先验缺失 → 旧行为）===")
p = make(None); p.idx = 1; p.used = 1
p._apply("NEXT", FRAME)
t("无先验时正常推进", p.idx == 2, p.idx)

print("=== 表达方式 B：把先验本身修对（EXPAND_DWELL）===")
from stage3.vlm_planner import _expand_dwell
e = _expand_dwell("sweep_to_dustpan_of_size", ["approach","grasp","transfer","wipe"])
t("sweep 的 transfer 被展开成两段", e == ["approach","grasp","transfer","transfer","wipe"], e)
e2 = _expand_dwell("reach_and_drag", ["approach","grasp","lift","transfer","pose-adjust","slide"])
t("drag 的 transfer 被展开成两段", e2.count("transfer") == 2, e2)
e3 = _expand_dwell("open_drawer", ["approach","grasp","pull"])
t("open_drawer 逐字不变（无约束）", e3 == ["approach","grasp","pull"], e3)
e4 = _expand_dwell("no_such_task", ["approach","grasp"])
t("未知任务逐字不变", e4 == ["approach","grasp"], e4)

print("=== A 与 B 必须互斥（否则 transfer 会被要求待满 4 帧）===")
V.EXPAND_DWELL = True; V.MIN_DWELL = True
p6 = make({"transfer": 2}); p6.idx = 1; p6.used = 1
p6._apply("NEXT", FRAME)
t("先验已展开时不再硬拦截", p6.idx == 2 and p6.stats.get("next_too_early") is None,
  (p6.idx, p6.stats))
V.EXPAND_DWELL = False
p7 = make({"transfer": 2}); p7.idx = 1; p7.used = 1
p7._apply("NEXT", FRAME)
t("关掉展开后硬拦截恢复", p7.idx == 1 and p7.stats.get("next_too_early") == 1)
V.EXPAND_DWELL = True

print("\n" + ("全部通过 ✅" if ok else "有失败 ❌"))
sys.exit(0 if ok else 1)
