#!/usr/bin/env python
"""v3.4 位置回退 + gap 仪表的单元测试（CPU-only 门禁）。

回退是**唯一一处会绕过 PerAct 直接下发动作**的逻辑，而一次运动规划失败就会
让 `custom_rlbench_env.step` 置 terminal=True、整局记 0 分。所以它的触发条件、
次数上限、以及「关掉时必须逐字退化成旧行为」都要在 CPU 上先钉死。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

import stage3.rollout as R

ok = True
def t(name, cond, extra=""):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else "  " + str(extra)))
    ok = ok and cond


class FakePlanner:
    """按脚本推进段位；observe() 把收到的 robot 存下来供断言。"""
    _views = ()
    def __init__(self, moves):
        self._moves = list(moves)      # 每次 observe 之后 idx 变成什么
        self.idx = 0
        self.history = []
        self.subtasks = [{"action": "grasp", "instruction": f"s{i}",
                          "use_codebook": True, "k_global": 0, "k_detail": [0]*9}
                         for i in range(4)]
        self.budgets = [1]*4
        self.seen_robot = []
    @property
    def current(self):
        return self.subtasks[self.idx]
    def prime(self, g, frame=None):
        pass
    def observe(self, g, frame=None, robot=None):
        self.seen_robot.append(robot)
        if self._moves:
            self.idx = self._moves.pop(0)
    def note(self, step):
        self.history.append({"t": len(self.history), "subtask_index": self.idx})


class FakeInner:
    def __init__(self):
        self.calls = 0
    def act(self, step, obs, deterministic=False):
        self.calls += 1
        from yarr.agents.agent import ActResult
        return ActResult(np.array([9., 9., 9., 0., 0., 0., 1., 1., 0.], dtype=np.float32))


def make(moves, poses, rollback=True, maxn=3):
    R.RETRY_ROLLBACK = rollback
    R.ROLLBACK_MAX = maxn
    inner = FakeInner()
    pl = FakePlanner(moves)
    it = iter(poses)
    agent = R._SubtaskAgent(inner, pl, R.TokenCache(), with_codes=False,
                            code_source=None,
                            robot_state=lambda: {"gripper_pose": next(it),
                                                 "joint_velocities": [0.0]*7})
    return agent, pl, inner


def obs():
    return {"low_dim_state": torch.tensor([1.0, 0., 0., 0.])}


P = lambda x: np.array([x, 0., 0., 0., 0., 0., 1.], dtype=np.float32)

print("=== 锚点记录 ===")
a, pl, inner = make(moves=[1, 2], poses=[P(0.), P(1.), P(2.)])
for s in range(3):
    a.act(s, obs())
t("每个首次进入的段位都留了锚点", sorted(a._anchor) == [0, 1, 2], sorted(a._anchor))
t("锚点是 9 维动作格式", all(v.shape == (9,) for v in a._anchor.values()))
t("锚点 xyz 取自实际到达位姿", float(a._anchor[1][0]) == 1.0, a._anchor[1])

print("=== gap / waypoint 仪表 ===")
t("首帧无 gap（没有上一次下发）", "gap" not in pl.history[0], pl.history[0])
t("第 2 帧记了 gap", "gap" in pl.history[1], pl.history[1])
# 上一帧下发 (9,9,9)，这一帧实际到 (1,0,0) → gap = ‖(8,9,9)‖
t("gap 数值正确", abs(pl.history[1]["gap"] - float(np.linalg.norm(
    np.array([9., 9., 9.]) - np.array([1., 0., 0.])))) < 1e-3, pl.history[1]["gap"])
t("记录了 waypoint 与 achieved",
  pl.history[1].get("waypoint") == [9.0, 9.0, 9.0] and pl.history[1].get("achieved") == [1.0, 0.0, 0.0])
t("gap 随 robot 传给了 planner", pl.seen_robot[-1] is not None and "gap" in pl.seen_robot[-1])

print("=== 回退触发（段位回拨）===")
a, pl, inner = make(moves=[1, 2, 0], poses=[P(0.), P(1.), P(2.), P(3.)])
for s in range(4):
    r = a.act(s, obs())
t("回退了 1 次", len(a._rollback_log) == 1, a._rollback_log)
t("回退目标是段位 0 的锚点", a._rollback_log[0]["to_index"] == 0)
t("回退动作 = 锚点位姿（xyz=0）", float(r.action[0]) == 0.0, r.action)
t("回退这一帧没有问 PerAct", inner.calls == 3, inner.calls)
t("回退帧在轨迹里打了标记", pl.history[-1].get("rollback") is True, pl.history[-1])

print("=== 次数上限 ===")
a, pl, inner = make(moves=[1, 0, 1, 0, 1, 0], poses=[P(float(i)) for i in range(7)], maxn=2)
for s in range(7):
    a.act(s, obs())
t("超过 ROLLBACK_MAX 后不再回退", len(a._rollback_log) == 2, len(a._rollback_log))

print("=== 关掉开关必须逐字退化成旧行为 ===")
a, pl, inner = make(moves=[1, 2, 0], poses=[P(0.), P(1.), P(2.), P(3.)], rollback=False)
for s in range(4):
    a.act(s, obs())
t("不回退", a._rollback_log == [])
t("每一帧都问了 PerAct", inner.calls == 4, inner.calls)
t("仪表照常记录（与回退无关）", "gap" in pl.history[1])

print("=== 前进不触发回退 ===")
a, pl, inner = make(moves=[1, 2, 3], poses=[P(float(i)) for i in range(4)])
for s in range(4):
    a.act(s, obs())
t("段位只前进时不回退", a._rollback_log == [])

print("\n" + ("全部通过 ✅" if ok else "有失败 ❌"))
sys.exit(0 if ok else 1)
