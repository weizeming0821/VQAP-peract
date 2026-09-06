#!/usr/bin/env python
"""`vlm-plan`（真·在线 planner）的单测 —— 不打网络、不进仿真，只验状态机与契约。

重点验四件事，每一件都对应一种「不会报错但会毁掉成绩」的失败形态：
  1. 触发规则：该问的时候问、不该问的时候不问（问多了烧钱，问少了失去闭环）
  2. 四态决策落地：NEXT/RETRY/REPLAN 各自对 idx / code_epoch / done 的影响
  3. code_epoch：RETRY 时 idx 不变，但必须让 Adapter 重算码
     （只按 idx 缓存的话，RETRY 会拿到过期的码，而且悄无声息）
  4. 护栏：调用上限、重试上限、重规划上限
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
from stage3.online_planner import PlannerError                  # noqa: E402
from stage3.vlm_planner import (OnlineVLMPlanner, _parse_plan,  # noqa: E402
                                _parse_decision, MAX_CALLS, MAX_RETRY,
                                MAX_REPLAN, HEARTBEAT, STALL_LIMIT,
                                MAX_ADVANCE, QUIET_FRAMES)

OK = True
FRAME = {"front": np.zeros((3, 8, 8), np.uint8),
         "wrist": np.zeros((3, 8, 8), np.uint8)}


def check(name, cond, extra=""):
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


class FakeClient:
    """按脚本回放：第一次调用返回计划，之后依次返回给定决策。"""

    def __init__(self, decisions, plan=None):
        self.plan = plan or [{"action": "approach", "instruction": "move toward the red block"},
                             {"action": "grasp", "instruction": "grasp the red block"},
                             {"action": "lift", "instruction": "lift the red block"},
                             {"action": "place", "instruction": "place the red block down"}]
        self.decisions = list(decisions)
        self.n_plan = 0
        self.n_mon = 0

    def chat(self, msgs):
        sysmsg = msgs[0]["content"]
        if "progress monitor" in sysmsg:
            self.n_mon += 1
            d = self.decisions.pop(0) if self.decisions else "CONTINUE"
            return {"content": '{"decision": "%s", "reason": "t"}' % d}
        self.n_plan += 1
        import json
        return {"content": json.dumps({"plan": self.plan})}


def mk(decisions, plan=None, heartbeat=HEARTBEAT):
    p = OnlineVLMPlanner("close_jar", "close the red jar", FakeClient(decisions, plan),
                         prior=["grasp", "lift"], n_repeat=None, heartbeat=heartbeat)
    p.prime(1.0, FRAME)
    return p


def main() -> int:
    print("=== 1. 开局现场规划（不读模板库）===")
    p = mk([])
    check("prime 时生成了计划", len(p.subtasks) == 4, str(len(p.subtasks)))
    check("PLAN 调用了 1 次", p.stats["plan_call"] == 1)
    check("动作带 use_codebook", p.subtasks[1]["use_codebook"] is True)
    check("码占位为 -1（必须由 Adapter 出）", p.subtasks[0]["k_global"] == -1)
    try:
        OnlineVLMPlanner("t", "i", FakeClient([])).prime(1.0, None)
        check("没有画面时 prime 必须报错", False, "居然没抛异常")
    except PlannerError:
        check("没有画面时 prime 必须报错", True)

    print("=== 2. 触发规则 ===")
    p = mk(["NEXT"], heartbeat=99)          # 心跳关掉，只看夹爪
    p.observe(1.0, FRAME)
    check("夹爪没变、未到心跳 -> 不调用", p.stats["monitor_call"] == 0)
    p.observe(0.0, FRAME)                   # 翻转
    check("夹爪翻转 -> 调用一次", p.stats["monitor_call"] == 1)
    check("触发原因记为 flip", p.stats["trigger_flip"] == 1)

    p = mk(["CONTINUE"], heartbeat=3)
    for _ in range(3):
        p.observe(1.0, FRAME)
    check(f"心跳 {3} 帧 -> 调用一次", p.stats["monitor_call"] == 1)
    check("触发原因记为 heartbeat", p.stats["trigger_heartbeat"] == 1)

    print("=== 3. 四态决策 ===")
    p = mk(["NEXT"], heartbeat=1)
    e0 = p.code_epoch
    p.observe(1.0, FRAME)
    check("NEXT -> idx 前进", p.idx == 1)
    check("NEXT -> code_epoch 前进", p.code_epoch == e0 + 1)
    check("NEXT -> 记入 done", len(p.done) == 1)

    # 🔴 最关键：RETRY 时 idx 不变，但码必须重算
    p = mk(["RETRY"], heartbeat=1)
    e0 = p.code_epoch
    p.observe(1.0, FRAME)
    check("RETRY -> idx 不变", p.idx == 0)
    check("RETRY -> code_epoch 仍前进（让 Adapter 重算）", p.code_epoch == e0 + 1)

    p = mk(["CONTINUE"], heartbeat=1)
    e0, i0 = p.code_epoch, p.idx
    p.observe(1.0, FRAME)
    check("CONTINUE -> idx 与 code_epoch 都不变",
          p.idx == i0 and p.code_epoch == e0)

    p = mk(["NEXT", "REPLAN"], heartbeat=1)
    p.observe(1.0, FRAME); p.observe(1.0, FRAME)
    check("REPLAN -> 重新规划", p.stats["plan_call"] == 2)
    check("REPLAN -> idx 归零", p.idx == 0)
    # 🔴 v3.1 起 REPLAN **不再**声称任何段已完成。实测：grasp 失败了
    #    （VLM 自己都说 “lid is on table, not held”），但 idx 已推到 5，
    #    done 于是骗 VLM「前 5 步都做完了」，它只规划出一步 rotate，计划直接废掉。
    #    画面才是真相，REPLAN 时让它从看到的场景重新规划全部剩余动作。
    check("REPLAN -> done 清空（不声称任何段已完成）", p.done == [], str(p.done))

    print("=== 3b. v3.1：RETRY 可以回退 ===")
    class _RetryBack:
        def __init__(self, plan): self.plan = plan
        def chat(self, msgs):
            import json
            if "progress monitor" in msgs[0]["content"]:
                return {"content": '{"decision":"RETRY","index":1,"reason":"grasp failed"}'}
            return {"content": json.dumps({"plan": self.plan})}
    plan5 = [{"action": "approach", "instruction": "move toward the red block"},
             {"action": "grasp", "instruction": "grasp the red block"},
             {"action": "lift", "instruction": "lift the red block"},
             {"action": "transfer", "instruction": "move the red block over the box"},
             {"action": "place", "instruction": "place the red block in the box"}]
    p = OnlineVLMPlanner("t", "i", _RetryBack(plan5), heartbeat=1)
    p.prime(1.0, FRAME)
    p.idx = 3
    p.done = plan5[:3]
    e0 = p.code_epoch
    p.observe(1.0, FRAME)
    check("RETRY 带 index 可退回更早的一步", p.idx == 1, f"idx={p.idx}")
    check("回退时把 done 里对应的段撤掉", len(p.done) == 1, str(len(p.done)))
    check("回退也让 Adapter 重算码", p.code_epoch == e0 + 1)

    class _RetryFwd:
        def __init__(self, plan): self.plan = plan
        def chat(self, msgs):
            import json
            if "progress monitor" in msgs[0]["content"]:
                return {"content": '{"decision":"RETRY","index":4,"reason":"x"}'}
            return {"content": json.dumps({"plan": self.plan})}
    p = OnlineVLMPlanner("t", "i", _RetryFwd(plan5), heartbeat=1)
    p.prime(1.0, FRAME); p.idx = 2
    p.observe(1.0, FRAME)
    check("RETRY 的 index 不得用来往前跳", p.idx == 2, f"idx={p.idx}")

    print("=== 3c. v3.1：末段只在停滞/静止时问 ===")
    class _Cnt:
        def __init__(self, plan): self.plan, self.n = plan, 0
        def chat(self, msgs):
            import json
            if "progress monitor" in msgs[0]["content"]:
                self.n += 1
                return {"content": '{"decision":"CONTINUE","reason":"x"}'}
            return {"content": json.dumps({"plan": self.plan})}
    c = _Cnt(plan5)
    p = OnlineVLMPlanner("t", "i", c, heartbeat=1)
    p.prime(1.0, FRAME); p.idx = len(plan5) - 1      # 已在末段
    for _ in range(3):
        p.observe(1.0, FRAME)                        # 只有心跳，无停滞/静止
    check("末段时心跳不再触发（省调用）", c.n == 0, f"n={c.n}")
    for _ in range(STALL_LIMIT + 1):
        p.observe(1.0, FRAME)
    check("末段时停滞仍然要问（可能需要 RETRY/REPLAN）", c.n >= 1, f"n={c.n}")

    print("=== 4. 护栏 ===")
    p = mk(["RETRY"] * 10, heartbeat=1)
    for _ in range(MAX_RETRY + 2):
        p.observe(1.0, FRAME)
    check(f"重试超过 {MAX_RETRY} 次就放弃该段", p.idx >= 1, f"idx={p.idx}")

    p = mk(["REPLAN"] * 10, heartbeat=1)
    for _ in range(MAX_REPLAN + 3):
        p.observe(1.0, FRAME)
    check(f"重规划不超过 {MAX_REPLAN} 次",
          p.stats["plan_call"] <= MAX_REPLAN + 1, str(p.stats["plan_call"]))

    p = mk(["CONTINUE"] * 40, heartbeat=1)
    for _ in range(MAX_CALLS + 6):
        p.observe(1.0, FRAME)
    total = p.stats["plan_call"] + p.stats["monitor_call"]
    check(f"每局调用不超过 {MAX_CALLS} 次", total <= MAX_CALLS, str(total))
    check("触顶被计数", p.stats["budget_exhausted"] > 0)

    print("=== 4b. v3：停滞 -> 触发 MONITOR 并给出警告（不再本地强制前进）===")
    # 🔴 v2 是「停滞就本地强制前进」，实测把 NEXT 从 42% 推到 72%，
    #    成绩反而从 26.67% 掉到 22.33% —— 盲目往前推不解决问题。
    #    v3 改成照常问 VLM，但把「已经耗了 N 帧」作为**新信息**塞进 prompt，
    #    因为 v1 实测连判 6 次 REPLAN、每次生成几乎相同的计划。
    class _Spy:
        """记下每次 MONITOR 的 user 文本，用来检查警告有没有真的塞进去。"""
        def __init__(self, plan, decision="CONTINUE"):
            self.plan, self.decision = plan, decision
            self.user_texts, self.n_mon, self.n_plan = [], 0, 0
        def chat(self, msgs):
            import json
            if "progress monitor" in msgs[0]["content"]:
                self.n_mon += 1
                self.user_texts.append(msgs[1]["content"][0]["text"])
                return {"content": '{"decision":"%s","reason":"t"}' % self.decision}
            self.n_plan += 1
            return {"content": json.dumps({"plan": self.plan})}

    plan4 = [{"action": "approach", "instruction": "move toward the red block"},
             {"action": "grasp", "instruction": "grasp the red block"},
             {"action": "lift", "instruction": "lift the red block"},
             {"action": "place", "instruction": "place the red block down"}]
    spy = _Spy(plan4)
    p = OnlineVLMPlanner("t", "i", spy, heartbeat=99)   # 心跳关掉，只看停滞
    p.prime(1.0, FRAME)
    for _ in range(STALL_LIMIT + 2):
        p.observe(1.0, FRAME)
    check(f"占用超过 {STALL_LIMIT} 帧 -> 触发 MONITOR", spy.n_mon >= 1, f"n_mon={spy.n_mon}")
    check("触发原因记为 stall", p.stats.get("trigger_stall", 0) >= 1, str(p.stats))
    check("不再本地强制前进（VLM 说 CONTINUE 就不动）", p.idx == 0, f"idx={p.idx}")
    check("PROGRESS WARNING 真的进了 prompt",
          any("PROGRESS WARNING" in t for t in spy.user_texts))
    check("警告里带了已耗帧数",
          any("keyframes on this step" in t for t in spy.user_texts))

    print("=== 4c. v3：关节速度 / 末端位姿静止也触发 ===")
    spy = _Spy(plan4)
    p = OnlineVLMPlanner("t", "i", spy, heartbeat=99)
    p.prime(1.0, FRAME)
    still = {"joint_velocities": np.zeros(7, np.float32),
             "gripper_pose": np.array([0.3, 0.0, 0.4, 0, 0, 0, 1], np.float32)}
    p.observe(1.0, FRAME, robot=still)          # 第 1 帧静止，未达 QUIET_FRAMES
    check(f"静止不足 {QUIET_FRAMES} 帧 -> 不触发", spy.n_mon == 0, f"n_mon={spy.n_mon}")
    p.observe(1.0, FRAME, robot=still)          # 连续第 2 帧
    check(f"连续 {QUIET_FRAMES} 帧静止 -> 触发", spy.n_mon == 1, f"n_mon={spy.n_mon}")
    check("触发原因记为 quiet", p.stats.get("trigger_quiet", 0) == 1, str(p.stats))
    check("警告里说明了关节速度为零",
          any("joint velocities are ~0" in t for t in spy.user_texts))

    moving = {"joint_velocities": np.full(7, 0.5, np.float32),
              "gripper_pose": np.array([0.9, 0.0, 0.4, 0, 0, 0, 1], np.float32)}
    spy2 = _Spy(plan4)
    p2 = OnlineVLMPlanner("t", "i", spy2, heartbeat=99)
    p2.prime(1.0, FRAME)
    for _ in range(4):
        p2.observe(1.0, FRAME, robot=moving)
    check("机械臂在动就不该因静止而触发", p2.stats.get("trigger_quiet", 0) == 0,
          str(p2.stats))

    # 拿不到 robot 状态时要计数，而不是静默当成静止
    spy3 = _Spy(plan4)
    p3 = OnlineVLMPlanner("t", "i", spy3, heartbeat=99)
    p3.prime(1.0, FRAME)
    p3.observe(1.0, FRAME, robot=None)
    check("拿不到 robot 状态 -> 计数且不误判为静止",
          p3.stats["no_robot_state"] == 1 and p3.stats.get("trigger_quiet", 0) == 0)

    print("=== 4d. v3：单次只前进 1 段 ===")
    class _Jump2:
        def __init__(self, plan): self.plan = plan
        def chat(self, msgs):
            import json
            if "progress monitor" in msgs[0]["content"]:
                return {"content": '{"decision":"NEXT","index":3,"reason":"far"}'}
            return {"content": json.dumps({"plan": self.plan})}
    p = OnlineVLMPlanner("t", "i", _Jump2(plan4 + plan4), heartbeat=1)
    p.prime(1.0, FRAME)
    p.observe(1.0, FRAME)
    check(f"即使 VLM 要跳到 3，也只前进 {MAX_ADVANCE} 段",
          p.idx == MAX_ADVANCE, f"idx={p.idx}")

    print("=== 4e. v3：候选指令清单进 prompt ===")
    spy = _Spy(plan4)
    p = OnlineVLMPlanner("t", "i", spy, heartbeat=99,
                         phrasings=["grasp: grasp the red block",
                                    "lift: lift the red block"])
    p.prime(1.0, FRAME)
    # PLAN 的 user 文本没被 _Spy 记下，直接检查构造函数
    from planner.prompts import build_plan_user_content
    txt = build_plan_user_content("t", "i", [("front", "u")],
                                  phrasings=["grasp: grasp the red block"])[0]["text"]
    check("候选清单进了 PLAN 的 prompt", "phrasings the controller was trained on" in txt)
    check("候选内容本身在里面", "grasp the red block" in txt)

    print("=== 5. 解析容错 ===")
    check("计划：裸 JSON", len(_parse_plan('{"plan":[{"action":"grasp","instruction":"grasp it"}]}')) == 1)
    check("计划：代码围栏", len(_parse_plan('```json\n{"plan":[{"action":"lift","instruction":"lift it"}]}\n```')) == 1)
    check("计划：散文包裹", len(_parse_plan('thinking... {"plan":[{"action":"push","instruction":"push it"}]} done')) == 1)
    for txt, want in [('{"decision":"NEXT"}', "NEXT"),
                      ('```\n{"decision":"RETRY","reason":"x"}\n```', "RETRY"),
                      ('I think we should REPLAN here', "REPLAN")]:
        check(f"决策：{txt[:26]!r} -> {want}", _parse_decision(txt)[0] == want)
    check("决策带 index 时能解析出来",
          _parse_decision('{"decision":"NEXT","index":2}')[1] == 2)
    try:
        _parse_decision("nothing useful")
        check("无法解析决策必须 raise", False, "没抛异常")
    except PlannerError:
        check("无法解析决策必须 raise", True)

    print("=== 6. 跨进程可 pickle ===")
    import pickle as pk
    try:
        from stage3.vlm_planner import OnlinePlanFactory
        f = OnlinePlanFactory("val", verbose=False)
        f2 = pk.loads(pk.dumps(f))
        check("OnlinePlanFactory 可 pickle 且不带 client",
              f2._client is None and f2._priors is None)
    except Exception as e:
        check("OnlinePlanFactory 可 pickle", False, repr(e)[:90])

    print()
    print("vlm-plan 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
