#!/usr/bin/env python
"""在线 Planner（P6）的单测 —— 不依赖仿真，只验逻辑与注入契约。

B2/B3 的评测全靠这条链路。它一旦出错的形态很危险：
`act()` 的语言防线只在**字段缺失**时报错，若字段在但内容不对
（比如注入了整任务指令、或码全是同一个值），训练/评测照跑不误，
跑出来的数字说不清是什么。所以这里逐项验注入的**内容**，不只验存在性。

七项断言：
  1. 状态机：夹爪翻转推进
  2. 状态机：预算耗尽推进
  3. 状态机：索引单调不减（长程任务「鬼打墙」的主要来源）
  4. 状态机：走到最后一段就停住，不越界
  5. 注入内容：B3 拿到子任务指令 token + 三个码字段，且**与整任务指令不同**
  6. 注入内容：B2 只拿语言、不拿码
  7. 计划缺失必须 raise（缺了就等于退化成 B1 的行为）
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

from stage3.online_planner import DeterministicPlanner, PlanBank, PlannerError  # noqa: E402
from stage3.rollout import TokenCache, _SubtaskAgent             # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def mk(n: int, kf: int = 1) -> list[dict]:
    return [{"action": f"a{i}", "instruction": f"do step {i}",
             "k_global": i, "k_detail": [i] * 9,
             "use_codebook": i != 1, "n_keyframes": kf} for i in range(n)]


class _Recorder:
    """假 agent：只记下最后一次 act() 收到的 observation。"""

    def __init__(self) -> None:
        self.seen = None

    def act(self, step, observation, deterministic=False):
        self.seen = observation
        return "ACT_RESULT"


def main() -> int:
    print("=== 1-4. 推进状态机 ===")
    # 夹爪翻转推进
    p = DeterministicPlanner(mk(3, kf=10))      # 预算很大，只能靠翻转推进
    p.observe(1.0)                              # 首次只记录基线
    check("初始停在第 0 段", p.idx == 0)
    p.observe(1.0)
    check("夹爪没变则不推进", p.idx == 0)
    p.observe(0.0)                              # 翻转
    check("夹爪翻转 -> 推进到第 1 段", p.idx == 1, f"idx={p.idx}")

    # 预算耗尽推进
    p = DeterministicPlanner(mk(3, kf=1))       # 预算 = max(1, 1*1.0) = 1
    # 🔴 这两条是 20.9% 那个 bug 的回归防线：下限曾经是 2，
    #    把每个单关键帧段强行占住两帧，逐段累积落后。
    check("单关键帧段的预算就是 1（不是 2）", p.budgets == [1, 1, 1], str(p.budgets))
    check("累计边界 = [1,2,3]", p.cum == [1, 2, 3], str(p.cum))
    p.observe(None)
    check("走完第 0 段的预算 -> 推进到第 1 段", p.idx == 1, f"idx={p.idx}")
    p.observe(None)
    check("再走一帧 -> 第 2 段", p.idx == 2, f"idx={p.idx}")

    # 单调不减
    p = DeterministicPlanner(mk(3, kf=1))
    seen = []
    for g in (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0):
        p.observe(g); seen.append(p.idx)
    check("索引单调不减", all(b >= a for a, b in zip(seen, seen[1:])), str(seen))

    # 最后一段不越界
    p = DeterministicPlanner(mk(2, kf=1))
    for _ in range(20):
        p.observe(0.0 if _ % 2 else 1.0)
    check("走到最后一段就停住，不越界", p.idx == 1, f"idx={p.idx}")

    print("=== 5. B3 的注入内容 ===")
    tokens = TokenCache()
    rec = _Recorder()
    pl = DeterministicPlanner(mk(3, kf=5))
    ag = _SubtaskAgent(rec, pl, tokens, with_codes=True)
    task_tok = torch.as_tensor(tokens("close the orange jar")).reshape(1, 1, -1)
    obs = {"lang_goal_tokens": task_tok,
           "low_dim_state": torch.zeros(1, 1, 4)}
    out = ag.act(0, obs)
    check("透传 inner 的返回值", out == "ACT_RESULT")
    got = rec.seen
    for k in ("subtask_lang_goal_tokens", "subtask_k_global",
              "subtask_k_detail", "subtask_code_mask"):
        check(f"注入了 {k}", k in got)
    check("子任务 token 与整任务 token **形状相同**",
          got["subtask_lang_goal_tokens"].shape == task_tok.shape,
          f"{got['subtask_lang_goal_tokens'].shape} vs {task_tok.shape}")
    # 🔴 最关键的一条：内容必须真的不一样。字段在但内容是整任务指令，
    #    是最危险的失败形态 —— 防线不会报错，成绩会静静地退化成 B1。
    check("子任务 token 与整任务 token **内容不同**",
          not torch.equal(got["subtask_lang_goal_tokens"], task_tok))
    exp = tokens("do step 0")
    check("子任务 token 就是第 0 段的指令",
          torch.equal(got["subtask_lang_goal_tokens"].reshape(-1),
                      torch.as_tensor(exp)))
    check("k_global 取自当前段", int(got["subtask_k_global"].reshape(-1)[0]) == 0)
    check("k_detail 宽度为 9", got["subtask_k_detail"].reshape(-1).numel() == 9)
    check("code_mask 反映 use_codebook",
          float(got["subtask_code_mask"].reshape(-1)[0]) == 1.0)

    # 推进后注入的内容要跟着变
    obs["low_dim_state"] = torch.ones(1, 1, 4)      # 夹爪翻转
    ag.act(1, obs)
    check("推进后注入的指令随之改变",
          not torch.equal(rec.seen["subtask_lang_goal_tokens"],
                          torch.as_tensor(exp).reshape(1, 1, -1)))
    check("推进后 k_global 随之改变",
          int(rec.seen["subtask_k_global"].reshape(-1)[0]) == 1)
    check("第 1 段 use_codebook=False -> code_mask=0",
          float(rec.seen["subtask_code_mask"].reshape(-1)[0]) == 0.0)

    print("=== 6. B2 只拿语言、不拿码 ===")
    rec2 = _Recorder()
    ag2 = _SubtaskAgent(rec2, DeterministicPlanner(mk(2, kf=5)), tokens,
                        with_codes=False)
    ag2.act(0, {"lang_goal_tokens": task_tok, "low_dim_state": torch.zeros(1, 1, 4)})
    check("B2 注入了子任务指令", "subtask_lang_goal_tokens" in rec2.seen)
    check("B2 **不**注入码字段",
          not any(k in rec2.seen for k in
                  ("subtask_k_global", "subtask_k_detail", "subtask_code_mask")))

    print("=== 7. 计划缺失必须 raise ===")
    from stage3.planners import plan_file, make_factory, KINDS
    pf = plan_file("val", "template")
    if pf.is_file():
        bank = PlanBank(pf)
        print(f"       {bank.summary()}")
        check("已有计划能取到", len(bank.get("close_jar", 0)) > 0)
        try:
            bank.get("close_jar", 99999)
            check("缺计划 -> raise", False, "没有抛异常")
        except PlannerError as e:
            check("缺计划 -> raise", True)
            print(f"           ↳ {str(e).splitlines()[0][:80]}")
    else:
        check("plans_val_template.json 存在", False,
              "先跑 stage3_eval.py plans --split val")

    print("=== 8. planner 可切换（P6 的可回撤保证）===")
    # 断言的是**全集**而非「至少包含」：新增一种 planner 却忘了在评测侧接上，
    # 这条会立刻红，比事后发现某个 --planner 值被静默忽略要早得多。
    # 2026-09-06 新增 vlm-plan（现场规划版）后这里从四种变五种。
    check("登记了 template/flat/oracle/vlm/vlm-plan 五种",
          set(KINDS) == {"template", "flat", "oracle", "vlm", "vlm-plan"},
          str(KINDS))
    try:
        vf = make_factory("vlm", "val", verbose=False)
        check("vlm 工厂可构造", vf.kind == "vlm")
        check("vlm 复用模板库的子任务序列（B2/B3 候选集合相同）",
              len(vf.bank.get("close_jar", 0)) > 0)
        import pickle as _pk0
        vf2 = _pk0.loads(_pk0.dumps(vf))
        check("vlm 工厂可 pickle（client/bank 不随行）",
              vf2._client is None and vf2._bank is None)
    except Exception as e:
        check("vlm 工厂可构造", False, repr(e)[:100])

    print("=== 8b. VLM planner 的解析与退化 ===")
    from stage3.vlm_planner import VLMPlanner, _parse, MAX_ADVANCE
    for txt, want in [('{"index": 3, "reason": "lid grasped"}', 3),
                      ('```json\n{"index": 2}\n```', 2),
                      ('the answer is {"index": 5} ok', 5),
                      ('7', 7)]:
        try:
            got, _ = _parse(txt)
            check(f"解析 {txt[:28]!r} -> {want}", got == want, f"得到 {got}")
        except Exception as e:
            check(f"解析 {txt[:28]!r}", False, repr(e)[:60])
    try:
        _parse("no number here at all")
        check("无法解析时必须 raise", False, "没抛异常")
    except PlannerError:
        check("无法解析时必须 raise", True)

    class _Boom:
        def chat(self, msgs):
            raise RuntimeError("模拟限流")
    # 🔴 VLM 挂掉时必须退回模板法，而不是让整局评测崩掉 ——
    #    否则「VLM 方案效果如何」会变成「VLM 方案能不能跑完」。
    # 🔴 失败必须**直接抛错**，不做静默退化：退回模板法会让「VLM 方案」的
    #    成绩单里混进模板法产生的步骤，数字说不清是什么。
    vp = VLMPlanner(mk(4, kf=1), "close_jar", "close the jar", _Boom())
    vp.prime(1.0)
    try:
        vp.observe(1.0, {"front": np.zeros((3, 8, 8), np.uint8)})
        check("VLM 调用失败 -> 抛错", False, "居然没抛异常")
    except PlannerError:
        check("VLM 调用失败 -> 抛错", True)
    check("失败被计数", vp.stats["fail"] == 1, str(vp.stats))
    check("索引没有被偷偷推进", vp.idx == 0, f"idx={vp.idx}")
    vp2 = VLMPlanner(mk(4, kf=1), "t", "i", _Boom())
    vp2.prime(1.0)
    try:
        vp2.observe(1.0, None)
        check("拿不到画面 -> 抛错", False, "居然没抛异常")
    except PlannerError:
        check("拿不到画面 -> 抛错", True)

    class _Jump:
        def chat(self, msgs):
            return {"content": '{"index": 99}'}
    vp3 = VLMPlanner(mk(8, kf=1), "t", "i", _Jump())
    vp3.prime(1.0)
    vp3.observe(1.0, {"front": np.zeros((3, 8, 8), np.uint8)})
    check(f"单次最多前进 {MAX_ADVANCE} 步（防乱跳）", vp3.idx == MAX_ADVANCE,
          f"idx={vp3.idx}")
    check("越界被计数", vp3.stats["clamped"] == 1, str(vp3.stats))

    class _Back:
        def chat(self, msgs):
            return {"content": '{"index": 0}'}
    vp4 = VLMPlanner(mk(4, kf=1), "t", "i", _Back())
    vp4.prime(1.0); vp4.idx = 2
    vp4.observe(1.0, {"front": np.zeros((3, 8, 8), np.uint8)})
    check("索引不得回退", vp4.idx == 2, f"idx={vp4.idx}")
    if plan_file("val", "template").is_file():
        f = make_factory("template", "val", verbose=False)
        pl1, pl2 = f("close_jar", 0), f("close_jar", 0)
        check("工厂每次给出**独立**实例（局内状态不能串）", pl1 is not pl2)
        pl1.observe(1.0); pl1.observe(0.0)
        check("推进一个不影响另一个", pl2.idx == 0, f"pl2.idx={pl2.idx}")
    ff = plan_file("val", "flat")
    if ff.is_file():
        ft = make_factory("flat", "val", verbose=False)
        tp = make_factory("template", "val", verbose=False)
        a_, b_ = ft("close_jar", 0), tp("close_jar", 0)
        check("flat 与 template 的**分段数相同**",
              len(a_.subtasks) == len(b_.subtasks))
        check("flat 与 template 的**码逐段相同**",
              all(x["k_global"] == y["k_global"] and x["k_detail"] == y["k_detail"]
                  for x, y in zip(a_.subtasks, b_.subtasks)))
        check("flat 的各段指令**彼此相同**（都是整任务指令）",
              len({x["instruction"] for x in a_.subtasks}) == 1,
              str({x["instruction"] for x in a_.subtasks})[:80])
        check("flat 的指令与 template 第 0 段**不同**",
              a_.subtasks[0]["instruction"] != b_.subtasks[0]["instruction"])
        print(f"           ↳ flat=“{a_.subtasks[0]['instruction']}”  "
              f"template=“{b_.subtasks[0]['instruction']}”")
    else:
        print("  [skip] flat 计划尚未生成")

    print("=== 9. 跨进程可 pickle（eval.py 用 spawn 起子进程）===")
    # 🔴 回归测试。第一次跑真实 rollout 就死在这里：
    #    AttributeError: Can't pickle local object 'make_factory.<locals>.factory'
    #    replay_dataset._seal 踩过同一个坑（那次是 DataLoader 的 spawn worker）。
    import pickle as _pk
    from stage3.rollout import Stage3RolloutGenerator
    if plan_file("val", "template").is_file():
        f = make_factory("template", "val", verbose=False)
        try:
            f2 = _pk.loads(_pk.dumps(f))
            check("工厂可 pickle", True)
            pl = f2("close_jar", 0)
            check("反序列化后仍能出计划", len(pl.subtasks) > 0)
        except Exception as e:
            check("工厂可 pickle", False, repr(e)[:90])
        try:
            rg = Stage3RolloutGenerator(f, with_codes=True, verbose=False)
            rg2 = _pk.loads(_pk.dumps(rg))
            check("Stage3RolloutGenerator 可 pickle", rg2._with_codes is True)
        except Exception as e:
            check("Stage3RolloutGenerator 可 pickle", False, repr(e)[:90])

    print("=== 10. 分段一致率回归（20.9% 那个 bug 的防线）===")
    # 拿 train 的真值分段离线比对：状态机走一遍 demo 的关键帧夹爪序列，
    # 看它给出的段号和 cache 的 keypoint_to_segment 对不对得上。
    # 这一条是最重要的回归测试 —— 分段错位不会让任何断言失败，
    # 只会让评测成绩静静地垮掉（实测 B3 10.0% vs 应有的水平）。
    try:
        from stage3.cache_join import PlannerCache
        from stage3.online_planner import _episode_plan
        cache = PlannerCache(REPO_ROOT / "aavla_data" / "planner_cache" / "train")
        tot = ag = 0
        for task in ("close_jar", "open_drawer", "stack_blocks", "place_cups"):
            for e in range(20):
                ep = cache.get(task, "train", e)
                if ep is None:
                    continue
                truth, grips = ep["keypoint_to_segment"], ep["gripper_at_keypoints"]
                if not truth or len(grips) != len(truth):
                    continue
                pl = DeterministicPlanner(_episode_plan(ep))
                pl.prime(1.0)
                pred = [pl.idx]
                for t in range(1, len(grips)):
                    pl.observe(float(grips[t - 1]))
                    pred.append(pl.idx)
                for a, b in zip(pred, truth):
                    tot += 1
                    ag += (a == b)
        rate = ag / max(tot, 1)
        check(f"逐关键帧分段一致率 ≥ 85%（实测 {rate:.1%}，{tot} 帧）", rate >= 0.85)
    except Exception as e:
        check("分段一致率回归", False, repr(e)[:100])

    print()
    print("在线 Planner 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
