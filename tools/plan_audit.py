#!/usr/bin/env python
"""在线 PLAN 的离线预检 —— 不烧一张 GPU，先判断「VLM 现场规划的计划靠不靠谱」。

# 为什么先做这个

真·在线 planner（`vlm-plan`）的核心假设是：**VLM 只看初始那一帧，就能推理出
一条接近真值的子任务序列**。这个假设不成立的话，后面的 rollout 全都白跑 ——
而验证它根本不需要仿真：拿 train episode 的初始帧喂 VLM.PLAN，
和离线 cache 里 VLM **看完整条 demo 之后**做的后验分段比对即可。

后验分段是这里的参照系，不是绝对真理，但它是同一个模型在信息更充分时的输出，
所以「在线单帧 vs 离线全 demo」的差距，就是在线规划要付出的信息代价。

# 三项指标

    动作序列完全一致率   最严格：逐位相同
    动作集合 F1          宽松：动作对了、顺序/数量略有出入也算部分正确
    段数差               在线规划倾向于多切还是少切
    指令合法率           落在 PerAct/Adapter 训练分布内的比例（<100% 就是硬伤）

# 成本

每个 episode 一次调用。默认 10 局 × 12 任务 = 120 次，约 10 分钟、20 元上下。
`PlannerClient` 按 sha256(model + messages) 记忆化，重跑不再付费。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 任务名单的真源在 stage3/tasks.py。--tasks 默认审全部 Seen18。
from stage3.tasks import SEEN18                                       # noqa: E402
VIEWS = ("front", "wrist")


def parse_plan(text: str) -> list[dict]:
    """容错解析。

    实测 qwen 在 PLAN 这种开放式任务上会先写一大段散文推理再给答案
    （MONITOR 那种「选一个下标」的封闭问题则不会）。所以除了直接 json.loads，
    还要能从散文里把最后一个完整的 JSON 对象抠出来。
    """
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        d = json.loads(s)
    except Exception:
        # 从右往左找第一个能解析成功的 {...}
        d = None
        for m in reversed(list(re.finditer(r"\{", s))):
            try:
                d = json.loads(s[m.start():
                                 s.rfind("}") + 1])
                break
            except Exception:
                continue
        if d is None:
            raise ValueError(f"输出里找不到 JSON: {s[:120]!r}")
    plan = d["plan"] if isinstance(d, dict) else d
    out = []
    for st in plan:
        out.append({"action": str(st["action"]).strip(),
                    "instruction": str(st["instruction"]).strip()})
    return out


def f1(pred: list[str], truth: list[str]) -> float:
    """动作**多重集**的 F1 —— 顺序不计，但数量计。"""
    cp, ct = collections.Counter(pred), collections.Counter(truth)
    inter = sum((cp & ct).values())
    if not inter:
        return 0.0
    p, r = inter / max(sum(cp.values()), 1), inter / max(sum(ct.values()), 1)
    return 2 * p * r / (p + r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10, help="每任务多少局")
    ap.add_argument("--tasks", nargs="+", default=SEEN18)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "result" / "p7" / "plan_audit.json"))
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    from planner.client import PlannerClient, image_data_url_from_array
    from planner.prompts import build_plan_system_prompt, build_plan_user_content
    from planner.contract import (check_instruction, CODEBOOK_ACTIONS,
                                  NON_CODEBOOK_ACTIONS, ACTION_VERBS,
                                  normalize_instruction, expand_prior,
                                  n_repeat_prior)
    from planner.offline import load_priors
    from stage3.cache_join import PlannerCache

    allowed = set(CODEBOOK_ACTIONS) | set(NON_CODEBOOK_ACTIONS)
    priors = load_priors()
    cache = PlannerCache(REPO_ROOT / "aavla_data" / "planner_cache" / a.split)
    root = REPO_ROOT / "aavla_data" / "rlbench" / a.split
    # response_format=json_object 让服务端强制 JSON；max_tokens 给足，
    # 免得模型先写推理再被截断（实测 882 token 全是散文，答案还没开始）。
    client = PlannerClient(model=a.model or "qwen3.8-max", max_tokens=2048,
                           extra_body={"enable_thinking": False,
                                       "response_format": {"type": "json_object"}})
    sysmsg = build_plan_system_prompt()

    rows, per_task = [], collections.defaultdict(list)
    t0 = time.time()
    for task in a.tasks:
        for e in range(a.episodes):
            ep = cache.get(task, a.split, e)
            if ep is None or not ep.get("segments"):
                continue
            ep_dir = root / task / "all_variations" / "episodes" / f"episode{e}"
            imgs = []
            for v in VIEWS:
                f = ep_dir / f"{v}_rgb" / "0.png"          # 初始帧
                if not f.is_file():
                    imgs = []
                    break
                imgs.append((v, image_data_url_from_array(
                    np.array(Image.open(f).convert("RGB")))))
            if not imgs:
                continue
            truth = [s["action"] for s in ep["segments"]]
            var = int(ep["variation"])
            pr = expand_prior(task, var, priors.get(task, []))
            nrep = n_repeat_prior(task, var)
            msgs = [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": build_plan_user_content(
                        task, ep["task_instruction"], imgs,
                        prior=pr, n_repeat=nrep)}]
            try:
                plan = parse_plan(client.chat(msgs)["content"])
            except Exception as exc:
                rows.append({"task": task, "episode": e,
                             "error": f"{type(exc).__name__}: {exc}"[:150]})
                print(f"  [ERR] {task}/{e}: {type(exc).__name__}: {exc}"[:110])
                continue
            pred = [s["action"] for s in plan]
            bad_act = [x for x in pred if x not in allowed]
            bad_ins, bad_verb = [], []
            for s in plan:
                if check_instruction(s["instruction"], ep["task_instruction"],
                                     len(plan)):
                    bad_ins.append(s["instruction"])
                verbs = ACTION_VERBS.get(s["action"], (s["action"],))
                if not any(normalize_instruction(s["instruction"]).startswith(v)
                           for v in verbs):
                    bad_verb.append(f"{s['action']}:{s['instruction']}")
            r = {"task": task, "episode": e,
                 "pred": pred, "truth": truth,
                 "exact": pred == truth, "f1": round(f1(pred, truth), 3),
                 "n_pred": len(pred), "n_truth": len(truth),
                 "bad_action": bad_act, "bad_instruction": bad_ins,
                 "bad_verb": bad_verb,
                 "plan": plan}
            rows.append(r)
            per_task[task].append(r)
            print(f"  {task}/{e}  {'✅' if r['exact'] else '  '} "
                  f"F1={r['f1']:.2f}  pred={len(pred)} truth={len(truth)}"
                  + ("  ⚠️非法动作" if bad_act else "")
                  + ("  ⚠️指令违规" if bad_ins else "")
                  + ("  ⚠️动词错" if bad_verb else ""), flush=True)

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("❌ 一条都没跑成", file=sys.stderr)
        return 2
    n = len(ok)
    n_steps = sum(r["n_pred"] for r in ok)
    summary = {
        "n_episodes": n, "n_failed_calls": len(rows) - n,
        "exact_seq_rate": sum(r["exact"] for r in ok) / n,
        "mean_f1": sum(r["f1"] for r in ok) / n,
        "mean_len_pred": sum(r["n_pred"] for r in ok) / n,
        "mean_len_truth": sum(r["n_truth"] for r in ok) / n,
        "illegal_action_rate": sum(len(r["bad_action"]) for r in ok) / max(n_steps, 1),
        "illegal_instruction_rate": sum(len(r["bad_instruction"]) for r in ok) / max(n_steps, 1),
        "wrong_verb_rate": sum(len(r["bad_verb"]) for r in ok) / max(n_steps, 1),
        "minutes": round((time.time() - t0) / 60, 1),
        "client_stats": client.stats,
    }
    print()
    print("=" * 62)
    print(f"  episode 数            {n}（调用失败 {summary['n_failed_calls']}）")
    print(f"  动作序列完全一致率     {summary['exact_seq_rate']:.1%}")
    print(f"  动作多重集 F1（均值）  {summary['mean_f1']:.3f}")
    print(f"  段数  在线 {summary['mean_len_pred']:.1f}  vs  离线真值 {summary['mean_len_truth']:.1f}")
    print(f"  非法动作率             {summary['illegal_action_rate']:.1%}")
    print(f"  指令违规率             {summary['illegal_instruction_rate']:.1%}")
    print(f"  起始动词错误率         {summary['wrong_verb_rate']:.1%}")
    print(f"  耗时 {summary['minutes']} min   client={summary['client_stats']}")
    print("=" * 62)
    print(f"\n{'任务':<32}{'一致率':>8}{'F1':>7}{'段数 在线/真值':>16}")
    for t in a.tasks:
        rs = per_task.get(t, [])
        if not rs:
            continue
        print(f"{t:<32}{sum(r['exact'] for r in rs)/len(rs):>8.0%}"
              f"{sum(r['f1'] for r in rs)/len(rs):>7.2f}"
              f"{sum(r['n_pred'] for r in rs)/len(rs):>9.1f}"
              f" /{sum(r['n_truth'] for r in rs)/len(rs):>5.1f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1))
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"\n明细 -> {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
