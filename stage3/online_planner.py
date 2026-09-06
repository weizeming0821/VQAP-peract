"""在线 Planner（P6）—— 评测时给 B2/B3 提供子任务指令与码。

# 为什么需要

`act()` 的语言输入只有 `observation['lang_goal_tokens']`，而它由
`yarr/envs/rlbench_env.py` 从 `self._lang_goal` 生成，那**永远是整任务指令**。
B2/B3 是用子任务指令训练的，评测时必须实时拿到：

    subtask_lang_goal_tokens        子任务指令的 CLIP token
    subtask_k_global / k_detail     码索引
    subtask_code_mask               该段是否启用码注入

拿不到就 `raise`（BUG-1 与 P0-3 的硬防线），所以**没有本模块，B2/B3 一局都评不了**。

# 为什么是确定性的，而不是每步调 VLM

离线建 cache 用 VLM 是必要的（要看完整 demo 做后验分段）。但**评测时不需要**：

* B2 与 B3 用同一个 planner，planner 的能力在 B3−B2 的差值里本来就抵消。
  引入 VLM 的随机性只会让两臂看到**不同的**子任务序列，把干净的对照弄脏。
* 闭环 VLM 实测代价：约 9000 次调用、1550 元，且每次调用串在 rollout 里
  无法并行、无法预生成（两臂的轨迹不同，连缓存都不能共用），
  每局要多花 3–10 分钟。
* 确定性方案可**预生成**：同一份计划喂给 B2 和 B3，两臂看到的子任务序列
  逐字相同 —— 这比文档原方案更强的公平性保证。

# 计划从哪来

`(task, variation)` → 模板。模板取自 **train split 的离线 cache**：
同一 variation 的多条 train episode 里，取**众数动作序列**对应的代表 episode，
把它的 segments（指令 / k_global / k_detail / use_codebook）整段搬过来。

按 variation 匹配是关键：指令里带着 variation 特有的物体与颜色
（"grasp the grey jar lid" / "place the lid on the azure jar"），
同 variation 的 val episode 场景构成相同，指令可以直接沿用。

# 推进规则（`DeterministicPlanner`）

PerAct 逐关键帧执行，所以推进按**关键帧**计，不按控制步：

    夹爪状态翻转      最可靠的信号，标志抓取/释放完成
    关键帧预算耗尽    每段的预算 = 该段在 train 里的关键帧数中位数 × 富余系数
    子任务索引单调不减  杜绝「鬼打墙」式回退
    最后一段兜底      走完就停在最后一段，不越界

VLM 闭环版本留作后续可选消融（`Exp_Design` 里作为 planner 消融维度），
主线不依赖它。
"""

from __future__ import annotations

import collections
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
RLBENCH_ROOT = REPO_ROOT / "aavla_data" / "rlbench"

#: 每段关键帧预算的富余系数。
#: 🔴 曾经是 2.0，实测是错的：离线拿 train 的真值分段对比，逐关键帧一致率只有
#:    20.9%，且 79% 的帧「落后」真值 1–2 段 —— 模型看着「已经放好盖子」的画面，
#:    收到的却是「抬起盖子」。B3@2500 因此只有 10.0%（B1@2500 是 37.3%）。
#:    改成 1.0 后一致率升到 92.5%（用模板；用本局分段是 97.5%）。
BUDGET_SLACK = 1.0
#: 预算下限。
#: 🔴 曾经是 2，这才是真正的元凶：多数段的关键帧中位数就是 1，
#:    下限 2 会把每个单帧段**强行占住两帧**，误差逐段累积。
#:    原注释说「避免过早切段」，方向恰好搞反了 —— 一个关键帧的段，
#:    预算就该是 1。单独把 slack 调回 1.0 而不动这里，一致率仍然是 21.0%。
MIN_BUDGET = 1


class PlannerError(RuntimeError):
    pass


# ---------------------------------------------------------------- 数据结构

def _subtask(seg: dict, n_keyframes: int) -> dict:
    """从 cache 的一个 segment 抽出在线 planner 需要的字段。"""
    return {
        "action": seg["action"],
        "instruction": seg["instruction"],
        "k_global": int(seg["k_global"]),
        "k_detail": [int(x) for x in seg["k_detail"]],
        "use_codebook": bool(seg["use_codebook"]),
        "n_keyframes": int(n_keyframes),
    }


def _episode_plan(ep: dict) -> list[dict]:
    """把一条 cache episode 转成子任务序列（含每段的关键帧数）。"""
    per_seg = collections.Counter(ep["keypoint_to_segment"])
    return [_subtask(s, per_seg.get(s["segment_index"], 1)) for s in ep["segments"]]


# ---------------------------------------------------------------- 模板库

def build_template_bank(cache, tasks: Iterable[str], split: str = "train",
                        max_episodes: int = 100) -> dict:
    """`(task, variation)` → 代表性子任务序列。

    同一 variation 通常有多条 train episode（实测每 variation 2–50 条）。
    取**众数动作序列**——即出现次数最多的那个 `(action, ...)` 元组——
    再从中挑一条代表 episode；每段的关键帧预算取该组内的中位数。

    众数而非任取一条，是因为个别 episode 的分段可能是 VLM 的异常输出；
    多数投票能把它们滤掉。
    """
    bank: dict[str, dict] = {}
    stats = {"tasks": {}, "n_entries": 0}
    for task in tasks:
        by_var: dict[int, list[dict]] = collections.defaultdict(list)
        for e in range(max_episodes):
            ep = cache.get(task, split, e)
            if ep is not None:
                by_var[int(ep["variation"])].append(ep)
        per_task = {}
        for var, eps in by_var.items():
            plans = [_episode_plan(ep) for ep in eps]
            sigs = [tuple(s["action"] for s in p) for p in plans]
            modal_sig, n_modal = collections.Counter(sigs).most_common(1)[0]
            group = [p for p, s in zip(plans, sigs) if s == modal_sig]
            rep = group[len(group) // 2]                    # 组内取中间那条作代表
            # 每段预算用组内中位数，比单条更稳
            for i, st in enumerate(rep):
                kfs = sorted(p[i]["n_keyframes"] for p in group)
                st["n_keyframes"] = kfs[len(kfs) // 2]
            per_task[str(var)] = {
                "subtasks": rep,
                "n_source_episodes": len(eps),
                "n_modal": n_modal,
                "modal_share": round(n_modal / len(eps), 3),
                "actions": list(modal_sig),
            }
            stats["n_entries"] += 1
        bank[task] = per_task
        shares = [v["modal_share"] for v in per_task.values()]
        stats["tasks"][task] = {
            "n_variations": len(per_task),
            "mean_modal_share": round(sum(shares) / max(len(shares), 1), 3),
        }
    return {"bank": bank, "stats": stats, "split": split}


# ---------------------------------------------------------- val/test 变体号

def episode_variation(task: str, split: str, ep_idx: int) -> int | None:
    """读某个 episode 的 variation 号（RLBench 把它存成 pickle）。"""
    p = (RLBENCH_ROOT / split / task / "all_variations" / "episodes" /
         f"episode{ep_idx}" / "variation_number.pkl")
    if not p.is_file():
        return None
    with p.open("rb") as f:
        return int(pickle.load(f))


def episode_descriptions(task: str, split: str, ep_idx: int) -> list[str]:
    p = (RLBENCH_ROOT / split / task / "all_variations" / "episodes" /
         f"episode{ep_idx}" / "variation_descriptions.pkl")
    if not p.is_file():
        return []
    with p.open("rb") as f:
        return list(pickle.load(f))


# ---------------------------------------------------------------- 计划预生成

def build_plans(bank_doc: dict, tasks: Iterable[str], split: str,
                n_episodes: int) -> dict:
    """给 `split` 的每个 episode 生成子任务计划。

    🔴 预生成而非在线现算，是为了让 **B2 与 B3 读到逐字相同的计划** ——
    两臂的差值里只剩「有没有注入码」这一件事。

    variation 在模板库里找不到时，退回该任务的**任务级众数模板**并打上
    `fallback=True`。指令里的物体/颜色会与实际场景不符，所以这个比例必须
    统计出来并如实报告；若比例高，说明模板方案不成立，要改用别的路子。
    """
    bank = bank_doc["bank"]
    plans: dict[str, dict] = {}
    cover = {"total": 0, "exact": 0, "fallback": 0, "missing": 0,
             "per_task": {}}
    for task in tasks:
        per_var = bank.get(task, {})
        # 任务级兜底模板：所有 variation 里出现最多的那个动作序列
        fb = None
        if per_var:
            sig_count = collections.Counter(
                tuple(v["actions"]) for v in per_var.values())
            if sig_count:
                top = sig_count.most_common(1)[0][0]
                fb = next(v for v in per_var.values() if tuple(v["actions"]) == top)
        t_stat = {"exact": 0, "fallback": 0, "missing": 0}
        for e in range(n_episodes):
            var = episode_variation(task, split, e)
            cover["total"] += 1
            if var is None:
                cover["missing"] += 1; t_stat["missing"] += 1
                continue
            entry = per_var.get(str(var))
            fallback = entry is None
            if fallback:
                entry = fb
            if entry is None:
                cover["missing"] += 1; t_stat["missing"] += 1
                continue
            plans[f"{task}/{e}"] = {
                "task": task, "episode": e, "variation": var,
                "fallback": fallback,
                "subtasks": entry["subtasks"],
            }
            if fallback:
                cover["fallback"] += 1; t_stat["fallback"] += 1
            else:
                cover["exact"] += 1; t_stat["exact"] += 1
        cover["per_task"][task] = t_stat
    return {"plans": plans, "coverage": cover, "split": split,
            "bank_split": bank_doc["split"]}


# ---------------------------------------------------------------- oracle

def build_oracle_plans(cache, tasks, split: str, n_episodes: int) -> dict:
    """用每一局**自己的真值分段**做计划 —— 任何 planner 的上界。

    真值来自离线 cache 的 `keypoint_to_segment`：VLM 看完整条 demo 之后做的
    后验分段，也**正是训练时每个 replay 样本用的标签**。所以 oracle 组的
    (画面, 子任务指令) 配对与训练时逐字相同，分段误差为零。

    ⚠️ 只有 train split 有 cache，所以 oracle 只能在 train split 上做。
       train 的 demo 模型训练时见过，三组的绝对成功率都会偏高，
       **不能与 val 的数字横向比**；要看的是三组之间的相对差。

    ⚠️ 它是**乐观上界**而非理论极限：评测时模型走自己的轨迹，一旦偏离 demo，
       按关键帧序号索引的「真值」也会失准。
    """
    plans, cover = {}, {"total": 0, "exact": 0, "fallback": 0, "missing": 0,
                        "per_task": {}}
    for task in tasks:
        t_stat = {"exact": 0, "fallback": 0, "missing": 0}
        for e in range(n_episodes):
            cover["total"] += 1
            ep = cache.get(task, split, e)
            if ep is None or not ep.get("segments"):
                cover["missing"] += 1; t_stat["missing"] += 1
                continue
            plans[f"{task}/{e}"] = {
                "task": task, "episode": e,
                "variation": int(ep["variation"]), "fallback": False,
                "subtasks": _episode_plan(ep),
            }
            cover["exact"] += 1; t_stat["exact"] += 1
        cover["per_task"][task] = t_stat
    return {"plans": plans, "coverage": cover, "split": split,
            "bank_split": split, "planner": "oracle"}


# ---------------------------------------------------------------- flat 对照

def flatten_plans(doc: dict, split: str) -> dict:
    """把计划里的子任务指令**全部换成整任务指令**，其余一律不动。

    这是判断「模板法到底有没有用」的对照组：分段数、每段的
    k_global / k_detail / use_codebook / n_keyframes 与 template 版逐字相同，
    唯一的差别是语言。于是

        B3(template) − B3(flat)  =  子任务分解本身值多少分

    若这个差 ≈ 0，说明模板法没起作用，该考虑换在线 VLM planner；
    若为正，说明模板法在干活，B3 分数低就不是 planner 的锅。
    没有这个对照，「该不该回撤」只能靠猜。

    整任务指令取 `variation_descriptions.pkl[0]` —— 与
    `yarr/envs/rlbench_env.py:153` 里 `self._lang_goal = descriptions[0]`
    完全同源，保证 flat 局看到的语言和 B0/B1 看到的一模一样。
    """
    out = {}
    n_missing = 0
    for key, plan in doc["plans"].items():
        descs = episode_descriptions(plan["task"], split, plan["episode"])
        if not descs:
            n_missing += 1
            continue
        goal = descs[0]
        subs = [dict(s, instruction=goal) for s in plan["subtasks"]]
        out[key] = dict(plan, subtasks=subs)
    cov = dict(doc["coverage"])
    cov["flat_missing_descriptions"] = n_missing
    return {"plans": out, "coverage": cov, "split": doc["split"],
            "bank_split": doc["bank_split"], "planner": "flat"}


# ---------------------------------------------------------------- 状态机

class DeterministicPlanner:
    """按关键帧推进子任务序列的确定性状态机。

    每个 rollout 一个实例。`observe()` 在**每个关键帧动作执行之后**调用一次，
    由它决定下一步喂哪个子任务。
    """

    def __init__(self, subtasks: list[dict], slack: float = BUDGET_SLACK) -> None:
        if not subtasks:
            raise PlannerError("子任务序列为空")
        self.subtasks = subtasks
        self.budgets = [max(MIN_BUDGET, int(round(s["n_keyframes"] * slack)))
                        for s in subtasks]
        # 累计边界：段 i 覆盖关键帧 [cum[i-1], cum[i])。
        # 用累计位置定位、而不是「每帧最多推进一段」的增量计数，
        # 是因为增量式追不上真值的跳段：模板说某段占 3 帧、实际只占 1 帧时，
        # 增量式会永久落后，且误差逐段累积（实测 86.1% vs 累计式 92.5%）。
        self.cum: list[int] = []
        acc = 0
        for b in self.budgets:
            acc += b
            self.cum.append(acc)
        self.idx = 0
        self.t = 0                     # 已执行的关键帧数（= act() 调用次数 - 1）
        self.used = 0                  # 当前段已用掉的关键帧数（仅供诊断）
        self.prev_gripper: float | None = None
        self.history: list[dict] = []  # 供事后分析：每步用了哪个子任务

    @property
    def current(self) -> dict:
        return self.subtasks[self.idx]

    @property
    def done(self) -> bool:
        return (self.idx >= len(self.subtasks) - 1
                and self.t >= self.cum[-1])

    def prime(self, gripper_open: float | None, frame=None) -> None:
        """建立夹爪基线，但**不推进**。

        `frame` 模板法用不到，留在签名里是为了和 vlm-plan 共用协议
        （那边要在第一帧现场规划）。

        rollout 的第一次 `act()` 面对的是初始状态，没有「上一个动作」可评判，
        所以不能推进。但必须把夹爪基线记下来 —— 否则下一次 observe() 时
        `prev_gripper` 还是 None，**第一次真正的翻转会被漏掉**，
        白白浪费一次推进机会（长程任务里这会让后面每一段都错位）。
        """
        if gripper_open is not None:
            self.prev_gripper = float(gripper_open)

    def note(self, step: int) -> None:
        """记一笔轨迹。

        `step` 是 YARR 的 step_signal —— 评测时它恒为 -1，没有信息量，
        所以记的是 `act()` 在本局内的**调用序号**（= 已执行的关键帧数）。
        夹爪值一并记下：推进逻辑全靠它，出问题时要能回看。
        """
        self.history.append({"t": len(self.history), "subtask_index": self.idx,
                             "action": self.current["action"],
                             "gripper": self.prev_gripper,
                             "instruction": self.current["instruction"]})

    def observe(self, gripper_open: float | None, frame=None, robot=None) -> None:
        """执行完一个关键帧后推进状态。

        `frame` / `robot` 模板法都用不到（它是开环的，只看夹爪与预算），
        留在签名里是为了让所有 planner 共用一个协议。

        `frame` 是当前观测的相机图像，模板法用不到（开环，只看夹爪与预算），
        留在签名里是为了让所有 planner 共用一个协议 —— 在线 VLM planner
        要用它。见 `stage3/vlm_planner.py`。

        定位规则 —— 取下面两者的较大者，且**单调不减**：

          1. **累计位置**：已执行 t 个关键帧，落在哪一段的 [cum[i-1], cum[i]) 区间里。
             这是开环的「按计划走」。
          2. **夹爪翻转**：抓取或释放完成的可靠信号，强制至少推进到下一段。
             这是闭环的「按反馈走」—— 模型比 demo 慢时靠它重新对齐。

        只增不减：长程任务里「回退重做」是鬼打墙的主要来源，
        宁可停在最后一段耗尽步数，也不回头。

        ⚠️ 这套规则是实测选出来的，不是拍的。拿 train 的真值分段离线比对
        （7840 个关键帧、12 任务、用模板而非本局分段）：

            现行前的增量式 slack=2.0 min=2   20.9%   ← 79% 的帧落后真值
            增量式 slack=1.0 min=1           86.1%
            累计式 slack=1.0 min=1           92.5%   ← 现行
            累计式 无夹爪                     94.1%

        无夹爪那档在 demo 上略高，但它是纯开环的：模型卡住时它照样按计划推进。
        评测时轨迹是模型自己走的（实测 22–25 步 vs 计划预算 ~12），
        闭环信号的价值不在这张表里，所以保留夹爪。
        """
        self.t += 1
        self.used += 1
        flip = (self.prev_gripper is not None and gripper_open is not None
                and abs(float(gripper_open) - float(self.prev_gripper)) > 0.5)
        if gripper_open is not None:
            self.prev_gripper = float(gripper_open)
        last = len(self.subtasks) - 1
        pos = next((i for i, c in enumerate(self.cum) if self.t < c), last)
        if flip:
            pos = max(pos, self.idx + 1)
        pos = min(max(self.idx, pos), last)
        if pos != self.idx:
            self.idx = pos
            self.used = 0


# ---------------------------------------------------------------- 载入

class PlanBank:
    """预生成计划的只读视图，供 rollout 时按 (task, episode) 取用。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise PlannerError(
                f"找不到计划文件 {self.path}。"
                f"先跑 python scripts/build_plans.py --split val")
        doc = json.loads(self.path.read_text())
        self.plans = doc["plans"]
        self.coverage = doc["coverage"]
        self.split = doc["split"]

    def get(self, task: str, episode: int) -> list[dict]:
        key = f"{task}/{episode}"
        p = self.plans.get(key)
        if p is None:
            raise PlannerError(
                f"计划里没有 {key}（split={self.split}）。"
                f"B2/B3 的评测不允许缺计划 —— 缺了就等于拿不到子任务指令，"
                f"会退化成 B1 的行为。")
        return p["subtasks"]

    def summary(self) -> str:
        c = self.coverage
        return (f"PlanBank({self.path.name}, split={self.split}): "
                f"{len(self.plans)} 个计划；variation 精确命中 {c['exact']}、"
                f"回退 {c['fallback']}、缺失 {c['missing']}")
