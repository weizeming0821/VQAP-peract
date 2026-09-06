"""Planner 策略的**唯一选择点** —— P6 可回撤性的落点。

# 为什么要有这一层

主线用模板法（`template`）：`(task, variation)` → train cache 的众数子任务序列。
这是个赌注 —— 赌「同 variation 的场景构成相同，指令可以直接沿用」。
万一赌输了（评测结果不佳，或轨迹显示状态机没在干活），要能换成
贴近原设计的在线 VLM planner，而**不用回滚代码**。

于是把「用哪种 planner」收敛到这一个模块。`rollout.py` / `eval.py` /
agent 都只认工厂函数，换 planner 时它们一行都不用改。

# 三种

    template   (task, variation) 模板库 + 确定性状态机。默认。
    oracle     每一局自己的真值分段（只有 train split 有）。**任何 planner 的上界**：
               分段零误差时能拿多少分。用来判断「换更好的 planner 值不值」。
    flat       对照组：子任务指令**全部替换成整任务指令**，码与分段完全不变。
               `B3(template) − B3(flat)` = 子任务分解本身值多少分。
               没有它，「该不该换 planner」只能靠猜。
    vlm        在线 VLM 闭环：子任务序列仍取自模板库，VLM 每个关键帧看画面
               决定「现在该做第几步」。已实现并跑通（120 局 1200 次调用 0 失败）。
    vlm-plan   真·在线 planner：**完全不读模板库**，开局由 VLM 看画面现场规划
               子任务序列，之后按触发规则做 CONTINUE/NEXT/RETRY/REPLAN 判定。
               CSV 先验只作参考送进 prompt。必须配 `--codes adapter`。

# 工厂而不是实例

每一局 rollout 需要一个**独立**的状态机（`idx`/`used`/`prev_gripper` 都是局内状态）。
所以这里返回的是 `factory(task, episode) -> planner`，由 rollout 每局调一次。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from stage3.online_planner import DeterministicPlanner, PlanBank, PlannerError

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "aavla_data" / "planner_cache"

#: 允许的 planner 种类。`vlm` 已登记但未实现，见模块 docstring。
KINDS = ("template", "flat", "oracle", "vlm", "vlm-plan")
DEFAULT_KIND = "template"


class SubtaskPlanner(Protocol):
    """所有 planner 必须满足的协议 —— 换实现时照着这三样做即可。"""

    def prime(self, gripper_open: float | None) -> None: ...
    def observe(self, gripper_open: float | None) -> None: ...
    def note(self, step: int) -> None: ...
    @property
    def current(self) -> dict: ...


PlannerFactory = Callable[[str, int], SubtaskPlanner]


def plan_file(split: str, kind: str = DEFAULT_KIND) -> Path:
    """计划文件按 planner 分开存 —— 两种方案的产物永不互相覆盖。"""
    if kind not in KINDS:
        raise PlannerError(f"未知 planner 种类 {kind!r}，可选 {KINDS}")
    # vlm 不另建计划文件：它复用模板库的子任务序列，只把「选下标」换成 VLM。
    kind = "template" if kind == "vlm" else kind
    return CACHE_DIR / f"plans_{split}_{kind}.json"


class PlanFactory:
    """按 (kind, split) 取计划、每局造一个新状态机。

    🔴 必须是**模块级类**，不能是 `make_factory` 里的闭包。
    `eval.py:215` 用 spawn 起评测子进程，`Stage3RolloutGenerator`（连同它
    持有的工厂）会被 pickle 过去，而局部函数 pickle 不了：
        AttributeError: Can't pickle local object 'make_factory.<locals>.factory'
    `stage3/replay_dataset.py` 的 `_seal` 踩过一模一样的坑 —— 那次是
    DataLoader 的 spawn worker。凡是要跨进程的可调用对象，一律模块级。

    bank 不随 pickle 过去（见 `__getstate__`）：父进程加载一次是为了**尽早失败**
    （计划文件缺了要在起 12 个分片之前就报错），子进程各自重新读，
    省掉 12 份 550 KB 的序列化。
    """

    def __init__(self, kind: str, split: str, verbose: bool = True) -> None:
        self.kind = kind
        self.split = split
        self._verbose = verbose
        self._bank: PlanBank | None = None
        self.bank                                  # 父进程立刻加载 = 尽早失败

    @property
    def bank(self) -> PlanBank:
        if self._bank is None:
            self._bank = PlanBank(plan_file(self.split, self.kind))
            if self._verbose:
                print(f"[planner] kind={self.kind} {self._bank.summary()}",
                      flush=True)
        return self._bank

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_bank"] = None                          # 子进程自己重新读
        return d

    def __call__(self, task: str, episode: int) -> SubtaskPlanner:
        return DeterministicPlanner(self.bank.get(task, episode))

    def summary(self) -> str:
        return self.bank.summary()


def make_factory(kind: str, split: str, verbose: bool = True) -> PlannerFactory:
    """按种类造一个「每局一个 planner」的工厂。"""
    if kind == "vlm":
        # 子任务序列仍取自模板库，VLM 只负责选下标 —— 这样 B2/B3 的候选集合
        # 完全相同，差值里不会混进「两臂看到不同子任务文本」这个额外变量。
        from stage3.vlm_planner import VLMPlanFactory
        return VLMPlanFactory(split, verbose=verbose)
    if kind == "vlm-plan":
        from stage3.vlm_planner import OnlinePlanFactory
        return OnlinePlanFactory(split, verbose=verbose)
    return PlanFactory(kind, split, verbose)
