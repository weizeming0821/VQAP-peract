"""Stage 3 四臂定义与 replay 字段隔离。

四臂逐级只多一样东西，差值才能干净归因：

    B0 --(+微调 K 步)--> B1 --(+子任务指令)--> B2 --(+码注入)--> B3

    B0 本身   -> 验证评测管线正确性（E0，所有结论的前提门禁）
    B1 - B0   -> 「在这批数据上微调动作头」本身的效果
    B2 - B1   -> Planner 的功劳，**不计入 VQAP 贡献**
    B3 - B2   -> 码本本身的功劳，**全文的核心主张**

字段隔离（VLA_Design §8.6.2）：三臂共享同一份 replay（152 GB，建三份不可接受，
且共享才能保证同种子下看到逐条相同、顺序相同的样本）。仅靠 `subtask_` 命名约定
不够，必须运行时强制——每个臂声明可读字段白名单，越权访问**第一次就抛异常**，
而不是悄悄训出一个说不清是什么的模型。

依据是本项目的真实教训：P1 阶段调度器的快速失败逻辑曾静默跳过 10 个作业、
P3 阶段修复脚本曾留下 208 条缺 keypoints 的残缺记录，两次都不报任何错。
凡是「多套数据共存、按配置选用」的结构，都必须有硬校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

# ---------------------------------------------------------------- 字段分组
# PerAct 原生的语言字段：永远承载**整任务**指令，不改名不改义
TASK_LANG_FIELDS = frozenset({"lang_goal_emb", "lang_token_embs", "lang_goal"})

# Stage 3 新增，一律 subtask_ 前缀
SUBTASK_LANG_FIELDS = frozenset({
    "subtask_lang_goal_emb", "subtask_lang_token_embs", "subtask_lang_goal",
})
SUBTASK_CODE_FIELDS = frozenset({
    "subtask_k_global", "subtask_k_detail", "subtask_code_mask",
})
SUBTASK_DIAG_FIELDS = frozenset({"subtask_index", "subtask_action"})
ALL_SUBTASK_FIELDS = SUBTASK_LANG_FIELDS | SUBTASK_CODE_FIELDS | SUBTASK_DIAG_FIELDS


@dataclass(frozen=True)
class Arm:
    name: str
    lang: str            # "task" | "subtask"
    codes: bool          # 是否启用码注入
    train: bool          # 是否参与微调（B0 零训练）
    description: str
    isolates: str        # 该臂相对前一臂多出的那一样东西

    @property
    def uses_subtask_lang(self) -> bool:
        return self.lang == "subtask"


ARMS: dict[str, Arm] = {
    "B0": Arm("B0", "task", False, False,
              "官方 peract_600k 原样，零训练",
              "基准点：验证评测管线（E0）"),
    "B1": Arm("B1", "task", False, True,
              "整任务指令 + 微调 K 步",
              "微调本身的效果"),
    "B2": Arm("B2", "subtask", False, True,
              "子任务指令 + 微调 K 步",
              "Planner 的功劳（不计入 VQAP 贡献）"),
    "B3": Arm("B3", "subtask", True, True,
              "子任务指令 + 码注入 + 微调 K 步",
              "码本本身的功劳（核心主张）"),
}


def allowed_fields(arm: str, base_fields: set[str]) -> frozenset[str]:
    """给定 replay 的全部字段名，返回该臂**允许读取**的子集。

    base_fields 传入 replay 的实际字段全集（含 subtask_*），由调用方从
    replay_buffer.get_storage_signature() 或一个样本的键集合得到。
    """
    a = ARMS[arm]
    native = set(base_fields) - ALL_SUBTASK_FIELDS
    if not a.uses_subtask_lang:
        # B0/B1：只看 PerAct 原生字段，一个 subtask_* 都不给
        return frozenset(native)
    # B2/B3：整任务语言字段被换成子任务语言字段
    allow = (native - TASK_LANG_FIELDS) | (SUBTASK_LANG_FIELDS & set(base_fields))
    if a.codes:
        allow |= SUBTASK_CODE_FIELDS & set(base_fields)
    return frozenset(allow)


class FieldAccessError(KeyError):
    """越权访问 replay 字段。"""


class GuardedSample(Mapping):
    """replay_sample 的薄包装：访问白名单外的键直接抛异常。

    只做键级隔离，不复制数据，开销可忽略。
    """

    def __init__(self, data: Mapping[str, Any], arm: str,
                 allowed: frozenset[str] | None = None) -> None:
        self._data = data
        self._arm = arm
        self._allowed = allowed if allowed is not None else allowed_fields(arm, set(data))

    def __getitem__(self, key: str) -> Any:
        if key not in self._allowed:
            reason = ("该字段承载整任务指令，B2/B3 必须用 subtask_ 版本"
                      if key in TASK_LANG_FIELDS else
                      "该字段属于子任务信息，本臂不得读取"
                      if key in ALL_SUBTASK_FIELDS else
                      "不在本臂白名单内")
            raise FieldAccessError(
                f"[{self._arm}] 禁止读取 replay 字段 '{key}'：{reason}。"
                f"若确需使用，请修改 stage3/arms.py 的白名单，不要绕过本守卫。")
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._allowed))

    def __len__(self) -> int:
        return len(self._allowed)

    def __contains__(self, key: object) -> bool:
        return key in self._allowed

    @property
    def arm(self) -> str:
        return self._arm

    def raw(self) -> Mapping[str, Any]:
        """绕过守卫的逃生舱，仅供调试与日志使用。"""
        return self._data


def lang_fields_for(arm: str) -> tuple[str, str]:
    """返回该臂应当喂给 PerAct 的 (句向量字段名, token 向量字段名)。"""
    if ARMS[arm].uses_subtask_lang:
        return "subtask_lang_goal_emb", "subtask_lang_token_embs"
    return "lang_goal_emb", "lang_token_embs"
