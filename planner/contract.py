"""Planner 契约层：白名单、动作定义、指令规范、use_codebook 规则、cache schema。

这是离线建 cache 与在线评测**共用**的唯一真相来源。`VLA_Design §6.1` 的一致性硬约束
要求两侧共用同一段格式规则文本与同一套后处理路径——所以规则只在本文件定义一次。
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------- 动作白名单
# 码本实际训练用的 17 类（= config/global.yaml::train_actions）。
# 这 17 类之外的动作没有对应的码字，注入层必须关断。
CODEBOOK_ACTIONS: tuple[str, ...] = (
    "approach", "grasp", "lift", "place", "push", "pull", "press", "rotate",
    "slide", "insert", "hang", "wipe", "flip-open", "flip-close",
    "revolve-in", "revolve-out", "transfer",
)

# pose-adjust 在 AtomAction_Dataset 中有 1307 段，但被排除在 train_actions 之外，
# 即**码本从未见过它**。因此它是合法输出，但强制 use_codebook=false。
# SEMANTIC_ACTION_STANDAR_v2.xlsx 里的 `unhang` 同样不在码本训练集中，
# 且未出现在 PerAct 18 任务中，按用户决定**整体忽略**（不进白名单、不允许输出）。
NON_CODEBOOK_ACTIONS: tuple[str, ...] = ("pose-adjust",)

ALLOWED_ACTIONS: frozenset[str] = frozenset(CODEBOOK_ACTIONS) | frozenset(NON_CODEBOOK_ACTIONS)


def use_codebook(action: str) -> bool:
    """VLA_Design §3.3：仅 17 类码本动作启用码注入，其余恒等关断。"""
    return action in CODEBOOK_ACTIONS


# ---------------------------------------------------------------- 指令规范
# 词数上下界取 PerAct 与 Adapter 两个训练分布的交集：
#   PerAct  descriptions[0]  205 条：min 3 / 中位 6 / max 14
#   AtomAction short         586 条：min 3 / 中位 4 / max 12
# 交集 3~8 覆盖 AtomAction 的 99%、PerAct 的 69%。
INSTRUCTION_MIN_WORDS = 3
INSTRUCTION_MAX_WORDS = 8
INSTRUCTION_WARN_MAX_WORDS = 10   # 9~10 词告警放行，>10 拒绝

# 每个 action 在 AtomAction short 指令里实际使用的首词（Adapter 的训练分布）。
# 关键差异：approach 88% 用 "move"，transfer 65% 用 "move" / 26% 用 "carry"，
# pose-adjust 93% 用 "adjust"——写成与标签同名反而是分布外。
ACTION_VERBS: dict[str, tuple[str, ...]] = {
    "approach": ("move toward", "approach"),
    "grasp": ("grasp",),
    "lift": ("lift",),
    "transfer": ("move", "carry"),
    "place": ("place",),
    "push": ("push",),
    "pull": ("pull",),
    "press": ("press",),
    "rotate": ("rotate", "tilt"),
    "slide": ("slide",),
    "insert": ("insert",),
    "hang": ("hang",),
    "wipe": ("sweep", "wipe"),
    "flip-open": ("flip",),
    "flip-close": ("flip", "lower"),
    "revolve-in": ("swing",),
    "revolve-out": ("swing", "pull"),
    "pose-adjust": ("adjust", "align"),
}

_TRAILING_PUNCT = ".。!！?？,，;；"


def normalize_instruction(text: str) -> str:
    """canonical 化：去首尾空白 → 去句尾标点 → 全小写 → 压缩空白。

    离线与在线必须走这同一个函数（VLA_Design §6.1）。
    """
    s = " ".join(str(text).strip().split())
    while s and s[-1] in _TRAILING_PUNCT:
        s = s[:-1].rstrip()
    return s.lower()


def check_instruction(text: str, task_instruction: str | None = None,
                      n_segments: int = 2) -> list[str]:
    """返回违规项列表；空列表表示通过。硬性项与告警项都返回，调用方自行分级。"""
    issues: list[str] = []
    s = normalize_instruction(text)
    if s != str(text).strip():
        issues.append("not_canonical")
    n = len(s.split())
    if n < INSTRUCTION_MIN_WORDS:
        issues.append(f"too_short({n})")
    elif n > INSTRUCTION_WARN_MAX_WORDS:
        issues.append(f"too_long({n})")
    elif n > INSTRUCTION_MAX_WORDS:
        issues.append(f"long_warn({n})")
    if not re.fullmatch(r"[a-z0-9 ,'\-]+", s or "x"):
        issues.append("bad_charset")
    # 单段 episode 的子任务指令天然 ≈ 任务指令，不算抄袭
    if (task_instruction is not None and n_segments >= 2
            and s == normalize_instruction(task_instruction)):
        issues.append("copies_task_instruction")
    return issues


# ---------------------------------------------------------------- 结构校验
def check_segments(segments: list[dict[str, Any]], n_keypoints: int,
                   task_instruction: str,
                   gripper_at_keypoints: list[int] | None = None) -> dict[str, Any]:
    """VLA_Design §5.1 的 C1–C4 + 本项目新增的 C8（夹爪一致性）。

    返回 {"errors": [...], "warnings": [...]}；errors 非空即该 episode 不合格。
    """
    errors: list[str] = []
    warnings: list[str] = []

    # C1：keypoint_indices 构成 [0, M) 的完整无重叠划分
    idx = [i for s in segments for i in s.get("keypoint_indices", [])]
    if sorted(idx) != list(range(n_keypoints)):
        missing = sorted(set(range(n_keypoints)) - set(idx))
        dup = sorted({i for i in idx if idx.count(i) > 1})
        errors.append(f"C1_partition(missing={missing[:6]},dup={dup[:6]},"
                      f"got={len(idx)},want={n_keypoints})")

    # C2：segment_index 递增、段内有序、段间时间不交叉
    if [s.get("segment_index") for s in segments] != list(range(len(segments))):
        errors.append("C2_segment_index_not_sequential")
    if idx != sorted(idx):
        errors.append("C2_keypoints_not_time_ordered")
    for s in segments:
        ki = s.get("keypoint_indices", [])
        if ki != sorted(ki):
            errors.append(f"C2_unsorted_within_segment({s.get('segment_index')})")
        if not ki:
            errors.append(f"C2_empty_segment({s.get('segment_index')})")

    # C3：action ∈ 白名单
    for s in segments:
        if s.get("action") not in ALLOWED_ACTIONS:
            errors.append(f"C3_action_not_allowed({s.get('action')})")

    # C4：指令格式
    seen: set[str] = set()
    for s in segments:
        ins = s.get("instruction", "")
        for issue in check_instruction(ins, task_instruction, len(segments)):
            (warnings if issue.startswith("long_warn") else errors).append(
                f"C4_{issue}@seg{s.get('segment_index')}")
        norm = normalize_instruction(ins)
        if norm in seen:
            warnings.append(f"C4_duplicate_instruction@seg{s.get('segment_index')}")
        seen.add(norm)

    # C8：夹爪状态一致性（用户口径）
    #   transfer    = 夹爪抓住物体时的移动  → 关键帧上 gripper_open == 0
    #   pose-adjust = 夹爪未抓住物体时的调整 → 关键帧上 gripper_open == 1
    # 设为告警：存在「夹爪闭合但未持物」等边缘情况，不宜硬拒。
    if gripper_at_keypoints is not None:
        for s in segments:
            act = s.get("action")
            if act not in ("transfer", "pose-adjust"):
                continue
            want_open = 1 if act == "pose-adjust" else 0
            got = [gripper_at_keypoints[i] for i in s.get("keypoint_indices", [])
                   if i < len(gripper_at_keypoints)]
            if got and any(g != want_open for g in got):
                warnings.append(
                    f"C8_gripper_mismatch@seg{s.get('segment_index')}"
                    f"({act},want_open={want_open},got={got})")

    return {"errors": errors, "warnings": warnings}


def gripper_consistent(action: str, keypoint_indices: list[int],
                       gripper_at_keypoints: list[int]) -> bool | None:
    """单段的 C8 结论，写进 cache 供统计体检使用。非 transfer/pose-adjust 返回 None。"""
    if action not in ("transfer", "pose-adjust"):
        return None
    want_open = 1 if action == "pose-adjust" else 0
    got = [gripper_at_keypoints[i] for i in keypoint_indices
           if i < len(gripper_at_keypoints)]
    return bool(got) and all(g == want_open for g in got)


# ---------------------------------------------------------------- cache schema
CACHE_SCHEMA_VERSION = "vqap_peract_cache_v1"

# 重复次数可由 variation 索引解析推出的任务（P1 实测规律），
# 作为可信字段传给 Planner，不让模型从图里数。
REPEAT_RULES: dict[str, Any] = {
    "push_buttons": lambda v: v % 3 + 1,     # 按 1/2/3 个键
    "stack_blocks": lambda v: v % 3 + 2,     # 堆 2/3/4 个块
    "place_cups": lambda v: v + 1,           # 放 1/2/3 个杯子
}

# 展开先验时必须用**单个循环**去乘，不能拿 CSV 里的整条序列去乘。
# 踩过的坑：CSV 里 stack_blocks 的先验本身已经是 2 个循环（approach…place ×2），
# 再乘以 n_repeat=3 就变成 6 个循环，传给模型的先验长了一倍。
# （实测模型对此完全鲁棒，100/100 条仍按图像判出正确的 2/3/4 个循环，
#   但代码不该依赖模型的鲁棒性。）
REPEAT_UNIT: dict[str, list[str]] = {
    "push_buttons": ["approach", "press"],
    "stack_blocks": ["approach", "grasp", "lift", "transfer", "place"],
    "place_cups": ["approach", "grasp", "lift", "transfer", "hang"],
}


def n_repeat_prior(task: str, variation: int) -> int | None:
    fn = REPEAT_RULES.get(task)
    return int(fn(variation)) if fn else None


def expand_prior(task: str, variation: int, csv_prior: list[str]) -> list[str]:
    """把 CSV 先验展开成该 variation 应有的完整序列。"""
    n = n_repeat_prior(task, variation)
    if n is None or n <= 0:
        return list(csv_prior)
    unit = REPEAT_UNIT.get(task)
    if unit is None:                      # 没登记单元循环时退回原行为
        return list(csv_prior) * n
    return list(unit) * n
