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
    # 码向量注入层版本，**臂是唯一真源**（不启用码的臂此字段无意义）。
    # 曾经它只写在 conf/stage3.yaml 里，靠命令行 stage3.injector=v2 传进去 ——
    # 那样一旦漏传，就会在 B4/ 目录下静默训出一个 v1 模型且不报任何错。
    # 现在 launch_utils.create_agent 一律从这里读，配置里写了不一致的值直接抛异常。
    injector: str = "v1"
    #: 细节码支路开关。实测 9 个槽位在样本内恒等占 99.7%，该支路退化成常量偏置。
    #: 🔴 必须写在臂上、不能放环境变量：2026-09-11 实测事故 —— B4X 用
    #:    AAVLA_USE_DETAIL=0 训练，评测时忘了传，模型多出一条**从未训练**的
    #:    随机初始化支路往 latents 注噪声，成绩从 38% 掉到 22%，
    #:    而 load_weights 对「模型有、ckpt 没有」的键**只保持随机初始化、不报警**。
    use_detail: bool = True
    #: 原子支撑度门控（stage3/atom_support.py）。同理必须随臂走 ——
    #: 训练与评测的门控表不一致，训出来的模型和评测看到的就是两回事。
    atom_gate: bool = False
    #: 相位嵌入 —— 注入「当前是计划里的第几段」（subtask_index）。
    #: 依据：段数 vs 成功率 r=−0.41(B0)/−0.46(B1)；短程(≤3段) B0 均值 64.0%，
    #: 长程(≥5段) 只有 32.9%，差 31.1 pp。PerAct 在一个关键帧只看到当前视觉 +
    #: 一句**恒定**的整任务指令，长程任务里同一视觉状态会出现在不同阶段，
    #: 模型无从判断走到哪一步 —— 这正是 planner 知道、模型拿不到的信息。
    #: 任务无关（只是「第几段」），所以能迁移到 UnSeen；参数仅 24×128=3,072。
    use_phase: bool = False

    @property
    def uses_subtask_lang(self) -> bool:
        return self.lang == "subtask"

    def __post_init__(self) -> None:
        if self.injector not in ("v1", "v2"):
            raise ValueError(
                f"{self.name}: injector 只能是 v1/v2，收到 {self.injector!r}")


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
              "子任务指令 + v1 码注入（FiLM + 门控 cross-attn）+ 微调 K 步",
              "码本本身的功劳（核心主张）",
              injector="v1"),
    # B4 = B3 的字段权限，只把注入层换成 v2（纯残差相加，无门控无 FiLM）。
    # 为什么独立成臂而不是「改 logdir、arm 仍写 B3」：归档文件名、
    # stage3_eval.py --arm、结果汇总表全靠臂名区分，混用一定出乱子；而且
    # v1/v2 参数名不同，独立臂能让「拿 B3 的 ckpt 去续 B4」报错而非静默加载错。
    "B4": Arm("B4", "subtask", True, True,
              "子任务指令 + v2 码注入（纯残差相加，无门控无 FiLM）+ 微调 K 步",
              "注入层形式（对照 B3：码相同、注入机制不同）",
              injector="v2"),
    # ---- 2026-09-11 新增：把「语言替换」这个变量从码注入臂里摘出去 ----
    # 实测依据：flat 对照（分段与码逐局逐段相同、只换指令文本）
    #   B4@40000 子任务指令 32.44%  →  整任务指令 36.22%   Δ=+3.78 pp（450 局 test）
    # 也就是说「把整任务指令换成子任务指令」这一步本身在**扣分**。
    # B4L 保留整任务指令，于是它与 B1 的唯一差别就是**有没有注入码** ——
    # 这是本项目第一次能干净地测出码本的净贡献。
    "B4L": Arm("B4L", "task", True, True,
               "整任务指令 + v2 码注入 + 微调 K 步",
               "码本的净贡献（与 B1 的唯一差别就是码）",
               injector="v2"),
    # B4X：在 B4L 之上再叠三项注入侧改动（都由构造参数/环境变量控制，
    # 见 conf/stage3.yaml 的 stage3.injector_opts）。参数量净减 0.27 M。
    "B4X": Arm("B4X", "task", True, True,
               "B4L + 关细节码支路 + 原子支撑度门控",
               "注入侧两项改动的合计效果",
               injector="v2", use_detail=False, atom_gate=True),
    "B4P": Arm("B4P", "task", True, True,
               "B4L + 相位嵌入（注入当前是计划里的第几段）",
               "补上 PerAct 拿不到的阶段信息，针对长程任务",
               injector="v2", use_phase=True),
}


def allowed_fields(arm: str, base_fields: set[str]) -> frozenset[str]:
    """给定 replay 的全部字段名，返回该臂**允许读取**的子集。

    base_fields 传入 replay 的实际字段全集（含 subtask_*），由调用方从
    replay_buffer.get_storage_signature() 或一个样本的键集合得到。
    """
    a = ARMS[arm]
    native = set(base_fields) - ALL_SUBTASK_FIELDS
    if a.uses_subtask_lang:
        # B2/B3/B4：整任务语言字段被换成子任务语言字段
        allow = (native - TASK_LANG_FIELDS) | (SUBTASK_LANG_FIELDS & set(base_fields))
    else:
        # B0/B1/B4L：保留 PerAct 原生的整任务语言字段
        allow = set(native)
    # 🔴 码的权限与语言无关。这里曾经写在 uses_subtask_lang 分支**内部**，
    #    于是 lang="task" + codes=True 的臂（B4L）会在上面那个分支直接返回，
    #    一个 subtask_k_* 都拿不到 —— 训出来是个没有码的 B1，而且**不报错**，
    #    从 loss 和成绩上都看不出来。
    if a.codes:
        allow |= SUBTASK_CODE_FIELDS & set(base_fields)
        # 原子支撑度门控要按段的动作名决定注不注入码（stage3/atom_support.py）。
        # subtask_action 是诊断字段，只给码臂开，语言通路不受影响。
        allow |= {"subtask_action"} & set(base_fields)
        # 相位嵌入要读「当前是第几段」。同样只给码臂。
        if a.use_phase:
            allow |= {"subtask_index"} & set(base_fields)
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
