"""语言通路的解冻档位（lang tier）—— 单一真源。

# 背景

PerAct 里语言的通路是：

    CLIP RN50 ❄️  →  lang_preprocess ✅(Linear 512→128, 0.066 M)
        →  拼进体素 token  →  cross_attend_blocks ❄️  →  Perceiver layers ❄️
        →  decoder_cross_attn ❄️

原始 Stage 3 冻结边界（`VLA_Design §8.2`）只放开 `lang_preprocess` 一层线性。
`VLA_Design §8.8 P1` 把这条列为已知瓶颈：**从「整任务指令」到「子任务指令」是
语言输入分布的实质改变，单层线性能否承载存疑**，并留了两个备选：

    (b) 同时解冻 decoder_cross_attn        +0.083 M
    (c) lang_preprocess 扩为残差 MLP       +0.16 M（初始化为精确复现原 Linear）

实测佐证这条瓶颈是真的：`B2 − B1 = −5.33 pp`（子任务指令反而更差），而 `flat`
消融显示模型对子任务指令的**内容**无反应 —— 两者合起来更像「容量不够，适应不了
新的语言分布」，而不是「子任务信息没用」。

# 档位

    L0   官方边界，只训 lang_preprocess                       基线
    L3   L0 + 残差 MLP + 解冻 decoder_cross_attn              +0.25 M
    L4   L3 + 解冻 cross_attend_blocks（语言真正进入主干那层）  +3.24 M

**档位是所有臂共享的训练配方，不是某一臂的特权。** 若只给码注入臂放开，
`B4 vs B1` 会把「码的作用」和「多了几 M 可训参数」混在一起，归因就废了。

# 为什么用「旁路增量」而不是原地替换 lang_preprocess

残差 MLP 若实现成 `lang_preprocess = Residual(原 Linear)`，state_dict 的键会从
`lang_preprocess.weight` 变成 `lang_preprocess.base.weight`。而 `load_weights`
是**合并式**加载（模型有、ckpt 没有的键保持原值，只打一行 warning），于是从官方
ckpt 起步时那层线性会**静默地保持随机初始化** —— 训练照常跑完，不报任何错。

所以这里改成加一个**旁路模块** `lang_preprocess_delta`，前向是
`lang_preprocess(x) + lang_preprocess_delta(x)`：

    · 官方 ckpt 的 `lang_preprocess.*` 键原封不动，照常命中
    · 旁路末层零初始化 ⇒ 训练起点与原模型**逐位等价**
    · 低档位的 ckpt 装进高档位模型：旁路缺失 → 保持零 → 行为等同低档位（安全）
    · 高档位的 ckpt 装进低档位模型：旁路的键无处可去 → **必须硬失败**
      （这个方向会静默丢掉学到的参数，见 `assert_ckpt_tier_compatible`）
"""

from __future__ import annotations

#: 各档位相对 L0 **额外**放开的模块名（用于冻结白名单的子串匹配）。
TIER_EXTRA_MODULES: dict[str, tuple[str, ...]] = {
    "L0": (),
    "L3": ("lang_preprocess_delta", "decoder_cross_attn"),
    "L4": ("lang_preprocess_delta", "decoder_cross_attn", "cross_attend_blocks"),
}

#: 该档位是否需要构造 lang_preprocess_delta 旁路。
TIER_HAS_DELTA: dict[str, bool] = {k: "lang_preprocess_delta" in v
                                   for k, v in TIER_EXTRA_MODULES.items()}

#: 可训参数量的验收区间（下界按无码臂、上界按带码注入臂）。
#: 实测：L0 B1/B2 = 2,117,917、B3/B4 = 2,550,685；
#:       旁路 MLP +164,224、decoder_cross_attn +83,328、cross_attend_blocks +3,235,072。
TIER_PARAM_RANGE: dict[str, tuple[float, float]] = {
    "L0": (2.0e6, 2.7e6),
    "L3": (2.3e6, 2.9e6),
    "L4": (5.5e6, 6.2e6),
}

#: 旁路 MLP 的隐藏层宽度。512→256→128 = 164,224 参数。
DELTA_HIDDEN = 256

DEFAULT_TIER = "L0"


def normalize(tier: str | None) -> str:
    """校验并规范化档位名。给了不认识的值就抛，绝不静默退回默认。"""
    t = (tier or DEFAULT_TIER).strip().upper()
    if t not in TIER_EXTRA_MODULES:
        raise ValueError(
            f"未知的 lang_tier {tier!r}，可选 {sorted(TIER_EXTRA_MODULES)}。"
            f"档位决定哪些语言模块参与训练，写错会静默训出另一个配方。")
    return t


def trainable_modules(tier: str, base: tuple[str, ...]) -> tuple[str, ...]:
    """基线可训模块 + 该档位额外放开的。"""
    return tuple(base) + TIER_EXTRA_MODULES[normalize(tier)]


def assert_param_count(tier: str, n_train: int) -> None:
    lo, hi = TIER_PARAM_RANGE[normalize(tier)]
    if not (lo < n_train < hi):
        raise AssertionError(
            f"lang_tier={tier} 的可训参数 {n_train:,} 不在预期区间 "
            f"({lo:,.0f}, {hi:,.0f})。冻结边界与档位不符 —— 训下去得到的不是"
            f"你以为的那个配方。")


def assert_ckpt_tier_compatible(tier: str, missing_in_model: list[str]) -> None:
    """ckpt 里有、当前模型里没有的键 —— 高档位 ckpt 装进低档位模型的信号。

    这个方向会**静默丢掉已经训好的参数**（合并式加载只打一行 warning），
    是本项目反复吃过亏的那类失败，必须硬失败。
    """
    orphan = [k for k in missing_in_model
              if any(m in k for m in TIER_EXTRA_MODULES["L4"])]
    if orphan:
        raise RuntimeError(
            f"checkpoint 含语言档位模块 {orphan[:4]}…（共 {len(orphan)} 个键），"
            f"而当前 lang_tier={tier} 的模型里没有它们。这会静默丢掉已训练的参数。"
            f"请把 stage3.lang_tier 设成训练该 ckpt 时用的档位。")
