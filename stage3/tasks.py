"""任务划分的**唯一真源** —— Seen18 / UnSeen 候选池 / Tier 分层。

2026-09 改版（`VLA_Design §2.4`、`Exp_Design §1.1`）把划分整个换掉了：

    旧：Seen12 / UnSeen6   在官方 18 任务**内部**按「是否参与 Stage 3 微调」切
    新：Seen18 / UnSeen8   UnSeen 移出官方 18 之外，取自 `peract_600k` 的
                           21 个预训练任务**都没有**的任务

换的理由是实测出来的：`peract_600k` 的 config.yaml 显示主干在 600k 预训练里
见过全部 21 个任务（官方 18 + 3 个），所以旧 UnSeen6 只能声称「未参与 Stage 3
微调」，撑不起「泛化到未见任务」。新划分下四臂在 UnSeen 上全部零训练。

# 为什么集中在这里

改版前 `SEEN12` / `UNSEEN6` 在四个文件里各抄了一份
（stage3_eval / planner_cache / plan_audit / preflight_stage3）。任务集是评测
口径的一部分，抄四份意味着改一处漏三处，而漏掉的那处**不会报错**，只会安静地
评错任务集。本项目已经因为「多套数据共存、按配置选用」的结构吃过多次亏，
所以这里只留一份，其余文件一律 import。

# UnSeen 的两层设计（【声明 B′】）

    Tier-A  码本在 AtomAction 69 任务里**见过**其原子片段 —— 这正是被检验的
            假设（原子知识能否跨任务迁移），不是数据污染。主结果。
    Tier-B  码本也**没见过** —— 排除「码本恰好见过该任务」这一替代解释。对照。

两层**分列报告，不合并求均值**。

# 定案规则事先冻结（【声明 E】）

最终 8 个从 10 个候选中筛出：先剔 E1 天花板为 0 的任务（该任务在关键帧动作
模式下本就执行不了），再剔 B0 零样本 sr ≥ 90% 的任务（无信号）。筛选**只用
B0 与天花板**，在看到 B1/B2/B3 任何结果之前完成。定案结果写进
`UNSEEN8_FINAL`（现在是 None —— 还没筛）。
"""

from __future__ import annotations

# ---------------------------------------------------------------- Seen 18
#: 官方 PerAct 18 任务全集，全部参与 Stage 3 微调。
#: 顺序与 source/peract/conf/stage3.yaml 的 rlbench.tasks 一致，便于逐行比对。
SEEN18 = [
    "close_jar", "light_bulb_in", "open_drawer", "place_cups",
    "place_shape_in_shape_sorter", "push_buttons", "put_groceries_in_cupboard",
    "reach_and_drag", "slide_block_to_color_target", "stack_blocks",
    "place_wine_at_rack_location", "sweep_to_dustpan_of_size",
    "insert_onto_square_peg", "meat_off_grill", "put_item_in_drawer",
    "put_money_in_safe", "stack_cups", "turn_tap",
]

# ---------------------------------------------------------------- UnSeen
#: Tier-A：主干没见过，码本见过其原子片段。E3 的主结果。
UNSEEN_TIER_A = [
    "close_drawer", "pick_up_cup", "open_jar",
    "phone_on_base", "lamp_on", "basketball_in_hoop",
]

#: Tier-B：主干与码本都没见过。用来排除「码本恰好见过该任务」这一替代解释。
UNSEEN_TIER_B = ["take_money_out_safe", "take_lid_off_saucepan"]

#: 备选：数据已生成，若定案筛掉了 Tier-A/B 中的任务则从这里补，否则进附录。
UNSEEN_SPARE = ["press_switch", "put_knife_on_chopping_board"]

#: 公开的 10 个候选池。test 数据已全部生成（各 25 局），无论怎么筛都不必补生成。
UNSEEN10 = UNSEEN_TIER_A + UNSEEN_TIER_B + UNSEEN_SPARE

#: 【声明E】定案后的 8 个。**筛选完成前必须是 None** —— 用 None 而不是先填一个
#: 猜测值，是为了让「还没定案就去读它」当场失败，而不是安静地用一份未冻结的名单。
UNSEEN8_FINAL: list[str] | None = None

#: 与 Seen 共享物体与场景、动作方向相反的任务（【声明 D】物体级熟悉度）。
#: 逐任务分列时必须标注，不能在总均值里掩盖。
UNSEEN_REVERSED = {"close_drawer", "open_jar", "take_money_out_safe"}

#: 主评测实际覆盖的任务：Seen18 + UnSeen **候选 10** = 28 个。
#: Exp_Design 写的「26 个任务」是按定案后的 8 个算的口径；数据与评测都按 10 个
#: 候选跑（多跑的 2 个进附录），这样无论怎么筛都不必回头补评测。
ALL_EVAL_TASKS = SEEN18 + UNSEEN10

# ------------------------------------------------------- 历史划分（只读）
#: 🔴 2026-09 改版前的划分，**已作废**。保留仅为读懂 result/ 里的旧结果与
#: Exp_Design 第八节的历史表格。新代码一律不要用它做评测口径。
LEGACY_SEEN12 = [
    "close_jar", "light_bulb_in", "open_drawer", "place_cups",
    "place_shape_in_shape_sorter", "push_buttons", "put_groceries_in_cupboard",
    "reach_and_drag", "slide_block_to_color_target", "stack_blocks",
    "place_wine_at_rack_location", "sweep_to_dustpan_of_size",
]
LEGACY_UNSEEN6 = [
    "insert_onto_square_peg", "meat_off_grill", "put_item_in_drawer",
    "put_money_in_safe", "stack_cups", "turn_tap",
]

#: 命令行 --tasks 接受的组名。值为 None 的组尚未定案，解析时会报错。
GROUPS: dict[str, list[str] | None] = {
    "seen18": SEEN18,
    "unseen10": UNSEEN10,
    "unseen8": UNSEEN8_FINAL,
    "unseen_a": UNSEEN_TIER_A,
    "unseen_b": UNSEEN_TIER_B,
    "all28": ALL_EVAL_TASKS,
    # 历史组：跑旧口径复现时才用
    "seen12": LEGACY_SEEN12,
    "unseen6": LEGACY_UNSEEN6,
}


def resolve(tasks: list[str]) -> list[str]:
    """把 `--tasks` 的取值解析成任务名列表。

    单个组名（`["seen18"]`）展开成该组；其余情况原样返回，视作显式任务名。
    组名与任务名混写会被拒绝 —— 那种写法的意图无法判断，静默猜一个比报错更坏。
    """
    if len(tasks) == 1 and tasks[0] in GROUPS:
        got = GROUPS[tasks[0]]
        if got is None:
            raise SystemExit(
                f"❌ 任务组 '{tasks[0]}' 尚未定案。UnSeen 的最终 8 个要先跑完 E1 "
                f"天花板与 B0 零样本摸底，按 Exp_Design【声明E】的规则筛出，"
                f"再把结果填进 stage3/tasks.py 的 UNSEEN8_FINAL。"
                f"在那之前请用 --tasks unseen10 跑全部候选。")
        return list(got)
    bad = [t for t in tasks if t in GROUPS]
    if bad and len(tasks) > 1:
        raise SystemExit(f"❌ --tasks 里混写了组名 {bad} 与任务名，意图不明确。"
                         f"要么只给一个组名，要么全写任务名。")
    return list(tasks)
