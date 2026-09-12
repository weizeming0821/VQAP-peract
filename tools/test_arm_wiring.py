#!/usr/bin/env python
"""门禁：`stage3.arm=<臂>` 真的装配出了那个臂应有的模型。

# 为什么需要这条

`launch_utils.create_agent` 曾把臂的能力写死成 `use_codes = arm == 'B3'` 与
`do_freeze = arm in ('B1','B2','B3')`。于是新增 B4 时，只要忘了同步这两行，
训练就会**一路跑完、不报任何错**，却在 `B4/` 目录下训出一个没有码注入、
也没冻结主干的四不像 —— 事后从成绩上根本看不出来。

本测试从**真实的 conf/stage3.yaml** 出发，逐臂装配 agent（纯 CPU，不占显卡、
不建 replay），逐条核对：

    ① 码本装没装        必须 == ARMS[arm].codes
    ② 注入层是哪个类     None / CodeInjector(v1) / CodeInjectorV2(v2)
    ③ 冻结后可训参数量   B1/B2 = 2,117,917
                        B3    = 2,117,917 + 432,768 = 2,550,685
                        B4    = 2,117,917 + 399,744 = 2,517,661
    ④ 配置里写了与臂不符的 injector 必须抛异常，不许静默

③ 的数字就是训练日志第一屏那行 `Stage 3 冻结边界：可训参数 …` ——
开训后拿它对一眼，就能确认这一轮训的确实是想训的那个臂。

用法：source run/env.sh && python tools/test_arm_wiring.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.WARNING)          # 静掉 create_agent 的 INFO

from hydra import compose, initialize_config_dir    # noqa: E402
from hydra.core.global_hydra import GlobalHydra      # noqa: E402
from omegaconf import open_dict                      # noqa: E402

from stage3.arms import ARMS                        # noqa: E402
from model.code_injector import CodeInjector, CodeInjectorV2   # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name
          + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


#: 冻结边界之下、注入层之外的可训参数（up0/final/dense0/dense1/
#: trans_decoder/rot_grip_collision_ff/lang_preprocess），实测值。
BASE_TRAINABLE = 2_117_917
#: 各注入层版本的参数量（tools/test_injector_v2.py 实测）。
INJECTOR_PARAMS = {"v1": 432_768, "v2": 399_744}
#: 关掉细节码支路后省下的参数（slot_embed / ln_h / w_q,k,v,o）。
#: 实测 CodeInjectorV2: use_detail=True 399,744 → False 231,040。
DETAIL_BRANCH_PARAMS = 168_704
#: 相位嵌入表 MAX_PHASE(24) × dim(128)。
PHASE_EMBED_PARAMS = 3_072


def load_cfg(arm: str, injector_override=None):
    # 本仓库的 hydra 版本没有 version_base 参数；每次 compose 前要清掉全局单例，
    # 否则第二次 initialize_config_dir 会报 "GlobalHydra is already initialized"。
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO_ROOT / "source" / "peract" / "conf")):
        cfg = compose(config_name="stage3")
    with open_dict(cfg):
        cfg.stage3.arm = arm
        if injector_override is not None:
            cfg.stage3.injector = injector_override
    return cfg


def encoder_of(agent):
    """PreprocessAgent -> QAttentionStackAgent -> 第一层的 perceiver_encoder。"""
    return agent._pose_agent._qattention_agents[0]._perceiver_encoder


def expected_injector(arm: str) -> int:
    """该臂的注入层参数量 —— 细节码支路按臂的 use_detail 决定（arms.py 是真源）。"""
    a = ARMS[arm]
    n = INJECTOR_PARAMS[a.injector]
    if not getattr(a, "use_detail", True):
        n -= DETAIL_BRANCH_PARAMS
    if getattr(a, "use_phase", False):
        n += PHASE_EMBED_PARAMS
    return n


def expected_trainable(arm: str) -> int:
    a = ARMS[arm]
    return BASE_TRAINABLE + (expected_injector(arm) if a.codes else 0)


def frozen_trainable(enc) -> int:
    """复刻 QAttentionPerActBCAgent.build 里的冻结逻辑，不建 QFunction/DDP。"""
    from agents.peract_bc.qattention_peract_bc_agent import STAGE3_TRAINABLE_MODULES
    n = 0
    for name, p in enc.named_parameters():
        if any(f'.{m}.' in f'.{name}.' for m in STAGE3_TRAINABLE_MODULES):
            n += p.numel()
    return n


def main() -> int:
    from agents.peract_bc.launch_utils import create_agent

    print("=== 1. 逐臂装配（纯 CPU，读真实的 conf/stage3.yaml）===")
    for arm, a in ARMS.items():
        if not a.train:                       # B0 零训练，不走 create_agent 的冻结分支
            print(f"       {arm}: 零训练臂，跳过装配检查")
            continue
        enc = encoder_of(create_agent(load_cfg(arm)))
        inj = getattr(enc, "code_injector", None)

        # ① 码本 / 注入层的有无
        check(f"{arm}: 注入层{'存在' if a.codes else '不存在'}",
              (inj is not None) == a.codes, f"code_injector={type(inj).__name__}")

        # ② 注入层是哪个类
        if a.codes:
            want = {"v1": CodeInjector, "v2": CodeInjectorV2}[a.injector]
            check(f"{arm}: 注入层类 == {want.__name__}（{a.injector}）",
                  isinstance(inj, want), f"实得 {type(inj).__name__}")
            _want_inj = expected_injector(arm)
            check(f"{arm}: 注入层参数 {_want_inj:,}"
                  + ("（细节码支路已关）" if not getattr(a, "use_detail", True) else "")
                  + ("（含相位嵌入）" if getattr(a, "use_phase", False) else ""),
                  inj.n_trainable() == _want_inj,
                  f"实得 {inj.n_trainable():,}")

        # ③ 冻结后可训参数量 —— 训练日志第一屏那个数
        got, want_n = frozen_trainable(enc), expected_trainable(arm)
        check(f"{arm}: 冻结后可训参数 {want_n:,}", got == want_n, f"实得 {got:,}")

    print("=== 2. B3 与 B4 必须能被这个数区分开 ===")
    check("B3 与 B4 的可训参数量不同（否则日志分不出训的是哪个臂）",
          expected_trainable("B3") != expected_trainable("B4"),
          f'B3={expected_trainable("B3"):,} B4={expected_trainable("B4"):,}')

    print("=== 3. 配置与臂不符时必须抛异常（不许静默）===")
    for arm, bad in (("B4", "v1"), ("B3", "v2")):
        try:
            create_agent(load_cfg(arm, injector_override=bad))
            check(f"{arm} 配 stage3.injector={bad} 必须抛异常", False, "居然装配成功了")
        except ValueError:
            check(f"{arm} 配 stage3.injector={bad} 必须抛异常", True)

    print("=== 4. 未知臂名必须抛异常 ===")
    try:
        create_agent(load_cfg("B9"))
        check("未知臂 B9 必须抛异常", False, "居然装配成功了")
    except ValueError:
        check("未知臂 B9 必须抛异常", True)

    print("\n臂装配门禁: " + ("PASS" if OK else "FAIL"))
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
