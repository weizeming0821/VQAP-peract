#!/usr/bin/env python
"""语言通路解冻档位（lang_tier）的门禁。

四条必须成立，任一条不成立都会导致「训出来的不是你以为的那个配方」，
而且都不会自己报错：

  1. 各档位的可训模块与参数量恰如预期
  2. 旁路末层零初始化 ⇒ L3/L4 的**前向输出与 L0 逐位相同**
     （否则「训练起点与原模型等价」这条红线就破了）
  3. 官方 ckpt 的 `lang_preprocess.*` 键在高档位下仍然命中
     （旁路方案存在的全部理由；原地替换会让它静默保持随机初始化）
  4. 高档位 ckpt 装进低档位模型 -> 硬失败，而不是静默丢参数

    source run/env.sh && python tools/test_lang_tier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage3 import lang_tier as lt                                  # noqa: E402
from agents.peract_bc.perceiver_lang_io import (                    # noqa: E402
    PerceiverVoxelLangEncoder,
)
from agents.peract_bc.qattention_peract_bc_agent import (           # noqa: E402
    STAGE3_TRAINABLE_MODULES,
)

FAILED: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   {extra}"))
    if not cond:
        FAILED.append(name)


def build(tier: str) -> PerceiverVoxelLangEncoder:
    """按官方 conf 的关键取值造一个 encoder（只为数参数与比前向）。"""
    return PerceiverVoxelLangEncoder(
        depth=6, iterations=1, voxel_size=100, initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4, layer=0, num_rotation_classes=72, num_grip_classes=2,
        num_collision_classes=2, input_axis=3, num_latents=2048, latent_dim=512,
        cross_heads=1, latent_heads=8, cross_dim_head=64, latent_dim_head=64,
        weight_tie_layers=False, activation='lrelu',
        pos_encoding_with_lang=False, input_dropout=0.1, attn_dropout=0.1,
        decoder_dropout=0.0, lang_fusion_type='seq', voxel_patch_size=5,
        voxel_patch_stride=5, no_skip_connection=False, no_perceiver=False,
        no_language=False, final_dim=64, lang_tier=tier,
    )


def trainable_count(net, tier: str) -> int:
    allow = lt.trainable_modules(tier, STAGE3_TRAINABLE_MODULES)
    return sum(p.numel() for n, p in net.named_parameters()
               if any(f'.{m}.' in f'.{n}.' for m in allow))


def main() -> int:
    print("=" * 74)
    print("① 各档位的可训参数量")
    print("=" * 74)
    nets = {}
    counts = {}
    for tier in ("L0", "L3", "L4"):
        net = build(tier)
        nets[tier] = net
        counts[tier] = trainable_count(net, tier)
        lo, hi = lt.TIER_PARAM_RANGE[tier]
        print(f"     {tier}: 可训 {counts[tier]:,}")
        check(f"{tier} 落在预期区间 ({lo:,.0f}, {hi:,.0f})",
              lo < counts[tier] < hi)

    check("L0 == 2,117,917（与设计文档记录一致）", counts["L0"] == 2_117_917,
          f"实测 {counts['L0']:,}")
    check("L3 − L0 == 247,552（旁路 164,224 + decoder_cross_attn 83,328）",
          counts["L3"] - counts["L0"] == 247_552,
          f"实测 {counts['L3'] - counts['L0']:,}")
    check("L4 − L3 == 3,235,072（cross_attend_blocks）",
          counts["L4"] - counts["L3"] == 3_235_072,
          f"实测 {counts['L4'] - counts['L3']:,}")

    print("\n" + "=" * 74)
    print("② 旁路零初始化 ⇒ 前向与 L0 逐位相同")
    print("=" * 74)
    x = torch.randn(2, 77, 512, dtype=torch.float32)
    with torch.no_grad():
        base = nets["L0"]._lang_pre(x)
        for tier in ("L3", "L4"):
            # 把 lang_preprocess 的权重对齐到 L0，只留旁路作为差异来源
            nets[tier].lang_preprocess.load_state_dict(
                nets["L0"].lang_preprocess.state_dict())
            got = nets[tier]._lang_pre(x)
            check(f"{tier} 的 _lang_pre 输出与 L0 逐位相同",
                  torch.equal(got, base),
                  f"max|Δ|={float((got - base).abs().max()):.3e}")
        check("L0 没有旁路模块", nets["L0"].lang_preprocess_delta is None)
        check("L4 有旁路模块", nets["L4"].lang_preprocess_delta is not None)

    print("\n" + "=" * 74)
    print("③ 官方 ckpt 的 lang_preprocess.* 键在高档位下仍然命中")
    print("=" * 74)
    keys = set(nets["L4"].state_dict())
    check("lang_preprocess.weight 仍在（未被改名成 .base.weight）",
          "lang_preprocess.weight" in keys)
    check("lang_preprocess.bias 仍在", "lang_preprocess.bias" in keys)
    check("旁路的键是独立前缀 lang_preprocess_delta.*",
          any(k.startswith("lang_preprocess_delta.") for k in keys))
    official = (PERACT_ROOT / "ckpts" / "multi" / "PERACT_BC" / "seed0" /
                "weights" / "600000" / "QAttentionAgent_layer0.pt")
    if official.is_file():
        sd = torch.load(official, map_location="cpu", weights_only=True)
        want = {k.replace("_qnet.module.", "") for k in sd
                if "lang_preprocess" in k}
        check(f"官方 ckpt 的语言键 {sorted(want)} 全部能对上",
              want and want <= keys, f"缺 {sorted(want - keys)}")
    else:
        print(f"  ⚠️  官方 ckpt 不在（{official}），跳过这一项")

    print("\n" + "=" * 74)
    print("④ 高档位 ckpt 装进低档位模型 -> 硬失败")
    print("=" * 74)
    orphan = ["_qnet.lang_preprocess_delta.0.weight",
              "_qnet.cross_attend_blocks.0.fn.to_q.weight"]
    try:
        lt.assert_ckpt_tier_compatible("L0", orphan)
        check("L0 模型遇到 L4 的键时抛异常", False, "没有抛")
    except RuntimeError as exc:
        check("L0 模型遇到 L4 的键时抛异常", True)
        print(f"       {str(exc)[:96]}…")
    try:
        lt.assert_ckpt_tier_compatible("L4", [])
        check("键齐全时不抛", True)
    except Exception as exc:                                  # noqa: BLE001
        check("键齐全时不抛", False, str(exc)[:60])
    try:
        lt.normalize("L9")
        check("未知档位名当场抛", False, "没有抛")
    except ValueError:
        check("未知档位名当场抛", True)

    print("\n" + "=" * 74)
    print(("通过" if not FAILED else f"失败 {len(FAILED)} 项: {FAILED}"))
    print("=" * 74)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
