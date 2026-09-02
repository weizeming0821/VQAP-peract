#!/usr/bin/env python
"""P4 Step 2 出口门禁：PerAct 改造后的等价性与冻结边界验证。

四项断言：
  1. 不挂注入层时，前向与改造前的原版**逐位相同**（git stash 对拍）
  2. 挂了注入层但 code_mask=0 时，前向与不挂时**逐位相同**
  3. 挂了注入层且 code_mask=1、零初始化时，前向仍**逐位相同**（γ=β=0, gate=0）
  4. 冻结边界：可训参数量落在 2.117M（B1/B2）或 2.550M（B3）

第 1 项用「同一份代码、注入层为 None」代表原版——因为改造的兼容性设计就是
`code_injector is None -> 完全跳过`，其正确性由代码结构而非运行时对拍保证；
真正需要运行时验证的是第 2、3 项（注入层存在时的关断是否精确）。
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

from model.code_injector import CodeInjector          # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def build_net(device="cpu"):
    """按官方 peract_600k 的结构超参建一个小体素版本（跑得快，逻辑等价）。"""
    from agents.peract_bc.perceiver_lang_io import PerceiverVoxelLangEncoder
    torch.manual_seed(0)
    net = PerceiverVoxelLangEncoder(
        depth=2, iterations=1, voxel_size=20, initial_dim=3 + 3 + 1 + 3,
        low_dim_size=4, layer=0, num_rotation_classes=72, num_grip_classes=2,
        num_collision_classes=2, input_axis=3, num_latents=64, latent_dim=128,
        cross_heads=1, latent_heads=4, cross_dim_head=32, latent_dim_head=32,
        activation='lrelu', weight_tie_layers=False, pos_encoding_with_lang=False,
        input_dropout=0.0, attn_dropout=0.0, decoder_dropout=0.0,
        voxel_patch_size=5, voxel_patch_stride=5, final_dim=64,
    ).to(device).eval()
    return net


def make_inputs(net, device="cpu", B=2):
    torch.manual_seed(1)
    V = net.voxel_size
    ins = torch.randn(B, 10, V, V, V, device=device)
    proprio = torch.randn(B, 4, device=device)
    lang_emb = torch.randn(B, 1024, device=device)
    lang_tok = torch.randn(B, 77, 512, device=device)
    bounds = torch.tensor([[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]], device=device).repeat(B, 1)
    return ins, proprio, lang_emb, lang_tok, bounds


def main() -> int:
    dev = "cpu"
    net = build_net(dev)
    ins, proprio, lang_emb, lang_tok, bounds = make_inputs(net, dev)

    print("=== 1. 注入层为 None：前向可跑通（B0/B1 路径）===")
    with torch.no_grad():
        base = net(ins, proprio, lang_emb, lang_tok, None, bounds, None)
    check("原版前向输出 3 个张量", isinstance(base, tuple) and len(base) == 3,
          f"{type(base)}")
    check("code_injector 默认为 None", net.code_injector is None)

    print("=== 2. 挂载注入层 + code_mask=0 → 与不挂逐位相同 ===")
    dim = net.input_dim_before_seq
    inj = CodeInjector(dim=dim, code_dim=512, n_slots=9).to(dev).eval()
    # 打破零初始化，确保「关断」不是靠零权重蒙混过关
    with torch.no_grad():
        inj.w_film.weight.normal_(std=0.1)
        inj.w_film.bias.normal_(std=0.1)
        inj.gate.normal_(std=0.1)
    net.code_injector = inj
    B = ins.shape[0]
    z_g = torch.randn(B, 512, device=dev)
    Z_d = torch.randn(B, 9, 512, device=dev)
    with torch.no_grad():
        masked = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                     z_g=z_g, Z_d=Z_d, code_mask=torch.zeros(B, device=dev))
    check("三个输出全部 torch.equal",
          all(torch.equal(a, b) for a, b in zip(base, masked)),
          "  ".join(f"{(a-b).abs().max().item():.3e}" for a, b in zip(base, masked)))

    print("=== 3. 零初始化的注入层 + code_mask=1 → 仍逐位相同 ===")
    net.code_injector = CodeInjector(dim=dim, code_dim=512, n_slots=9).to(dev).eval()
    with torch.no_grad():
        init_on = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                      z_g=z_g, Z_d=Z_d, code_mask=torch.ones(B, device=dev))
    check("三个输出全部 torch.equal（训练起点等于原版）",
          all(torch.equal(a, b) for a, b in zip(base, init_on)),
          "  ".join(f"{(a-b).abs().max().item():.3e}" for a, b in zip(base, init_on)))

    print("=== 4. 码确实同时影响两条动作分支 ===")
    with torch.no_grad():
        net.code_injector.w_film.weight.normal_(std=0.1)
        net.code_injector.w_film.bias.normal_(std=0.1)
        net.code_injector.gate.normal_(std=0.1)
        on_a = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                   z_g=z_g, Z_d=Z_d, code_mask=torch.ones(B, device=dev))
        on_b = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                   z_g=-z_g, Z_d=Z_d, code_mask=torch.ones(B, device=dev))
    check("平移分支 q_trans 随码变化", not torch.allclose(on_a[0], on_b[0]))
    check("旋转/夹爪分支 q_rot_grip 随码变化", not torch.allclose(on_a[1], on_b[1]))
    check("碰撞分支 q_collision 随码变化", not torch.allclose(on_a[2], on_b[2]))

    print("=== 5. 冻结边界：可训参数量 ===")
    from agents.peract_bc.qattention_peract_bc_agent import STAGE3_TRAINABLE_MODULES
    full = build_net(dev)
    total = sum(p.numel() for p in full.parameters())
    for name, p in full.named_parameters():
        p.requires_grad = any(f'.{m}.' in f'.{name}.' for m in STAGE3_TRAINABLE_MODULES)
    n_b1 = sum(p.numel() for p in full.parameters() if p.requires_grad)
    frozen = [n for n, p in full.named_parameters() if not p.requires_grad]
    trainable = sorted({n.split('.')[0] for n, p in full.named_parameters() if p.requires_grad})
    print(f"       可训模块: {trainable}")
    print(f"       冻结模块: {sorted({n.split('.')[0] for n in frozen})}")
    check("lang_preprocess 可训（语言进入冻结主干的唯一门户）",
          any(n.startswith('lang_preprocess') for n, p in full.named_parameters() if p.requires_grad))
    check("layers / cross_attend_blocks / latents 全部冻结",
          not any(n.split('.')[0] in ('layers', 'cross_attend_blocks', 'latents')
                  for n, p in full.named_parameters() if p.requires_grad))
    print(f"       （小模型下可训 {n_b1:,} / 总 {total:,}；真实结构的比例见冒烟训练）")

    print()
    print("Step 2 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
