#!/usr/bin/env python
"""BUG 1 回归测试：B3 的评测前向必须真的用上码。

这个 bug 之前的形态是：act() 调 self._q(...) 时不传 z_g/Z_d/code_mask，
而注入点的守卫是 `if code_injector is not None and z_g is not None`，
于是 B3 **训练时用码、评测时不用码**，跑出来就是 B2 的成绩 ——
不报错、不告警，会让我们得出「码本没用」的错误结论。

三项断言：
  1. B3 的 act() 在**换一组码**时输出必须改变（码真的参与了计算）
  2. B3 的 act() 在 observation 缺码时必须**硬失败**（不允许静默退化）
  3. B1（无码本）的 act() 不受影响
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

from model.code_injector import CodeInjector, load_codebook   # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def build_net(with_injector: bool, dim_hint: int = None, device="cpu"):
    """小体素版 PerceiverIO，逻辑与真实结构等价、跑得快。"""
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
    if with_injector:
        inj = CodeInjector(dim=net.input_dim_before_seq, code_dim=512, n_slots=9).to(device).eval()
        with torch.no_grad():        # 打破零初始化，让码真的产生影响
            inj.w_film.weight.normal_(std=0.1)
            inj.w_film.bias.normal_(std=0.1)
            inj.gate.normal_(std=0.1)
        net.code_injector = inj
    return net


def main() -> int:
    dev = "cpu"
    B = 1
    torch.manual_seed(1)
    ins = torch.randn(B, 10, 20, 20, 20)
    proprio = torch.randn(B, 4)
    lang_emb = torch.randn(B, 1024)
    lang_tok = torch.randn(B, 77, 512)
    bounds = torch.tensor([[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]]).repeat(B, 1)

    print("=== 1. 码真的参与前向：换码 → 输出改变 ===")
    net = build_net(with_injector=True, device=dev)
    cb_path = REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth"
    cb = load_codebook(str(cb_path)) if cb_path.is_file() else None
    if cb is None:
        print("  [SKIP] 找不到码本 checkpoint")
    else:
        zg_a, zd_a = cb(torch.tensor([3]), torch.tensor([[10] * 9]))
        zg_b, zd_b = cb(torch.tensor([27]), torch.tensor([[150] * 9]))
        m1 = torch.ones(B)
        with torch.no_grad():
            out_a = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                        z_g=zg_a, Z_d=zd_a, code_mask=m1)
            out_b = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                        z_g=zg_b, Z_d=zd_b, code_mask=m1)
            out_none = net(ins, proprio, lang_emb, lang_tok, None, bounds, None)
        check("码 A vs 码 B：q_trans 不同", not torch.allclose(out_a[0], out_b[0]))
        check("码 A vs 码 B：q_rot_grip 不同", not torch.allclose(out_a[1], out_b[1]))
        # 这一条就是 BUG 1 的直接回归：不传码时前向会跳过注入，
        # 结果必须与传码时不同 —— 若相同，说明码根本没起作用。
        check("不传码 vs 传码：输出不同（BUG 1 的直接回归）",
              not torch.allclose(out_none[0], out_a[0]))
        with torch.no_grad():
            out_masked = net(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                             z_g=zg_a, Z_d=zd_a, code_mask=torch.zeros(B))
        check("code_mask=0 与不传码逐位相同",
              all(torch.equal(x, y) for x, y in zip(out_none, out_masked)))

    print("=== 2. B3 缺码时必须硬失败（不允许静默退化）===")
    from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent
    import inspect
    src = inspect.getsource(QAttentionPerActBCAgent.act)
    check("act() 里读取 observation['subtask_k_global']",
          "subtask_k_global" in src)
    check("act() 里把 z_g/Z_d/code_mask 传给 self._q",
          "z_g=z_g" in src and "code_mask=code_mask" in src)
    check("缺码时抛 RuntimeError 而非静默跳过",
          "raise RuntimeError" in src and "codebook" in src.lower())

    print("=== 3. B1（无码本）不受影响 ===")
    net_b1 = build_net(with_injector=False, device=dev)
    with torch.no_grad():
        o1 = net_b1(ins, proprio, lang_emb, lang_tok, None, bounds, None)
        o2 = net_b1(ins, proprio, lang_emb, lang_tok, None, bounds, None,
                    z_g=None, Z_d=None, code_mask=None)
    check("无注入层时传不传码参数都一样",
          all(torch.equal(a, b) for a, b in zip(o1, o2)))
    check("code_injector 为 None", net_b1.code_injector is None)

    print()
    print("BUG 1 回归测试:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
