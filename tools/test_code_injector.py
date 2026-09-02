#!/usr/bin/env python
"""CodeInjector 单测（P4 Step 1 出口门禁）。

六项断言，每一项对应设计里的一条红线或一个易错点。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.code_injector import CodebookLookup, CodeInjector, load_codebook  # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def make(dim=128, code=512, slots=9, seed=0):
    torch.manual_seed(seed)
    inj = CodeInjector(dim=dim, code_dim=code, n_slots=slots).double()
    B, X = 3, 5                                   # 用小空间尺寸跑得快，与 20³ 等价
    latents = torch.randn(B, dim, X, X, X, dtype=torch.float64)
    z_g = torch.randn(B, code, dtype=torch.float64)
    Z_d = torch.randn(B, slots, code, dtype=torch.float64)
    return inj, latents, z_g, Z_d


def main() -> int:
    print("=== 1. 参数量 ===")
    inj, latents, z_g, Z_d = make()
    n = inj.n_trainable()
    check(f"参数量 {n:,} 落在 0.40M~0.46M", 0.40e6 < n < 0.46e6, f"实际 {n}")

    print("=== 2. code_mask=0 → 逐位恒等（红线 3）===")
    mask0 = torch.zeros(3, dtype=torch.float64)
    out = inj(latents, z_g, Z_d, mask0)
    check("torch.equal(out, latents)", torch.equal(out, latents),
          f"最大差 {(out-latents).abs().max().item():.3e}")

    print("=== 3. 零初始化下 mask=1 也恒等（红线 1：γ=β=0, gate=0）===")
    mask1 = torch.ones(3, dtype=torch.float64)
    out1 = inj(latents, z_g, Z_d, mask1)
    check("初始状态 torch.equal(out, latents)", torch.equal(out1, latents),
          f"最大差 {(out1-latents).abs().max().item():.3e}")

    print("=== 4. gate 有非零梯度（红线 2：w_o 不能与 gate 同时零初始化）===")
    inj2, lat2, zg2, zd2 = make(seed=1)
    o = inj2(lat2, zg2, zd2, torch.ones(3, dtype=torch.float64))
    o.sum().backward()
    g = inj2.gate.grad
    check("gate.grad 存在且非零", g is not None and float(g.abs().max()) > 0,
          f"grad max = {float(g.abs().max()) if g is not None else None}")
    wf = inj2.w_film.weight.grad
    check("w_film.weight.grad 非零（FiLM 支路可学）",
          wf is not None and float(wf.abs().max()) > 0)

    print("=== 5. batch 内 mask 混合 ===")
    inj3, lat3, zg3, zd3 = make(seed=2)
    with torch.no_grad():          # 打破零初始化，制造真实扰动
        inj3.w_film.weight.normal_(std=0.05)
        inj3.w_film.bias.normal_(std=0.05)
        inj3.gate.normal_(std=0.05)
    mm = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    out3 = inj3(lat3, zg3, zd3, mm)
    check("mask=0 的样本逐位恒等", torch.equal(out3[1], lat3[1]),
          f"最大差 {(out3[1]-lat3[1]).abs().max().item():.3e}")
    check("mask=1 的样本确实被改动", not torch.equal(out3[0], lat3[0]))

    print("=== 6. 码同时影响两条动作分支（守护注入点位置）===")
    # 模拟 PerAct 的两条读出：ss1/global_maxp（→旋转）与 up0（→平移）
    inj4, lat4, zg4, zd4 = make(seed=3)
    with torch.no_grad():
        inj4.w_film.weight.normal_(std=0.05)
        inj4.w_film.bias.normal_(std=0.05)
        inj4.gate.normal_(std=0.05)
    m1 = torch.ones(3, dtype=torch.float64)
    a = inj4(lat4, zg4, zd4, m1)
    b = inj4(lat4, zg4 * -1.0, zd4, m1)                # 换一组码
    rot_a = a.flatten(2).max(-1).values                # ≈ global_maxp → 旋转分支
    rot_b = b.flatten(2).max(-1).values
    trans_a = a.mean(1)                                # ≈ up0 前的空间张量 → 平移分支
    trans_b = b.mean(1)
    check("旋转分支读出随码变化", not torch.allclose(rot_a, rot_b))
    check("平移分支读出随码变化", not torch.allclose(trans_a, trans_b))

    print("=== 7. 码本查表（真实 checkpoint）===")
    cb_path = REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth"
    if cb_path.is_file():
        cb = load_codebook(str(cb_path))
        check(f"全局码本 {cb.n_global} 个（期望 36）", cb.n_global == 36)
        check(f"细节码本 {cb.n_detail} 个（期望 192）", cb.n_detail == 192)
        zg, zd = cb(torch.tensor([0, 35]), torch.tensor([[0] * 9, [191] * 9]))
        check("查表形状 z_g [2,512] / Z_d [2,9,512]",
              tuple(zg.shape) == (2, 512) and tuple(zd.shape) == (2, 9, 512),
              f"{tuple(zg.shape)} {tuple(zd.shape)}")
        try:
            cb(torch.tensor([36]), torch.tensor([[0] * 9]))
            check("越界索引应抛 IndexError", False)
        except IndexError:
            check("越界索引抛 IndexError", True)
    else:
        print(f"  [SKIP] 找不到 {cb_path}")

    print()
    print("Step 1 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
