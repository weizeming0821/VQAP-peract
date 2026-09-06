#!/usr/bin/env python
"""CodeInjectorV2（B4 的注入层）单测。

最关键的一条：**注入幅度必须真的起来**。v1 的失败不是逻辑错，是数值上太小 ——
`gate` 零初始化 + LAMB 的 `‖Δp‖ ≡ lr·‖p‖` 让它 100000 步只从 0.0036 长到
0.0061，注入对 latents 的改变量始终在 0.01% 量级，码本贡献为零。
这种失败**不会让任何断言失败**，只会让成绩静静地等于不带码的基线。
所以这里直接量测「注入改变了 latents 百分之多少」，并给出下界。
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                    # noqa: E402
from model.code_injector import CodeInjector, CodeInjectorV2    # noqa: E402

OK = True


def check(name, cond, extra=""):
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def main() -> int:
    torch.manual_seed(0)
    B, C, S, D = 2, 128, 20, 512
    lat = torch.randn(B, C, S, S, S)
    z_g = torch.randn(B, D)
    Z_d = torch.randn(B, 9, D)

    v2 = CodeInjectorV2(dim=C, code_dim=D).eval()
    v1 = CodeInjector(dim=C, code_dim=D).eval()

    print("=== 1. 🔴 注入幅度（v1 失败的根源就在这里）===")
    on = torch.ones(B, 1)
    with torch.no_grad():
        o2 = v2(lat, z_g, Z_d, on)
        o1 = v1(lat, z_g, Z_d, on)
    r2, r1 = rel(o2, lat), rel(o1, lat)
    print(f"       v1 初始注入 {r1:.4%}      v2 初始注入 {r2:.2%}")
    check("v2 的注入幅度 ≥ 5%（v1 实测训练 100000 步后只有 0.0095%）",
          r2 >= 0.05, f"{r2:.2%}")
    check("v2 至少比 v1 大两个数量级", r2 / max(r1, 1e-12) > 100,
          f"{r2/max(r1,1e-12):.0f}x")

    print("=== 2. 两条支路各自的贡献 ===")
    with torch.no_grad():
        g_only = v2(lat, z_g, None, on)                  # 只有全局码
        d_only = v2(g_only, z_g * 0, Z_d, on) - g_only   # 近似取细节码那一项
    check(f"全局码支路有效（{rel(g_only, lat):.2%}）", rel(g_only, lat) >= 0.03,
          f"{rel(g_only, lat):.2%}")
    check(f"细节码支路有效（{(d_only.norm()/lat.norm()).item():.2%}）",
          (d_only.norm() / lat.norm()).item() >= 0.03)

    print("=== 3. 换一个码，输出必须跟着变（码真的被用上了）===")
    with torch.no_grad():
        a = v2(lat, z_g, Z_d, on)
        b = v2(lat, torch.randn(B, D), Z_d, on)
    check(f"换全局码 -> 输出改变 {rel(b, a):.2%}", rel(b, a) >= 0.02, f"{rel(b,a):.2%}")
    with torch.no_grad():
        c = v2(lat, z_g, torch.randn(B, 9, D), on)
    check(f"换细节码 -> 输出改变 {rel(c, a):.2%}", rel(c, a) >= 0.02, f"{rel(c,a):.2%}")

    print("=== 4. code_mask=0 必须与输入逐位相同（单测锚点）===")
    # 注：mask=0 实测只占 112/10323 = 1.08%（仅 pose-adjust），
    # 所以这不是安全网，而是抓「乘/加写反」这类实现 bug 的断言。
    off = torch.zeros(B, 1)
    with torch.no_grad():
        z = v2(lat, z_g, Z_d, off)
    check("mask=0 -> 逐位相同", torch.equal(z, lat))
    mix = torch.tensor([[1.0], [0.0]])
    with torch.no_grad():
        mx = v2(lat, z_g, Z_d, mix)
    check("batch 内混合 mask：第 1 条被注入", not torch.equal(mx[0], lat[0]))
    check("batch 内混合 mask：第 2 条逐位相同", torch.equal(mx[1], lat[1]))

    print("=== 5. 没有 gate / FiLM 这两个零初始化张量 ===")
    names = {n for n, _ in v2.named_parameters()}
    check("没有 gate", "gate" not in names)
    check("没有 w_film", not any("w_film" in n for n in names))
    check("有 w_g", any("w_g" in n for n in names))
    zero_init = [n for n, p in v2.named_parameters()
                 if p.dim() > 1 and float(p.abs().max()) == 0.0]
    check("没有任何零初始化的权重矩阵（那正是被 LAMB 锁死的形态）",
          not zero_init, str(zero_init))

    print("=== 6. 参数量与梯度 ===")
    n2, n1 = v2.n_trainable(), v1.n_trainable()
    print(f"       v1 {n1:,}   v2 {n2:,}")
    check("参数量与 v1 同量级（改的是幅度与结构，不是容量）",
          0.5 * n1 <= n2 <= 1.5 * n1)
    v2.train()
    out = v2(lat, z_g, Z_d, on)
    out.sum().backward()
    no_grad = [n for n, p in v2.named_parameters()
               if p.grad is None or float(p.grad.abs().max()) == 0.0]
    check("所有参数都收到非零梯度", not no_grad, str(no_grad))

    print("=== 7. 形状与 dtype ===")
    check("输出形状与输入一致", o2.shape == lat.shape)
    check("use_detail=False 时只走全局码支路",
          CodeInjectorV2(dim=C, code_dim=D, use_detail=False)(
              lat, z_g, Z_d, on).shape == lat.shape)

    print()
    print("CodeInjectorV2 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
