#!/usr/bin/env python
"""B4 阶段性体检 —— 在早期就判断「这一轮训下去还有没有意义」。

# 为什么需要

v1（B3）的失败**不会让任何断言失败**：注入幅度从 0.0056% 长到 0.0095%，
训练一路正常跑完 100000 步，成绩静静地等于不带码的基线。等到跑完评测才发现，
已经烧掉几十小时。B4 换成 v2 注入层就是去修这一点，但**同样可能悄悄退化** ——
优化器完全可以学着把 w_g / w_o 压向零，把 v2 退化成 v1 的形态。

本脚本只读已保存的 checkpoint，**纯 CPU、不占显卡**，可在训练进行中随时跑。

# 四项检查与判据（提前定死，避免事后找理由）

  ① 注入幅度  把 ckpt 里的注入层权重装回 CodeInjectorV2，前向量测
              ‖inject(latents) − latents‖ / ‖latents‖。
              初始 16.79%（tools/test_injector_v2.py 同口径同随机种子）。
                  ≥ 3%      OK    注入仍然实质存在
                  1% ~ 3%   WARN  在快速衰减，需盯住
                  < 1%      FAIL  正在退化成 v1 的形态（v1 实测 0.0095%）

  ② 权重在动  相邻 ckpt 之间的相对变化 ‖Δp‖/‖p‖。
              LAMB 下 ‖Δp‖ ≡ lr·‖p‖，2500 步的理论上限约 2500×1e-4 = 25%。
                  ≥ 1%      OK    在学
                  < 0.1%    FAIL  被锁死 —— 正是 v1 gate 的病

  ③ loss     与 B2 同步数的平滑 loss 对比。B4 起点带约 17% 扰动，早期高于 B2
              是预期的；关键看有没有收敛回去。
                  step ≥ 10000 时仍 > B2 的 1.5 倍  → WARN（没恢复过来）
                  出现 NaN / Inf                     → FAIL

  ④ 参数完整 ckpt 必须含 w_g、且**不含** gate / w_film。
              不满足 = 装成了 v1 或压根没挂注入层 → FAIL

# 用法

    source run/env.sh && python tools/b4_health.py                # 查 B4 全部 ckpt
    python tools/b4_health.py --arm B3                            # 同口径看 v1 做对照
    python tools/b4_health.py --step 2500 --step 5000             # 只查指定步
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                     # noqa: E402
from model.code_injector import (CodeInjector, CodeInjectorV2,   # noqa: E402
                                 load_codebook)
from stage3.arms import ARMS                                      # noqa: E402

CKPT_ROOT = REPO_ROOT / "checkpoints" / "stage3_main"
CODEBOOK = REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth"
PREFIX = "_qnet.module.code_injector."

# ---- 判据（改这里就等于改判据，别在别处另写一套）----
INJECT_OK, INJECT_WARN = 0.03, 0.01      # ① 注入幅度
MOVE_OK, MOVE_FAIL = 0.01, 0.001         # ② 相邻 ckpt 相对变化
LOSS_RATIO_WARN = 1.5                    # ③ loss 相对基线臂的倍数
LOSS_CHECK_FROM = 10000                  # ③ 从这一步起才判 loss

VERDICT_RANK = {"OK": 0, "WARN": 1, "FAIL": 2}


def ckpt_steps(arm: str) -> list[int]:
    wd = CKPT_ROOT / arm / "seed0" / "weights"
    if not wd.is_dir():
        return []
    return sorted(int(p.name) for p in wd.iterdir() if p.name.isdigit())


def injector_state(arm: str, step: int) -> dict[str, torch.Tensor]:
    f = CKPT_ROOT / arm / "seed0" / "weights" / str(step) / "QAttentionAgent_layer0.pt"
    sd = torch.load(f, map_location="cpu", weights_only=False)
    return {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}


def measure_injection(inj_sd: dict, kind: str, codebook) -> float:
    """把 ckpt 权重装回注入层，量测它把 latents 改变了百分之多少。

    随机种子与张量形状**必须**与 tools/test_injector_v2.py 一致，否则
    「初始 16.79%」这个参照系就不成立、跨 ckpt 的数也没法比。
    """
    torch.manual_seed(0)
    B, C, S, D = 2, 128, 20, 512
    lat = torch.randn(B, C, S, S, S)
    # 码用**真实码本**里的向量，不用随机数 —— 随机向量的范数与真实码相差很多，
    # 量出来的注入幅度会系统性偏离实际训练时的量级。
    g_idx = torch.tensor([0, 1]) % codebook.n_global
    d_idx = (torch.arange(B * 9) % codebook.n_detail).reshape(B, 9)
    z_g, Z_d = codebook(g_idx, d_idx)

    cls = CodeInjectorV2 if kind == "v2" else CodeInjector
    inj = cls(dim=C, code_dim=D).eval()
    missing, unexpected = inj.load_state_dict(inj_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"注入层权重对不上：缺 {missing} 多 {unexpected}")
    with torch.no_grad():
        out = inj(lat, z_g, Z_d, torch.ones(B, 1))
    return ((out - lat).norm() / lat.norm()).item()


def rel_change(a: dict, b: dict) -> float:
    """两个 ckpt 之间注入层权重的整体相对变化 ‖Δp‖/‖p‖。"""
    num = sum(((a[k] - b[k]).float() ** 2).sum() for k in a if k in b)
    den = sum((b[k].float() ** 2).sum() for k in a if k in b)
    return (num.sqrt() / den.sqrt()).item() if den > 0 else 0.0


def smoothed_loss(arm: str, step: int, window: int = 5) -> float | None:
    """train_data.csv 里 step 附近若干行的 total_loss 均值。

    单步 loss 噪声极大（batch=4/卡，实测在 3~17 之间跳），必须平滑后再比。
    """
    f = CKPT_ROOT / arm / "seed0" / "train_data.csv"
    if not f.is_file():
        return None
    col = "QAttentionAgent_layer0/losses/total_loss"
    rows = []
    with f.open() as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((int(r["step"]), float(r[col])))
            except (KeyError, ValueError, TypeError):
                continue
    if not rows:
        return None
    rows.sort()
    near = sorted(rows, key=lambda t: abs(t[0] - step))[:window]
    return sum(v for _, v in near) / len(near)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B4")
    ap.add_argument("--baseline", default="B2",
                    help="loss 对照臂（同样是子任务指令、同样训练量）")
    ap.add_argument("--step", type=int, action="append",
                    help="只查这些步；不给就查全部 ckpt")
    a = ap.parse_args()

    arm = a.arm
    if arm not in ARMS:
        print(f"❌ 未知臂 {arm}；已定义 {sorted(ARMS)}", file=sys.stderr)
        return 2
    if not ARMS[arm].codes:
        print(f"❌ {arm} 不启用码注入，没有可体检的注入层", file=sys.stderr)
        return 2
    kind = ARMS[arm].injector

    steps = [s for s in ckpt_steps(arm) if s > 0]
    if a.step:
        steps = [s for s in steps if s in set(a.step)]
    if not steps:
        print(f"⏳ {arm} 还没有 step>0 的 checkpoint（save_freq=2500）。"
              f"现有：{ckpt_steps(arm) or '无'}")
        return 0

    codebook = load_codebook(str(CODEBOOK))
    print(f"=== {arm} 注入层体检（{kind}）· 纯 CPU · {len(steps)} 个 ckpt ===")
    print(f"    判据：注入幅度 ≥{INJECT_OK:.0%} OK / <{INJECT_WARN:.0%} FAIL；"
          f"相邻变化 ≥{MOVE_OK:.0%} OK / <{MOVE_FAIL:.1%} FAIL")
    print()
    print(f"    {'step':>7}  {'注入幅度':>9}  {'相邻变化':>9}  "
          f"{'loss':>7}  {'基线loss':>8}  判定")

    worst, prev = "OK", None
    for s in steps:
        sd = injector_state(arm, s)
        notes = []

        # ④ 参数完整性
        has_v2 = "w_g.weight" in sd
        has_v1 = "gate" in sd or "w_film.weight" in sd
        want_v2 = kind == "v2"
        if not sd:
            notes.append("ckpt 里没有 code_injector.*（注入层压根没挂）")
        elif want_v2 and (not has_v2 or has_v1):
            notes.append("ckpt 里是 v1 的参数名（装错版本）")
        elif not want_v2 and not has_v1:
            notes.append("ckpt 里不是 v1 的参数名")

        if notes:
            print(f"    {s:>7}  {'—':>9}  {'—':>9}  {'—':>7}  {'—':>8}  FAIL")
            for n in notes:
                print(f"             ❌ {n}")
            worst = "FAIL"
            prev = sd
            continue

        # ① 注入幅度
        amp = measure_injection(sd, kind, codebook)
        v = ("OK" if amp >= INJECT_OK else
             "WARN" if amp >= INJECT_WARN else "FAIL")
        if v != "OK":
            notes.append(f"注入幅度 {amp:.2%} —— "
                         + ("在快速衰减，盯住下一个 ckpt"
                            if v == "WARN" else
                            "已接近 v1 的形态（v1 实测 0.0095%），继续训无意义"))

        # ② 权重在动
        mv = rel_change(sd, prev) if prev is not None else float("nan")
        if prev is not None:
            mv_v = ("OK" if mv >= MOVE_OK else
                    "WARN" if mv >= MOVE_FAIL else "FAIL")
            if mv_v != "OK":
                notes.append(f"相邻 ckpt 相对变化仅 {mv:.3%} —— "
                             f"注入层几乎不动，疑似被优化器锁死")
            if VERDICT_RANK[mv_v] > VERDICT_RANK[v]:
                v = mv_v

        # ③ loss
        lo = smoothed_loss(arm, s)
        lb = smoothed_loss(a.baseline, s)
        if lo is not None and (lo != lo or lo in (float("inf"), float("-inf"))):
            notes.append("loss 出现 NaN/Inf")
            v = "FAIL"
        elif (lo and lb and s >= LOSS_CHECK_FROM
              and lo > lb * LOSS_RATIO_WARN):
            notes.append(f"loss {lo:.2f} 是 {a.baseline} 的 {lo/lb:.2f} 倍 —— "
                         f"起点扰动没恢复过来")
            if v == "OK":
                v = "WARN"

        print(f"    {s:>7}  {amp:>8.2%}  "
              + (f"{mv:>8.3%}" if prev is not None else f"{'—':>9}")
              + f"  {lo:>7.2f}" if lo is not None else f"  {'—':>7}",
              end="")
        print(f"  {lb:>8.2f}" if lb is not None else f"  {'—':>8}", end="")
        print(f"  {v}")
        for n in notes:
            print(f"             {'⚠️' if v == 'WARN' else '❌'} {n}")

        if VERDICT_RANK[v] > VERDICT_RANK[worst]:
            worst = v
        prev = sd

    print()
    print(f"    参照：v2 初始注入 16.79%（未训练） · v1 训练 100000 步后 0.0095%")
    print(f"\n{arm} 体检结论: {worst}")
    if worst == "FAIL":
        print("  → 建议停训。注入层已退化，继续训只是重演 B3。")
    elif worst == "WARN":
        print("  → 继续训，但下一个 ckpt 必须复查；若继续下滑就停。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
