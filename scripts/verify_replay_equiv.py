#!/usr/bin/env python
"""逐样本逐字段比对两份 replay 工件中同一个任务的样本 —— Step 1a「环境等价性」的判据。

# 为什么需要这个脚本

整仓从旧机迁到新机时，torch 2.4.1+cu121 → 2.8.0+cu128、GPU 换成 sm_120。
若新环境产出的样本与旧环境**不完全一致**，而我们又只重建了一部分任务，
replay 里就会混进两套不同规则生成的数据 —— 这种错误不报任何异常，
只会让训练结果悄悄失真。

做法：用**新环境**把一个**已存在于旧工件中**的任务单独重建一遍，
与旧文件逐字段比对。全等（嵌入字段在容差内）⇒ 新环境 == 旧环境。

# 对齐方式

任务内的 cursor 由单个 fill 进程顺序分配，所以把两边的 `task_idxs[task]`
各自升序排列后，第 k 个对第 k 个即为同源样本。
（跨任务的 cursor 是多进程抢占分配的，全量重建两次也不会一样，
  所以绝对 cursor 值本身不可比 —— 只有任务内序号可比。）

用法：
    python scripts/verify_replay_equiv.py \
        --old aavla_data/replay/seen12/multi/PERACT_BC/seed0 \
        --new /tmp/replay_g1/open_drawer/PERACT_BC/seed0 \
        --task open_drawer
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

#: 允许存在微小数值差异的字段：CLIP RN50 **以 fp16 前向**，在不同 GPU 架构 /
#: torch 版本上最后一两位不保证一致。字段本身存成 <f4，但精度只有 fp16。
EMB_FIELDS = {
    "lang_goal_emb", "lang_token_embs",
    "subtask_lang_goal_emb", "subtask_lang_token_embs",
}
#: 方向容差。实测新旧环境 1-cos ≈ 2.5e-6 ~ 4.6e-6，取 1e-5 留 2 倍余量。
COS_MIN = 1.0 - 1e-5
#: 绝对容差按 **fp16 ULP** 度量 —— 用固定的 1e-5 之类是错的：
#: 量级 25 处 fp16 的 1 ULP 就是 1.56e-2，任何 fp32 尺度的阈值都不可能满足。
#: 实测差异为 0.4 ~ 2.5 ULP，取 4 ULP。
ULP_MAX = 4.0

#: 未初始化的填充字段：`add_final` 帧不写 reward，落盘的是未初始化内存。
#: 这些帧 terminal = -1，被 `is_valid_transition` 永久排除，训练中永不被采样，
#: 所以两边不同不构成问题 —— 但**必须**验证差异只出现在这些帧上。
PAD_FIELDS = {"reward"}


def load_index(d: Path) -> dict:
    with open(d / "_INDEX.pkl", "rb") as f:
        return pickle.load(f)


def cmp_field(name, a, b) -> tuple[bool, str]:
    """返回 (是否通过, 描述)。"""
    a = np.asarray(a) if not isinstance(a, np.ndarray) else a
    b = np.asarray(b) if not isinstance(b, np.ndarray) else b
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        return False, f"dtype {a.dtype} vs {b.dtype}"
    if a.dtype == object or a.dtype.kind in "USO":
        eq = bool((a == b).all())
        return eq, "" if eq else "字符串/对象字段不等"
    if np.array_equal(a, b):
        return True, ""
    if name not in EMB_FIELDS:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        return False, f"逐位不等，max|Δ|={d.max():.3e}"
    # 嵌入字段：方向看 cos，幅度按 fp16 ULP 度量
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    cos = float(np.dot(x, y) / denom) if denom > 0 else 1.0
    amax = float(np.abs(x - y).max())
    scale = float(np.abs(x).max())
    ulp = float(np.spacing(np.float16(scale))) if scale > 0 else 1.0
    n_ulp = amax / ulp if ulp > 0 else 0.0
    ok = cos >= COS_MIN and n_ulp <= ULP_MAX
    return ok, f"1-cos={1-cos:.2e} max|Δ|={amax:.2e} ({n_ulp:.1f} fp16 ULP)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只比前 N 个样本（0=全部）")
    ap.add_argument("--partial", action="store_true",
                    help="前哨模式：新工件样本数少于旧工件时，只比对旧工件的前 N 条")
    a = ap.parse_args()

    old, new = Path(a.old), Path(a.new)
    io, inx = load_index(old), load_index(new)
    co = sorted(int(c) for c in io["task_idxs"].get(a.task, []))
    cn = sorted(int(c) for c in inx["task_idxs"].get(a.task, []))

    print(f"任务 {a.task}：旧 {len(co):,} 样本 / 新 {len(cn):,} 样本")
    if len(co) != len(cn):
        if a.partial and len(cn) < len(co):
            # 前哨模式：新工件只建了前几条 demo。fill_replay 按 d_idx 顺序写，
            # 所以新工件的 N 条对应旧工件的前 N 条。
            print(f"[partial] 前哨模式：只比对旧工件的前 {len(cn):,} 条")
            co = co[:len(cn)]
        else:
            print("\n❌ 样本数不同 —— 说明关键帧发现或 cache 过滤在新环境下变了。"
                  "（若新工件是少量 demo 的前哨构建，请加 --partial）", file=sys.stderr)
            return 2
    if not co:
        print(f"\n❌ 任务 {a.task} 在工件中不存在。", file=sys.stderr)
        return 2

    n = len(co) if a.limit <= 0 else min(a.limit, len(co))
    # add_final 填充帧（terminal = -1）：PAD_FIELDS 在这些帧上允许不同
    term_o, term_n = np.asarray(io["terminal"]), np.asarray(inx["terminal"])
    # 逐字段统计：通过数 / 容差内通过数 / 失败样例
    stats: dict[str, dict] = {}
    bad_samples = 0
    pad_exempt = 0
    for k in range(n):
        with open(old / f"{co[k]}.replay", "rb") as f:
            so = pickle.load(f)
        with open(new / f"{cn[k]}.replay", "rb") as f:
            sn = pickle.load(f)
        if set(so) != set(sn):
            print(f"\n❌ 第 {k} 个样本字段集合不同："
                  f"旧多 {sorted(set(so)-set(sn))[:5]} / 新多 {sorted(set(sn)-set(so))[:5]}",
                  file=sys.stderr)
            return 3
        sample_bad = False
        is_pad = int(term_o[co[k]]) == -1
        if is_pad != (int(term_n[cn[k]]) == -1):
            print(f"\n❌ 第 {k} 个样本的 terminal 语义不同："
                  f"旧={int(term_o[co[k]])} 新={int(term_n[cn[k]])}", file=sys.stderr)
            return 3
        for f_ in sorted(so):
            st = stats.setdefault(f_, {"exact": 0, "tol": 0, "fail": 0, "pad": 0, "worst": ""})
            ok, msg = cmp_field(f_, so[f_], sn[f_])
            if not ok and f_ in PAD_FIELDS and is_pad:
                # add_final 帧不写这些字段，落盘的是未初始化内存；
                # 这些帧 terminal=-1，永远采不到，差异无意义。
                st["pad"] += 1
                pad_exempt += 1
                continue
            if ok and not msg:
                st["exact"] += 1
            elif ok:
                st["tol"] += 1
                st["worst"] = msg if not st["worst"] else st["worst"]
            else:
                st["fail"] += 1
                sample_bad = True
                if st["fail"] <= 1:
                    st["worst"] = f"样本#{k}: {msg}"
        bad_samples += sample_bad
        if (k + 1) % 500 == 0:
            print(f"  已比对 {k+1:,}/{n:,} …", flush=True)

    print(f"\n{'字段':34s} {'逐位相等':>9s} {'容差内':>7s} {'填充豁免':>9s} {'失败':>6s}  备注")
    print("-" * 116)
    failed = []
    for f_, st in sorted(stats.items()):
        flag = "" if st["fail"] == 0 else "  ❌"
        print(f"{f_:34s} {st['exact']:>9,} {st['tol']:>7,} {st['pad']:>9,} "
              f"{st['fail']:>6,}  {st['worst'][:40]}{flag}")
        if st["fail"]:
            failed.append(f_)

    print()
    if failed:
        print(f"❌ 不等价：{len(failed)} 个字段有失败样本 —— {failed}", file=sys.stderr)
        print(f"   共 {bad_samples:,}/{n:,} 个样本存在差异。", file=sys.stderr)
        return 1
    tol_only = [f for f, s in stats.items() if s["tol"]]
    print(f"✅ 等价：{n:,} 个样本、{len(stats)} 个字段全部通过。")
    if pad_exempt:
        pads = [f for f, s in stats.items() if s["pad"]]
        print(f"   {pads} 在 {pad_exempt:,} 个 add_final 填充帧（terminal=-1）上不同 —— "
              f"已逐帧核实差异**只**出现在这些永不被采样的帧上。")
    if tol_only:
        print(f"   其中 {tol_only} 走了容差（1-cos ≤ {1-COS_MIN:.0e} 且 ≤ {ULP_MAX:.0f} 个 "
              f"fp16 ULP）—— CLIP RN50 以 fp16 前向，跨 GPU 架构最后一两位不保证一致。"
              f"须在新 manifest 中记录迁移备注。")
    else:
        print("   全部逐位相等，无需迁移备注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
