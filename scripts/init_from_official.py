#!/usr/bin/env python
"""BUG 6 修复：把官方 peract_600k 权重装成某个实验臂的 iteration-0 检查点。

PerAct 没有「从外部 checkpoint 初始化」的机制 —— `framework.load_existing_weights`
只会去**自己的** weightsdir 里找。所以要从 peract_600k 起步，就把官方权重复制成
`<logdir>/<arm>/seed0/weights/0/QAttentionAgent_layer0.pt`，训练启动时
OfflineTrainRunner 会把它当作「已有的 iteration 0」加载，并从 iteration 1 继续。

两个前提已实测成立：
  * 官方 ckpt 的键是 `_qnet.module.*`（127 个），与 DDP 包装后的训练模型格式一致；
  * agent.load_weights 是**合并式**加载 —— 模型里有、ckpt 里没有的键保持原值，
    所以 B3 的 code_injector.* 会保持零初始化，正是所需。

用法：
    source run/env.sh
    python scripts/init_from_official.py --arm B1
    python scripts/init_from_official.py --arm B1 --arm B2 --arm B3
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OFFICIAL = (REPO_ROOT / "source" / "peract" / "ckpts" / "multi" / "PERACT_BC" /
            "seed0" / "weights" / "600000" / "QAttentionAgent_layer0.pt")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_logdir() -> str:
    """训练配置里的 framework.logdir —— 权重必须装进它，否则训练找不到。"""
    conf = REPO_ROOT / "source" / "peract" / "conf" / "stage3.yaml"
    try:
        import yaml
        d = yaml.safe_load(conf.read_text())
        v = d["framework"]["logdir"]
        if v:
            return str(v)
    except Exception as exc:                      # 配置读不到就退回约定路径
        print(f"  ⚠️ 未能从 {conf} 读出 framework.logdir（{exc}），"
              f"退回默认 checkpoints/stage3_main", file=sys.stderr)
    return str(REPO_ROOT / "checkpoints" / "stage3_main")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="可重复，如 --arm B1 --arm B3")
    # 🔴 默认值直接从 conf/stage3.yaml 的 framework.logdir 读，不写死。
    #    原来写死成 REPO_ROOT/"logs"/"stage3" —— 与训练真正使用的
    #    checkpoints/stage3_main 不是同一个地方，漏传 --logdir 就会把权重
    #    装到一个没人读的目录里，然后训练从随机初始化起步且**不报错**。
    ap.add_argument("--logdir", default=_default_logdir())
    ap.add_argument("--official", default=str(OFFICIAL))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.official)
    if not src.is_file():
        print(f"❌ 找不到官方权重: {src}", file=sys.stderr)
        return 1

    import torch
    sd = torch.load(src, map_location="cpu", weights_only=False)
    bad = [k for k in sd if not k.startswith("_qnet.module.")]
    if bad:
        print(f"❌ 官方 ckpt 的键前缀不是 _qnet.module.*：{bad[:3]}", file=sys.stderr)
        return 1
    print(f"官方权重 {src.name}：{len(sd)} 个张量，sha256 {sha256(src)[:16]}…")

    for arm in args.arm:
        dst_dir = Path(args.logdir) / arm / f"seed{args.seed}" / "weights" / "0"
        dst = dst_dir / "QAttentionAgent_layer0.pt"
        if dst.is_file() and not args.force:
            print(f"  skip {arm}: {dst} 已存在（--force 可覆盖）")
            continue
        if dst_dir.parent.exists():
            others = sorted(p.name for p in dst_dir.parent.iterdir() if p.name != "0")
            if others and not args.force:
                print(f"  ⚠️ {arm} 的 weights/ 下已有 {others[:5]}，"
                      f"说明已经训过。装 iteration-0 会让恢复逻辑回到起点，已跳过。"
                      f"确需重来请加 --force。")
                continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  ✅ {arm} -> {dst}")

    print("\n训练时 OfflineTrainRunner 会打印 "
          "'Resuming training from iteration 0'，即从官方权重起步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
