#!/usr/bin/env python
"""端到端训练冒烟 —— 走 `train.py` 的**真实入口**跑过所有周期性代码路径。

# 为什么必须有这个

P5 启动训练时连炸两次，两次都是「跑起来了才崩」，而 9 套单测全绿：

  1. `AttributeError: Can't pickle local object '_seal.<locals>._blocked'`
     —— 死在 `iter(dataset)`，即 DataLoader 用 spawn 起 worker 的那一刻。
  2. `NotImplementedError: Got <class 'NoneType'>` （tensorboard add_histogram）
     —— **跑满 100 步**之后死在第一次 `update_summaries()`：Stage 3 冻结了 94%
     的参数，它们的 `param.grad` 恒为 None。
  3. 训练产出的 ckpt **无法用于评测**：VoxelGrid 的 register_buffer 按 batch_size
     定形，训练存 batch=4、评测建 batch=1，load_state_dict 形状不符。训练全程正常，
     直到第一次拿训练产物去评测才暴露。
  4. `XIO: fatal IO error on X server` —— 训练跑到 15,600 步被一个**刚结束的
     评测进程**带死：`xvfb-run -a` 选 display 号有竞态，两个进程撞号后，
     先退出的那个把另一个的 X server 一并杀掉。体素可视化关掉后训练不再需要 X，
     本测试**刻意裸跑**来守住这一点。

两次的根因是同一个：**没有任何测试走过真实训练入口**。P4 的冒烟脚本直接组装
`create_replay -> fill_replay -> create_agent -> agent.update()`，绕开了
DataLoader、OfflineTrainRunner 和日志路径 —— 而 bug 恰恰都在被绕开的那部分。

所以这个冒烟不自己拼流程，**直接调 `train.py`**，只把迭代数压到刚好跨过
`log_freq`（摘要+体素可视化）与 `save_freq`（存 ckpt + 优化器状态）两道坎。

# 用法

    source run/env.sh
    xvfb-run -a python tools/test_train_smoke.py --arm B1        # 约 3-4 分钟
    xvfb-run -a python tools/test_train_smoke.py --arm B3        # 带码注入的那条路

⚠️ 必须 xvfb-run：第 100 步的 `update_summaries` 会调 `visualise_voxel` ->
   pyrender.OffscreenRenderer，无头环境下抛 NoSuchDisplayException 并终止训练。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"

ITERS = 121          # 跨过 log_freq=100，并让 save_freq=120 触发一次存盘
LOG_FREQ = 100
SAVE_FREQ = 120

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="B1", choices=["B1", "B2", "B3"])
    ap.add_argument("--src-exp", default="stage3_main",
                    help="从哪个实验目录借 iteration-0 权重")
    ap.add_argument("--keep", action="store_true", help="保留临时目录便于排查")
    a = ap.parse_args()

    src0 = (REPO_ROOT / "checkpoints" / a.src_exp / a.arm / "seed0" /
            "weights" / "0" / "QAttentionAgent_layer0.pt")
    if not src0.is_file():
        print(f"❌ 找不到 iteration-0 权重 {src0}\n"
              f"   先跑 scripts/init_from_official.py", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="train_smoke_"))
    try:
        # 造一个与正式训练同构的 logdir，权重用硬链接（不复制 157 MB）
        w0 = tmp / a.arm / "seed0" / "weights" / "0"
        w0.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src0, w0 / src0.name)
        except OSError:
            shutil.copy2(src0, w0 / src0.name)

        overrides = [
            "--config-name=stage3",
            f"stage3.arm={a.arm}",
            f"framework.logdir={tmp}",
            f"framework.training_iterations={ITERS}",
            f"framework.log_freq={LOG_FREQ}",
            f"framework.save_freq={SAVE_FREQ}",
            "framework.num_workers=2",      # 走 DataLoader 的 spawn worker 路径
            "ddp.num_devices=1",
            "hydra.output_subdir=null",
        ]
        # 🔴 **刻意不套 xvfb-run**，并把 DISPLAY 清空 —— 这是一条回归守卫。
        #    训练原本每 log_freq 步调 visualise_voxel -> pyrender，依赖 X，
        #    于是必须套 xvfb-run；而 `xvfb-run -a` 自动选 display 号有竞态，
        #    并发时两个进程可能选中同一个号，其中一个退出就把另一个的
        #    X server 一起杀掉 —— 实测让 B3 训练在 15,600 步被带死。
        #    体素可视化关掉之后训练不再需要 X。这里裸跑就是在验证这一点：
        #    一旦有人把 X 依赖加回来，本测试立刻失败。
        prefix = []
        env = dict(os.environ)
        env.pop("DISPLAY", None)
        print(f"=== 跑真实 train.py（{ITERS} 步，arm={a.arm}，**不套 xvfb**）===")
        print(f"    logdir={tmp}")
        log = tmp / "train.out"
        with log.open("wb") as f:
            rc = subprocess.call(prefix + [sys.executable, "-u", "train.py"] + overrides,
                                 cwd=PERACT_ROOT, env=env, stdout=f, stderr=f)
        text = log.read_text(errors="ignore")

        check("train.py 正常退出", rc == 0, f"退出码 {rc}")
        check("挂载了共享 replay 工件（没有重填）",
              "[replay] 已挂载工件" in text)
        check("从 iteration 0 恢复（官方权重起步）",
              "Resuming training from iteration 0" in text)
        check(f"跨过 log_freq={LOG_FREQ}：写出了第一次摘要（旧 bug 2 死在这）",
              f"Train Step {LOG_FREQ:06d}" in text)
        d = tmp / a.arm / "seed0"
        ck = d / "weights" / str(SAVE_FREQ)
        check(f"跨过 save_freq={SAVE_FREQ}：存下 ckpt", ck.is_dir())
        check("ckpt 含权重", (ck / "QAttentionAgent_layer0.pt").is_file())
        check("ckpt 含优化器/调度器状态（续训无损的前提）",
              (ck / "QAttentionAgent_layer0_optim.pt").is_file())
        check("写出了 tensorboard 事件文件",
              any(p.name.startswith("events.out.tfevents") for p in d.glob("*")))
        for bad in ("Traceback", "ProcessRaisedException", "NotImplementedError"):
            check(f"日志中无 {bad}", bad not in text)

        # 🔴 训完能存 ≠ 存下来能用。VoxelGrid 的缓冲按 batch_size 定形
        #    （build 里 `batch_size if training else 1`），训练存的是 batch=4 的形状，
        #    评测建模型时是 batch=1 —— 早期这些 buffer 会进 state_dict，于是
        #    **所有训练产物都无法评测**，而训练本身一切正常。
        #    这一步就是把「训练 -> 评测」的交接补上。
        if ck.is_dir():
            try:
                import torch
                from omegaconf import OmegaConf
                sys.path.insert(0, str(PERACT_ROOT))
                from agents.peract_bc.launch_utils import create_agent
                cfg = OmegaConf.load(d / "config.yaml")
                ag = create_agent(cfg)
                ag.build(training=False,
                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                ag.load_weights(str(ck))
                check("训练产出的 ckpt 能在**评测模式**下加载", True)
            except Exception as exc:
                check("训练产出的 ckpt 能在**评测模式**下加载", False,
                      f"{type(exc).__name__}: {str(exc)[:160]}")

        if not OK:
            print("\n--- 日志尾部 ---")
            print("\n".join(text.splitlines()[-25:]))
    finally:
        if a.keep:
            print(f"\n（--keep）临时目录保留于 {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("端到端训练冒烟:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
