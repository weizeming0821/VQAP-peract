#!/usr/bin/env python
"""用一份冻结权重在 replay 上跑前向，产出逐分量 loss —— Step 1b/1f「新 replay 可用性」的判据。

# 为什么需要这个脚本

Seen18 全量重建必须先删掉旧的 seen12 工件（两份 293+203 GB 不可并存），
删了就再也无法比对。所以在删除**之前**先用冻结的 B1@40k 在旧 replay 上量一次
loss（Step 1b），重建之后在新 replay 的**同 12 个任务**上用**同样的样本**再量一次
（Step 1f）。两个数字对得上 ⇒ 新 replay 与旧的等价、可用。

# 为什么是完全确定性的

- **样本**：不走随机采样，而是对每个任务按 `sorted(task_idxs[task])` 的**位置**
  等距取样，再用 `sample_transition_batch(indices=...)` 显式指定。
  任务内的 cursor 由单进程顺序分配，所以「第 k 个位置」在两份工件里指向**同一条样本**
  —— 尽管它们的绝对 cursor 值完全不同（跨任务 cursor 是多进程抢占分配的）。
- **增广**：SE3 与 crop 用 torch RNG，每个 batch 前重置种子。
- **权重**：`_optimizer.step` 被替换成空操作，前向不会改动权重。

用法：
    python scripts/replay_loss_probe.py \
        --replay aavla_data/replay/seen12/multi/PERACT_BC/seed0 \
        --weights checkpoints/stage3_main/B1/seed0/weights/40000 \
        --arm B1 --batches 500 --out result/p5_train/loss_probe_old.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PERACT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from stage3 import replay_dataset

LOSS_KEYS = ["losses/total_loss", "losses/trans_loss", "losses/rot_loss",
             "losses/grip_loss", "losses/collision_loss"]


def pick_indices(replay, tasks: list[str], n_per_task: int) -> list[int]:
    """按任务内**位置**等距取样，返回可用的 cursor 列表。

    位置而非 cursor —— 这是新旧两份工件之间唯一可对齐的坐标。
    """
    out: list[int] = []
    for t in sorted(tasks):
        cur = sorted(int(c) for c in replay._task_idxs[t])
        # add_final 帧 terminal=-1，永远不是合法 transition，先滤掉
        valid = [c for c in cur if replay.is_valid_transition(c)]
        if not valid:
            raise RuntimeError(f"任务 {t} 没有合法 transition")
        # 等距取 n_per_task 个「位置」
        k = min(n_per_task, len(valid))
        step = max(1, len(valid) // k)
        out.extend(valid[i * step] for i in range(k))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="replay 工件目录")
    ap.add_argument("--weights", required=True, help="权重目录，如 .../weights/40000")
    ap.add_argument("--arm", default="B1", choices=["B0", "B1", "B2", "B3"])
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="限定任务（默认取工件里的全部）。做新旧配对时必须限定为共同的 12 个。")
    ap.add_argument("--batches", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--config", default=str(PERACT_ROOT / "conf" / "stage3.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    if "method" not in cfg or "voxel_sizes" not in cfg.get("method", {}):
        base = OmegaConf.load(PERACT_ROOT / "conf" / "method" / "PERACT_BC.yaml")
        cfg.method = OmegaConf.merge(base, cfg.get("method", {}))
    cfg.stage3.arm = a.arm
    # 🔴 voxelizer 在 create_agent 时按 cfg.replay.batch_size 固定了 batch 维，
    #    喂进去的 batch 大小必须与之一致，否则在 coords_to_bounding_voxel_grid 处炸。
    cfg.replay.batch_size = a.batch_size

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29521")
        dist.init_process_group("gloo", rank=0, world_size=1)

    from agents import peract_bc

    replay = peract_bc.launch_utils.create_replay(
        a.batch_size, int(cfg.replay.timesteps), bool(cfg.replay.prioritisation),
        bool(cfg.replay.task_uniform), a.replay, list(cfg.rlbench.cameras),
        list(cfg.method.voxel_sizes), list(cfg.rlbench.camera_resolution),
        stage3_subtask_fields=True)
    man = replay_dataset.attach(replay, a.replay, verbose=True)

    tasks = a.tasks or sorted(replay._task_idxs.keys())
    missing = [t for t in tasks if t not in replay._task_idxs]
    if missing:
        print(f"\n❌ 工件里没有这些任务：{missing}", file=sys.stderr)
        return 2

    n_needed = a.batches * a.batch_size
    n_per_task = max(1, -(-n_needed // len(tasks)))          # 向上取整
    idxs = pick_indices(replay, tasks, n_per_task)
    print(f"[probe] {len(tasks)} 个任务 × 每任务 {n_per_task} 个位置 → 候选 {len(idxs):,}")

    agent = peract_bc.launch_utils.create_agent(cfg)
    # 🔴 device 必须是 int rank，不能是 torch.device ——
    #    qattention_peract_bc_agent.load_weights 里是 'cuda:%d' % self._device。
    dev_idx = int(a.device.split(":")[-1]) if ":" in a.device else 0
    agent.build(training=True, device=dev_idx)
    agent.load_weights(a.weights)
    print(f"[probe] 权重已加载：{a.weights}")

    # create_agent 返回的是 PreprocessAgent（负责 rgb 归一化与去掉 timesteps 维），
    # 真正持有 optimizer / summaries 的是它包着的 QAttentionStackAgent。
    qas = agent._pose_agent._qattention_agents
    # 🔴 前向不得改动权重：把优化器的 step 换成空操作。
    #    （backward 仍会跑，只是白算一次梯度，换取与训练完全相同的代码路径。）
    for qa in qas:
        qa._optimizer.step = lambda *_a, **_k: None

    rows: list[dict] = []
    for i in range(a.batches):
        sel = idxs[i * a.batch_size:(i + 1) * a.batch_size]
        if len(sel) < a.batch_size:
            print(f"[probe] 候选耗尽，实际跑了 {i} 个 batch")
            break
        np.random.seed(a.seed + i)
        torch.manual_seed(a.seed + i)
        torch.cuda.manual_seed_all(a.seed + i)
        # batch_size 必须显式给，上游 assert len(indices) == batch_size
        sample = replay.sample_transition_batch(
            batch_size=len(sel), indices=sel, pack_in_dict=True)
        # 只转数值数组：task / lang_goal 是 object dtype，from_numpy 会抛异常
        batch = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray) and v.dtype != object and v.dtype.kind not in "USO":
                batch[k] = torch.from_numpy(v).to(a.device)
        agent.update(i, batch)
        s = qas[0]._summaries
        rows.append({k: float(s[k]) if torch.is_tensor(s[k]) else float(s[k])
                     for k in LOSS_KEYS})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{a.batches} … total_loss 均值 "
                  f"{st.mean(r['losses/total_loss'] for r in rows):.4f}", flush=True)

    res = {
        "replay": str(a.replay), "weights": str(a.weights), "arm": a.arm,
        "tasks": tasks, "n_batches": len(rows), "batch_size": a.batch_size,
        "seed": a.seed, "replay_n_samples": int(man["n_samples"]),
        "per_task": {k: v for k, v in sorted(man.get("per_task", {}).items())},
        "loss": {k: {"mean": st.mean(r[k] for r in rows),
                     "sd": st.pstdev([r[k] for r in rows]),
                     "n": len(rows)} for k in LOSS_KEYS},
    }
    print("\n=== 结果 ===")
    for k in LOSS_KEYS:
        d = res["loss"][k]
        print(f"  {k:28s} mean={d['mean']:.6f}  sd={d['sd']:.6f}  n={d['n']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"\n已写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
