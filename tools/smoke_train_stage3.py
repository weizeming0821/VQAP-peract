#!/usr/bin/env python
"""P4 Step 5/6：三臂冒烟训练 + 显存扫描。

不走 PerAct 的 train.py（那要起 DDP + env runner），直接组装
create_replay -> fill_replay -> create_agent -> agent.update()，
用的是**完全相同**的代码路径。

用法（必须 xvfb-run，否则 update_summaries 里的 pyrender 会挂）：
    source run/env.sh
    xvfb-run -a python tools/smoke_train_stage3.py --arms B1 B2 B3 --steps 200
    xvfb-run -a python tools/smoke_train_stage3.py --arms B3 --batch-scan 2 4 6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage3.cache_join import PlannerCache, TextEmbedCache      # noqa: E402


def build_cfg(arm: str, batch: int, lr: float) -> OmegaConf:
    cfg = OmegaConf.create({
        "rlbench": {
            "demo_path": str(REPO_ROOT / "data_rlbench" / "train"),
            "episode_length": 25,
            "cameras": ["front", "left_shoulder", "right_shoulder", "wrist"],
            "camera_resolution": [128, 128],
            "scene_bounds": [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
            "include_lang_goal_in_obs": True,
        },
        "method": {
            # 与官方 peract_600k 的 config.yaml 对齐
            "voxel_sizes": [100], "bounds_offset": [0.15],
            "image_crop_size": 64, "num_latents": 2048, "latent_dim": 512,
            "transformer_depth": 6, "transformer_iterations": 1,
            "cross_heads": 1, "cross_dim_head": 64,
            "latent_heads": 8, "latent_dim_head": 64,
            "pos_encoding_with_lang": False,        # 🔴 官方 ckpt 是 false，仓库默认 True
            "conv_downsample": True, "lang_fusion_type": "seq",
            "voxel_patch_size": 5, "voxel_patch_stride": 5, "final_dim": 64,
            "input_dropout": 0.1, "attn_dropout": 0.1, "decoder_dropout": 0.0,
            "lr": lr, "lr_scheduler": False, "num_warmup_steps": 3000,
            "optimizer": "lamb", "lambda_weight_l2": 1e-6,
            "trans_loss_weight": 1.0, "rot_loss_weight": 1.0,
            "grip_loss_weight": 1.0, "collision_loss_weight": 1.0,
            "rotation_resolution": 5, "crop_augmentation": True,
            "activation": "lrelu", "norm": None,
            "transform_augmentation": {"apply_se3": True,
                                       "aug_xyz": [0.125, 0.125, 0.125],
                                       "aug_rpy": [0.0, 0.0, 0.0],
                                       "aug_rot_resolution": 5},
            "demo_augmentation": True, "demo_augmentation_every_n": 10,
            "no_skip_connection": False, "no_perceiver": False, "no_language": False,
            "keypoint_method": "heuristic",
        },
        "replay": {"batch_size": batch, "timesteps": 1},
        "framework": {"training_iterations": 100000, "logging_level": 30},
        "ddp": {"num_devices": 1},
    })
    if arm != "B0":
        cfg.stage3 = OmegaConf.create({
            "arm": arm,
            "codebook": str(REPO_ROOT / "checkpoints" / "vqap_pretrain" /
                            "stage1" / "codebook.pth"),
        })
    return cfg


def run_arm(arm: str, task: str, n_demos: int, steps: int, batch: int,
            lr: float, device: str, ckpt: str | None) -> dict:
    from agents.peract_bc.launch_utils import create_agent, create_replay, fill_replay
    from helpers.clip.core.clip import build_model, load_clip
    from helpers import utils

    cfg = build_cfg(arm, batch, lr)
    need_subtask = arm in ("B2", "B3")
    tmp = Path(f"/data0/xiexiao/VQAP/.tmp_smoke/{arm}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    replay = create_replay(
        batch_size=batch, timesteps=1, prioritisation=False, task_uniform=True,
        save_dir=str(tmp), cameras=cfg.rlbench.cameras,
        voxel_sizes=cfg.method.voxel_sizes, replay_size=int(2e4),
        stage3_subtask_fields=need_subtask)

    clip_model, _ = load_clip("RN50", jit=False, device=device)
    clip_model = build_model(clip_model.state_dict()).to(device)
    obs_config = utils.create_obs_config(
        cfg.rlbench.cameras, cfg.rlbench.camera_resolution, "PERACT_BC")
    cache = PlannerCache() if need_subtask else None
    tcache = TextEmbedCache(clip_model, device) if need_subtask else None

    t0 = time.time()
    fill_replay(cfg=cfg, obs_config=obs_config, rank=0, replay=replay, task=task,
                num_demos=n_demos, demo_augmentation=True,
                demo_augmentation_every_n=10, cameras=cfg.rlbench.cameras,
                rlbench_scene_bounds=cfg.rlbench.scene_bounds,
                voxel_sizes=cfg.method.voxel_sizes,
                bounds_offset=cfg.method.bounds_offset,
                rotation_resolution=cfg.method.rotation_resolution,
                crop_augmentation=cfg.method.crop_augmentation,
                clip_model=clip_model, device=device,
                keypoint_method="heuristic",
                planner_cache=cache, split="train", text_embed_cache=tcache)
    fill_s = time.time() - t0
    n_samples = replay.add_count
    del clip_model
    torch.cuda.empty_cache()

    agent = create_agent(cfg)
    agent.build(training=True, device=torch.device(device))

    # 载入官方权重（B0/B1/B2 完全匹配；B3 的注入层是新增模块，strict=False）
    loaded = "random-init"
    if ckpt and Path(ckpt).is_file():
        sd = torch.load(ckpt, map_location=device, weights_only=False)
        inner = agent._pose_agent._qattention_agents[0]
        missing, unexpected = inner._q.load_state_dict(sd, strict=False)
        # 允许缺失的两类：
        #   code_injector.*  —— B3 新增模块，官方权重里当然没有
        #   _voxelizer.*     —— register_buffer 的位置网格（VLA_Design §8.2 记录的
        #                       6.024M 非可学习参数），官方 ckpt 本就不保存，运行时重建
        ALLOW = ("code_injector", "_voxelizer")
        extra = [k for k in missing if not any(a in k for a in ALLOW)]
        assert not unexpected, f"官方权重出现 unexpected keys: {unexpected[:5]}"
        assert not extra, f"官方权重加载缺失可学习参数: {extra[:5]}"
        n_inj = sum(1 for k in missing if "code_injector" in k)
        loaded = (f"peract_600k (missing {len(missing)}: "
                  f"{n_inj} 个 code_injector + {len(missing)-n_inj} 个 voxelizer buffer)")

    inner = agent._pose_agent._qattention_agents[0]
    n_train = sum(p.numel() for p in inner._q.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in inner._q.parameters())

    torch.cuda.reset_peak_memory_stats()
    losses, step_t = [], []
    for i in range(steps):
        batch_d = replay.sample_transition_batch(pack_in_dict=True)
        # 复刻正式训练路径（YARR offline_train_runner.py:138）：
        #   batch = {k: v.to(dev) for k, v in sampled_batch.items() if type(v)==torch.Tensor}
        # 即**只保留 tensor**，object/str 字段（lang_goal / subtask_lang_goal /
        # subtask_action 等诊断字段）根本不会传给 agent。
        conv = {}
        for k, v in batch_d.items():
            if isinstance(v, torch.Tensor):
                conv[k] = v.to(device)
            elif isinstance(v, np.ndarray) and v.dtype.kind in "biufc":
                conv[k] = torch.from_numpy(v).to(device)
        batch_d = conv
        ts = time.time()
        out = agent.update(i, batch_d)
        step_t.append(time.time() - ts)
        losses.append(float(out["total_losses"]))
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    shutil.rmtree(tmp, ignore_errors=True)
    return {
        "arm": arm, "weights": loaded, "batch": batch, "steps": steps,
        "n_samples": n_samples, "fill_seconds": round(fill_s, 1),
        "trainable": n_train, "total": n_total,
        "loss_first10": round(float(np.mean(losses[:10])), 4),
        "loss_last10": round(float(np.mean(losses[-10:])), 4),
        "loss_min": round(float(np.min(losses)), 4),
        "peak_mem_gb": round(peak, 2),
        "sec_per_step": round(float(np.mean(step_t[5:])), 3),
        "text_cache": tcache.stats() if tcache else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B1", "B2", "B3"])
    ap.add_argument("--task", default="close_jar")
    ap.add_argument("--demos", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--batch-scan", nargs="*", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt", default=str(PERACT_ROOT / "ckpts" / "multi" / "PERACT_BC" /
                                          "seed0" / "weights" / "600000" /
                                          "QAttentionAgent_layer0.pt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "result" / "p4_stage3" / "smoke.json"))
    args = ap.parse_args()

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29518")
        dist.init_process_group("gloo", rank=0, world_size=1)

    def _py(o):
        """numpy 标量/数组 -> 原生 python，供 json 序列化。"""
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"不可序列化: {type(o).__name__}")

    results = []
    combos = ([(a, b) for a in args.arms for b in args.batch_scan]
              if args.batch_scan else [(a, args.batch) for a in args.arms])
    for arm, batch in combos:
        print(f"\n{'='*76}\n### {arm}  batch={batch}  task={args.task} "
              f"demos={args.demos} steps={args.steps}\n{'='*76}", flush=True)
        try:
            r = run_arm(arm, args.task, args.demos, args.steps, batch,
                        args.lr, args.device, args.ckpt)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            r = {"arm": arm, "batch": batch, "error": "CUDA OOM"}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"arm": arm, "batch": batch, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=1, default=_py), flush=True)
        torch.cuda.empty_cache()

    print(f"\n{'='*90}")
    print(f"{'臂':4} {'batch':>5} {'样本':>6} {'可训参数':>11} {'loss首10':>9} "
          f"{'loss末10':>9} {'峰值显存':>9} {'秒/步':>7}")
    print("=" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['arm']:4} {r['batch']:>5}  ❌ {r['error']}")
            continue
        trend = "↓" if r["loss_last10"] < r["loss_first10"] else "↑"
        print(f"{r['arm']:4} {r['batch']:>5} {r['n_samples']:>6} {r['trainable']:>11,} "
              f"{r['loss_first10']:>9.3f} {r['loss_last10']:>9.3f}{trend} "
              f"{r['peak_mem_gb']:>8.1f}G {r['sec_per_step']:>7.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=_py))
    print(f"\n结果写入 {out}")
    bad = [r for r in results if "error" in r or r.get("loss_last10", 1e9) >= r.get("loss_first10", 0)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
