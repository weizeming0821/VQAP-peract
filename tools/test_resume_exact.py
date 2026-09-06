#!/usr/bin/env python
"""中断恢复的等价性单测 —— 验证优化器/调度器状态确实被持久化了。

# 为什么需要

YARR 的「断点恢复」只恢复权重：`offline_train_runner.py` 只调 `agent.load_weights`，
而上游的 `save_weights` 只 `torch.save(self._q.state_dict())`。于是每次停-续都会
静默地把 LAMB 的一阶/二阶动量归零（若开了调度器，warmup 与 cosine 相位也归零）。
本项目允许中途换卡续训，这条路径必须是无损的。

# 做法

固定同一批样本、关掉一切随机性（dropout=0、SE3 增广关闭、cudnn 确定性），
跑三条路径再比：

    A  连续训 2N 步                                    -> W_cont
    B  训 N 步 -> 存 -> 新建 agent -> 载(**带**优化器状态) -> 再训 N 步  -> W_warm
    C  同 B，但把优化器状态文件删掉（即修复前的行为）      -> W_cold

断言：
    1. 优化器 state_dict 往返后**逐位一致**
    2. 冷启动后优化器动量为**空**、热启动**完整** —— 这是「修复确有必要」的
       确定性证据（修复前每次续训都会静默丢掉动量）
    3. ‖W_warm − W_cont‖ 落在 GPU 非确定性的噪声底之内

⚠️ 权重侧只能做弱断言：体素化用 atomic 累加，同样的代码同样的数据跑两遍，
   权重差就有 ~5e-4，与 LAMB 十几步内造成的变化同量级。所以先实测噪声底，
   再以它为尺子判读，而不是拍一个绝对阈值。
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OK = True
N_HALF = 12         # 每段步数；总共训 2*N_HALF 步
                    # 太短则 LAMB 动量还没积累起来，冷/热启动看不出差别
ARM = "B3"          # 用 B3：可训参数最多（含注入层），最能暴露问题


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def build_cfg(batch: int) -> OmegaConf:
    """与 smoke_train_stage3.build_cfg 同源，但关掉全部随机性。"""
    return OmegaConf.create({
        "rlbench": {"demo_path": str(REPO_ROOT / "aavla_data/rlbench" / "train"),
                    "episode_length": 25,
                    "cameras": ["front", "left_shoulder", "right_shoulder", "wrist"],
                    "camera_resolution": [128, 128],
                    "scene_bounds": [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
                    "include_lang_goal_in_obs": True},
        "method": {
            "voxel_sizes": [100], "bounds_offset": [0.15],
            "image_crop_size": 64, "num_latents": 2048, "latent_dim": 512,
            "transformer_depth": 6, "transformer_iterations": 1,
            "cross_heads": 1, "cross_dim_head": 64,
            "latent_heads": 8, "latent_dim_head": 64,
            "pos_encoding_with_lang": False, "conv_downsample": True,
            "lang_fusion_type": "seq", "voxel_patch_size": 5,
            "voxel_patch_stride": 5, "final_dim": 64,
            # 关掉 dropout —— 否则两条路径的随机数流不同，比不出东西
            "input_dropout": 0.0, "attn_dropout": 0.0, "decoder_dropout": 0.0,
            "lr": 1.0e-4, "lr_scheduler": False, "num_warmup_steps": 500,
            "optimizer": "lamb", "lambda_weight_l2": 1e-6,
            "trans_loss_weight": 1.0, "rot_loss_weight": 1.0,
            "grip_loss_weight": 1.0, "collision_loss_weight": 1.0,
            "rotation_resolution": 5, "crop_augmentation": True,
            "activation": "lrelu", "norm": None,
            # 关掉 SE3 增广 —— 它在 update() 内部采样，是另一路随机性
            "transform_augmentation": {"apply_se3": False,
                                       "aug_xyz": [0.0, 0.0, 0.0],
                                       "aug_rpy": [0.0, 0.0, 0.0],
                                       "aug_rot_resolution": 5},
            "demo_augmentation": True, "demo_augmentation_every_n": 10,
            "no_skip_connection": False, "no_perceiver": False,
            "no_language": False, "keypoint_method": "heuristic",
        },
        "replay": {"batch_size": batch, "timesteps": 1},
        "framework": {"training_iterations": 50001, "logging_level": 30},
        "ddp": {"num_devices": 1},
        "stage3": {"arm": ARM,
                   "codebook": str(REPO_ROOT / "checkpoints" / "vqap_pretrain" /
                                   "stage1" / "codebook.pth")},
    })


def make_agent(cfg, device: int):
    from agents.peract_bc.launch_utils import create_agent
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    agent = create_agent(cfg)
    # device 传 int，与正式训练一致（OfflineTrainRunner 的 train_device = rank）。
    # load_weights 里有 `torch.device('cuda:%d' % self._device)`，传 torch.device 会崩。
    agent.build(training=True, device=device)
    return agent


def inner_of(agent):
    return agent._pose_agent._qattention_agents[0]


def flat_weights(agent) -> torch.Tensor:
    """只取**可训练**参数。

    state_dict 里还有 voxelizer 的坐标网格等 register_buffer，量级到 1e6 且
    全程不变；把它们算进来，绝对差的最大值就只反映 buffer 的量级，毫无意义。
    """
    q = inner_of(agent)._q
    return torch.cat([p.detach().float().reshape(-1).cpu()
                      for _, p in sorted(q.named_parameters())
                      if p.requires_grad])


def train(agent, batches, device: int, start: int = 0) -> None:
    for i, b in enumerate(batches):
        torch.manual_seed(1000 + start + i)          # 两条路径的第 k 步种子一致
        agent.update(start + i, {k: v.to(device) for k, v in b.items()})


def main() -> int:
    import os
    import torch.distributed as dist
    from agents.peract_bc.launch_utils import create_replay, fill_replay
    from helpers.clip.core.clip import build_model, load_clip
    from helpers import utils

    if not torch.cuda.is_available():
        print("需要 GPU，跳过。")
        return 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device, batch, task, n_demos = 0, 2, "close_jar", 2
    cfg = build_cfg(batch)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29519")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    tmp = Path(tempfile.mkdtemp(prefix="resume_exact_"))
    ck = tmp / "ck"
    ck.mkdir(parents=True, exist_ok=True)
    try:
        print("=== 0. 准备固定的样本序列 ===")
        clip_model, _ = load_clip("RN50", jit=False, device=f"cuda:{device}")
        clip_model = build_model(clip_model.state_dict()).to(f"cuda:{device}")
        obs_config = utils.create_obs_config(
            cfg.rlbench.cameras, cfg.rlbench.camera_resolution, "PERACT_BC")
        replay = create_replay(
            batch_size=batch, timesteps=1, prioritisation=False, task_uniform=True,
            save_dir=str(tmp / "replay"), cameras=cfg.rlbench.cameras,
            voxel_sizes=cfg.method.voxel_sizes, replay_size=int(1e4),
            stage3_subtask_fields=True)
        (tmp / "replay").mkdir(parents=True, exist_ok=True)
        fill_replay(cfg=cfg, obs_config=obs_config, rank=0, replay=replay,
                    task=task, num_demos=n_demos, demo_augmentation=True,
                    demo_augmentation_every_n=10, cameras=cfg.rlbench.cameras,
                    rlbench_scene_bounds=cfg.rlbench.scene_bounds,
                    voxel_sizes=cfg.method.voxel_sizes,
                    bounds_offset=cfg.method.bounds_offset,
                    rotation_resolution=cfg.method.rotation_resolution,
                    crop_augmentation=cfg.method.crop_augmentation,
                    clip_model=clip_model, device=f"cuda:{device}",
                    keypoint_method="heuristic",
                    planner_cache_dir=str(REPO_ROOT / "aavla_data" / "planner_cache" / "train"),
                    data_split="train")
        # 预先采好 2N 个 batch，三条路径喂完全相同的数据
        np.random.seed(0)
        batches = []
        for _ in range(2 * N_HALF):
            raw = replay.sample_transition_batch(pack_in_dict=True)
            conv = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    conv[k] = v
                elif isinstance(v, np.ndarray) and v.dtype.kind in "biufc":
                    conv[k] = torch.from_numpy(v)
            batches.append(conv)
        check(f"预采 {len(batches)} 个固定 batch", len(batches) == 2 * N_HALF)

        print(f"=== 1. A 路径：连续训 {2*N_HALF} 步（跑两遍，测噪声底）===")
        a = make_agent(cfg, device)
        train(a, batches, device, start=0)
        w_cont = flat_weights(a)
        del a
        torch.cuda.empty_cache()

        # 同样的代码、同样的数据、同样的种子再跑一遍。两次之差就是 GPU 非确定性
        # （cudnn 之外仍有 atomics / SDPA 等）的噪声底 —— 后面的判据都以它为尺子，
        # 而不是拍一个绝对阈值。
        a2 = make_agent(cfg, device)
        train(a2, batches, device, start=0)
        noise = float((flat_weights(a2) - w_cont).abs().max())
        del a2
        torch.cuda.empty_cache()
        print(f"       噪声底 max|W_A2 - W_A1| = {noise:.4e}")

        print(f"=== 2. 训 {N_HALF} 步并存盘（含优化器状态）===")
        b = make_agent(cfg, device)
        train(b, batches[:N_HALF], device, start=0)
        b.save_weights(str(ck))
        opt_before = inner_of(b)._optimizer.state_dict()
        files = sorted(p.name for p in ck.iterdir())
        check(f"存盘产生 {files}", any(f.endswith("_optim.pt") for f in files),
              "没写出优化器状态文件")
        del b
        torch.cuda.empty_cache()

        print("=== 3. B 路径：载入(带优化器状态)后续训 ===")
        bw = make_agent(cfg, device)
        bw.load_weights(str(ck))
        opt_after = inner_of(bw)._optimizer.state_dict()

        # 断言 1：优化器状态逐位往返
        same, checked = True, 0
        for pid, st in opt_before["state"].items():
            for key, v in st.items():
                if isinstance(v, torch.Tensor):
                    got = opt_after["state"][pid][key]
                    same &= torch.equal(v.cpu(), got.cpu())
                    checked += 1
        check(f"优化器状态逐位往返（比对 {checked} 个张量）", same and checked > 0)

        train(bw, batches[N_HALF:], device, start=N_HALF)
        w_warm = flat_weights(bw)
        del bw
        torch.cuda.empty_cache()

        print("=== 4. C 路径：删掉优化器状态后续训（= 修复前的行为）===")
        for p in ck.iterdir():
            if p.name.endswith("_optim.pt"):
                p.unlink()
        cw = make_agent(cfg, device)
        cw.load_weights(str(ck))

        # 断言 2（确定性证据，不受 GPU 非确定性影响）：
        # 冷启动后优化器 state 必须是空的 —— 这正是修复前每次续训都会发生的事，
        # 且它不报错。用「有没有动量」来判，比用权重差判干净得多。
        cold_state = inner_of(cw)._optimizer.state_dict()["state"]
        n_cold = sum(1 for st in cold_state.values()
                     for v in st.values() if isinstance(v, torch.Tensor))
        n_warm = sum(1 for st in opt_after["state"].values()
                     for v in st.values() if isinstance(v, torch.Tensor))
        check(f"冷启动后优化器动量为空（{n_cold} 个张量），"
              f"热启动完整（{n_warm} 个张量）",
              n_cold == 0 and n_warm > 0)

        train(cw, batches[N_HALF:], device, start=N_HALF)
        w_cold = flat_weights(cw)
        del cw
        torch.cuda.empty_cache()

        print("=== 5. 比对（只看可训练参数）===")
        d_warm = float((w_warm - w_cont).abs().max())
        d_cold = float((w_cold - w_cont).abs().max())
        scale = float(w_cont.abs().max())
        print(f"       可训练权重量级 max|W| = {scale:.4e}")
        print(f"       噪声底                = {noise:.4e}")
        print(f"       max|W_warm - W_cont|  = {d_warm:.4e}   ({d_warm/max(noise,1e-30):.1f}× 噪声)")
        print(f"       max|W_cold - W_cont|  = {d_cold:.4e}   ({d_cold/max(noise,1e-30):.1f}× 噪声)")
        # 判据以噪声底为尺子：热启动应当落在噪声量级内，冷启动应当远超它。
        check("带优化器状态续训 ≈ 连续训练（差异在 GPU 非确定性噪声量级内）",
              d_warm <= max(noise * 5, 1e-9),
              f"warm={d_warm:.3e} 噪声底={noise:.3e}")
        # 权重侧只做弱断言：体素化的 atomic 累加带来的非确定性噪声底就有 ~5e-4，
        # 与 LAMB 在十几步内造成的权重变化同量级，信噪比撑不起强断言。
        # 「修复确有必要」由上面那条确定性的动量断言承担。
        check("冷启动的权重偏离大于热启动（弱证据，强证据见上条动量断言）",
              d_cold > d_warm * 1.5,
              f"cold={d_cold:.3e} warm={d_warm:.3e} 噪声底={noise:.3e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("中断恢复单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
