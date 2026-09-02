#!/usr/bin/env python
"""P4 Step 3 出口门禁：真跑一次 fill_replay，验证 subtask_* 字段与六条断言。

不用整个任务（那要几 GB），只填少量 demo，但走的是**完全相同**的代码路径。
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

from stage3.arms import ALL_SUBTASK_FIELDS, FieldAccessError, GuardedSample  # noqa: E402
from stage3.cache_join import PlannerCache, TextEmbedCache                   # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def main() -> int:
    from agents.peract_bc.launch_utils import create_replay, fill_replay
    from helpers.clip.core.clip import build_model, load_clip
    from helpers import utils

    task = "close_jar"
    n_demos = 3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.create({
        "rlbench": {"demo_path": str(REPO_ROOT / "data_rlbench" / "train"),
                    "episode_length": 25, "cameras": ["front", "left_shoulder",
                                                      "right_shoulder", "wrist"],
                    "camera_resolution": [128, 128], "scene_bounds":
                        [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
                    "include_lang_goal_in_obs": True},
        "method": {"voxel_sizes": [100], "bounds_offset": [0.15],
                   "rotation_resolution": 5, "crop_augmentation": True,
                   "demo_augmentation": True, "demo_augmentation_every_n": 10,
                   "keypoint_method": "heuristic"},
        "framework": {"logging_level": 30},
    })

    # YARR 的 replay buffer 依赖分布式进程组（正式训练由 run_seed_fn 初始化），
    # 单测里起一个单进程 gloo group 即可。
    import os
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group("gloo", rank=0, world_size=1)

    cache = PlannerCache()
    print(f"       {cache.summary()}")

    tmp = Path(tempfile.mkdtemp(prefix="stage3_replay_"))
    try:
        replay = create_replay(
            batch_size=2, timesteps=1, prioritisation=False, task_uniform=True,
            save_dir=str(tmp), cameras=cfg.rlbench.cameras,
            voxel_sizes=cfg.method.voxel_sizes, replay_size=int(1e4),
            stage3_subtask_fields=True)

        print("=== 1. replay schema 含 8 个 subtask_* 字段 ===")
        obs_el, _ = replay.get_storage_signature()
        names = {e.name for e in obs_el}
        missing = ALL_SUBTASK_FIELDS - names
        check("8 个字段全部声明", not missing, f"缺 {sorted(missing)}")
        check("PerAct 原生 lang_goal_emb 仍在（承载整任务指令）",
              "lang_goal_emb" in names)

        print("=== 2. 真跑 fill_replay ===")
        clip_model, _ = load_clip("RN50", jit=False, device=device)
        clip_model = build_model(clip_model.state_dict()).to(device)
        text_cache = TextEmbedCache(clip_model, device)
        obs_config = utils.create_obs_config(
            cfg.rlbench.cameras, cfg.rlbench.camera_resolution, "PERACT_BC")

        fill_replay(cfg=cfg, obs_config=obs_config, rank=0, replay=replay,
                    task=task, num_demos=n_demos, demo_augmentation=True,
                    demo_augmentation_every_n=10, cameras=cfg.rlbench.cameras,
                    rlbench_scene_bounds=cfg.rlbench.scene_bounds,
                    voxel_sizes=cfg.method.voxel_sizes,
                    bounds_offset=cfg.method.bounds_offset,
                    rotation_resolution=cfg.method.rotation_resolution,
                    crop_augmentation=cfg.method.crop_augmentation,
                    clip_model=clip_model, device=device,
                    keypoint_method="heuristic",
                    planner_cache=cache, split="train",
                    text_embed_cache=text_cache)
        n = replay.add_count
        check(f"写入 {n} 个样本", n > 0)
        print(f"       {text_cache.stats()}")
        check("文本编码有缓存命中（唯一指令数 << 样本数）",
              text_cache.misses < max(text_cache.hits, 1),
              f"unique={text_cache.misses} calls={text_cache.hits+text_cache.misses}")

        print("=== 3. 采样一个 batch，验证字段内容 ===")
        batch = replay.sample_transition_batch(pack_in_dict=True)
        check("subtask_lang_goal 非空",
              all(str(x[0]).strip() for x in batch["subtask_lang_goal"]),
              str(batch["subtask_lang_goal"][:2]))
        kg = np.asarray(batch["subtask_k_global"]).reshape(-1)
        kd = np.asarray(batch["subtask_k_detail"]).reshape(-1, 9)
        check(f"k_global ∈ [0,36)  实测 [{kg.min()},{kg.max()}]",
              bool((kg >= 0).all() and (kg < 36).all()))                       # A4
        check(f"k_detail ∈ [0,192) 实测 [{kd.min()},{kd.max()}]，宽度 9",
              bool((kd >= 0).all() and (kd < 192).all() and kd.shape[1] == 9))  # A4
        check("整任务与子任务指令确实不同",
              any(str(a[0]) != str(b[0]) for a, b in
                  zip(batch["lang_goal"], batch["subtask_lang_goal"])),
              f"{batch['lang_goal'][0]} vs {batch['subtask_lang_goal'][0]}")
        emb_t = np.asarray(batch["lang_goal_emb"])
        emb_s = np.asarray(batch["subtask_lang_goal_emb"])
        check("两套语言嵌入数值不同", not np.allclose(emb_t, emb_s))

        print("=== 4. 字段白名单在真实样本上生效 ===")
        keys = set(batch.keys())
        for arm, bad in (("B1", "subtask_lang_goal_emb"), ("B2", "lang_goal_emb")):
            g = GuardedSample(batch, arm, None)
            try:
                _ = g[bad]
                check(f"{arm} 读 {bad} 应抛异常", False)
            except FieldAccessError:
                check(f"{arm} 读 {bad} 抛 FieldAccessError", True)
        check("B3 能读码字段", GuardedSample(batch, "B3")["subtask_k_global"] is not None)
        check("B1 能读原生语言字段", GuardedSample(batch, "B1")["lang_goal_emb"] is not None)

        print("=== 5. 样本数与 baseline 一致（三臂对比不被样本量污染）===")
        replay2 = create_replay(
            batch_size=2, timesteps=1, prioritisation=False, task_uniform=True,
            save_dir=str(tmp / "base"), cameras=cfg.rlbench.cameras,
            voxel_sizes=cfg.method.voxel_sizes, replay_size=int(1e4),
            stage3_subtask_fields=False)
        fill_replay(cfg=cfg, obs_config=obs_config, rank=0, replay=replay2,
                    task=task, num_demos=n_demos, demo_augmentation=True,
                    demo_augmentation_every_n=10, cameras=cfg.rlbench.cameras,
                    rlbench_scene_bounds=cfg.rlbench.scene_bounds,
                    voxel_sizes=cfg.method.voxel_sizes,
                    bounds_offset=cfg.method.bounds_offset,
                    rotation_resolution=cfg.method.rotation_resolution,
                    crop_augmentation=cfg.method.crop_augmentation,
                    clip_model=clip_model, device=device,
                    keypoint_method="heuristic")
        check(f"带 subtask {n} == baseline {replay2.add_count}",
              n == replay2.add_count)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("Step 3 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
