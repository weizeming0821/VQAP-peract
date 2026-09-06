#!/usr/bin/env python
"""replay 工件（stage3/replay_dataset.py）的单测 —— 重点验**失败模式**。

工件机制的价值全在「出错时必须炸」上：三臂共享同一份 replay，
一旦挂错了（字段不符 / 目录被别的构建覆盖 / 文件半截写入），
训练照跑不误，我们会拿到一个说不清是什么的模型。

五项断言：
  1. save -> attach 往返：add_count / task_idxs / terminal 逐位还原
  2. **字段签名不符必须 raise**（拿不含 subtask_* 的 replay 去训 B3）
  3. **样本文件损坏必须 raise**（半截 pickle）
  4. **索引与目录内容不同源必须 raise**（cursor 的 task 对不上）
  5. attach 后 replay 只读，再 add() 必须 raise
  6. **封存后的 buffer 仍能进 DataLoader 的 spawn worker**（真实事故的回归）
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 🔴 必须在**模块导入时**就把启动方式设成 spawn，与 train.py 同序。
#    实测：等到用 DataLoader 时才 set_start_method(force=True)，
#    此前已建好 CUDA 上下文与若干 replay buffer，worker 会直接段错误。
#    这不是被测代码的问题，是测试自己的顺序不对 —— 但足以伪装成 bug，故记于此。
from torch.multiprocessing import get_start_method, set_start_method  # noqa: E402
try:
    if get_start_method(allow_none=True) != "spawn":
        set_start_method("spawn", force=True)
except RuntimeError:
    pass

from stage3 import replay_dataset                                    # noqa: E402
from stage3.replay_dataset import ReplayArtifactError                # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def expect_raise(name: str, fn) -> None:
    try:
        fn()
        check(name, False, "没有抛异常")
    except ReplayArtifactError as exc:
        check(name, True)
        print(f"           ↳ {str(exc).splitlines()[0][:90]}")
    except Exception as exc:                                # 抛错了，但类型不对
        check(name, False, f"抛的是 {type(exc).__name__}: {str(exc)[:80]}")


def main() -> int:
    import torch
    import torch.distributed as dist
    from agents.peract_bc.launch_utils import create_replay, fill_replay
    from helpers.clip.core.clip import build_model, load_clip
    from helpers import utils

    task, n_demos = "close_jar", 2
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = OmegaConf.create({
        "rlbench": {"demo_path": str(REPO_ROOT / "aavla_data/rlbench" / "train"),
                    "episode_length": 25,
                    "cameras": ["front", "left_shoulder", "right_shoulder", "wrist"],
                    "camera_resolution": [128, 128],
                    "scene_bounds": [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6],
                    "include_lang_goal_in_obs": True},
        "method": {"voxel_sizes": [100], "bounds_offset": [0.15],
                   "rotation_resolution": 5, "crop_augmentation": True,
                   "demo_augmentation": True, "demo_augmentation_every_n": 10,
                   "keypoint_method": "heuristic"},
        "framework": {"logging_level": 30},
    })

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29518")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

    tmp = Path(tempfile.mkdtemp(prefix="replay_artifact_"))
    try:
        def new_buffer(sub: bool, save_dir: Path):
            return create_replay(
                batch_size=2, timesteps=1, prioritisation=False, task_uniform=True,
                save_dir=str(save_dir), cameras=cfg.rlbench.cameras,
                voxel_sizes=cfg.method.voxel_sizes, replay_size=int(1e4),
                stage3_subtask_fields=sub)

        print("=== 0. 建一个小 replay（2 demo）===")
        clip_model, _ = load_clip("RN50", jit=False, device=device)
        clip_model = build_model(clip_model.state_dict()).to(device)
        obs_config = utils.create_obs_config(
            cfg.rlbench.cameras, cfg.rlbench.camera_resolution, "PERACT_BC")

        src = new_buffer(True, tmp)
        fill_replay(cfg=cfg, obs_config=obs_config, rank=0, replay=src,
                    task=task, num_demos=n_demos, demo_augmentation=True,
                    demo_augmentation_every_n=10, cameras=cfg.rlbench.cameras,
                    rlbench_scene_bounds=cfg.rlbench.scene_bounds,
                    voxel_sizes=cfg.method.voxel_sizes,
                    bounds_offset=cfg.method.bounds_offset,
                    rotation_resolution=cfg.method.rotation_resolution,
                    crop_augmentation=cfg.method.crop_augmentation,
                    clip_model=clip_model, device=device,
                    keypoint_method="heuristic",
                    planner_cache_dir=str(REPO_ROOT / "aavla_data" / "planner_cache" / "train"),
                    data_split="train")
        n = int(src.add_count)
        check(f"填了 {n} 个样本", n > 0)
        man = replay_dataset.save(src, tmp, extra={"tasks": [task], "demos": n_demos})
        check("is_built 为真", replay_dataset.is_built(tmp))

        print("=== 1. save -> attach 往返 ===")
        got = new_buffer(True, tmp)
        replay_dataset.attach(got, tmp, verbose=False)
        check("add_count 一致", int(got.add_count) == n)
        check("task_idxs 一致",
              {k: sorted(v) for k, v in got._task_idxs.items()} ==
              {k: sorted(map(int, v)) for k, v in src._task_idxs.items()})
        import numpy as np
        check("terminal 数组逐位一致",
              bool(np.array_equal(np.asarray(got._store["terminal"]),
                                  np.asarray(src._store["terminal"]))))
        batch = got.sample_transition_batch(pack_in_dict=True)
        check("attach 后能采样且带码字段", batch["subtask_k_global"] is not None)

        print("=== 2. 字段签名不符必须 raise ===")
        wrong = new_buffer(False, tmp)          # 不含 subtask_* 字段
        expect_raise("用不含 subtask_* 的 buffer attach -> raise",
                     lambda: replay_dataset.attach(wrong, tmp, verbose=False))

        print("=== 3. 样本文件损坏必须 raise ===")
        dmg = Path(tempfile.mkdtemp(prefix="replay_dmg_"))
        shutil.copytree(tmp, dmg, dirs_exist_ok=True)
        idx = pickle.loads((dmg / replay_dataset.INDEX).read_bytes())
        victim = idx["task_idxs"][task][0]
        (dmg / f"{victim}.replay").write_bytes(b"\x80\x04\x95truncated")
        expect_raise("半截 pickle -> raise",
                     lambda: replay_dataset.attach(new_buffer(True, dmg), dmg,
                                                   spot_check=10**6, verbose=False))
        shutil.rmtree(dmg, ignore_errors=True)

        print("=== 4. 索引与目录内容不同源必须 raise ===")
        mis = Path(tempfile.mkdtemp(prefix="replay_mis_"))
        shutil.copytree(tmp, mis, dirs_exist_ok=True)
        idx = pickle.loads((mis / replay_dataset.INDEX).read_bytes())
        # 把索引改成「这些 cursor 属于另一个任务」——模拟目录被别的构建覆盖
        idx["task_idxs"] = {"some_other_task": idx["task_idxs"][task]}
        with open(mis / replay_dataset.INDEX, "wb") as f:
            pickle.dump(idx, f, protocol=4)
        expect_raise("cursor 的 task 对不上 -> raise",
                     lambda: replay_dataset.attach(new_buffer(True, mis), mis,
                                                   spot_check=10**6, verbose=False))
        shutil.rmtree(mis, ignore_errors=True)

        print("=== 5. attach 后只读 ===")
        expect_raise("挂载后再 add() -> raise",
                     lambda: got.add(None, 0.0, False, False))

        print("=== 6. 封存后的 buffer 必须能进 DataLoader 的 spawn worker ===")
        # 🔴 这条是真实事故的回归：_seal 曾用局部闭包替换 add/add_final，
        #    普通用法一切正常，但 DataLoader 用 spawn 起 worker 时要 pickle
        #    整个 replay buffer，于是训练在 `iter(dataset)` 处崩：
        #        AttributeError: Can't pickle local object '_seal.<locals>._blocked'
        #    此前所有单测都只在主进程里用 replay，没有一条走这条路。
        check("start_method 是 spawn（与 train.py 一致）",
              get_start_method() == "spawn")
        from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
        try:
            dl = PyTorchReplayBuffer(got, num_workers=2).dataset()
            b2 = next(iter(dl))
            check("num_workers=2 能取到 batch", b2["subtask_k_global"] is not None)
        except Exception as exc:
            check("num_workers=2 能取到 batch", False,
                  f"{type(exc).__name__}: {str(exc)[:160]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print("replay 工件单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
