#!/usr/bin/env python
"""Stage 3 训练前的全面预检。

在启动任何长任务之前跑一遍，把「环境 / 数据 / cache / 权重 / 代码 / 配置」六类
前提一次性验完。任一项失败即返回非零。

    source run/env.sh && python tools/preflight_stage3.py
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for _p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OK = True
WARN = 0


def check(name: str, cond: bool, extra: str = "") -> bool:
    global OK
    print(("  ✅ " if cond else "  ❌ ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)
    return bool(cond)


def warn(name: str, cond: bool, extra: str = "") -> None:
    global WARN
    if not cond:
        WARN += 1
    print(("  ✅ " if cond else "  ⚠️  ") + name + ("" if cond else f"   {extra}"))


def section(t: str) -> None:
    print(f"\n{'='*78}\n{t}\n{'='*78}")


def main() -> int:
    # ---------------------------------------------------------------- 环境
    section("① 环境")
    import torch
    check(f"torch {torch.__version__}", torch.__version__.startswith("2.4"))
    check(f"CUDA 可用，{torch.cuda.device_count()} 卡", torch.cuda.is_available())
    import numpy as np
    check(f"numpy {np.__version__}（必须 1.x）", np.__version__.startswith("1."))
    for mod in ("pyrep", "rlbench", "yarr", "pytorch3d"):
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as exc:
            check(f"import {mod}", False, str(exc)[:80])
    check("COPPELIASIM_ROOT 已设", bool(os.environ.get("COPPELIASIM_ROOT")))
    check("QT_QPA_PLATFORM_PLUGIN_PATH 已设（否则 CoppeliaSim core dump）",
          bool(os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")))
    check("xvfb-run 可用（训练/评测/数据生成都要）", shutil.which("xvfb-run") is not None)

    free = torch.cuda.device_count()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        idle = [l.split(",")[0].strip() for l in out.strip().splitlines()
                if int(l.split(",")[1]) > 40000]
        print(f"       空闲 GPU（>40 GB 可用）: {idle}")
        check(f"至少 2 张空闲卡（当前 {len(idle)}）", len(idle) >= 2,
              "无法训练，需等待其他用户释放")
        warn(f"有 6 张空闲卡（当前 {len(idle)}，配置 ddp.num_devices=6）", len(idle) >= 6,
             "这是共享机器，可用卡数会波动。🔴 三臂必须用**相同的** num_devices，"
             "否则有效 batch 不同会成为混淆项 —— 请选一个三臂都能保证的卡数，"
             "并用 CUDA_VISIBLE_DEVICES 锁定具体的卡")
    except Exception as exc:
        warn("GPU 空闲检测", False, str(exc)[:60])

    # ---------------------------------------------------------------- 磁盘
    section("② 磁盘")
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1024 ** 3
    print(f"       /data0 可用 {free_gb:.0f} GB")
    check("≥ 220 GB（replay 176 + ckpt 30 + 余量）", free_gb >= 220,
          f"仅 {free_gb:.0f} GB")
    warn("≥ 300 GB（宽松余量；其他用户在持续消耗）", free_gb >= 300)

    # ---------------------------------------------------------------- 数据
    section("③ RLBench 数据")
    SEEN12 = ["close_jar", "light_bulb_in", "open_drawer", "place_cups",
              "place_shape_in_shape_sorter", "push_buttons",
              "put_groceries_in_cupboard", "reach_and_drag",
              "slide_block_to_color_target", "stack_blocks",
              "place_wine_at_rack_location", "sweep_to_dustpan_of_size"]
    bad = []
    for t in SEEN12:
        d = REPO_ROOT / "data_rlbench" / "train" / t / "all_variations" / "episodes"
        n = len([e for e in d.glob("episode*") if (e / "low_dim_obs.pkl").is_file()]) if d.is_dir() else 0
        if n != 100:
            bad.append(f"{t}={n}")
    check(f"Seen12 train 每任务 100 episode", not bad, f"异常: {bad}")
    for split, want in (("val", 25), ("test", 25)):
        n = len(glob.glob(str(REPO_ROOT / "data_rlbench" / split / "*" /
                              "all_variations" / "episodes" / "episode*" / "low_dim_obs.pkl")))
        check(f"{split} 共 {18*want} episode（实测 {n}）", n == 18 * want)

    # ---------------------------------------------------------------- cache
    section("④ Planner cache")
    try:
        from stage3.cache_join import PlannerCache
        c = PlannerCache()
        print(f"       {c.summary()}")
        check("18 个任务", len(c.tasks()) == 18)
        check(f"可用 episode {len(c._eps)} ≥ 1780", len(c._eps) >= 1780)
        n_seen12 = sum(1 for (t, s, _) in c._eps if t in SEEN12 and s == "train")
        check(f"Seen12 train 可用 {n_seen12} ≥ 1180", n_seen12 >= 1180)
        ep = next(iter(c._eps.values()))
        seg = ep["segments"][0]
        check("segment 已补码（k_global / k_detail）",
              "k_global" in seg and len(seg.get("k_detail", [])) == 9)
        check("cache meta 记录了 adapter/codebook 的 sha256",
              "adapter_ckpt_sha256" in c.meta and "codebook_ckpt_sha256" in c.meta)
    except Exception as exc:
        check("PlannerCache 加载", False, f"{type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 权重
    section("⑤ 权重")
    official = (PERACT_ROOT / "ckpts" / "multi" / "PERACT_BC" / "seed0" /
                "weights" / "600000" / "QAttentionAgent_layer0.pt")
    check(f"官方 peract_600k 存在", official.is_file())
    for f, name in ((REPO_ROOT / "checkpoints" / "vqap_adapter" / "best.pth", "Adapter"),
                    (REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth", "码本")):
        check(f"{name} checkpoint 存在", f.is_file())
    check("CLIP RN50 权重已缓存（urllib 走代理会 SSL 失败）",
          (Path.home() / ".cache" / "clip" / "RN50.pt").is_file())
    for arm in ("B1", "B2", "B3"):
        p = REPO_ROOT / "logs" / "stage3" / arm / "seed0" / "weights" / "0" / "QAttentionAgent_layer0.pt"
        check(f"{arm} 的 iteration-0 权重已装（从官方起步）", p.is_file(),
              "跑 scripts/init_from_official.py")

    # ---------------------------------------------------------------- 配置
    section("⑥ 训练配置")
    from omegaconf import OmegaConf
    cfg_p = PERACT_ROOT / "conf" / "stage3.yaml"
    if check("conf/stage3.yaml 存在", cfg_p.is_file()):
        cfg = OmegaConf.load(cfg_p)
        check("pos_encoding_with_lang == False（官方 ckpt 如此，仓库默认 True）",
              cfg.method.pos_encoding_with_lang is False)
        check("load_existing_weights == True（否则断点恢复失效）",
              cfg.framework.load_existing_weights is True)
        check("aug_rpy == [0,0,0]（对齐官方 ckpt）",
              list(cfg.method.transform_augmentation.aug_rpy) == [0.0, 0.0, 0.0])
        check("task_uniform == True（拉平 stack_blocks 的 35% 占比）",
              cfg.replay.task_uniform is True)
        check(f"Seen12 共 12 个任务（实测 {len(cfg.rlbench.tasks)}）",
              len(cfg.rlbench.tasks) == 12)
        check("replay.path 三臂共享（路径里不含 arm）",
              "${" not in str(cfg.replay.path) and "B1" not in str(cfg.replay.path))
        print(f"       LR={cfg.method.lr}  batch/卡={cfg.replay.batch_size}  "
              f"卡数={cfg.ddp.num_devices}  有效 batch="
              f"{cfg.replay.batch_size * cfg.ddp.num_devices}")

    # ---------------------------------------------------------------- 代码
    section("⑦ 代码不变式（六个已修 bug 的回归）")
    ag = (PERACT_ROOT / "agents" / "peract_bc" / "qattention_peract_bc_agent.py").read_text()
    lu = (PERACT_ROOT / "agents" / "peract_bc" / "launch_utils.py").read_text()
    rs = (PERACT_ROOT / "run_seed_fn.py").read_text()
    pio = (PERACT_ROOT / "agents" / "peract_bc" / "perceiver_lang_io.py").read_text()
    check("BUG1: act() 从 observation 取码并传给 _q",
          "subtask_k_global" in ag and ag.count("code_mask=code_mask") >= 2)
    check("BUG1: 缺码时硬失败", "raise RuntimeError" in ag and "codebook" in ag.lower())
    check("BUG2: run_seed_fn 传 stage3_subtask_fields 与 planner_cache_dir",
          "stage3_subtask_fields" in rs and "planner_cache_dir" in rs)
    check("BUG3: fill_replay 收 planner_cache_dir 路径（而非对象）",
          "planner_cache_dir = None" in lu and "data_split" in lu)
    check("BUG5: run_seed_fn 设随机种子", "_set_all_seeds" in rs)
    check("purge_replay_on_shutdown=False（否则训完 replay 被删）",
          "purge_replay_on_shutdown=False" in lu)
    check("注入点在 feats.extend 之前",
          pio.index("self.code_injector(latents") < pio.index("feats.extend([self.ss1"))
    check("冻结在 QFunction 构造之前（否则 DDP 等不到梯度）",
          ag.index("STAGE3_TRAINABLE_MODULES) \n" if False else "self._perceiver_encoder.named_parameters()")
          < ag.index("self._q = QFunction("))

    # ---------------------------------------------------------------- 单测
    section("⑧ 单测")
    for t in ("test_code_injector", "test_peract_integration", "test_stage3_arms",
              "test_act_codes", "test_planner_contract"):
        r = subprocess.run([sys.executable, str(REPO_ROOT / "tools" / f"{t}.py")],
                           capture_output=True, text=True, timeout=900)
        check(f"{t}", r.returncode == 0, (r.stdout + r.stderr)[-200:])

    # ---------------------------------------------------------------- replay
    section("⑨ 共享 replay（尚未构建则提示）")
    try:
        rp = Path(OmegaConf.load(cfg_p).replay.path)
        marker = rp / "_BUILD_COMPLETE.json"
        if marker.is_file():
            info = json.loads(marker.read_text())
            check(f"replay 已完整：{info['n_samples']:,} 样本 / {info['size_gb']:.0f} GB",
                  info["n_samples"] > 130000)
            check("含 subtask 字段", info.get("subtask_fields") is True)
        else:
            warn("replay 尚未构建", False,
                 "跑 python scripts/build_replay.py（约 176 GB / 30 分钟）")
    except Exception as exc:
        warn("replay 检查", False, str(exc)[:60])

    print(f"\n{'='*78}")
    print(f"预检结果: {'PASS' if OK else 'FAIL'}   （{WARN} 项警告）")
    print("=" * 78)
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
