"""把 YARR 的 replay buffer 固化成**不可变数据集工件**，供多次实验复用。

# 为什么需要这个模块

PerAct 的 replay 名义上是 RL 的经验回放，实际是「预处理后的训练数据集」：
1800 个 episode 经关键帧解析 + 点云重建 + CLIP 编码 + Planner cache 查表，
展开成 140,613 条样本（约 176 GB）。这套处理跑一遍 30 分钟，训练要随机读它两万多次。

上游的用法有三个问题，本模块逐一堵掉：

* **P0-1 路径不一致**：`scripts/build_replay.py` 曾写到 `cfg.replay.path`，而
  `run_seed_fn.py` 读的是 `cfg.replay.path/<task_folder>/<method>/seed<N>`。
  预建的 176 GB 永远用不上。→ `replay_dir()` 成为两边唯一的路径来源。

* **P0-2 无跳过逻辑 + 多 rank 并发重填**：`fill_multi_task_replay` 无条件
  `add_count = 0` 后重填，而 `train.py` 用 `mp.spawn` 起 N 个 rank，**每个都调它**，
  写的是同一批 `<cursor>.replay` 文件名。后果是 (a) 每次启动/续训重建 176 GB；
  (b) 同名文件被两个进程同时 `open('wb')` → pickle 半截写入 → 训练跑到几小时后
  突然 `UnpicklingError`；(c) 各 rank 的 `_task_idxs` 与磁盘内容错配 →
  task_uniform 采样静默失真。→ `attach()` 让 replay 只建一次、之后只读打开。

* **公平性**：三臂必须看到逐字节相同的样本。共享同一份不可变工件是最直接的保证。

# 工件结构

    <replay_dir>/
        0.replay … <N-1>.replay     样本本体（pickle）
        _MANIFEST.json              人可读：样本数 / 逐任务计数 / 字段签名 / 来源
        _INDEX.pkl                  机器可读：重开 replay 所需的全部内存状态

`_INDEX.pkl` 里只有四样东西，因为 `use_disk=True` 时 `_store` 除 `terminal`
以外什么都不存（`uniform_replay_buffer.py:226`）：
    task_idxs / terminal 数组 / add_count / 结构参数

# 设计红线

`attach()` 的每一道校验失败都 **raise**，绝不静默降级。本项目已有两次
「静默跳过 → 拿到残缺数据 → 得出错误结论」的教训（P1 调度器静默跳过 10 个作业、
P3 修复脚本留下 208 条缺 keypoints 的记录），凡是「多套数据共存、按配置选用」的
结构都必须硬校验。尤其是字段签名：拿不含 subtask_* 的 replay 去训 B3，
必须当场报错，而不是训出一个说不清是什么的模型。
"""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA_VERSION = "vqap_replay_artifact_v1"
MANIFEST = "_MANIFEST.json"
INDEX = "_INDEX.pkl"

#: attach() 时随机抽验多少个样本文件，确认目录内容与索引确实同源
SPOT_CHECK_N = 32


class ReplayArtifactError(RuntimeError):
    """工件缺失、损坏，或与当前配置不匹配。"""


# ---------------------------------------------------------------- 路径

def replay_dir(replay_path: str | Path, tasks: Iterable[str],
               method_name: str = "PERACT_BC", seed: int = 0) -> Path:
    """工件目录 —— 必须与 `run_seed_fn.run_seed` 的计算逐字符一致。

    上游（run_seed_fn.py:57）：
        task_folder = task if not multi_task else 'multi'
        replay_path = os.path.join(cfg.replay.path, task_folder,
                                   cfg.method.name, 'seed%d' % seed)
    其中 multi_task = len(cfg.rlbench.tasks) > 1。

    构建脚本与训练脚本都必须走这个函数，不许各自拼路径 —— P0-1 就是这么来的。
    """
    tasks = list(tasks)
    task_folder = "multi" if len(tasks) > 1 else tasks[0]
    return Path(replay_path) / task_folder / method_name / f"seed{seed}"


# ---------------------------------------------------------------- 签名

def _dtype_str(t: Any) -> str:
    """ReplayElement.type 可能是 numpy 标量类型、`object` 或 `str`，统一成字符串。"""
    try:
        return np.dtype(t).str
    except TypeError:
        return getattr(t, "__name__", str(t))


def signature(replay_buffer) -> str:
    """replay 的字段结构指纹：名字 + shape + dtype，排序后取 sha256。

    B0/B1 与 B2/B3 用的是同一份含 subtask_* 字段的 replay，所以签名相同；
    但若有人误建了一份不含 subtask_* 的 replay 去训 B3，签名会立刻对不上。
    """
    storage, _ = replay_buffer.get_storage_signature()
    items = sorted(f"{e.name}|{tuple(e.shape)}|{_dtype_str(e.type)}" for e in storage)
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def _describe_fields(replay_buffer) -> list[str]:
    storage, _ = replay_buffer.get_storage_signature()
    return sorted(f"{e.name}{tuple(e.shape)}:{_dtype_str(e.type)}" for e in storage)


# ---------------------------------------------------------------- 保存

def save(replay_buffer, out: str | Path, *, extra: dict | None = None) -> dict:
    """把内存索引落盘，使该目录成为可复用的工件。

    调用时机：`fill_multi_task_replay` 刚跑完、replay 还在内存里的那一刻。
    """
    out = Path(out)
    if not out.is_dir():
        raise ReplayArtifactError(f"目录不存在：{out}")

    task_idxs = {str(k): [int(i) for i in v]
                 for k, v in replay_buffer._task_idxs.items()}
    n = int(replay_buffer.add_count)
    if n == 0:
        raise ReplayArtifactError("add_count 为 0，不写工件（极可能 fill 子进程全崩）")
    n_indexed = sum(len(v) for v in task_idxs.values())
    if n_indexed != n:
        raise ReplayArtifactError(
            f"索引条数 {n_indexed:,} != add_count {n:,}，replay 内部状态不自洽")

    terminal = np.asarray(replay_buffer._store["terminal"])

    with open(out / INDEX, "wb") as f:
        pickle.dump({
            "schema_version": SCHEMA_VERSION,
            "task_idxs": task_idxs,
            "terminal": terminal,
            "add_count": n,
            "replay_capacity": int(replay_buffer._replay_capacity),
            "timesteps": int(replay_buffer._timesteps),
            "update_horizon": int(replay_buffer._update_horizon),
        }, f, protocol=4)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "n_samples": n,
        "per_task": {k: len(v) for k, v in sorted(task_idxs.items())},
        "signature": signature(replay_buffer),
        "fields": _describe_fields(replay_buffer),
        "replay_capacity": int(replay_buffer._replay_capacity),
        "timesteps": int(replay_buffer._timesteps),
        "update_horizon": int(replay_buffer._update_horizon),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest.update(extra or {})
    (out / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


# ---------------------------------------------------------------- 读取

def is_built(out: str | Path) -> bool:
    out = Path(out)
    return (out / MANIFEST).is_file() and (out / INDEX).is_file()


def read_manifest(out: str | Path) -> dict:
    p = Path(out) / MANIFEST
    if not p.is_file():
        raise ReplayArtifactError(f"没有 {MANIFEST}：{p}")
    return json.loads(p.read_text())


def attach(replay_buffer, out: str | Path, *,
           spot_check: int = SPOT_CHECK_N, verbose: bool = True) -> dict:
    """把已建好的工件挂到一个空 replay_buffer 上，**跳过 fill**。

    六道校验，任一失败即 raise：
        1. schema 版本
        2. 字段签名逐位一致（防止拿错 replay 训错臂）
        3. 结构参数（capacity / timesteps / update_horizon）一致
        4. 索引自洽：逐任务计数之和 == add_count == manifest.n_samples
        5. 抽验 N 个样本文件可反序列化，且其 `task` 字段与索引记录相符
        6. 挂载后 replay 转为只读，再调 add() 直接报错
    """
    out = Path(out)
    man = read_manifest(out)

    if man.get("schema_version") != SCHEMA_VERSION:
        raise ReplayArtifactError(
            f"schema 版本不符：工件 {man.get('schema_version')} vs 代码 {SCHEMA_VERSION}")

    sig_now = signature(replay_buffer)
    if man.get("signature") != sig_now:
        want = set(man.get("fields", []))
        have = set(_describe_fields(replay_buffer))
        raise ReplayArtifactError(
            "replay 字段签名不匹配 —— 这份 replay 不是给当前配置建的。\n"
            f"  工件缺少（当前配置需要）: {sorted(have - want)[:6]}\n"
            f"  工件多出（当前配置不需要）: {sorted(want - have)[:6]}\n"
            "  常见原因：建 replay 时没传 planner_cache_dir，"
            "导致没有 subtask_* 字段，却拿去训 B2/B3。")

    with open(out / INDEX, "rb") as f:
        idx = pickle.load(f)

    for key in ("replay_capacity", "timesteps", "update_horizon"):
        if int(idx[key]) != int(getattr(replay_buffer, "_" + key)):
            raise ReplayArtifactError(
                f"{key} 不一致：工件 {idx[key]} vs 当前 {getattr(replay_buffer, '_' + key)}")

    task_idxs = {str(k): list(map(int, v)) for k, v in idx["task_idxs"].items()}
    n = int(idx["add_count"])
    n_indexed = sum(len(v) for v in task_idxs.values())
    if not (n == n_indexed == int(man["n_samples"])):
        raise ReplayArtifactError(
            f"样本数不自洽：add_count={n:,} 索引={n_indexed:,} "
            f"manifest={man['n_samples']:,}")

    terminal = np.asarray(idx["terminal"])
    if terminal.shape[0] != int(idx["replay_capacity"]):
        raise ReplayArtifactError(
            f"terminal 数组长度 {terminal.shape[0]} != capacity {idx['replay_capacity']}")

    # ---- 5. 抽验：目录里的样本文件确实与索引同源 ----
    # 只验 cursor -> task 的对应关系。这能抓住「目录被另一次构建覆盖」
    # （P0-2 里多 rank 并发写同名文件正是这种情形）。
    if spot_check > 0:
        cursor_to_task = {}
        for t, idxs in task_idxs.items():
            for c in idxs:
                cursor_to_task[c] = t
        rng = random.Random(0)                      # 固定种子，抽验可复现
        picks = rng.sample(sorted(cursor_to_task), min(spot_check, len(cursor_to_task)))
        for c in picks:
            fp = out / f"{c}.replay"
            if not fp.is_file():
                raise ReplayArtifactError(f"索引指向的样本文件不存在：{fp}")
            try:
                with open(fp, "rb") as f:
                    sample = pickle.load(f)
            except Exception as exc:               # 半截写入 / 损坏
                raise ReplayArtifactError(
                    f"样本文件损坏，无法反序列化：{fp}（{type(exc).__name__}: {exc}）"
                ) from None
            got = sample.get("task")
            if got != cursor_to_task[c]:
                raise ReplayArtifactError(
                    f"样本 {c}.replay 的 task='{got}'，索引却记为 "
                    f"'{cursor_to_task[c]}' —— 目录内容与索引不同源，"
                    "极可能被另一次构建覆盖过。")

    # ---- 恢复内存状态 ----
    # use_disk=True 时 _store 里只有 terminal 有意义，其余字段一律在磁盘上
    # （uniform_replay_buffer.py:226）。这里用普通 ndarray 而非 Manager 代理：
    # attach 路径没有写入子进程，普通数组更快，且 DataLoader 的 spawn worker
    # 能直接 pickle 过去。
    from yarr.replay_buffer.uniform_replay_buffer import invalid_range
    replay_buffer._store["terminal"] = terminal
    replay_buffer._task_idxs = task_idxs
    replay_buffer.add_count = n
    replay_buffer.invalid_range = invalid_range(
        replay_buffer.cursor(), replay_buffer._replay_capacity,
        replay_buffer._timesteps, replay_buffer._update_horizon)

    # ---- 6. 转只读 ----
    _seal(replay_buffer, out)

    # 🔴 立刻验证「封存后的 buffer 仍能被 spawn worker 接收」。
    #    DataLoader 用 spawn 起 worker 时会 pickle 整个 replay buffer；
    #    只要 add/add_final 上挂了不可 pickle 的对象，训练就会在
    #    `iter(dataset)` 处崩掉 —— 而那时冻结、attach、权重加载都已跑完，
    #    日志看起来一切正常，很容易误判成别的问题。这里当场炸，信息清楚得多。
    #    注意不能直接 pickle.dumps(replay_buffer)：它持有 mp.Lock，
    #    只有在**真正 spawn 时**才可传递（普通 pickle 会报
    #    "Lock objects should only be shared between processes through inheritance"）。
    #    所以只验我们自己挂上去的那两个属性。
    for attr in ("add", "add_final"):
        try:
            pickle.loads(pickle.dumps(getattr(replay_buffer, attr)))
        except Exception as exc:
            raise ReplayArtifactError(
                f"封存后的 replay.{attr} 无法 pickle（{type(exc).__name__}: {exc}）。"
                "DataLoader 的 spawn worker 会因此起不来，训练必然在 iter(dataset) 崩溃。"
            ) from None

    if verbose:
        per = man.get("per_task", {})
        print(f"[replay] 已挂载工件 {out}")
        print(f"[replay]   {n:,} 样本 / {len(task_idxs)} 任务 / "
              f"建于 {man.get('built_at')}")
        print(f"[replay]   逐任务: " + ", ".join(f"{k}={v}" for k, v in
                                                 sorted(per.items())[:4]) + " …")
    return man


def _blocked_write(save_dir: str, *_a, **_kw):
    """被 _seal 绑到 replay.add / add_final 上的哨兵。

    🔴 必须是**模块级**函数，不能是 _seal 里的闭包。
    replay buffer 会被 DataLoader 的 spawn worker pickle 过去
    （offline_train_runner.py:123 的 `iter(dataset)`），而局部函数不可 pickle：
        AttributeError: Can't pickle local object '_seal.<locals>._blocked'
    这条曾让 B1 训练在启动两分钟后崩掉 —— 前面的冻结、attach、权重加载全部成功，
    偏偏死在 worker 启动这一步。functools.partial 绑定模块级函数则可以 pickle。
    """
    raise ReplayArtifactError(
        f"这是一份只读 replay 工件（{save_dir}），不允许再 add()。"
        "若要重建，请显式删除目录或用 build_replay.py --force。")


def _seal(replay_buffer, out: Path) -> None:
    """挂载后禁止再写 —— 工件是不可变的，任何写入都说明调用方搞错了。"""
    blocked = functools.partial(_blocked_write, str(out))
    replay_buffer.add = blocked
    replay_buffer.add_final = blocked
    replay_buffer._is_readonly_artifact = True
