#!/usr/bin/env python
"""Stage 3 评测与 K 选择的唯一入口。

    python scripts/stage3_eval.py setup-b0                      # 造 B0 伪实验目录
    python scripts/stage3_eval.py bench  --arm B0               # 量单局耗时
    python scripts/stage3_eval.py run    --arm B0 --episodes 25 # 跑评测
    python scripts/stage3_eval.py run    --arm B1 --ckpt missing --watch
    python scripts/stage3_eval.py prune  --arm B1               # 只留 best + latest
    python scripts/stage3_eval.py report                        # 汇总进 result/p5_train/p5.json

# 设计说明

**不重造评测轮子**：底层直接调官方 `source/peract/eval.py`，它已经处理了
checkpoint 选择（missing/best/last/具体步数）、`eval_data.csv` 累积、
录像、多任务统计。本脚本只做三件官方没做的事：

  1. **路径对齐**。eval.py 把 checkpoint 目录拼成
     `framework.logdir / rlbench.task_name / method.name / seed<N>`，
     而我们的布局是 `checkpoints/<exp>/<arm>/seed0`。三段分别喂
     `logdir=checkpoints`、`task_name=<exp>`、`method.name=<arm>` 即可对上。
     `eval_cfg.method.name` 只在这一处拼路径用到（eval.py:207），
     建 agent 走的是 train_cfg.method.name，所以这样借用是安全的。

  2. **B0 没有训练过，也就没有 config.yaml 和 weights/ 目录**。
     `setup-b0` 用官方权重**硬链接**造一个伪实验目录（不复制 157 MB），
     并合成一份 `stage3.arm=B0` 的 config.yaml。

  3. **计时与汇总**。单局耗时是整个评测排期的唯一未知数，这里每次都量并记下来。

# 并行度

官方的并行粒度是**按 checkpoint 分进程**（`framework.eval_envs`），
多任务在每个进程内部串行。所以：
  * 扫 8 个 ckpt 时，`--parallel 2` 意味着同时评 2 个 ckpt；
  * 只评 1 个 ckpt 时，`--parallel` 无效——那种情况只能靠减局数提速。
`--gpu` 通过 CUDA_VISIBLE_DEVICES 限定用哪张卡，便于把评测放到训练之外的卡上。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# 本脚本多处按需 import stage3.*（臂定义、planner、replay 工件）。
# 在模块级一次性入路径，免得某个调用路径漏了 sys.path.insert 就 ImportError。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PERACT_ROOT = REPO_ROOT / "source" / "peract"
CKPT_ROOT = REPO_ROOT / "checkpoints"
RESULT = REPO_ROOT / "result" / "p5_train"
OFFICIAL = (PERACT_ROOT / "ckpts" / "multi" / "PERACT_BC" / "seed0" /
            "weights" / "600000" / "QAttentionAgent_layer0.pt")

# 任务划分的真源在 stage3/tasks.py。这里曾经抄过一份 SEEN12/UNSEEN6，
# 而同一份名单在 planner_cache / plan_audit / preflight 里还各有一份 ——
# 任务集是评测口径的一部分，改一处漏三处不会报错，只会安静地评错任务集。
from stage3.tasks import resolve as resolve_tasks     # noqa: E402

#: B0 的伪 checkpoint 步数。用官方的 600000 以示它来自 peract_600k，未经本项目训练。
B0_STEP = 600000


def arm_dir(exp: str, arm: str, seed: int = 0) -> Path:
    return CKPT_ROOT / exp / arm / f"seed{seed}"


def rotate_eval_csv_if_schema_changed(d: Path, tasks: list[str]) -> Path | None:
    """任务集与现有 `eval_data.csv` 的表头不符时，把旧文件归档，让表头重写。

    🔴 YARR 的 CSVWriter 只在**文件不存在**时写表头
       （`log_writer.py` 里 `should_write_train_header = not os.path.exists(...)`）。
       换一批任务再评，列数变了而表头不变 —— 新行会与旧表头**逐列错位**，
       按列名取成绩全部落空（`snapshot()` 拿到一堆 NaN，均值成 None 后直接抛
       `TypeError: unsupported format string passed to NoneType.__format__`）。
       更坏的情况是列数恰好相同：错位悄无声息，读出来的是**另一个任务的成绩**。

    分片路径也需要它：`_merge_shards` 会把新结果与既有 `eval_data.csv` 外连接，
    旧口径的任务列会作为 NaN 列留下来，把「列数 == 期望任务数」的硬校验顶掉。

    2026-09 的 Seen12 → Seen18/UnSeen 改版正踩在这个点上，所以必须常备。

    归档而不是删除：旧文件里可能存着不可再现的历史成绩。
    返回归档路径；无需轮转时返回 None。
    """
    csv_path = d / "eval_data.csv"
    if not csv_path.is_file():
        return None
    head = csv_path.read_text().splitlines()
    header = head[0].split(",") if head else []
    need = [f"eval_envs/return/{t}" for t in tasks]
    extra = [c for c in header if c.startswith("eval_envs/return/") and c not in need]
    missing = [c for c in need if c not in header]
    # 缺列会错位，多列会顶掉完整性校验 —— 两种都要轮转
    if not missing and not extra:
        return None
    archive = d / f"eval_data.{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path.rename(archive)
    why = []
    if missing:
        why.append(f"缺 {[c.rsplit('/', 1)[-1] for c in missing][:3]}…")
    if extra:
        why.append(f"多出 {[c.rsplit('/', 1)[-1] for c in extra][:3]}…")
    print(f"  ⚠️ eval_data.csv 的表头与本次任务集不符（{'；'.join(why)}），"
          f"已归档为 {archive.name}，本轮重写表头。")
    return archive


# ------------------------------------------------------------------ setup-b0

def cmd_setup_b0(a) -> int:
    """造 B0 的伪实验目录：官方权重硬链接 + 合成 config.yaml。"""
    from omegaconf import OmegaConf

    if not OFFICIAL.is_file():
        print(f"❌ 找不到官方权重 {OFFICIAL}", file=sys.stderr)
        return 1

    d = arm_dir(a.exp, "B0", a.seed)
    wd = d / "weights" / str(B0_STEP)
    wd.mkdir(parents=True, exist_ok=True)
    dst = wd / OFFICIAL.name
    if not dst.exists():
        try:
            os.link(OFFICIAL, dst)          # 硬链接：同一份 inode，不占额外空间
            how = "硬链接"
        except OSError:
            shutil.copy2(OFFICIAL, dst)     # 跨文件系统时退回复制
            how = "复制"
        print(f"  ✅ B0 权重已{how}: {dst}")
    else:
        print(f"  已存在: {dst}")

    # 合成 train_cfg —— 与三臂共用同一份 stage3.yaml，只把 arm 改成 B0。
    # arm='B0' 时 create_agent 不冻结、不装码本；act() 的语言防线也放行整任务指令。
    cfg = OmegaConf.load(PERACT_ROOT / "conf" / "stage3.yaml")
    base = OmegaConf.load(PERACT_ROOT / "conf" / "method" / "PERACT_BC.yaml")
    cfg.method = OmegaConf.merge(base, cfg.get("method", {}))
    cfg.stage3.arm = "B0"
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    (d / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(f"  ✅ config.yaml (stage3.arm=B0) -> {d/'config.yaml'}")
    return 0


# ---------------------------------------------------------------------- run

def pick_displays(n: int, base: int | None = None) -> list[int]:
    """给 n 个分片分配**互不相同**的 X display 号。

    🔴 不能用 `xvfb-run -a`。它的分配是「扫 /tmp/.X<n>-lock 找空号」和
       「起 Xvfb 占住这个号」两步，中间没有锁。并发启动的分片会挑中同一个号：
       先退出的那个在 trap 里 kill 自己的 Xvfb、删掉 lock，
       把另一个分片的 X server 一并带走 —— 对方立刻死于
       `XIO: fatal IO error on X server ":<n>"`。
       这不是假想：上一轮我的评测分片就是这样把 B3 训练打死在 step 15600。
       改成父进程一次性分配互不相交的号，再用 `xvfb-run -n <num>` 钉死。

    base 从 130 起：机器上 :99..:120 残留着 22 个陈旧 lock（对应的 Xvfb 早已不在），
    避开它们，也避开别人常用的低号段。

    🔴 **并发跑多个 stage3_eval 时必须用 `--display-base` 错开号段。**
    本函数只扫「现在有没有 lock」，两个父进程同时扫会挑到同一批号；
    它们各自的 12 个分片随后互相踩，先退出的那批会把对方的 X server
    一起带走。号段由调用方显式分开是唯一可靠的办法
    （例如 ckpt A 用 130、ckpt B 用 160）。
    """
    if base is None:
        base = int(os.environ.get("AAVLA_DISPLAY_BASE", "130"))
    out, d = [], base
    while len(out) < n and d < base + 400:
        if not (Path(f"/tmp/.X{d}-lock").exists()
                or Path(f"/tmp/.X11-unix/X{d}").exists()):
            out.append(d)
        d += 1
    if len(out) < n:
        raise SystemExit(f"❌ 找不到 {n} 个空闲 X display（从 :{base} 起扫了 400 个）")
    return out


def _eval_once(exp: str, arm: str, tasks: list[str], split: str, episodes: int,
               ckpt, parallel: int, gpu: str | None, seed: int,
               record_every_n: int, log_path: Path | None,
               display: int | None = None, planner: str = "template",
               trace_dir: Path | None = None,
               codes: str = "plan") -> tuple[int, float]:
    """调一次官方 eval.py。返回 (returncode, 墙钟秒数)。"""
    d = arm_dir(exp, arm, seed)
    if not (d / "config.yaml").is_file():
        raise SystemExit(f"❌ {d}/config.yaml 不存在。B0 请先跑 setup-b0；"
                         f"其余臂要先训练过。")

    # eval.py:207 把目录拼成
    #     framework.logdir / rlbench.task_name / method.name / seed<N>
    # 我们的布局是 checkpoints/<exp>/<arm>/seed0，于是三段分别喂
    #     logdir=checkpoints   task_name=<exp>   method.name=<arm>
    # —— 不用空串（subprocess 传列表参数时 shell 引号会变成字面量），
    #    也不影响功能：eval_cfg.method.name 只用于这一处拼路径，
    #    建 agent 用的是 train_cfg.method.name（从 <arm>/seed0/config.yaml 读）。
    overrides = [
        f"framework.logdir={CKPT_ROOT}",
        f"rlbench.task_name={exp}",
        f"method.name={arm}",
        f"framework.start_seed={seed}",
        f"framework.eval_episodes={episodes}",
        "framework.eval_from_eps_number=0",
        f"framework.eval_envs={parallel}",
        f"framework.eval_type={ckpt}",
        f"framework.record_every_n={record_every_n}",
        f"rlbench.demo_path={REPO_ROOT / 'aavla_data' / 'rlbench' / split}",
        "rlbench.tasks=[" + ",".join(tasks) + "]",
        "rlbench.headless=True",
        # 别让 hydra 每次都新建 outputs/<日期>/<时刻> 目录
        f"hydra.run.dir={REPO_ROOT / 'log' / 'p5' / 'hydra_eval'}",
        "hydra.output_subdir=null",
    ]
    # B2/B3 需要在线 Planner（B0/B1 走不到这个分支，eval.py 里按臂判断）。
    # 计划文件由 (split, planner) 推出，见 stage3/planners.py。
    overrides.append(f"stage3.planner={planner}")
    overrides.append(f"stage3.codes={codes}")
    if trace_dir is not None:
        overrides.append(f"stage3.trace_dir={trace_dir}")
    env = dict(os.environ)
    # `python -u` 只让**父进程**无缓冲；eval.py 内部用 Process(spawn) 起的子进程
    # 会重新执行 python 而不带这个标志，rollout 的 "Evaluating <task> | Episode N"
    # 全在子进程里打，于是日志长时间看不到任何进度（误以为卡死）。
    # PYTHONUNBUFFERED 是环境变量，会被子进程继承。
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    # -u：子进程 stdout 指向文件时 Python 默认块缓冲，日志要攒满 8 KB 才落盘，
    # 长任务期间完全看不到进度。加 -u 强制行缓冲。
    # -n <num>：显式钉死 display 号（见 pick_displays 的说明）。
    # 故意不加 -a：号被占时就该响亮地失败，而不是悄悄漂到别人的号上去。
    xvfb = ["xvfb-run", "-n", str(display)] if display is not None else ["xvfb-run", "-a"]
    cmd = xvfb + [sys.executable, "-u", "eval.py"] + overrides

    print(f"  $ CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES','')} "
          f"{' '.join(xvfb)} python eval.py {' '.join(overrides[:6])} …")
    t0 = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # 🔴 "wb" 而非 "ab"：追加模式下上一轮的日志会留在文件里，
        #    按行数统计进度时会把旧内容算进来 —— 我就因此把「本轮 0 局」
        #    误读成「已评 300 局」，白白排查了半天。每轮从头写。
        with log_path.open("wb") as f:
            rc = subprocess.call(cmd, cwd=PERACT_ROOT, env=env, stdout=f, stderr=f)
    else:
        rc = subprocess.call(cmd, cwd=PERACT_ROOT, env=env)
    return rc, time.time() - t0


#: 单个评测进程的显存占用（8 路并行实测 32705 MiB / 8）。留 15% 余量。
EVAL_MEM_MB = 4100
EVAL_MEM_MARGIN = 1.15
#: 卡上有**别人**的进程时，给他们留出的不动用显存。
#: 机会卡策略允许与别人共用一张卡（只要显存够），但共用不等于吃满 ——
#: 别人的作业显存占用会随 batch/阶段波动，我们把余量吃光就会把他们挤爆。
#: 8 GB 是经验值：够别人的训练做一次显存峰值波动。
OTHERS_KEEP_MB = 8 * 1024


def gpu_free_mb() -> dict[int, int]:
    """各卡的空闲显存（MiB）。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    free = {}
    for line in out.strip().splitlines():
        i, used, total = (int(x.strip()) for x in line.split(","))
        free[i] = total - used
    return free


def plan_gpus(n_shards: int, allow: list[int] | None = None) -> list[str] | None:
    """按**实测空闲显存**给 n_shards 个分片分配卡；装不下则返回 None。

    🔴 这道预检是真实事故的产物：共享集群上别的用户随时会吃掉显存，
    上一次 6 个分片里 4 个跑到一半 CUDA OOM 死掉、2 个还活着，
    合并出来是一份**残缺但看起来正常**的 eval_data.csv —— 正是本项目最忌讳的
    静默错误。宁可启动前就明确拒绝。
    """
    free = gpu_free_mb()
    if allow:
        free = {i: v for i, v in free.items() if i in allow}
    # 共用卡上给别人留余量（OTHERS_KEEP_MB），绝不把卡吃满把别人挤爆。
    try:
        from tools.gpu_reserver import others_on
        shared = {i for i in free if others_on(i)}
    except Exception:                        # 取不到就按「都是共用卡」保守处理
        shared = set(free)
    if shared:
        print("  共用卡 " + ",".join(f"GPU{i}" for i in sorted(shared))
              + f"：各预留 {OTHERS_KEEP_MB} MiB 给别人，不吃满")
        free = {i: (v - OTHERS_KEEP_MB if i in shared else v)
                for i, v in free.items()}
        free = {i: v for i, v in free.items() if v > 0}
    need = int(EVAL_MEM_MB * EVAL_MEM_MARGIN)
    # 每张卡能放几个；**跨卡轮转**而不是填满一张再填下一张 ——
    # 这些卡上通常跑着别人的训练，摊开能把干扰降到最低
    # （评测实测 GPU 利用率近 0%，瓶颈在 CPU 侧的运动规划与仿真）。
    cap = {i: mb // need for i, mb in sorted(free.items(), key=lambda kv: -kv[1]) if mb // need}
    slots, rnd = [], 0
    while cap and len(slots) < n_shards:
        progressed = False
        for i in list(cap):
            if cap[i] > rnd:
                slots.append(str(i)); progressed = True
                if len(slots) >= n_shards:
                    break
        if not progressed:
            break
        rnd += 1
    if len(slots) < n_shards:
        print(f"❌ 显存不足：{n_shards} 个分片需要每个 {need} MiB，"
              f"当前全机只能放下 {len(slots)} 个。各卡空闲：" +
              ", ".join(f"GPU{i}={mb}" for i, mb in sorted(free.items())), file=sys.stderr)
        return None
    return slots[:n_shards]


def _make_shard(exp: str, arm: str, k: int, seed: int) -> str:
    """造一个分片实验目录：config 复制 + 所有 ckpt 权重**硬链接**（零额外磁盘）。"""
    src = arm_dir(exp, arm, seed)
    name = f"{arm}__sh{k}"
    dst = arm_dir(exp, name, seed)
    (dst / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "config.yaml", dst / "config.yaml")
    for step in (src / "weights").iterdir():
        if not step.name.isdigit():
            continue
        d = dst / "weights" / step.name
        d.mkdir(exist_ok=True)
        for f in step.iterdir():
            t = d / f.name
            if not t.exists():
                try:
                    os.link(f, t)
                except OSError:
                    shutil.copy2(f, t)
    return name


def _merge_shards(exp: str, arm: str, shards: list[str], seed: int,
                  groups: list[list[str]] | None = None) -> int:
    """把各分片的 eval_data.csv 按 step 外连接合并回主 arm 目录。

    每个分片只评了任务全集的一个子集，按 step 外连接即可拼回完整的一行。

    🔴 列名要自己补任务后缀。YARR 只在**多任务 env** 下给列名加后缀
    （`_independent_env_runner.py:272` 的 `if eval_task_name and multi_task`）。
    分片数 == 任务数时每片只有一个任务，env 走的是单任务分支，
    12 份 CSV 的列名全都叫 `eval_envs/return`，外连接会把它们**塌成一列**，
    合出来一份「只有 4 列、看起来却很正常」的结果 —— 正是本项目最忌讳的静默错误
    （实测就这么发生过一次：12 任务的结果合成了 1 列）。
    """
    import pandas as pd
    dfs = []
    for i, name in enumerate(shards):
        f = arm_dir(exp, name, seed) / "eval_data.csv"
        if not f.is_file():
            continue
        d = pd.read_csv(f)
        tasks = groups[i] if groups else []
        if len(tasks) == 1:                      # 单任务分片：列名缺后缀，补上
            t = tasks[0]
            d = d.rename(columns={c: f"{c}/{t}" for c in d.columns
                                  if c.startswith("eval_envs/") and "/" not in c[10:]})
        dfs.append(d)
    if not dfs:
        return 0
    out = dfs[0]
    for d in dfs[1:]:
        dup = [c for c in d.columns if c in out.columns and c != "step"]
        out = out.merge(d.drop(columns=dup), on="step", how="outer")
    dst = arm_dir(exp, arm, seed) / "eval_data.csv"
    if dst.is_file():                      # 与既有结果合并，不覆盖
        prev = pd.read_csv(dst)
        keep = prev[~prev["step"].isin(out["step"])]
        out = pd.concat([keep, out], ignore_index=True)
    out = out.sort_values("step")
    out.to_csv(dst, index=False)
    # 硬校验。分两层，缺一不可：
    #   ① 列数 —— 防列名塌陷（12 个分片各跑 1 任务时 YARR 不加任务后缀）
    #   ② 值非空 —— 🔴 列是按分片任务名补出来的，**分片没产出数据时列照样存在**，
    #      只是值为空。只查列数会放过「11/12 个任务有数据」这种残缺结果：
    #      实测就发生过一次，put_groceries_in_cupboard 整列为空，
    #      而退出码 0、列数 12、一切看起来正常，均值却是拿 11 个任务算的。
    n_task_cols = sum(1 for c in out.columns if c.startswith("eval_envs/return/"))
    n_expected = sum(len(g) for g in groups) if groups else n_task_cols
    bad = n_task_cols != n_expected
    if bad:
        print(f"  ❌ 合并后只有 {n_task_cols} 个任务的 return 列，应有 {n_expected} 个"
              f" —— 列名塌陷，结果不可用。", file=sys.stderr)
    if groups:
        want = [t for g in groups for t in g]
        for _, row in out.iterrows():
            miss = [t for t in want
                    if pd.isna(row.get(f"eval_envs/return/{t}", float("nan")))]
            if miss:
                bad = True
                print(f"  ❌ step {int(row['step'])}: {len(miss)} 个任务没有结果 "
                      f"-> {miss} —— 这一行是残缺的，**不要拿它算均值**。",
                      file=sys.stderr)
    if not bad:
        print(f"  ✅ 校验通过：{n_task_cols} 个任务列，全部有值")
    return len(out)


def snapshot(a, tasks: list[str]) -> dict | None:
    """把这一轮的逐任务结果**立刻**归档成一份带配置标签的 JSON。

    🔴 为什么必须归档而不是回头读 CSV：
    `checkpoints/<exp>/<arm>/seed0/eval_data.csv` 是**单一数据流，按 step 行**，
    不同 planner / codes 配置跑同一个 step 会**互相覆盖**。
    实测踩过：CSV 里 step 2500 的值 13.33 其实是 vlm 那轮的，
    把 template 那轮的 15.83 覆盖了，回头看根本分不出是谁的成绩。

    另外这里**逐任务检查有没有值**。分片失败时 YARR 照样返回 0、列也照样在，
    只是值为空 —— 实测 put_groceries_in_cupboard 就这样整列缺失，
    而"11 个任务的均值"被当成了 12 任务的结果。complete=False 时必须重跑。
    """
    import pandas as pd
    f = arm_dir(a.exp, a.arm, a.seed) / "eval_data.csv"
    if not f.is_file():
        return None
    df = pd.read_csv(f)
    want_step = None
    try:
        want_step = int(a.ckpt)
    except (TypeError, ValueError):
        pass
    rows = df[df["step"] == want_step] if want_step is not None else df.tail(1)
    if rows.empty:
        return None
    row = rows.iloc[-1]
    per, miss = {}, []
    for t in tasks:
        v = row.get(f"eval_envs/return/{t}", float("nan"))
        if pd.isna(v):
            miss.append(t)
        else:
            per[t] = float(v)
    tag = a.planner + ("" if a.codes == "plan" else "+adapter")
    doc = {
        "arm": a.arm, "split": a.split, "ckpt": a.ckpt,
        "episodes_per_task": a.episodes, "planner": a.planner, "codes": a.codes,
        "n_tasks_expected": len(tasks), "n_tasks_got": len(per),
        "complete": not miss, "missing": miss,
        "mean": (sum(per.values()) / len(per)) if per else None,
        "per_task": per,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = (REPO_ROOT / "result" / "p7" /
           f"{a.arm}_{a.split}_ep{a.episodes}_{tag}_{a.ckpt}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    flag = "✅" if doc["complete"] else "❌ 残缺，勿用"
    # mean 可能是 None：CSV 里找不到这一 step 的行（例如单任务 bench 走的是
    # 非分片路径，YARR 不给列加任务后缀）。此时不能用 :.2f 格式化。
    mean_txt = f"{doc['mean']:.2f}%" if doc["mean"] is not None else "（无数据）"
    print(f"  {flag} {doc['n_tasks_got']}/{len(tasks)} 任务，均值 "
          f"{mean_txt}  -> {out.relative_to(REPO_ROOT)}")
    if miss:
        print(f"     缺失: {miss}", file=sys.stderr)
    return doc


def cmd_run(a) -> int:
    if getattr(a, "display_base", None):
        os.environ["AAVLA_DISPLAY_BASE"] = str(a.display_base)

    tasks = resolve_tasks(a.tasks)
    # 🔴 必须在任何分片启动之前轮转一次，且**只轮转一次**：分片是并发起的，
    #    每片各转一次会互相抢同一个文件。
    rotate_eval_csv_if_schema_changed(arm_dir(a.exp, a.arm, a.seed), tasks)
    log = (REPO_ROOT / "log" / "p5" / f"{a.arm}.eval.out") if not a.foreground else None

    # --wait-gpu：共享集群上显存随时被别人吃掉，与其失败退出不如等。
    if a.wait_gpu and a.shards > 1 and str(a.gpu) in ("auto",) or (
            a.wait_gpu and a.gpu and "," in str(a.gpu)):
        allow = None if str(a.gpu) == "auto" else [int(g) for g in str(a.gpu).split(",")]
        deadline = time.time() + a.wait_gpu
        while plan_gpus(a.shards, allow) is None:
            if time.time() > deadline:
                print(f"❌ 等待 {a.wait_gpu}s 后显存仍不足，放弃。", file=sys.stderr)
                return 2
            print(f"  显存不足，{60}s 后重试（剩余等待 "
                  f"{int(deadline - time.time())}s）…", flush=True)
            time.sleep(60)

    while True:
        if a.shards > 1 and len(tasks) > 1:
            rc, dt = _run_sharded(a, tasks, log, a.stagger)
        else:
            rc, dt = _eval_once(a.exp, a.arm, tasks, a.split, a.episodes, a.ckpt,
                                a.parallel, a.gpu, a.seed, a.record_every_n, log,
                                pick_displays(1)[0], a.planner, trace_dir(a),
                                a.codes)
        n_ep = len(tasks) * a.episodes
        print(f"  → 退出码 {rc}，墙钟 {dt/60:.1f} min"
              + (f"，约 {dt/n_ep:.1f} s/局（{n_ep} 局，未计 ckpt 数）" if n_ep else ""))
        doc = snapshot(a, tasks) if rc == 0 else None
        if not a.watch:
            # 结果残缺就以非零码退出：让调用方的脚本停下来，
            # 而不是把一份 11/12 的均值当成结果继续往下走。
            return rc if (doc is None or doc["complete"]) else 3
        # --watch：等训练吐出新 ckpt 再评。eval.py 在 eval_type=missing 且没有
        # 待评 ckpt 时会 sys.exit(0)，所以这里就是轮询。
        time.sleep(a.poll)


def _run_sharded(a, tasks: list[str], log, stagger: float = 25.0) -> tuple[int, float]:
    """把任务切成 a.shards 份并行评，再合并 CSV。

    官方 eval.py 的并行粒度是**按 checkpoint** 分进程（framework.eval_envs），
    多任务在进程内串行。所以只评一个 checkpoint 时（例如 B0 基准）它无法并行。
    这里按任务切分，每片一个临时实验目录（权重硬链接，零额外磁盘），
    跑完合并 eval_data.csv 再删掉分片。

    实测 rollout 是 CPU-bound（208 核、单个评测进程约 4.1 G 显存），
    4 路近线性、8 路仍有 1.7x 增益。
    """
    import threading
    n = min(a.shards, len(tasks))
    groups = [tasks[i::n] for i in range(n)]     # 交错切分，任务难度更均衡
    names, results = [], [None] * n

    # 分卡策略：
    #   --gpu auto           按实测空闲显存自动挑卡（推荐，共享集群上最稳）
    #   --gpu 6,7            指定候选卡，仍按空闲显存做预检与分配
    #   --gpu <单个> / 不给   沿用旧行为（不做预检）
    if str(a.gpu) == "auto":
        gpus = plan_gpus(n, None)
        if gpus is None:
            return 2, 0.0
    elif a.gpu and "," in str(a.gpu):
        allow = [int(g) for g in str(a.gpu).split(",")]
        gpus = plan_gpus(n, allow)
        if gpus is None:
            return 2, 0.0
    else:
        gpus = [a.gpu]

    # 每片一个专属 X display —— 在**起线程之前**一次性分配，
    # 保证我们自己的分片之间绝不撞号（见 pick_displays）。
    displays = pick_displays(n)

    def work(k: int) -> None:
        name = names[k]
        # 🔴 日志名带上 ckpt/planner/codes：原来固定叫 <arm>.eval.sh<k>.out，
        #    下一轮开跑就把上一轮截断掉。实测因此丢失了一次分片失败的现场
        #    （put_groceries_in_cupboard 整列为空，想回查时日志已被覆盖）。
        tag = f"{a.ckpt}_{a.planner}" + ("" if a.codes == "plan" else "+adapter")
        lg = (REPO_ROOT / "log" / "p5" / f"{a.arm}.eval.{tag}.sh{k}.out") if log else None
        results[k] = _eval_once(a.exp, name, groups[k], a.split, a.episodes,
                                a.ckpt, 1, gpus[k % len(gpus)], a.seed,
                                a.record_every_n, lg, displays[k],
                                a.planner, trace_dir(a), a.codes)

    t0 = time.time()
    try:
        names = [_make_shard(a.exp, a.arm, k, a.seed) for k in range(n)]
        print(f"  分片 {n} 路：" + " | ".join(
            f"sh{k}→GPU{gpus[k % len(gpus)]}/:{displays[k]}({len(g)}任务)"
            for k, g in enumerate(groups)))
        # 🔴 错峰启动。每个评测进程冷启动时要读 torch / CoppeliaSim / CLIP /
        #    模型权重共几 GB，12 个同时启动会把共享盘打穿：实测
        #    sda 队列深度冲到 242、781 MB/s，全部进程卡在
        #    D 状态（folio_wait_bit_common）19 分钟零输出。
        #    错开之后启动 I/O 被摊平，rollout 阶段本身是 CPU-bound、I/O 很轻。
        ts = [threading.Thread(target=work, args=(k,)) for k in range(n)]
        for i, t in enumerate(ts):
            if i:
                time.sleep(stagger)
            t.start()
        for t in ts:
            t.join()
        rows = _merge_shards(a.exp, a.arm, names, a.seed, groups)
        print(f"  已合并 {n} 个分片的结果 -> eval_data.csv（{rows} 行）")
    finally:
        for name in names:
            shutil.rmtree(arm_dir(a.exp, name, a.seed).parent, ignore_errors=True)
    rc = max((r[0] if r else 1) for r in results)
    return rc, time.time() - t0


# -------------------------------------------------------------------- prune

def _read_curve(d: Path) -> dict[int, float]:
    """从 eval_data.csv 读 step -> 总均值 sr。"""
    import pandas as pd
    f = d / "eval_data.csv"
    if not f.is_file():
        return {}
    df = pd.read_csv(f)
    cols = [c for c in df.columns if c.startswith("eval_envs/return/")]
    if not cols:
        cols = [c for c in df.columns if c == "eval_envs/return"]
    if not cols:
        return {}
    return {int(r["step"]): float(sum(r[c] for c in cols) / len(cols))
            for _, r in df.iterrows()}


def cmd_prune(a) -> int:
    d = arm_dir(a.exp, a.arm, a.seed)
    wd = d / "weights"
    steps = sorted(int(p.name) for p in wd.iterdir() if p.name.isdigit())
    if not steps:
        print(f"  {a.arm}: weights/ 为空，跳过")
        return 0
    curve = _read_curve(d)
    latest = steps[-1]
    best = max(curve, key=curve.get) if curve else None

    if best is None and not a.force:
        print(f"❌ {a.arm}: 没有 eval_data.csv，无法判定 best。"
              f"先跑 run 出曲线，或用 --force 只保留 latest。", file=sys.stderr)
        return 2

    keep = {latest} | ({best} if best is not None else set())
    drop = [s for s in steps if s not in keep]
    freed = sum(f.stat().st_size for s in drop
                for f in (wd / str(s)).rglob("*") if f.is_file()) / 1024 ** 3
    print(f"  {a.arm}: 共 {len(steps)} 个 ckpt；保留 "
          f"best={best}{'' if best is None else f'(sr={curve[best]:.3f})'} "
          f"latest={latest}；删除 {len(drop)} 个，释放 {freed:.2f} GB")
    if a.dry_run:
        print("   （--dry-run，未实际删除）")
        return 0
    for s in drop:
        shutil.rmtree(wd / str(s))
    return 0


# --------------------------------------------------------------------- plans

#: 预生成计划的落盘位置。放在 planner_cache 下，与离线 cache 同源同域。
def plan_path(split: str, planner: str = "template") -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    from stage3.planners import plan_file
    return plan_file(split, planner)


def trace_dir(a) -> Path | None:
    """这一轮评测的 planner 轨迹目录；`--no-trace` 时返回 None。

    目录名带上 planner 种类与 ckpt，让 template / flat 两组、不同 ckpt 的
    轨迹并存可比 —— 「该不该换 planner」就是靠对比它们来判的。
    用整任务指令的臂（B0/B1）没有 planner，直接不记。
    """
    from stage3.arms import ARMS
    if getattr(a, "no_trace", False) or not ARMS[a.arm].uses_subtask_lang:
        return None
    tag = a.planner + ("" if a.codes == "plan" else "+adapter")
    return (REPO_ROOT / "result" / f"p6_{tag}" / "traces" /
            f"{a.arm}_{a.split}_{a.ckpt}")


def cmd_plans(a) -> int:
    """为某个 split 预生成 B2/B3 评测用的子任务计划。

    🔴 预生成而非在线现算：B2 与 B3 读**同一份**计划，两臂看到的子任务序列
    逐字相同，差值里只剩「有没有注入码」。这比每步调 VLM 的方案公平性更强，
    而且零 API 成本、零时延、可复现。

    --planner flat 派生对照组：分段与码逐字不变，只把每段指令换成整任务指令。
    `B3(template) − B3(flat)` 就是子任务分解本身值多少分 —— 这是判断
    「要不要改用在线 VLM planner」的依据。
    """
    sys.path.insert(0, str(REPO_ROOT))
    from stage3.cache_join import PlannerCache
    from stage3.online_planner import (build_template_bank, build_plans,
                                       flatten_plans, build_oracle_plans)

    out = plan_path(a.split, a.planner)

    if a.planner == "flat":
        src = plan_path(a.split, "template")
        if not src.is_file():
            print(f"❌ 先生成 template 计划：plans --split {a.split}", file=sys.stderr)
            return 2
        doc = flatten_plans(json.loads(src.read_text()), a.split)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        miss = doc["coverage"].get("flat_missing_descriptions", 0)
        print(f"  flat 计划 {len(doc['plans'])} 条 -> {out}")
        ex = next(iter(doc["plans"].values()))
        print(f"  样例：{ex['task']}/{ex['episode']} 的 {len(ex['subtasks'])} 段"
              f"指令全部为 “{ex['subtasks'][0]['instruction']}”")
        if miss:
            print(f"  ⚠️  {miss} 局读不到 variation_descriptions，已跳过", file=sys.stderr)
            return 2
        return 0

    tasks = resolve_tasks(a.tasks)
    cache = PlannerCache(a.cache_dir)

    if a.planner == "oracle":
        # 真值分段只有 train split 的 cache 里有 —— 这也是 oracle 只能在
        # train 上做的原因（见 build_oracle_plans 的说明）。
        if a.split != "train":
            print("❌ oracle 计划只有 train split 有真值分段", file=sys.stderr)
            return 2
        doc = build_oracle_plans(cache, tasks, a.split, a.episodes)
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        c = doc["coverage"]
        print(f"  oracle 计划 {len(doc['plans'])} 条 -> {out}")
        print(f"  真值命中 {c['exact']}、缺失 {c['missing']}")
        return 2 if c["missing"] else 0

    print(f"  {cache.summary()}")
    bank = build_template_bank(cache, tasks, split=a.bank_split)
    print(f"  模板库：{bank['stats']['n_entries']} 个 (task,variation) 条目")
    doc = build_plans(bank, tasks, a.split, a.episodes)
    doc["planner"] = "template"
    c = doc["coverage"]
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"  计划 {len(doc['plans'])} 条 -> {out}")
    print(f"  variation 精确命中 {c['exact']} ({c['exact']/max(c['total'],1):.1%})、"
          f"回退 {c['fallback']}、缺失 {c['missing']}")
    if c["missing"]:
        print(f"  ❌ 有 {c['missing']} 局没有计划 —— B2/B3 评到这些局会硬失败。",
              file=sys.stderr)
        return 2
    if c["fallback"]:
        print(f"  ⚠️  {c['fallback']} 局的 variation 不在模板库里，退回任务级模板；"
              f"这些局的指令中物体/颜色可能与实际场景不符，判读时需注意。")
    return 0


def cmd_trace(a) -> int:
    """统计 planner 轨迹 —— 回答「状态机到底有没有在干活」。

    三个病症各有明确判据（见 P6 回撤条件）：
      · 从不推进   reached_index == 0 的局占比 > 30%  → 状态机失灵
      · 推进失控   前 25% 步数就冲到末段的局 > 30%
      · 推进过慢   reached_frac 中位数偏低
    没有这份统计，B3 成绩不好时无法区分是 planner 还是模型的问题。
    """
    d = Path(a.dir)
    files = sorted(d.glob("*.json"))
    if not files:
        print(f"❌ {d} 下没有轨迹文件", file=sys.stderr)
        return 2
    import statistics as st
    rows, per_task = [], collections.defaultdict(list)
    for f in files:
        r = json.loads(f.read_text())
        # 🔴 判据用「计划总预算」当基准，不能用 n_steps：
        #    失败的局会一直跑到 episode_length（25 步）才结束，
        #    用 n_steps 当分母会把「模型失败」误判成「planner 推进失控」。
        #    计划总预算 sum(budgets) 与模型表现无关，是干净的基准。
        budget = max(sum(r["budgets"]), 1)
        first_last = next((h["t"] for h in r["trace"]
                           if h["subtask_index"] == r["n_subtasks"] - 1), None)
        r["reached_last_at"] = first_last
        r["budget"] = budget
        r["steps_in_last"] = sum(1 for h in r["trace"]
                                 if h["subtask_index"] == r["n_subtasks"] - 1)
        r["early_last"] = (first_last is not None and r["n_subtasks"] > 1
                           and first_last < 0.5 * budget)
        rows.append(r); per_task[r["task"]].append(r)
    n = len(rows)
    never = sum(1 for r in rows if r["reached_index"] == 0 and r["n_subtasks"] > 1)
    early = sum(1 for r in rows if r["early_last"])
    multi = [r for r in rows if r["n_subtasks"] > 1]
    print(f"  轨迹 {n} 局（其中 {len(multi)} 局是多段任务）  <- {d}")
    print(f"  从不推进（reached_index==0）      {never:4d}  {never/max(len(multi),1):6.1%}"
          f"   {'🔴 >30%，状态机失灵' if never > 0.3*max(len(multi),1) else '✅'}")
    print(f"  推进失控（不到计划预算一半就到末段）{early:4d}  {early/max(len(multi),1):6.1%}"
          f"   {'🔴 >30%' if early > 0.3*max(len(multi),1) else '✅'}")
    if multi:
        il = sorted(r["steps_in_last"] / max(r["n_steps"], 1) for r in multi)
        print(f"  停在末段的步数占比  中位 {il[len(il)//2]:.2f}"
              f"   （偏高通常是模型没做完、局跑满长度，不一定是 planner 的问题）")
    if multi:
        fr = sorted(r["reached_frac"] for r in multi)
        print(f"  走完的段数占比  中位 {st.median(fr):.2f}  "
              f"（1.00 表示走到最后一段）")
    print()
    print(f"  {'任务':<32}{'局数':>5}{'段数':>6}{'到达段占比中位':>16}{'从不推进':>10}")
    for t in sorted(per_task):
        rs = per_task[t]; m = [r for r in rs if r["n_subtasks"] > 1]
        nv = sum(1 for r in m if r["reached_index"] == 0)
        med = st.median([r["reached_frac"] for r in m]) if m else float("nan")
        print(f"  {t:<32}{len(rs):>5}{st.median([r['n_subtasks'] for r in rs]):>6.0f}"
              f"{med:>16.2f}{nv:>10}")
    return 0


# ------------------------------------------------------------------- report

def cmd_report(a) -> int:
    """把配置、吞吐、卡数履历、val 曲线、选定的 K 汇总进**一个** p5.json。"""
    import re
    RESULT.mkdir(parents=True, exist_ok=True)
    out = RESULT / "p5.json"
    doc = json.loads(out.read_text()) if out.is_file() else {}

    # ---- 配置快照 ----
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(PERACT_ROOT / "conf" / "stage3.yaml")
    doc["config"] = {
        "lr": float(cfg.method.lr),
        "batch_per_gpu": int(cfg.replay.batch_size),
        "num_devices": int(cfg.ddp.num_devices),
        "effective_batch": int(cfg.replay.batch_size) * int(cfg.ddp.num_devices),
        "training_iterations": int(cfg.framework.training_iterations),
        "save_freq": int(cfg.framework.save_freq),
        "lr_scheduler": bool(cfg.method.lr_scheduler),
        "tasks": list(cfg.rlbench.tasks),
        "replay_path": str(cfg.replay.path),
    }
    doc["git"] = {
        r: subprocess.run(["git", "-C", str(p), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
        for r, p in (("main", REPO_ROOT), ("peract", PERACT_ROOT),
                     ("YARR", REPO_ROOT / "source" / "YARR"))
    }

    # ---- replay 工件 ----
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from stage3 import replay_dataset
        art = replay_dataset.replay_dir(cfg.replay.path, list(cfg.rlbench.tasks))
        if replay_dataset.is_built(art):
            m = replay_dataset.read_manifest(art)
            doc["replay"] = {k: m[k] for k in
                             ("n_samples", "per_task", "signature", "size_gb",
                              "build_seconds", "built_at") if k in m}
    except Exception as exc:
        doc["replay"] = {"error": str(exc)[:120]}

    # ---- 每臂：训练吞吐（直接从训练日志读，不另设脚本）+ val 曲线 ----
    doc.setdefault("arms", {})
    from stage3.arms import ARMS as _ARMS
    for arm in _ARMS:                      # 臂名单以 stage3/arms.py 为准，不在这里写死
        d = arm_dir(a.exp, arm, a.seed)
        if not d.exists():
            continue
        entry = doc["arms"].setdefault(arm, {})
        wd = d / "weights"
        if wd.is_dir():
            entry["checkpoints"] = sorted(int(p.name) for p in wd.iterdir()
                                          if p.name.isdigit())
        log = REPO_ROOT / "log" / "p5" / f"{arm}.train.out"
        if log.is_file():
            # YARR 每 log_freq 步打一行：
            #   Train Step 000100 | Loss: 2.1 | Sample time: 0.01 | Step time: 0.85.
            rows = re.findall(
                r"Train Step (\d+) \| Loss: ([\d.]+).*?Step time: ([\d.]+)",
                log.read_text(errors="ignore"))
            if rows:
                tail = rows[-20:]
                entry["train"] = {
                    "last_step": int(rows[-1][0]),
                    "loss_first20_mean": round(
                        sum(float(r[1]) for r in rows[:20]) / len(rows[:20]), 4),
                    "loss_last20_mean": round(
                        sum(float(r[1]) for r in tail) / len(tail), 4),
                    "sec_per_step_median": round(
                        sorted(float(r[2]) for r in rows)[len(rows) // 2], 4),
                    "n_log_rows": len(rows),
                }
        curve = _read_curve(d)
        if curve:
            entry["val_curve"] = {str(k): round(v, 4) for k, v in sorted(curve.items())}
            entry["val_best_step"] = max(curve, key=curve.get)
            entry["val_best_sr"] = round(curve[max(curve, key=curve.get)], 4)

    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"✅ 已写入 {out}")
    for arm, e in doc.get("arms", {}).items():
        bits = []
        if "train" in e:
            bits.append(f"step={e['train']['last_step']} "
                        f"loss={e['train']['loss_last20_mean']} "
                        f"{e['train']['sec_per_step_median']}s/step")
        if "val_best_step" in e:
            bits.append(f"best@{e['val_best_step']} sr={e['val_best_sr']}")
        print(f"   {arm}: " + ("；".join(bits) if bits else "（无数据）"))
    return 0


# ---------------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="stage3_main", help="checkpoints/<exp>/")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup-b0").set_defaults(fn=cmd_setup_b0)

    for name, defaults in (("run", {}),
                           ("bench", {"episodes": 3, "tasks": ["close_jar"],
                                      "ckpt": None})):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--tasks", nargs="+",
                       default=defaults.get("tasks", ["seen18"]),
                       help="任务名，或任务组：seen18 / unseen10 / unseen_a / "
                            "unseen_b / all28（历史：seen12 / unseen6）")
        p.add_argument("--split", default="val", choices=["train", "val", "test"])
        p.add_argument("--episodes", type=int, default=defaults.get("episodes", 25))
        p.add_argument("--ckpt", default=defaults.get("ckpt", "missing"),
                       help="missing | best | last | 具体步数")
        p.add_argument("--parallel", type=int, default=1,
                       help="同时评几个 ckpt（官方并行粒度是 ckpt，不是任务）")
        p.add_argument("--shards", type=int, default=1,
                       help="把任务切成几份并行评（单 ckpt 时唯一的提速手段）")
        p.add_argument("--gpu", default=None,
                       help="auto=按空闲显存自动挑卡（推荐）；也可 6,7 指定候选；"
                            "单个数字则不做预检")
        p.add_argument("--stagger", type=float, default=25.0, metavar="SEC",
                       help="分片之间的启动间隔秒数（避免冷启动惊群打穿磁盘）")
        p.add_argument("--wait-gpu", type=int, default=0, metavar="SEC",
                       help="显存不足时轮询等待的总秒数（0=不等，直接失败）")
        p.add_argument("--record-every-n", type=int, default=-1)
        p.add_argument("--watch", action="store_true", help="轮询等新 ckpt")
        p.add_argument("--poll", type=int, default=600)
        p.add_argument("--foreground", action="store_true", help="日志打到终端")
        p.add_argument("--planner", default="template",
                       choices=["template", "flat", "oracle", "vlm", "vlm-plan"],
                       help="B2/B3 用哪种在线 planner（见 stage3/planners.py）；"
                            "flat 是把子任务指令换成整任务指令的对照组")
        p.add_argument("--codes", default="plan", choices=["plan", "adapter"],
                       help="B3 的码来源：plan=模板库查表（旧行为）；"
                            "adapter=实时调 Stage 2 Adapter（对齐设计）")
        p.add_argument("--display-base", type=int, default=None,
                       help="X display 起始号。并发跑多个评测时必须错开，"
                            "如 130 / 160 / 190")
        p.add_argument("--no-trace", action="store_true",
                       help="不记录 planner 轨迹（默认记录，用于事后诊断）")
        p.set_defaults(fn=cmd_run)
    # bench 默认评最后一个 ckpt
    for act in sub._name_parser_map["bench"]._actions:
        if act.dest == "ckpt" and act.default is None:
            act.default = "last"

    p = sub.add_parser("prune")
    p.add_argument("--arm", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="没有 val 曲线时只留 latest")
    p.set_defaults(fn=cmd_prune)

    p = sub.add_parser("plans")
    p.add_argument("--planner", default="template",
                   choices=["template", "flat", "oracle"],
                   help="template 从 train cache 建模板库；"
                        "flat 由已有的 template 计划派生（只换语言）；"
                        "oracle 用每局自己的真值分段（仅 train split）")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=25, help="每任务多少局")
    p.add_argument("--tasks", nargs="+", default=["seen18"],
                   help="任务名，或任务组：seen18 / unseen10 / all28"
                        "（历史：seen12 / unseen6）")
    p.add_argument("--bank-split", default="train",
                   help="模板取自哪个 split 的离线 cache（默认 train）")
    p.add_argument("--cache-dir",
                   default=str(REPO_ROOT / "aavla_data" / "planner_cache" / "train"))
    p.set_defaults(fn=cmd_plans)

    p = sub.add_parser("trace")
    p.add_argument("--dir", required=True, help="result/p6_<planner>/traces/<arm>_<split>_<ckpt>")
    p.set_defaults(fn=cmd_trace)

    sub.add_parser("report").set_defaults(fn=cmd_report)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
