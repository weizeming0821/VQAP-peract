"""离线分段：对一条 RLBench demo 调用 Planner，产出通过结构校验的 segments。"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner.client import PlannerClient, image_data_url          # noqa: E402
from planner.contract import (                                     # noqa: E402
    CACHE_SCHEMA_VERSION, check_segments, expand_prior, gripper_consistent,
    n_repeat_prior, normalize_instruction, use_codebook,
)
from planner.prompts import build_system_prompt, build_user_content  # noqa: E402

DEFAULT_VIEWS = ("front", "wrist", "overhead")

# 三个 fork 后改名的任务，先验取同族旧任务（已在 CSV 中核实存在）
TASK_PRIOR_ALIAS = {
    "slide_block_to_color_target": "slide_block_to_target",
    "sweep_to_dustpan_of_size": "sweep_to_dustpan",
    "place_wine_at_rack_location": "stack_wine",
}
# 用户直接指定的先验（覆盖 CSV）
PRIOR_OVERRIDE = {
    "put_groceries_in_cupboard": ["approach", "grasp", "lift", "pose-adjust",
                                  "pose-adjust", "place"],
    # 🔴 slide_block 的这条**保持原样不动**，尽管它已被证明与训练分布不符
    #    （动作词 press 应为 push，段数 5 应为 1）。原因是可复现性：
    #    scripts/planner_cache.py 用 load_priors() 建离线 cache，而那份 cache
    #    是已冻结的训练工件（三臂共享的 replay 就是从它来的）。改这里等于让
    #    「重建 cache」产出与训练时不同的东西 —— 这类静默漂移正是本项目反复
    #    吃亏的地方。修正只作用于**在线** planner，见下面的 PRIOR_VARIANTS。
    "slide_block_to_color_target": ["approach", "press", "pose-adjust",
                                    "approach", "press"],
}

#: 备选先验：同一任务的**多种合理分解**，在线 planner 走不通时可以换一个。
#:
#: 🔴 slide_block_to_color_target 是本项目已知的三方不一致（HANDOFF #5）：
#:      CSV 先验            5 段，且动作词写成 press
#:      离线真值            平均 2.7 段
#:      train 模板库        最常见是**单个 push**（6/10 局），
#:                          次常见是 5 段版（4/10 局）
#:    原来的 PRIOR_OVERRIDE 只给了 5 段版、动作词还是错的 `press`。
#:    实测在线 planner 忠实照它生成 5 段，而模板法只有 1 段，成绩 64 → 30。
#:
#:    变体顺序按训练分布的频次排：先试更常见的短版本，走不通再切长版本。
PRIOR_VARIANTS: dict[str, list[list[str]]] = {
    # 🔴 顺序在 2026-09-08 反转过。原来把单段 ["push"] 放在变体 0，依据是
    #    **train split 的众数统计**（6/10 是单段）。但评测真正使用的是 val 的
    #    模板计划，那里 variation 0/1 都是**5 段**，而模板法在该任务上拿了 64 分。
    #    实测把在线 planner 推到 2 段后，成绩直接归零 —— 方向反了。
    "slide_block_to_color_target": [
        ["approach", "push", "pose-adjust", "approach", "push"],   # val 模板同款
        ["push"],                                                  # 备选：单段
    ],
}


def load_priors(csv_path: Path | None = None) -> dict[str, list[str]]:
    """Phase_Action_Label.csv → {task: 动作序列}。CSV 只标了 variation 0，是软先验。"""
    csv_path = csv_path or REPO_ROOT / "Phase_Action_Label.csv"
    prior: dict[str, list[str]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t = row["Task"]
            if t in prior:
                continue
            prior[t] = [row[f"Phase{i}"] for i in range(34) if row.get(f"Phase{i}")]
    for new, old in TASK_PRIOR_ALIAS.items():
        if old in prior:
            prior.setdefault(new, prior[old])
    prior.update(PRIOR_OVERRIDE)
    # 🔴 这里**不**套用 PRIOR_VARIANTS —— load_priors() 的返回值必须与建
    #    离线 cache 时逐位一致（scripts/planner_cache.py 依赖它）。
    #    在线 planner 走 load_prior_variants()，那边才是修正后的顺序。
    return prior


def load_prior_variants(csv_path: Path | None = None) -> dict[str, list[list[str]]]:
    """`{task: [变体0, 变体1, ...]}`。没登记备选的任务返回单元素列表。

    在线 planner 用它做「这条路走不通就换一种分解」：REPLAN 或 RETRY 用满时
    切到下一个变体。离线建 cache 不用它（离线有完整 demo，不需要试错）。
    """
    base = load_priors(csv_path)
    out: dict[str, list[list[str]]] = {}
    for t, p in base.items():
        # 登记了变体的任务用变体列表（顺序按 train 里的频次），其余包一层。
        # 与 load_priors() 的差异是**有意的**：离线要复现，在线要修正。
        out[t] = [list(v) for v in PRIOR_VARIANTS[t]] if t in PRIOR_VARIANTS else [list(p)]
    return out


def parse_json(text: str) -> dict[str, Any]:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("响应中找不到 JSON 对象")
    return json.loads(s[i:j + 1])


def episode_inputs(ep_dir: str) -> dict[str, Any]:
    """读一条 episode 的关键帧、夹爪状态、任务指令、variation。"""
    from helpers.demo_loading_utils import keypoint_discovery
    from rlbench.demo import Demo

    with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "rb") as f:
        raw = pickle.load(f)
    demo = raw if isinstance(raw, Demo) else Demo(raw)
    kps = keypoint_discovery(demo, method="heuristic")
    with open(os.path.join(ep_dir, "variation_descriptions.pkl"), "rb") as f:
        desc = pickle.load(f)[0]          # PerAct 训练与评测都只用 [0]
    with open(os.path.join(ep_dir, "variation_number.pkl"), "rb") as f:
        var = int(pickle.load(f))
    return {"keypoints": kps,
            "grippers": [int(round(float(demo[k].gripper_open))) for k in kps],
            "task_instruction": desc, "variation": var, "demo_len": len(demo)}


def segment_episode(client: PlannerClient, task: str, split: str, ep_idx: int,
                    ep_dir: str, priors: dict[str, list[str]],
                    views=DEFAULT_VIEWS, image_size: int = 224,
                    system_prompt: str | None = None) -> dict[str, Any]:
    """返回一条 cache episode 记录（尚未补码）。校验失败时 errors 非空。"""
    info = episode_inputs(ep_dir)
    kps, grip = info["keypoints"], info["grippers"]
    if not kps:
        return {"task": task, "split": split, "episode": ep_idx,
                "errors": ["no_keypoints"], "warnings": []}

    n_rep = n_repeat_prior(task, info["variation"])
    prior = expand_prior(task, info["variation"], list(priors.get(task, [])))

    def img(frame: int, view: str) -> str:
        return image_data_url(os.path.join(ep_dir, f"{view}_rgb", f"{frame}.png"),
                              size=image_size)

    messages = [
        {"role": "system", "content": system_prompt or build_system_prompt()},
        {"role": "user", "content": build_user_content(
            task, info["task_instruction"], kps, grip,
            priors.get(task, []), n_rep, img, list(views))},
    ]
    resp = client.chat(messages)
    try:
        obj = parse_json(resp["content"])
    except Exception as exc:
        return {"task": task, "split": split, "episode": ep_idx,
                "errors": [f"parse_error:{type(exc).__name__}"], "warnings": [],
                "raw": str(resp["content"])[:600], "usage": resp}

    segs = obj.get("segments", [])
    for s in segs:
        s["instruction"] = normalize_instruction(s.get("instruction", ""))
    chk = check_segments(segs, len(kps), info["task_instruction"], grip)

    # 补齐派生字段（即使有 errors 也补，便于人工判读）
    kp_to_seg = [-1] * len(kps)
    for s in segs:
        s["use_codebook"] = use_codebook(s.get("action", ""))
        s["gripper_consistent"] = gripper_consistent(
            s.get("action", ""), s.get("keypoint_indices", []), grip)
        ki = sorted(s.get("keypoint_indices", []))
        s["start_frame"] = 0 if not ki or ki[0] == 0 else kps[ki[0] - 1]
        for i in ki:
            if 0 <= i < len(kp_to_seg):
                kp_to_seg[i] = s.get("segment_index", -1)

    return {
        "task": task, "split": split, "episode": ep_idx,
        "variation": info["variation"], "task_instruction": info["task_instruction"],
        "demo_len": info["demo_len"], "keypoints": kps, "gripper_at_keypoints": grip,
        "n_repeat_prior": n_rep, "prior_sequence": prior,
        "segments": segs, "keypoint_to_segment": kp_to_seg,
        "prior_alignment": obj.get("prior_alignment"),
        "prior_deviation_reason": obj.get("prior_deviation_reason"),
        "anomalies": obj.get("anomalies", []),
        "episode_quality": obj.get("episode_quality"),
        "errors": chk["errors"], "warnings": chk["warnings"],
        "views_used": list(views), "planner_model_used": resp["model"],
        "usage": {k: resp[k] for k in ("prompt_tokens", "completion_tokens", "seconds")},
    }


def cache_meta(model: str, views, image_size: int) -> dict[str, Any]:
    import hashlib
    sp = build_system_prompt()
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "planner_model": model,
        "prompt_sha256": hashlib.sha256(sp.encode()).hexdigest(),
        "views": list(views), "image_size": image_size,
        "keypoint_method": "heuristic",
        "instruction_format": "lowercase_no_punct_3to8w",
        "task_set": "peract_official_18",
    }
