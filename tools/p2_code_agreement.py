#!/usr/bin/env python
"""D1 诊断：码本语义错位量化（临时工具，非流水线的一部分）。

回答 `VLA_Design §7 R1`：Adapter 学的是「AtomAction phase 起始帧 + short 指令 → 该 phase 轨迹
经 VQAP 编码器得到的码」，Stage 3 却要喂它「PerAct segment 起始帧 + Planner 指令」。
两者是否错位？错多远？

做法：对 RLBench demo 的每个 segment 同时取两路码——

    oracle : segment 低维轨迹 → normalize(traj_stats) → VQAP 编码器 → argmin → (k_g*, k_d*)
    pred   : segment 起始帧 (front, wrist) + 指令 → 冻结 Adapter          → (k_g^, k_d^)

一致率即 R1 的直接量化。oracle 部署时拿不到（没有未来轨迹），只作诊断参照。

**两种指令来源各跑一遍**，把误差来源拆开：
  - `atomaction` : AtomAction 的 short 指令，**按描述文本内容匹配**（不依赖 variation 索引对齐）
                   → Adapter 域内上界，差异只来自时间边界 + 图像域
  - `template`   : 由 RLBench 任务指令派生的模板指令，接近 Planner 实际会产出的措辞
                   → 再叠加措辞漂移的影响
若 atomaction 高而 template 低 → 问题在措辞，改 Planner prompt 即可；
两者都低 → 问题在时间边界/图像域，才需要动 Adapter。

用法：
    source run/env.sh
    python tools/p2_code_agreement.py --tasks open_drawer close_jar stack_cups --episodes 20
    python tools/p2_code_agreement.py --episodes 100 --out result/p2_d1_diagnosis/agreement.json
"""

from __future__ import annotations

import argparse
import collections
import difflib
import glob
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
PERACT_ROOT = REPO_ROOT / "source" / "peract"
for p in (str(REPO_ROOT), str(PERACT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.dataset import LOW_DIM_FIELDS                              # noqa: E402
from data.utils import AtomActionDataset_collate_fn, normalize       # noqa: E402
from model.adapter import Adapter                                    # noqa: E402
from model.module.atomaction_nsvq import AtomAction_NSVQ             # noqa: E402

# AtomAction ∩ PerAct18（VLA_Design §7 R2：另 5 个任务不在 AtomAction 中）
TASKS_IN_ATOMACTION = [
    "open_drawer", "meat_off_grill", "turn_tap", "put_item_in_drawer", "close_jar",
    "reach_and_drag", "stack_blocks", "light_bulb_in", "put_money_in_safe",
    "put_groceries_in_cupboard", "place_shape_in_shape_sorter", "push_buttons",
    "insert_onto_square_peg",
]
ATOMACTION_ROOT = REPO_ROOT / "AtomAction_Dataset"
MODULE_PREFIX = "atomaction_nsvq."


# ----------------------------------------------------------------------------- 指令
def load_atomaction_instructions() -> dict[str, list[tuple[str, int, str]]]:
    """{task: [(short 指令, phase_index, action), ...]}，跨 action 目录聚合。

    刻意**不按 variation 索引组织**——AtomAction 与 RLBench 的 variation 编号是否对齐
    并不影响 Stage 3（Stage 3 不用 AtomAction 数据，CSV 也只是参考），
    这里改为在使用时按任务指令的文本内容做匹配。
    """
    out: dict[str, list[tuple[str, int, str]]] = collections.defaultdict(list)
    for meta_path in glob.glob(str(ATOMACTION_ROOT / "*" / "*" / "variation*" / "variation_metadata.json")):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            continue
        task = meta.get("task_name")
        if task not in TASKS_IN_ATOMACTION:
            continue
        for pd in meta.get("phase_descriptions", []):
            short = next((d["text"] for d in pd.get("descriptions", [])
                          if d.get("style") == "short"), None)
            if short:
                out[task].append((short.rstrip(". "), int(pd.get("phase_index", 0)),
                                  pd.get("action_label", "")))
    return out


def instruction_atomaction(task: str, action: str, seg_idx: int,
                           pool: dict[str, list[tuple[str, int, str]]],
                           rlbench_desc: str) -> str | None:
    """按 (action 标签一致) + (与 RLBench 任务指令文本最相似) 选一条 AtomAction short 指令。"""
    cands = [s for s, _, a in pool.get(task, []) if a == action]
    if not cands:
        cands = [s for s, _, _ in pool.get(task, [])]
    if not cands:
        return None
    return max(cands, key=lambda s: difflib.SequenceMatcher(
        None, s.lower(), rlbench_desc.lower()).ratio())


# 由 RLBench 任务指令派生的模板指令：动词 + 任务指令的名词部分。
# 目的不是做得漂亮，而是模拟 Planner 会产出的「动作 + 区分性信息」的措辞。
_ACTION_VERB = {
    "approach": "approach", "grasp": "grasp", "lift": "lift", "place": "place",
    "push": "push", "pull": "pull", "press": "press", "rotate": "rotate",
    "slide": "slide", "insert": "insert", "hang": "hang", "wipe": "wipe",
    "transfer": "move", "flip-open": "open", "flip-close": "close",
    "revolve-in": "close", "revolve-out": "open", "pose-adjust": "adjust",
}
_STOP = {"the", "a", "an", "to", "in", "on", "at", "of", "then", "and", "with", "into"}


def instruction_template(action: str, rlbench_desc: str) -> str:
    verb = _ACTION_VERB.get(action, action)
    words = [w for w in rlbench_desc.lower().replace(",", " ").split() if w not in _STOP]
    # 丢掉任务指令自带的动词（首词），保留其余的区分性信息（颜色/序号/方位/物体）
    obj = " ".join(words[1:]) if len(words) > 1 else " ".join(words)
    obj = " ".join(obj.split()[:5])
    return f"{verb} the {obj}".strip()


# ----------------------------------------------------------------------------- 模型
def load_models(device: torch.device):
    ck_path = REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "latest.pth"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    nsvq = AtomAction_NSVQ(model_args=ck["model_args"])
    sd = {k[len(MODULE_PREFIX):]: v for k, v in ck["model"].items()
          if k.startswith(MODULE_PREFIX)}
    missing = set(nsvq.state_dict()) - set(sd)
    if missing:
        raise RuntimeError(f"VQAP encoder 权重缺失: {sorted(missing)[:5]}")
    nsvq.load_state_dict(sd, strict=True)
    nsvq.eval().to(device)

    ack = torch.load(REPO_ROOT / "checkpoints" / "vqap_adapter" / "best.pth",
                     map_location="cpu", weights_only=False)
    adapter = Adapter(ack["model_args"])
    _, unexpected = adapter.load_state_dict(ack["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Adapter 出现 unexpected keys: {unexpected[:5]}")
    adapter.eval().to(device)

    cb = torch.load(REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth",
                    map_location="cpu", weights_only=False)
    g = next(v for k, v in cb["global_codebook"].items() if v.ndim == 2)
    d = next(v for k, v in cb["detail_codebook"].items() if v.ndim == 2)
    return nsvq, adapter, g.float(), d.float(), str(ck_path)


def encode_oracle(nsvq, obs_slice, device) -> tuple[int, list[int]]:
    """一段 Observation 序列 → (k_global, k_detail[9])。"""
    traj = {f: [] for f in LOW_DIM_FIELDS}
    for o in obs_slice:
        for f in LOW_DIM_FIELDS:
            v = getattr(o, f, None)
            traj[f].append(v.tolist() if hasattr(v, "tolist") else v)
    traj = normalize(trajectory_data=traj, dataset_root=str(ATOMACTION_ROOT))
    # collate_fn 还要 Action/Task/Variation/selected_views 四个字段，
    # 它们只被原样打包、不参与编码，这里填占位值即可。
    batch = AtomActionDataset_collate_fn([{
        "trajectory_data": traj, "trajectory_length": len(obs_slice),
        "Action": "", "Task": "", "Variation": 0, "selected_views": [],
    }])
    td = {k: v.to(device) for k, v in batch["trajectory_data"].items()}
    tm = batch["trajectory_mask"].to(device)
    with torch.no_grad():
        out = nsvq.encode_codebook_indices(td, tm)
    return int(out["global_codeindex"][0]), out["detail_codeindices"][0].cpu().tolist()


def load_frame(ep: str, cam: str, idx: int) -> torch.Tensor:
    img = Image.open(os.path.join(ep, f"{cam}_rgb", f"{idx}.png")).convert("RGB")
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).contiguous()


# ----------------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=TASKS_IN_ATOMACTION)
    ap.add_argument("--episodes", type=int, default=20, help="每任务取多少条 episode")
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=str(REPO_ROOT / "result" / "p2_d1_diagnosis" / "agreement.json"))
    args = ap.parse_args()

    device = torch.device(args.device)
    from helpers.demo_loading_utils import keypoint_discovery
    from rlbench.demo import Demo

    nsvq, adapter, cb_g, cb_d, enc_ckpt = load_models(device)
    pool = load_atomaction_instructions()
    print(f"VQAP 编码器: {enc_ckpt}")
    print(f"AtomAction 指令池: {sum(len(v) for v in pool.values())} 条，覆盖 {len(pool)} 个任务")

    # 码本几何：平均码间距，用于归一化码向量误差
    with torch.no_grad():
        dg = torch.cdist(cb_g, cb_g)
        mean_d_g = float(dg[~torch.eye(len(cb_g), dtype=bool)].mean())
        dd = torch.cdist(cb_d, cb_d)
        mean_d_d = float(dd[~torch.eye(len(cb_d), dtype=bool)].mean())
    print(f"码本平均两两距离: global {mean_d_g:.2f} / detail {mean_d_d:.2f}")

    records: list[dict] = []
    for task in args.tasks:
        eps = sorted(glob.glob(f"aavla_data/rlbench/{args.split}/{task}/all_variations/episodes/episode*"),
                     key=lambda p: int(os.path.basename(p).removeprefix("episode")))
        eps = [e for e in eps if os.path.isfile(f"{e}/low_dim_obs.pkl")][:args.episodes]
        n_seg = 0
        for ep in eps:
            with open(f"{ep}/low_dim_obs.pkl", "rb") as f:
                raw = pickle.load(f)
            demo = raw if isinstance(raw, Demo) else Demo(raw)
            kps = keypoint_discovery(demo)
            if not kps:
                continue
            with open(f"{ep}/variation_descriptions.pkl", "rb") as f:
                desc = pickle.load(f)[0]
            with open(f"{ep}/variation_number.pkl", "rb") as f:
                var = int(pickle.load(f))

            # 每个 PerAct 关键帧区间 = 一个 segment（1 关键帧/段，最细粒度，
            # 无需 Planner 参与，诊断口径最干净）
            start = 0
            for si, k in enumerate(kps):
                seg = list(demo)[start:k + 1]
                if len(seg) < 2:
                    start = k
                    continue
                try:
                    kg, kd = encode_oracle(nsvq, seg, device)
                except Exception as exc:
                    print(f"  [skip] {task}/{os.path.basename(ep)} seg{si}: {exc}")
                    start = k
                    continue
                records.append({
                    "task": task, "variation": var, "episode": os.path.basename(ep),
                    "segment_index": si, "start_frame": start, "end_keypoint": k,
                    "seg_len": len(seg), "task_instruction": desc,
                    "k_global_oracle": kg, "k_detail_oracle": kd,
                    "_ep": ep,
                })
                n_seg += 1
                start = k
        print(f"  {task:32s} {len(eps):>3} ep → {n_seg:>5} segment")

    if not records:
        print("没有可用 segment")
        return 1

    # ---- Adapter 前向（两种指令来源）----
    # segment 的 action 标签在诊断阶段不可得，用 segment_index 到先验序列的粗映射，
    # 只为选一条措辞合理的指令；一致率统计与该标签无关。
    for r in records:
        r["action_guess"] = ("approach" if r["segment_index"] == 0 else
                             "grasp" if r["segment_index"] == 1 else "transfer")

    for source in ("atomaction", "template"):
        for r in records:
            if source == "atomaction":
                ins = instruction_atomaction(r["task"], r["action_guess"],
                                             r["segment_index"], pool, r["task_instruction"])
                if ins is None:
                    ins = instruction_template(r["action_guess"], r["task_instruction"])
            else:
                ins = instruction_template(r["action_guess"], r["task_instruction"])
            r[f"instruction_{source}"] = ins

        for i in range(0, len(records), args.batch):
            chunk = records[i:i + args.batch]
            front = torch.stack([load_frame(r["_ep"], "front", r["start_frame"]) for r in chunk]).to(device)
            wrist = torch.stack([load_frame(r["_ep"], "wrist", r["start_frame"]) for r in chunk]).to(device)
            with torch.no_grad():
                out = adapter({"front": front, "wrist": wrist},
                              [r[f"instruction_{source}"] for r in chunk])
            gl = out["global_logits"].float().cpu()
            dl = out["detail_logits"].float().cpu()
            top5 = gl.topk(5, dim=-1).indices
            for j, r in enumerate(chunk):
                r[f"k_global_pred_{source}"] = int(gl[j].argmax())
                r[f"k_global_top5_{source}"] = top5[j].tolist()
                r[f"k_detail_pred_{source}"] = dl[j].argmax(-1).tolist()
            if (i // args.batch) % 20 == 0:
                print(f"  [{source}] {i + len(chunk)}/{len(records)}", flush=True)

    # ---- 指标 ----
    def metrics(source: str) -> dict:
        top1 = np.mean([r[f"k_global_pred_{source}"] == r["k_global_oracle"] for r in records])
        top5 = np.mean([r["k_global_oracle"] in r[f"k_global_top5_{source}"] for r in records])
        dtop1 = np.mean([np.mean(np.array(r[f"k_detail_pred_{source}"]) == np.array(r["k_detail_oracle"]))
                         for r in records])
        err_g = np.mean([float(torch.norm(cb_g[r[f"k_global_pred_{source}"]] - cb_g[r["k_global_oracle"]]))
                         for r in records]) / mean_d_g
        # 一致性：同 (task, variation, segment_index) 内预测的众数占比
        grp = collections.defaultdict(list)
        for r in records:
            grp[(r["task"], r["variation"], r["segment_index"])].append(r[f"k_global_pred_{source}"])
        modes = [collections.Counter(v).most_common(1)[0][1] / len(v)
                 for v in grp.values() if len(v) >= 3]
        by_task = {}
        for t in sorted({r["task"] for r in records}):
            rs = [r for r in records if r["task"] == t]
            by_task[t] = {
                "n": len(rs),
                "top1": round(float(np.mean([r[f"k_global_pred_{source}"] == r["k_global_oracle"] for r in rs])), 4),
                "top5": round(float(np.mean([r["k_global_oracle"] in r[f"k_global_top5_{source}"] for r in rs])), 4),
            }
        return {
            "global_top1": round(float(top1), 4), "global_top5": round(float(top5), 4),
            "detail_slot_top1": round(float(dtop1), 4),
            "code_vector_norm_err": round(float(err_g), 4),
            "consistency_mode_share": round(float(np.mean(modes)), 4) if modes else None,
            "n_consistency_groups": len(modes),
            "by_task": by_task,
        }

    res = {source: metrics(source) for source in ("atomaction", "template")}

    print()
    print("=" * 92)
    print(f"D1 码本语义错位诊断  ({len(records)} 个 segment)")
    print("=" * 92)
    print(f"{'指标':28s} {'atomaction 指令':>16} {'template 指令':>16}   参照")
    ref = {"global_top1": "随机 2.8% / 域内 89.7%", "global_top5": "随机 13.9% / 域内 97.8%",
           "detail_slot_top1": "随机 0.5% / 域内 72.6%",
           "code_vector_norm_err": "随机 ≈1.0 / 域内 0.072",
           "consistency_mode_share": "健康 > 80%"}
    for k in ("global_top1", "global_top5", "detail_slot_top1",
              "code_vector_norm_err", "consistency_mode_share"):
        a, b = res["atomaction"][k], res["template"][k]
        fa = f"{a:.4f}" if a is not None else "-"
        fb = f"{b:.4f}" if b is not None else "-"
        print(f"{k:28s} {fa:>16} {fb:>16}   {ref[k]}")

    print()
    print(f"{'分任务 global top-1':28s} {'atomaction':>16} {'template':>16} {'n':>7}")
    for t in sorted(res["atomaction"]["by_task"]):
        print(f"{t:28s} {res['atomaction']['by_task'][t]['top1']:>16.4f} "
              f"{res['template']['by_task'][t]['top1']:>16.4f} "
              f"{res['atomaction']['by_task'][t]['n']:>7}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for r in records:
        r.pop("_ep", None)
    with out.open("w") as f:
        json.dump({
            "config": {"tasks": args.tasks, "episodes_per_task": args.episodes,
                       "split": args.split, "encoder_ckpt": enc_ckpt,
                       "segment_def": "1 keypoint per segment (PerAct keypoint_discovery)"},
            "codebook_geometry": {"mean_pairwise_dist_global": round(mean_d_g, 4),
                                  "mean_pairwise_dist_detail": round(mean_d_d, 4)},
            "metrics": res, "n_segments": len(records), "records": records,
        }, f, indent=1, ensure_ascii=False)
    print(f"\n明细写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
