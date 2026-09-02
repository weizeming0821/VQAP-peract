"""Cache 三道质量门禁（VLA_Design §5）。

① 结构校验：全量自动，任一 error 即该 episode 不合格
② 统计一致性体检：全量，产出报告供人工判读
③ 人工抽检：分层抽样清单（渲染由 tools/render_cache_review.py 负责）
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_cache(cache_dir: str | Path) -> tuple[dict, list[dict]]:
    """读所有分片，合并为 (meta, episodes)。分片间 meta 必须一致。"""
    metas, eps = [], []
    for f in sorted(glob.glob(str(Path(cache_dir) / "*.json"))):
        d = json.loads(Path(f).read_text())
        metas.append(d.get("meta", {}))
        eps.extend(d.get("episodes", []))
    if metas:
        keys = {json.dumps(m, sort_keys=True) for m in metas}
        if len(keys) > 1:
            raise ValueError(f"分片之间 meta 不一致，共 {len(keys)} 种——"
                             f"说明它们不是同一次配置下生成的，不能混用")
    return (metas[0] if metas else {}), eps


# ---------------------------------------------------------------- 门禁 ①
def gate1_structural(episodes: list[dict]) -> dict[str, Any]:
    """结构校验已在 offline.segment_episode 中逐条执行，这里做汇总 + C5/C6/C7 复核。"""
    from planner.contract import use_codebook

    bad, err_kinds = [], collections.Counter()
    c567 = collections.Counter()
    for ep in episodes:
        errs = list(ep.get("errors", []))
        # C7：use_codebook 必须与白名单规则一致（防止写入时被改动）
        for s in ep.get("segments", []):
            if s.get("use_codebook") != use_codebook(s.get("action", "")):
                errs.append(f"C7_use_codebook_mismatch@seg{s.get('segment_index')}")
                c567["C7"] += 1
        # C6：码索引范围（补码之后才有）
        for s in ep.get("segments", []):
            if "k_global" in s:
                if not (isinstance(s["k_global"], int) and 0 <= s["k_global"] < 36):
                    errs.append("C6_k_global_out_of_range")
                    c567["C6"] += 1
                kd = s.get("k_detail", [])
                if len(kd) != 9 or any(not (0 <= x < 192) for x in kd):
                    errs.append("C6_k_detail_invalid")
                    c567["C6"] += 1
        if errs:
            bad.append({"key": f"{ep['task']}/{ep.get('episode')}", "errors": errs})
            for e in errs:
                err_kinds[e.split("(")[0]] += 1
    return {"n_episodes": len(episodes), "n_bad": len(bad),
            "pass_rate": 1 - len(bad) / max(len(episodes), 1),
            "error_kinds": dict(err_kinds.most_common()),
            "bad_examples": bad[:30]}


# ---------------------------------------------------------------- 门禁 ②
def gate2_statistics(episodes: list[dict]) -> dict[str, Any]:
    ok = [e for e in episodes if not e.get("errors")]
    by_tv = collections.defaultdict(list)
    for e in ok:
        by_tv[(e["task"], e["variation"])].append(e)

    seg_counts = collections.defaultdict(list)      # (task,var) -> [段数]
    instr_uniq = collections.defaultdict(set)       # (task,var,seg_idx) -> {指令}
    kp_per_seg, actions = [], collections.Counter()
    align = collections.Counter()
    quality = collections.Counter()
    n_pose_adjust = 0
    n_seg_total = 0
    c8_violations = 0
    c8_applicable = 0
    warn_kinds = collections.Counter()

    for (t, v), eps in by_tv.items():
        for e in eps:
            seg_counts[(t, v)].append(len(e["segments"]))
            align[e.get("prior_alignment")] += 1
            quality[e.get("episode_quality")] += 1
            for w in e.get("warnings", []):
                warn_kinds[w.split("(")[0].split("@")[0]] += 1
            for s in e["segments"]:
                n_seg_total += 1
                actions[s["action"]] += 1
                if s["action"] == "pose-adjust":
                    n_pose_adjust += 1
                kp_per_seg.append(len(s.get("keypoint_indices", [])))
                instr_uniq[(t, v, s["segment_index"])].add(s["instruction"])
                gc = s.get("gripper_consistent")
                if gc is not None:
                    c8_applicable += 1
                    c8_violations += (not gc)

    def spread(d):
        return {k: {"values": sorted(set(v)), "n": len(v)} for k, v in d.items()}

    unstable = {f"{t}/var{v}": sorted(set(c))
                for (t, v), c in seg_counts.items() if len(set(c)) > 2}
    instr_drift = {f"{t}/var{v}/seg{i}": len(s)
                   for (t, v, i), s in instr_uniq.items() if len(s) > 3}

    # 同 (task,var,seg_idx) 内 k_global 的众数占比（§5.2 的核心红灯指标，补码后才有）
    kg = collections.defaultdict(list)
    for e in ok:
        for s in e["segments"]:
            if "k_global" in s:
                kg[(e["task"], e["variation"], s["segment_index"])].append(s["k_global"])
    mode_shares = [collections.Counter(v).most_common(1)[0][1] / len(v)
                   for v in kg.values() if len(v) >= 3]

    return {
        "n_ok_episodes": len(ok), "n_segments": n_seg_total,
        "segments_per_episode": {
            "mean": round(sum(len(e["segments"]) for e in ok) / max(len(ok), 1), 2)},
        "keypoints_per_segment": {
            "mean": round(sum(kp_per_seg) / max(len(kp_per_seg), 1), 2),
            "hist": dict(sorted(collections.Counter(kp_per_seg).items()))},
        "action_distribution": dict(actions.most_common()),
        "pose_adjust_share": round(n_pose_adjust / max(n_seg_total, 1), 4),
        "code_mask_zero_share": round(
            sum(1 for e in ok for s in e["segments"] if not s.get("use_codebook"))
            / max(n_seg_total, 1), 4),
        "prior_alignment": dict(align),
        "episode_quality": dict(quality),
        "warning_kinds": dict(warn_kinds.most_common()),
        "C8_gripper_violation_rate": (round(c8_violations / c8_applicable, 4)
                                      if c8_applicable else None),
        "C8_applicable_segments": c8_applicable,
        "unstable_segment_count_groups": unstable,
        "instruction_drift_groups": instr_drift,
        "k_global_mode_share_mean": (round(sum(mode_shares) / len(mode_shares), 4)
                                     if mode_shares else None),
        "k_global_mode_share_groups": len(mode_shares),
    }


# ---------------------------------------------------------------- 门禁 ③
def gate3_sampling(episodes: list[dict], per_task: int = 3, seed: int = 0) -> list[dict]:
    """分层抽样：每任务 ≥per_task 条；全部 modified / suspect / 有 warning 的进池。"""
    ok = [e for e in episodes if not e.get("errors")]
    pool = {f"{e['task']}/{e['episode']}": e for e in ok
            if e.get("prior_alignment") == "modified"
            or e.get("episode_quality") == "suspect"
            or e.get("warnings")}
    by_task = collections.defaultdict(list)
    for e in ok:
        by_task[e["task"]].append(e)
    for t, eps in sorted(by_task.items()):
        eps = sorted(eps, key=lambda e: hashlib.sha256(
            f"{seed}|{t}|{e['episode']}".encode()).hexdigest())
        for e in eps[:per_task]:
            pool[f"{t}/{e['episode']}"] = e
    return sorted(pool.values(), key=lambda e: (e["task"], e["episode"]))


def run_all(cache_dir: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    meta, eps = load_cache(cache_dir)
    g1 = gate1_structural(eps)
    g2 = gate2_statistics(eps)
    g3 = gate3_sampling(eps)
    report = {"meta": meta, "gate1_structural": g1, "gate2_statistics": g2,
              "gate3_review_pool": [{"task": e["task"], "episode": e["episode"],
                                     "variation": e["variation"],
                                     "reason": ("modified" if e.get("prior_alignment") == "modified"
                                                else "warning" if e.get("warnings") else "stratified")}
                                    for e in g3]}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
