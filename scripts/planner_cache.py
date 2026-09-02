#!/usr/bin/env python
"""离线 Planner Cache 的唯一入口：生成 → 修复 → 回填 → 补码 → 门禁。

子命令
    build     调用 VLM 把 PerAct 关键帧归组成原子动作段，逐 (task) 分片落盘
    repair    对结构校验失败的 episode 带纠正提示重跑（call_failed 不加纠正说明）
    backfill  纯本地补齐缺失的 episode 级字段并重新校验（不调用 API）
    codes     用冻结 Adapter 给每个 segment 补 (k_global, k_detail[9])
    gates     跑三道质量门禁并出报告
    all       按 build → repair → backfill → codes → gates 顺序全跑

典型用法
    source run/env.sh
    python scripts/planner_cache.py all --episodes 100 --concurrent 8
    python scripts/planner_cache.py build --tasks close_jar --episodes 5 --out-dir planner_cache/_try

抗中断设计
  * 每个 (task) 完成即写分片，不等全量跑完；
  * LLM 响应按 sha256(model+messages) 磁盘记忆化 —— 重跑不重复付费、同输入同输出；
  * build 会跳过已完整的分片；repair 只挑有 error 的；backfill 只挑缺字段的；
  * 零产出快速失败阈值 = max(fail_fast_after, concurrent)。

⚠️ 并发经验：18 路并发经代理会大量 APIConnectionError（实测 209/1800 失败），
   降到 6~8 路后降至 1 条。默认 8。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from planner.client import PlannerClient, image_data_url            # noqa: E402
from planner.contract import (                                       # noqa: E402
    check_segments, expand_prior, gripper_consistent, n_repeat_prior,
    normalize_instruction, use_codebook,
)
from planner.offline import (                                        # noqa: E402
    DEFAULT_VIEWS, cache_meta, episode_inputs, load_priors,
    parse_json, segment_episode,
)
from planner.prompts import build_system_prompt, build_user_content  # noqa: E402

SEEN12 = ["close_jar", "light_bulb_in", "open_drawer", "place_cups",
          "place_shape_in_shape_sorter", "push_buttons", "put_groceries_in_cupboard",
          "reach_and_drag", "slide_block_to_color_target", "stack_blocks",
          "place_wine_at_rack_location", "sweep_to_dustpan_of_size"]
UNSEEN6 = ["insert_onto_square_peg", "meat_off_grill", "put_item_in_drawer",
           "put_money_in_safe", "stack_cups", "turn_tap"]
PERACT_18 = SEEN12 + UNSEEN6

EPISODE_FIELDS = ("variation", "task_instruction", "keypoints", "gripper_at_keypoints",
                  "demo_len", "prior_sequence", "n_repeat_prior", "views_used")
ADAPTER_CAMERAS = ("front", "wrist")     # Adapter 图像管线写死的两路相机


def shard_path(out_dir: Path, split: str, task: str) -> Path:
    return out_dir / f"{split}__{task}.json"


def _episode_fields(task: str, info: dict, priors: dict, views) -> dict:
    n_rep = n_repeat_prior(task, info["variation"])
    return {"variation": info["variation"],
            "task_instruction": info["task_instruction"],
            "demo_len": info["demo_len"],
            "keypoints": info["keypoints"],
            "gripper_at_keypoints": info["grippers"],
            "n_repeat_prior": n_rep,
            "prior_sequence": expand_prior(task, info["variation"],
                                           list(priors.get(task, []))),
            "views_used": list(views)}


def _derive_segment_fields(segs: list[dict], kps: list[int], grip: list[int]) -> list[int]:
    """补齐 segment 的派生字段，返回 keypoint_to_segment。"""
    kp2seg = [-1] * len(kps)
    for s in segs:
        ki = sorted(s.get("keypoint_indices", []))
        s["use_codebook"] = use_codebook(s.get("action", ""))
        s["gripper_consistent"] = gripper_consistent(s.get("action", ""), ki, grip)
        s["start_frame"] = 0 if not ki or ki[0] == 0 else kps[ki[0] - 1]
        for i in ki:
            if 0 <= i < len(kp2seg):
                kp2seg[i] = s.get("segment_index", -1)
    return kp2seg


# ================================================================= build
def cmd_build(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    priors = load_priors()
    system_prompt = build_system_prompt()
    client = PlannerClient(model=args.model)

    todo = []
    for task in args.tasks:
        sp = shard_path(out_dir, args.split, task)
        if sp.is_file() and not args.force:
            try:
                if len(json.loads(sp.read_text()).get("episodes", [])) >= args.episodes:
                    print(f"  skip {args.split}/{task}")
                    continue
            except Exception:
                pass
        todo.append(task)
    print(f"[build] {len(todo)} 个任务 × {args.episodes} ep，模型 {args.model}，"
          f"视角 {args.views}，并发 {args.concurrent}", flush=True)
    if not todo:
        return 0

    abort, lock, t0 = threading.Event(), threading.Lock(), time.time()
    results = []

    def run_task(task: str) -> dict:
        base = Path(args.data_root) / args.split / task / "all_variations" / "episodes"
        recs = []
        for e in range(args.episode_offset, args.episode_offset + args.episodes):
            if abort.is_set():
                break
            ep_dir = base / f"episode{e}"
            if not (ep_dir / "low_dim_obs.pkl").is_file():
                continue
            try:
                r = segment_episode(client, task, args.split, e, str(ep_dir), priors,
                                    views=tuple(args.views), image_size=args.image_size,
                                    system_prompt=system_prompt)
            except Exception as exc:
                r = {"task": task, "split": args.split, "episode": e,
                     "errors": [f"call_failed:{type(exc).__name__}:{str(exc)[:160]}"],
                     "warnings": []}
            recs.append(r)
        shard_path(out_dir, args.split, task).write_text(json.dumps(
            {"meta": cache_meta(args.model, args.views, args.image_size),
             "episodes": recs}, ensure_ascii=False, indent=1))
        n_err = sum(1 for r in recs if r.get("errors"))
        with lock:
            print(f"  [{'OK' if recs and not n_err else 'PART' if recs else 'FAIL'}] "
                  f"{args.split}/{task}  {len(recs)} ep，{n_err} err  "
                  f"({(time.time()-t0)/60:.1f} min，缓存 {client.stats['hit']}/"
                  f"{client.stats['hit']+client.stats['miss']})", flush=True)
        return {"task": task, "n": len(recs), "n_error": n_err}

    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futs = {pool.submit(run_task, t): t for t in todo}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as exc:
                print(f"  [FAIL] {futs[f]}: {type(exc).__name__}: {exc}", flush=True)
                results.append({"task": futs[f], "n": 0, "n_error": -1})
            zero = [r for r in results if r["n"] == 0]
            if (not args.no_fail_fast and len(zero) >= max(args.fail_fast_after, args.concurrent)
                    and len(zero) == len(results)):
                abort.set()
                print(f"\n❌ 前 {len(results)} 个任务均零产出，判定环境/接口问题，已中止", flush=True)
    _print_usage(client, args.model)
    return 0


# ================================================================= repair
def _correction_note(errors: list[str], n_kp: int) -> str:
    """仅针对结构性错误生成纠正说明。

    call_failed 是网络层失败，模型从未产出答案、响应也从未被缓存 ——
    重发原 prompt 就是全新调用，追加「你上次答错了」纯属噪声。
    """
    structural = [e for e in errors if not e.startswith("call_failed")]
    if not structural:
        return ""
    lines = ["\nYOUR PREVIOUS ANSWER WAS REJECTED by the structural validator. Fix it."]
    for e in structural:
        if e.startswith("C1_partition"):
            lines.append(
                f"  * Invalid partition: {e}\n"
                f"    A keyframe is a MEMBER of exactly one segment, not a shared BOUNDARY "
                f"between two segments. Adjacent segments must NOT repeat an index.\n"
                f"    The union of all keypoint_indices must be exactly "
                f"[0, 1, ..., {n_kp - 1}], each appearing exactly once.")
        elif e.startswith("C3_"):
            lines.append(f"  * Illegal action label: {e}. Use only the whitelist.")
        elif e.startswith("C4_"):
            lines.append(f"  * Instruction format violation: {e}. 3-8 words, all lowercase, "
                         f"no trailing punctuation, describe the segment's own action.")
        elif e.startswith("C2_"):
            lines.append(f"  * Ordering problem: {e}. segment_index must be 0,1,2,... "
                         f"and keyframes strictly increasing.")
        else:
            lines.append(f"  * {e}")
    return "\n".join(lines)


def cmd_repair(args) -> int:
    priors = load_priors()
    system_prompt = build_system_prompt()
    client = PlannerClient(model=args.model)

    for rnd in range(1, args.max_rounds + 1):
        targets = []
        for sp in sorted(glob.glob(str(Path(args.cache_dir) / "*.json"))):
            d = json.loads(Path(sp).read_text())
            views = args.views or d["meta"].get("views", list(DEFAULT_VIEWS))
            size = d["meta"].get("image_size", 224)
            for k, ep in enumerate(d["episodes"]):
                if ep.get("errors"):
                    targets.append((sp, k, ep, views, size))
        if not targets:
            print(f"[repair] 第 {rnd} 轮：无待修复 episode")
            break
        print(f"[repair] 第 {rnd} 轮：{len(targets)} 条待修复", flush=True)

        def repair_one(item):
            sp, k, ep, views, size = item
            ep_dir = os.path.join(args.data_root, ep["split"], ep["task"],
                                  "all_variations", "episodes", f"episode{ep['episode']}")
            try:
                info = episode_inputs(ep_dir)
            except Exception as exc:
                return sp, k, None, f"load_failed:{exc}"
            kps, grip = info["keypoints"], info["grippers"]
            if not kps:
                return sp, k, None, "no_keypoints"
            n_rep = n_repeat_prior(ep["task"], info["variation"])

            def img(frame, view):
                return image_data_url(os.path.join(ep_dir, f"{view}_rgb", f"{frame}.png"),
                                      size=size)

            content = build_user_content(ep["task"], info["task_instruction"], kps, grip,
                                         priors.get(ep["task"], []), n_rep, img, list(views))
            note = _correction_note(ep.get("errors", []), len(kps))
            if note:
                content.append({"type": "text", "text": note})
            try:
                resp = client.chat([{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": content}])
                obj = parse_json(resp["content"])
            except Exception as exc:
                return sp, k, None, f"call_failed:{type(exc).__name__}"

            segs = obj.get("segments", [])
            for s in segs:
                s["instruction"] = normalize_instruction(s.get("instruction", ""))
            chk = check_segments(segs, len(kps), info["task_instruction"], grip)
            kp2seg = _derive_segment_fields(segs, kps, grip)
            new = dict(ep)
            # 失败记录只有 {task,split,episode,errors}，episode 级字段必须一并写回，
            # 否则会留下缺 keypoints 的残缺记录，而 Stage 3 的 A1 断言需要它。
            new.update(_episode_fields(ep["task"], info, priors, views))
            new.update({"segments": segs, "keypoint_to_segment": kp2seg,
                        "prior_alignment": obj.get("prior_alignment"),
                        "prior_deviation_reason": obj.get("prior_deviation_reason"),
                        "episode_quality": obj.get("episode_quality"),
                        "errors": chk["errors"], "warnings": chk["warnings"],
                        "repaired_round": rnd})
            return sp, k, new, ("OK" if not chk["errors"] else f"still_bad:{chk['errors'][:1]}")

        updates: dict[str, list] = {}
        n_ok = 0
        with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            for fut in as_completed([pool.submit(repair_one, t) for t in targets]):
                sp, k, new, status = fut.result()
                if new is not None:
                    updates.setdefault(sp, []).append((k, new))
                    n_ok += status == "OK"
                if status != "OK":
                    print(f"   {os.path.basename(sp)} #{k}: {status}", flush=True)
        for sp, items in updates.items():
            d = json.loads(Path(sp).read_text())
            for k, new in items:
                d["episodes"][k] = new
            Path(sp).write_text(json.dumps(d, ensure_ascii=False, indent=1))
        print(f"[repair] 第 {rnd} 轮成功 {n_ok}/{len(targets)}\n", flush=True)
        if n_ok == 0:
            break
    _print_usage(client, args.model)
    return 0


# ================================================================= backfill
def cmd_backfill(args) -> int:
    """纯本地补齐 episode 级字段并重新校验，不调用任何 API。"""
    priors = load_priors()
    n_fixed = n_scanned = 0
    for sp in sorted(glob.glob(str(Path(args.cache_dir) / "*.json"))):
        d = json.loads(Path(sp).read_text())
        views = d["meta"].get("views", list(DEFAULT_VIEWS))
        changed = False
        for ep in d["episodes"]:
            n_scanned += 1
            if all(k in ep for k in EPISODE_FIELDS) and not args.refresh_fields:
                continue
            ep_dir = os.path.join(args.data_root, ep["split"], ep["task"],
                                  "all_variations", "episodes", f"episode{ep['episode']}")
            try:
                info = episode_inputs(ep_dir)
            except Exception as exc:
                ep.setdefault("errors", []).append(f"backfill_failed:{type(exc).__name__}")
                changed = True
                continue
            ep.update(_episode_fields(ep["task"], info, priors, views))
            segs = ep.get("segments", [])
            ep["keypoint_to_segment"] = _derive_segment_fields(
                segs, info["keypoints"], info["grippers"])
            chk = check_segments(segs, len(info["keypoints"]),
                                 info["task_instruction"], info["grippers"])
            ep["errors"], ep["warnings"] = chk["errors"], chk["warnings"]
            n_fixed += 1
            changed = True
        if changed:
            Path(sp).write_text(json.dumps(d, ensure_ascii=False, indent=1))
            print(f"  {os.path.basename(sp)} 已更新")
    print(f"[backfill] 扫描 {n_scanned} 条，回填 {n_fixed} 条")
    return 0


# ================================================================= codes
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_codes(args) -> int:
    """VLA_Design §4.4：每个 (episode, segment) 预计算一组码。

    Adapter 输入 = 该 segment 起始帧的 (front, wrist) + 该 segment 指令。
    段内所有训练样本共享这组码，与部署时「切换时调用一次、段内不变」一致。
    """
    import numpy as np
    import torch
    from PIL import Image
    from model.adapter import Adapter

    device = torch.device(args.device)
    ck = torch.load(args.adapter, map_location="cpu", weights_only=False)
    adapter = Adapter(ck["model_args"])
    _, unexpected = adapter.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"Adapter unexpected keys: {unexpected[:5]}"
    adapter.eval().to(device)
    print(f"[codes] Adapter epoch {ck.get('epoch')} val_top1 {ck.get('best_val_top1'):.4f}")

    def load_frame(ep_dir, cam, frame):
        img = Image.open(os.path.join(ep_dir, f"{cam}_rgb", f"{frame}.png")).convert("RGB")
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()

    total = n_err = 0
    for sp in sorted(glob.glob(str(Path(args.cache_dir) / "*.json"))):
        d = json.loads(Path(sp).read_text())
        jobs = []
        for ep in d["episodes"]:
            if ep.get("errors"):
                continue
            ep_dir = os.path.join(args.data_root, ep["split"], ep["task"],
                                  "all_variations", "episodes", f"episode{ep['episode']}")
            for s in ep["segments"]:
                jobs.append((s, ep_dir, int(s["start_frame"]), s["instruction"]))
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i:i + args.batch]
            try:
                imgs = {c: torch.stack([load_frame(j[1], c, j[2]) for j in chunk]).to(device)
                        for c in ADAPTER_CAMERAS}
                with torch.no_grad():
                    out = adapter(imgs, [j[3] for j in chunk])
                kg = out["global_logits"].argmax(-1).cpu().tolist()
                kd = out["detail_logits"].argmax(-1).cpu().tolist()
            except Exception as exc:
                print(f"  [ERR] {os.path.basename(sp)} batch {i}: {type(exc).__name__}: {exc}")
                n_err += len(chunk)
                continue
            for (seg, _, _, _), g, dd in zip(chunk, kg, kd):
                assert 0 <= g < 36, f"k_global 越界: {g}"                       # A4
                assert len(dd) == 9 and all(0 <= x < 192 for x in dd), f"k_detail 非法: {dd}"
                seg["k_global"] = int(g)
                seg["k_detail"] = [int(x) for x in dd]
            total += len(chunk)
        d["meta"]["adapter_ckpt_sha256"] = _sha256_file(Path(args.adapter))
        d["meta"]["codebook_ckpt_sha256"] = _sha256_file(Path(args.codebook))
        d["meta"]["adapter_cameras"] = list(ADAPTER_CAMERAS)
        Path(sp).write_text(json.dumps(d, ensure_ascii=False, indent=1))
        print(f"  {os.path.basename(sp):48s} 补码完成")
    print(f"[codes] {total} 个 segment，{n_err} 失败")
    return 1 if n_err else 0


# ================================================================= gates
def cmd_gates(args) -> int:
    from planner.gates import run_all
    r = run_all(args.cache_dir, out=args.report)
    g1, g2 = r["gate1_structural"], r["gate2_statistics"]
    print(f"[gates] ① 结构 {g1['n_episodes']} ep，不合格 {g1['n_bad']}，"
          f"通过率 {g1['pass_rate']:.4f}  {g1['error_kinds']}")
    print(f"[gates] ② 统计 segment {g2['n_segments']}，段/ep {g2['segments_per_episode']['mean']}")
    print(f"        action {g2['action_distribution']}")
    print(f"        pose-adjust 段占比 {g2['pose_adjust_share']:.4f} | "
          f"code_mask=0 段占比 {g2['code_mask_zero_share']:.4f}")
    print(f"        C8 违反率 {g2['C8_gripper_violation_rate']} | "
          f"align {g2['prior_alignment']} | quality {g2['episode_quality']}")
    print(f"[gates] ③ 抽检池 {len(r['gate3_review_pool'])} 条  报告 -> {args.report}")
    return 0


def _print_usage(client: PlannerClient, model: str) -> None:
    pt, ct = client.stats["prompt_tokens"], client.stats["completion_tokens"]
    cost = pt / 1e6 * 12 + ct / 1e6 * 36 if model == "qwen3.8-max" else None
    print(f"token: prompt {pt:,} / completion {ct:,}"
          + (f"  ≈ {cost:.1f} 元" if cost is not None else ""))
    print(f"缓存: 命中 {client.stats['hit']} / 新调用 {client.stats['miss']} / "
          f"重试 {client.stats['retry']} / 失败 {client.stats['fail']}")


# ================================================================= main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "repair", "backfill", "codes", "gates", "all"])
    ap.add_argument("--data-root", default=str(REPO_ROOT / "data_rlbench"))
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "planner_cache" / "train"))
    ap.add_argument("--out-dir", default=None, help="build 的输出目录，默认同 --cache-dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tasks", nargs="+", default=PERACT_18)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--episode-offset", type=int, default=0)
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--model", default="qwen3.8-max")
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--fail-fast-after", type=int, default=3)
    ap.add_argument("--no-fail-fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refresh-fields", action="store_true",
                    help="backfill：即使字段齐全也重算 episode 级字段（先验展开规则变更后用）")
    ap.add_argument("--adapter", default=str(REPO_ROOT / "checkpoints" / "vqap_adapter" / "best.pth"))
    ap.add_argument("--codebook", default=str(REPO_ROOT / "checkpoints" / "vqap_pretrain" / "stage1" / "codebook.pth"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--report", default=str(REPO_ROOT / "result" / "p3_planner_cache" / "gates_report.json"))
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = args.cache_dir

    steps = {"build": cmd_build, "repair": cmd_repair, "backfill": cmd_backfill,
             "codes": cmd_codes, "gates": cmd_gates}
    if args.cmd == "all":
        for name in ("build", "repair", "backfill", "codes", "gates"):
            rc = steps[name](args)
            if rc:
                print(f"\n❌ {name} 返回 {rc}，中止")
                return rc
        return 0
    return steps[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
