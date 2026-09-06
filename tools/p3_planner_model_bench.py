#!/usr/bin/env python
"""P3 前置：Planner 模型选型对比（临时工具）。

在真实 RLBench episode 上跑候选模型，按 `VLA_Design §5.1` 的结构校验项自动打分，
再加上成本/延迟与跨模型一致性，作为选型依据。

评分项（全部自动，无需人工）：
  C1 keypoint_indices 构成 [0,M) 的完整无重叠划分
  C2 segments 按 segment_index 递增、时间有序
  C3 action ∈ 17 类白名单 ∪ {pose-adjust}
  C4 instruction 全小写 / 无尾标点 / 词数 ∈ [3,8]
  C4b instruction 不得等于任务指令（直接抄 = 没做子任务分解）
  C4c 同一 episode 内 instruction 互不相同
  P  与 CSV 先验动作序列的编辑距离

用法：
    source run/env.sh
    python tools/p3_planner_model_bench.py --episodes 3 --models qwen3-vl-plus qwen-vl-max
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if p not in sys.path:
        sys.path.insert(0, p)

WHITELIST = {"approach", "grasp", "lift", "place", "push", "pull", "press", "rotate",
             "slide", "insert", "hang", "wipe", "flip-open", "flip-close",
             "revolve-in", "revolve-out", "transfer"}
ALLOWED = WHITELIST | {"pose-adjust"}

# 用你确认过的先验（含新补的两条）
PRIOR = {
    "close_jar": ["grasp", "lift", "transfer", "place", "rotate"],
    "open_drawer": ["approach", "grasp", "pull"],
    "meat_off_grill": ["approach", "grasp", "lift", "transfer", "place"],
    "put_money_in_safe": ["approach", "grasp", "transfer", "transfer", "place"],
    "put_groceries_in_cupboard": ["approach", "grasp", "lift", "pose-adjust", "pose-adjust", "place"],
    "slide_block_to_color_target": ["approach", "press", "pose-adjust", "approach", "press"],
    "reach_and_drag": ["approach", "grasp", "lift", "transfer", "pose-adjust", "slide"],
    "light_bulb_in": ["approach", "grasp", "lift", "transfer", "place", "rotate"],
    "place_shape_in_shape_sorter": ["approach", "grasp", "lift", "transfer", "place"],
    "insert_onto_square_peg": ["approach", "grasp", "lift", "transfer", "place"],
    "turn_tap": ["approach", "grasp", "rotate"],
    "sweep_to_dustpan_of_size": ["approach", "grasp", "transfer", "wipe"],
    "place_wine_at_rack_location": ["approach", "grasp", "transfer", "insert"],
}

SYSTEM = (
    "你是 RLBench 机械臂演示轨迹的动作分段审核员。给定一条演示的候选关键帧"
    "（已由 PerAct 的关键帧算法自动提取），把它们归组为若干个连续的原子动作段。\n"
    "硬约束：\n"
    "1. 不得新增、删除或修改关键帧编号，只能把给定的关键帧分配到动作段。\n"
    "2. 每个关键帧必须且只能属于一个段；段按时间顺序连续，不得交叉或留空。\n"
    "3. action 必须取自白名单：approach, grasp, lift, place, push, pull, press, rotate, "
    "slide, insert, hang, wipe, flip-open, flip-close, revolve-in, revolve-out, transfer"
    "（另允许 pose-adjust 用于姿态微调段）。\n"
    "4. instruction 格式：全小写、无句尾标点、3~8 词祈使句。**必须描述该段自己的动作**，"
    "不得直接复制整任务指令；必须保留任务指令中的区分性信息（颜色、序号、方位）。\n"
    "动作定义：approach=空载接近目标尚未接触；grasp=夹爪由开变闭并与目标建立接触；"
    "lift=已持物竖直抬离；transfer=已持物水平移动到目标区域；place=已持物下放并释放；"
    "press=以末端或持有物施压触发目标；pull/push=沿直线拉/推；rotate=绕轴旋转目标；"
    "slide=使物体在平面滑移；insert=沿轴向插入孔位；hang=挂置于支撑结构；wipe=往复清扫；"
    "flip-open/flip-close=翻转打开/合上盖状结构；revolve-in/revolve-out=绕铰链向内/外转动门状结构。\n"
    "夹爪状态是最可靠的信号：由 open 变 closed 处必是 grasp 完成，由 closed 变 open 处必是释放。\n"
    "参考序列来自人工标注，是**软先验**：粒度不一定与关键帧一致，数量也可能不同。"
    "以实际图像与夹爪状态为准，不要为了凑先验强行分段；难以归类的姿态微调段标为 pose-adjust。\n"
    "只输出 JSON，不要任何额外文字，不要代码围栏。"
)


def data_url(path: str, size: int) -> str:
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(ep: str, task: str, kps, grippers, desc: str, size: int, n_repeat: int | None):
    prior = PRIOR.get(task, [])
    prior_txt = " → ".join(prior) if prior else "（无先验）"
    head = (f"任务：{task}\n官方任务指令：{desc}\n"
            f"参考动作序列（人工标注软先验）：{prior_txt}\n")
    if n_repeat and n_repeat > 1:
        head += (f"⚠️ 本 variation 需要重复该序列 {n_repeat} 次"
                 f"（由 variation 索引解析得出，可信）。\n")
    head += f"候选关键帧共 {len(kps)} 个（按时间顺序），每个给出 front 与 wrist 两个视角：\n"
    parts = [{"type": "text", "text": head}]
    for i, (k, g) in enumerate(zip(kps, grippers)):
        parts.append({"type": "text",
                      "text": f"[{i}] frame={k} gripper={'open' if g else 'closed'}"})
        parts.append({"type": "image_url", "image_url": {"url": data_url(f"{ep}/front_rgb/{k}.png", size)}})
        parts.append({"type": "image_url", "image_url": {"url": data_url(f"{ep}/wrist_rgb/{k}.png", size)}})
    parts.append({"type": "text", "text":
                  '输出 schema：{"segments":[{"segment_index":0,"action":"grasp",'
                  '"instruction":"grasp the red jar lid","keypoint_indices":[0,1]}],'
                  '"prior_alignment":"exact|modified","prior_deviation_reason":null,'
                  '"episode_quality":"ok|suspect|reject"}'})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": parts}]


def parse_json(txt: str):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.S)
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("找不到 JSON 对象")
    return json.loads(txt[i:j + 1])


def score(obj, n_kp: int, task: str, desc: str) -> dict:
    segs = obj.get("segments", [])
    idx = [i for s in segs for i in s.get("keypoint_indices", [])]
    c1 = sorted(idx) == list(range(n_kp))
    c2 = ([s.get("segment_index") for s in segs] == list(range(len(segs)))
          and all(s.get("keypoint_indices", []) == sorted(s.get("keypoint_indices", []))
                  for s in segs)
          and idx == sorted(idx))
    c3 = all(s.get("action") in ALLOWED for s in segs)
    ins = [str(s.get("instruction", "")) for s in segs]
    c4 = all(t == t.lower() and not t.endswith((".", "。", "!", "?"))
             and 3 <= len(t.split()) <= 8 for t in ins)
    c4b = all(t.strip().lower() != desc.strip().lower() for t in ins)
    c4c = len(set(ins)) == len(ins)
    acts = [s.get("action") for s in segs]
    prior = PRIOR.get(task, [])
    # 归一化编辑距离（先验只是参考，不是对错判据，仅作漂移监控）
    if prior:
        m, n = len(acts), len(prior)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for a in range(m + 1):
            for b in range(n + 1):
                dp[a][b] = (b if a == 0 else a if b == 0 else
                            min(dp[a-1][b] + 1, dp[a][b-1] + 1,
                                dp[a-1][b-1] + (acts[a-1] != prior[b-1])))
        ed = dp[m][n] / max(m, n)
    else:
        ed = None
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C4b": c4b, "C4c": c4c,
            "n_segments": len(segs), "actions": acts,
            "prior_edit_dist_norm": None if ed is None else round(ed, 3),
            "instructions": ins,
            "prior_alignment": obj.get("prior_alignment"),
            "episode_quality": obj.get("episode_quality")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["close_jar", "open_drawer", "put_groceries_in_cupboard",
                             "slide_block_to_color_target", "light_bulb_in", "turn_tap"])
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--models", nargs="+",
                    default=["qwen3.8-max", "qwen3-vl-plus", "qwen-vl-max", "qwen3-vl-flash"])
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(REPO_ROOT / "result" / "p3_planner_model" / "bench.json"))
    args = ap.parse_args()

    from helpers.demo_loading_utils import keypoint_discovery
    from rlbench.demo import Demo

    jobs = []
    for task in args.tasks:
        for e in range(args.episodes):
            ep = f"aavla_data/rlbench/train/{task}/all_variations/episodes/episode{e}"
            if not os.path.isfile(f"{ep}/low_dim_obs.pkl"):
                continue
            demo = Demo(pickle.load(open(f"{ep}/low_dim_obs.pkl", "rb")))
            kps = keypoint_discovery(demo)
            if not kps:
                continue
            desc = pickle.load(open(f"{ep}/variation_descriptions.pkl", "rb"))[0]
            var = int(pickle.load(open(f"{ep}/variation_number.pkl", "rb")))
            n_rep = ({"push_buttons": var % 3 + 1, "stack_blocks": var % 3 + 2,
                      "place_cups": var + 1}).get(task)
            jobs.append({"task": task, "ep": ep, "kps": kps,
                         "grip": [int(demo[k].gripper_open) for k in kps],
                         "desc": desc, "var": var, "n_repeat": n_rep})
    print(f"{len(jobs)} 条 episode × {len(args.models)} 个模型")

    client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                    base_url=os.environ["DASHSCOPE_BASE_URL"])

    def run_one(arg):
        model, j = arg
        msgs = build_messages(j["ep"], j["task"], j["kps"], j["grip"], j["desc"],
                              args.image_size, j["n_repeat"])
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, messages=msgs,
                                               temperature=0, max_tokens=4000)
            dt = time.time() - t0
            txt = r.choices[0].message.content
            rec = {"model": model, "task": j["task"], "episode": os.path.basename(j["ep"]),
                   "n_keypoints": len(j["kps"]), "seconds": round(dt, 1),
                   "prompt_tokens": r.usage.prompt_tokens,
                   "completion_tokens": r.usage.completion_tokens}
            try:
                rec.update(score(parse_json(txt), len(j["kps"]), j["task"], j["desc"]))
                rec["parse_ok"] = True
            except Exception as exc:
                rec.update({"parse_ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "raw": txt[:400]})
            return rec
        except Exception as exc:
            return {"model": model, "task": j["task"], "episode": os.path.basename(j["ep"]),
                    "parse_ok": False, "seconds": round(time.time() - t0, 1),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    todo = [(m, j) for m in args.models for j in jobs]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        recs = list(pool.map(run_one, todo))

    print()
    print("=" * 104)
    print(f"{'model':18s} {'解析':>5} {'C1划分':>7} {'C2有序':>7} {'C3白名单':>8} "
          f"{'C4格式':>7} {'C4b非抄':>8} {'C4c不重':>8} {'编辑距':>7} {'秒':>6} {'out_tok':>8}")
    print("=" * 104)
    summary = {}
    for m in args.models:
        rs = [r for r in recs if r["model"] == m]
        ok = [r for r in rs if r.get("parse_ok")]
        def rate(k):
            return sum(1 for r in ok if r.get(k)) / len(ok) if ok else 0.0
        eds = [r["prior_edit_dist_norm"] for r in ok if r.get("prior_edit_dist_norm") is not None]
        s = {"n": len(rs), "parse_rate": len(ok) / len(rs) if rs else 0,
             "C1": rate("C1"), "C2": rate("C2"), "C3": rate("C3"),
             "C4": rate("C4"), "C4b": rate("C4b"), "C4c": rate("C4c"),
             "prior_edit": sum(eds) / len(eds) if eds else None,
             "sec": sum(r["seconds"] for r in rs) / len(rs) if rs else 0,
             "out_tok": sum(r.get("completion_tokens", 0) for r in ok) / len(ok) if ok else 0,
             "in_tok": sum(r.get("prompt_tokens", 0) for r in ok) / len(ok) if ok else 0}
        summary[m] = s
        ed_txt = "-" if s["prior_edit"] is None else f"{s['prior_edit']:.2f}"
        print(f"{m:18s} {s['parse_rate']:>5.0%} {s['C1']:>7.0%} {s['C2']:>7.0%} {s['C3']:>8.0%} "
              f"{s['C4']:>7.0%} {s['C4b']:>8.0%} {s['C4c']:>8.0%} "
              f"{ed_txt:>7} {s['sec']:>6.1f} {s['out_tok']:>8.0f}")

    print("\n失败样例：")
    for r in recs:
        if not r.get("parse_ok"):
            print(f"  {r['model']:16s} {r['task']:28s} {r.get('error','')[:90]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": recs},
                              indent=1, ensure_ascii=False))
    print(f"\n明细写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
