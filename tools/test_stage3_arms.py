#!/usr/bin/env python
"""P4 Step 3/4 门禁：四臂字段隔离 + cache 连接层。"""
from __future__ import annotations
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path: sys.path.insert(0, _p)
from stage3.arms import ARMS, FieldAccessError, GuardedSample, allowed_fields, lang_fields_for
from stage3.cache_join import PlannerCache

OK = True
def check(name, cond, extra=""):
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)

FIELDS = {"front_rgb","low_dim_state","trans_action_indicies","gripper_pose",
          "lang_goal_emb","lang_token_embs","lang_goal",
          "subtask_lang_goal_emb","subtask_lang_token_embs","subtask_lang_goal",
          "subtask_k_global","subtask_k_detail","subtask_code_mask",
          "subtask_index","subtask_action"}
SAMPLE = {k: k for k in FIELDS}

print("=== 1. 四臂白名单 ===")
for arm in ("B0","B1","B2","B3"):
    a = allowed_fields(arm, FIELDS)
    print(f"       {arm}: {len(a)} 个字段")
check("B1 拿不到任何 subtask_*", not any(f.startswith("subtask_") for f in allowed_fields("B1", FIELDS)))
check("B2 拿不到整任务语言字段", "lang_goal_emb" not in allowed_fields("B2", FIELDS))
check("B2 拿得到子任务语言字段", "subtask_lang_goal_emb" in allowed_fields("B2", FIELDS))
check("B2 拿不到码字段", "subtask_k_global" not in allowed_fields("B2", FIELDS))
check("B3 拿得到码字段", "subtask_k_global" in allowed_fields("B3", FIELDS))
check("四臂都拿得到观测字段", all("front_rgb" in allowed_fields(a, FIELDS) for a in ARMS))

print("=== 2. 越权访问抛异常 ===")
for arm, bad in (("B1","subtask_lang_goal_emb"), ("B1","subtask_k_global"),
                 ("B2","lang_goal_emb"), ("B2","subtask_k_global")):
    g = GuardedSample(SAMPLE, arm)
    try:
        g[bad]; check(f"{arm} 读 {bad} 应抛异常", False)
    except FieldAccessError as e:
        check(f"{arm} 读 {bad} 抛 FieldAccessError", "禁止读取" in str(e))

print("=== 3. 合法访问正常 ===")
check("B1 读 lang_goal_emb", GuardedSample(SAMPLE,"B1")["lang_goal_emb"] == "lang_goal_emb")
check("B3 读 subtask_k_detail", GuardedSample(SAMPLE,"B3")["subtask_k_detail"] == "subtask_k_detail")
check("B3 读 front_rgb", GuardedSample(SAMPLE,"B3")["front_rgb"] == "front_rgb")

print("=== 4. lang_fields_for ===")
check("B0/B1 -> 整任务字段", lang_fields_for("B1") == ("lang_goal_emb","lang_token_embs"))
check("B2/B3 -> 子任务字段", lang_fields_for("B3") == ("subtask_lang_goal_emb","subtask_lang_token_embs"))

print("=== 5. PlannerCache 真实数据 ===")
try:
    c = PlannerCache()
    print("      ", c.summary())
    check("18 个任务", len(c.tasks()) == 18, c.tasks())
    check("跳过 1 条 error", len(c.skipped["errors"]) == 1, c.skipped["errors"])
    # 7 条而非 10 条：关键帧夹爪全程 OPEN 的共 10 条，但其中 3 条 turn_tap 是
    # 合理的（用张开的夹爪拨水龙头、从不抓握，模型也正确标成 approach->rotate，
    # 不含任何隐含持物的动作）。判据要求「全程 OPEN 且含 grasp/lift/place/transfer」。
    check("跳过 7 条夹爪异常", len(c.skipped["gripper_anomaly"]) == 7, c.skipped["gripper_anomaly"])
    check("剩余 1792 条可用", len(c._eps) == 1792, len(c._eps))
    ep = c.require("close_jar", "train", 0)
    seg = c.segment_of_keypoint(ep, ep["keypoints"][0])
    check("按绝对帧号查到 segment", seg["segment_index"] == ep["keypoint_to_segment"][0])
    check("segment 含码", "k_global" in seg and len(seg["k_detail"]) == 9)
    try:
        c.assert_keypoints(ep, [1,2,3], "close_jar", 0); check("关键帧失配应抛异常", False)
    except Exception as e:
        check("关键帧失配抛异常", "关键帧失配" in str(e))
    c.assert_keypoints(ep, ep["keypoints"], "close_jar", 0)
    check("关键帧一致时不抛", True)
except Exception as e:
    check(f"PlannerCache 加载: {type(e).__name__}: {e}", False)

print()
print("Step 3/4 单测:", "PASS" if OK else "FAIL")
sys.exit(0 if OK else 1)
