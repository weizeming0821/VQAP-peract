#!/usr/bin/env python
"""planner/contract.py 的单元测试（Step 0 出口门禁）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from planner.contract import (ALLOWED_ACTIONS, CODEBOOK_ACTIONS, check_instruction,
                              check_segments, gripper_consistent, n_repeat_prior,
                              normalize_instruction, use_codebook)

ok = True
def t(name, cond, extra=""):
    global ok
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else "  " + str(extra)))
    ok = ok and cond

print("=== use_codebook 规则（VLA_Design §3.3）===")
t("17 类码本动作 → true", all(use_codebook(a) for a in CODEBOOK_ACTIONS))
t("pose-adjust → false", use_codebook("pose-adjust") is False)
t("unhang 不在白名单（用户决定忽略）", "unhang" not in ALLOWED_ACTIONS)
t("白名单恰为 18 个", len(ALLOWED_ACTIONS) == 18, sorted(ALLOWED_ACTIONS))

print("=== 指令 canonical 化 ===")
t("去尾句号+小写", normalize_instruction("Grasp the jar lid.") == "grasp the jar lid")
t("压缩空白", normalize_instruction("  move   toward  the  tap  ") == "move toward the tap")
t("多重标点", normalize_instruction("Push the lid down!!") == "push the lid down")

print("=== 指令校验 ===")
t("正常指令通过", check_instruction("grasp the azure jar lid", "close the azure jar", 5) == [])
t("过短", any(x.startswith("too_short") for x in check_instruction("go now", None, 2)))
t("11 词拒绝", any(x.startswith("too_long") for x in check_instruction(" ".join(["w"]*11), None, 2)))
t("9 词仅告警", check_instruction(" ".join(["w"]*9), None, 2) == ["long_warn(9)"])
t("多段时抄袭被抓", "copies_task_instruction" in check_instruction("close the azure jar", "close the azure jar", 5))
t("单段时不算抄袭", "copies_task_instruction" not in check_instruction("close the azure jar", "close the azure jar", 1))

print("=== 结构校验 C1/C2/C3/C4 ===")
good = [{"segment_index":0,"action":"grasp","instruction":"grasp the azure jar lid","keypoint_indices":[0,1]},
        {"segment_index":1,"action":"lift","instruction":"lift the azure jar lid","keypoint_indices":[2]}]
r = check_segments(good, 3, "close the azure jar", [1,0,0])
t("合法输入无 error", r["errors"] == [], r)
bad = [{"segment_index":0,"action":"grasp","instruction":"grasp the lid","keypoint_indices":[0,1]},
       {"segment_index":1,"action":"lift","instruction":"lift the lid","keypoint_indices":[1,2]}]
t("重复关键帧被抓", any(x.startswith("C1_") for x in check_segments(bad,3,"x",[1,0,0])["errors"]))
miss = [{"segment_index":0,"action":"grasp","instruction":"grasp the lid","keypoint_indices":[0]}]
t("遗漏关键帧被抓", any(x.startswith("C1_") for x in check_segments(miss,3,"x",[1,0,0])["errors"]))
badact = [{"segment_index":0,"action":"unhang","instruction":"unhang the cup","keypoint_indices":[0]}]
t("unhang 被 C3 拒绝", any(x.startswith("C3_") for x in check_segments(badact,1,"x",[1])["errors"]))

print("=== C8 夹爪一致性（用户口径）===")
tr = [{"segment_index":0,"action":"transfer","instruction":"carry the lid to the jar","keypoint_indices":[0]}]
t("transfer 夹爪开 → 告警", any(x.startswith("C8_") for x in check_segments(tr,1,"x",[1])["warnings"]))
t("transfer 夹爪闭 → 无告警", not any(x.startswith("C8_") for x in check_segments(tr,1,"x",[0])["warnings"]))
pa = [{"segment_index":0,"action":"pose-adjust","instruction":"adjust the gripper pose","keypoint_indices":[0]}]
t("pose-adjust 夹爪闭 → 告警", any(x.startswith("C8_") for x in check_segments(pa,1,"x",[0])["warnings"]))
t("pose-adjust 夹爪开 → 无告警", not any(x.startswith("C8_") for x in check_segments(pa,1,"x",[1])["warnings"]))
t("gripper_consistent transfer", gripper_consistent("transfer",[0,1],[0,0]) is True)
t("gripper_consistent 非相关动作返回 None", gripper_consistent("grasp",[0],[1]) is None)

print("=== 重复次数解析（P1 实测规律）===")
t("push_buttons var0→1", n_repeat_prior("push_buttons",0)==1)
t("push_buttons var2→3", n_repeat_prior("push_buttons",2)==3)
t("stack_blocks var0→2", n_repeat_prior("stack_blocks",0)==2)
t("stack_blocks var2→4", n_repeat_prior("stack_blocks",2)==4)
t("place_cups var2→3", n_repeat_prior("place_cups",2)==3)
t("其它任务无规则", n_repeat_prior("close_jar",3) is None)

print("\nStep 0 单测:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
