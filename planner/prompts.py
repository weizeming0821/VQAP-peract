"""Planner 的 prompt 文本。离线建 cache 与在线评测共用同一段格式规则常量。

动作定义直接采用 `SEMANTIC_ACTION_STANDAR_v2.xlsx` 的英文原文（比中文转述精确），
外加用户给定的 transfer / pose-adjust 夹爪判据。
"""

from __future__ import annotations

from planner.contract import (
    ACTION_VERBS,
    CODEBOOK_ACTIONS,
    INSTRUCTION_MAX_WORDS,
    INSTRUCTION_MIN_WORDS,
    NON_CODEBOOK_ACTIONS,
)

# ---------------------------------------------------------------- 动作定义
# 逐条摘自 SEMANTIC_ACTION_STANDAR_v2.xlsx Sheet1 的 Description 列。
ACTION_DEFINITIONS: dict[str, str] = {
    "approach": "The end effector moves from its current position towards the target object "
                "(or a specific operating point). This process does NOT involve any physical "
                "contact with the object. It stops at a preparatory pose above or beside the target.",
    "grasp": "The end effector approaches the target further until it contacts it, then closes, "
             "firmly fixing the object inside the end effector through friction or geometric constraints.",
    "lift": "While maintaining a stable grip, the end effector overcomes gravity to move the object "
            "upwards, completely detaching it from its initial support surface and suspending it in the air.",
    "transfer": "The end effector moves the object from its current spatial position to a pre-place pose "
                "near the target position, while maintaining a stable grip and having already moved away "
                "from its initial position (Lift). NOTE: if the gripper releases at the very end of a "
                "transfer, that release is too short to be its own segment and still counts as transfer.",
    "place": "The end effector holds an object and moves vertically/downwards along the normal direction "
             "from the pre-place pose, bringing the bottom of the object into contact with the target "
             "support surface (tabletop, container bottom, or bracket).",
    "rotate": "While the end effector maintains a stable grip without loosening or shifting the grasp "
              "centre, it drives the target through a fixed-axis angular rotation, adjusting only the "
              "object's pose/orientation while its overall suspended position stays unchanged.",
    "push": "After establishing single or multi-point contact with the target surface, the end effector "
            "(usually closed or in a specific posture) applies horizontal pressure towards the object "
            "centre or a predetermined direction, causing translational or rotational displacement on "
            "the support surface. NOTE: striking a ball with a held cue also counts as push.",
    "pull": "The end effector stably grasps the target or a mechanism handle, keeps the grasp unchanged, "
            "and moves linearly along a constrained axis away from the initial position, typically "
            "separating the object from its original constrained or closed state.",
    "press": "After establishing stable contact between the end effector (directly or through a held tool) "
             "and the target surface, a controlled increasing pressure is applied along the surface normal. "
             "Displacement is bounded by physical limits such as button travel or surface stiffness.",
    "slide": "The end effector stably grasps or firmly contacts the handle or edge of a horizontally / "
             "vertically sliding mechanism, keeps that contact, and moves linearly along the sliding axis, "
             "moving the mechanism from a closed/retracted state to an open/extended state.",
    "insert": "The end effector holds an object (the insert) and guides it into another object (the base) "
              "with a matching hole or slot, applying displacement along the depth direction of that hole "
              "to establish a tight geometric nesting.",
    "hang": "The end effector holds an object with hooks, holes or loops, and through fine adjustment lets "
            "the object's mounting part pass through, fit into, or hook onto a fixed support component in "
            "the environment (hook, crossbar, bracket, nail).",
    "wipe": "The end effector stably grasps the target object, maintains continuous contact between that "
            "object and the supporting surface, and performs controlled linear movements to slide the "
            "object across the surface, without lifting it off the surface.",
    "flip-open": "After the end effector stably contacts a flippable component (lid, flip cover, flap), it "
                 "applies rotational torque about a HORIZONTAL axis, rotating the component more than 90° "
                 "from closed/covered to open/unfolded, following an arc trajectory without obvious "
                 "translational pulling or pushing.",
    "flip-close": "The end effector acts on an object that rotates about a HORIZONTAL axis (laptop cover, "
                  "box lid, toilet seat), rotating it downward from the open state until its edge or "
                  "surface fully contacts the base and reaches equilibrium.",
    "revolve-out": "After stably gripping an object with a fixed axial constraint (hinge, shaft), the end "
                   "effector applies tangential tension so the object moves along a circular arc about its "
                   "VERTICAL constraint axis, away from the closed plane or initial support structure.",
    "revolve-in": "The end effector (closed or in a rigid posture) establishes unidirectional contact "
                  "(thrust only) with an axially constrained object such as an oven or fridge door, and "
                  "drives it along a circular arc about its VERTICAL axis into the closed position.",
    "pose-adjust": "A pose/orientation correction performed while the gripper is NOT holding any object. "
                   "Use this whenever a segment cannot be confidently assigned to any of the atomic "
                   "actions above.",
}

# transfer 与 pose-adjust 最易混淆，给出决定性区分句（用户口径，可由夹爪状态直接验证）
DISAMBIGUATION = (
    "CRITICAL distinction between `transfer` and `pose-adjust`:\n"
    "  * `transfer`    — the gripper IS holding an object while moving (gripper=closed).\n"
    "  * `pose-adjust` — the gripper is NOT holding anything, it is only correcting its own "
    "pose (gripper=open).\n"
    "The gripper state is given for every keyframe; use it as the decisive signal."
)


def _verb_table() -> str:
    lines = []
    for act in list(CODEBOOK_ACTIONS) + list(NON_CODEBOOK_ACTIONS):
        verbs = ACTION_VERBS.get(act, (act,))
        lines.append(f"  {act:<12} -> start the instruction with: {' / '.join(verbs)}")
    return "\n".join(lines)


def _definitions_block() -> str:
    return "\n".join(f"  {a}: {d}" for a, d in ACTION_DEFINITIONS.items())


def build_system_prompt() -> str:
    allowed = ", ".join(list(CODEBOOK_ACTIONS) + list(NON_CODEBOOK_ACTIONS))
    return f"""You are an annotator that segments RLBench robot-arm demonstrations into atomic actions.

You are given the candidate keyframes of one demonstration (already extracted automatically by
PerAct's keyframe algorithm). Group them into consecutive atomic-action segments.

HARD CONSTRAINTS
1. Do NOT add, delete or modify any keyframe index. You may only assign the given keyframes to
   segments. Every keyframe must belong to exactly one segment; segments are consecutive in time
   and must not overlap or leave gaps.
2. `action` must be one of: {allowed}
3. `instruction` format: all lowercase, no trailing punctuation, {INSTRUCTION_MIN_WORDS}-{INSTRUCTION_MAX_WORDS} words,
   imperative mood. It MUST describe that segment's own action - never copy the whole-task
   instruction. It MUST keep the distinguishing information from the task instruction
   (colour, ordinal, spatial position), otherwise different variations become indistinguishable.
4. Start each instruction with the verb prescribed below. These verbs are fixed by the downstream
   model's training distribution, so do not substitute synonyms:
{_verb_table()}

ACTION DEFINITIONS (from the project's annotation standard)
{_definitions_block()}

{DISAMBIGUATION}

ON THE PRIORS
You will be given a reference action sequence from human annotation. Both that sequence and the
action definitions above are REFERENCES, not ground truth:
  * the keyframe algorithm's granularity need not match the human annotation - the counts may
    differ and the per-segment semantics may be misaligned;
  * judge from the actual images and gripper states, never force a segmentation just to match
    the reference;
  * if it aligns            -> prior_alignment = "exact"
  * if you merged / split / relabelled -> prior_alignment = "modified", and explain in
    prior_deviation_reason;
  * a segment you cannot confidently assign to any atomic action -> label it `pose-adjust`.

Output strict JSON only. No prose, no code fences.
"""


def build_user_content(task: str, task_instruction: str, keypoints, grippers,
                       prior_sequence, n_repeat, image_provider, views) -> list[dict]:
    """image_provider(frame_index, view_name) -> data URL 字符串。"""
    prior_txt = " -> ".join(prior_sequence) if prior_sequence else "(none available)"
    head = (f"task: {task}\n"
            f"official task instruction: {task_instruction}\n"
            f"reference action sequence (soft prior from human annotation): {prior_txt}\n")
    if n_repeat and n_repeat > 1:
        head += (f"NOTE: this variation repeats the reference sequence {n_repeat} times. "
                 f"This count is derived deterministically from the variation index and is reliable.\n")
    head += (f"\n{len(keypoints)} candidate keyframes in temporal order, "
             f"each shown from these views: {', '.join(views)}.\n")
    parts: list[dict] = [{"type": "text", "text": head}]
    for i, (k, g) in enumerate(zip(keypoints, grippers)):
        parts.append({"type": "text",
                      "text": f"[{i}] frame={k} gripper={'open' if g else 'closed'}"})
        for v in views:
            parts.append({"type": "image_url",
                          "image_url": {"url": image_provider(k, v)}})
    parts.append({"type": "text", "text": OUTPUT_SCHEMA})
    return parts


OUTPUT_SCHEMA = """
Output exactly this JSON structure:
{"segments":[{"segment_index":0,"action":"grasp","instruction":"grasp the red jar lid",
"keypoint_indices":[0,1]}],
 "prior_alignment":"exact|modified",
 "prior_deviation_reason":null,
 "anomalies":[],
 "episode_quality":"ok|suspect|reject"}
"""
