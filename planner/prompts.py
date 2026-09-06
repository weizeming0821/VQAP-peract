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


# ------------------------------------------------------------- 在线 Planner

def build_online_system_prompt() -> str:
    """在线 Planner 的 system prompt。

    与离线建 cache 的任务**不同**：离线是「看完整条 demo 做后验分段」，
    在线是「看当前这一帧，判断机器人正处在计划的第几步」。
    在线版本不允许改写计划，只能在给定的步骤序列里选一个下标 ——
    这样 B2/B3 拿到的候选集合完全相同，差值里仍然只有「有没有码」。
    """
    return """You are a progress monitor for a robot arm executing a multi-step plan.

You are given:
  * the overall task instruction;
  * an ordered list of plan steps (each with an index and a short instruction);
  * the index the controller currently believes it is on;
  * camera images of the CURRENT scene (front view, and a wrist close-up when available);
  * whether the gripper is currently open or closed.

Decide which plan step the robot should be executing RIGHT NOW, judging from the images.

HARD CONSTRAINTS
1. Answer with an index from the given list only. Never invent, merge, split or reword steps.
2. The index must NOT go backwards: it must be >= the controller's current index.
3. Advance only when the images show the current step is already finished. When in doubt, stay.
4. Do not jump more than 2 steps ahead of the current index in a single decision.

Output strict JSON only, no prose, no code fences:
  {"index": <int>, "reason": "<at most 12 words>"}
"""


def build_online_user_content(task: str, task_instruction: str,
                              steps: list[dict], cur: int,
                              gripper_open, images: list[tuple[str, str]]) -> list[dict]:
    """images: [(view_name, data_url), ...]"""
    lines = [f"task: {task}",
             f"task instruction: {task_instruction}",
             f"gripper: {'open' if (gripper_open is None or gripper_open > 0.5) else 'closed'}",
             f"controller's current step index: {cur}",
             "plan steps:"]
    for i, s in enumerate(steps):
        mark = "  <-- current" if i == cur else ""
        lines.append(f"  [{i}] {s['action']}: {s['instruction']}{mark}")
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for name, url in images:
        content.append({"type": "text", "text": f"{name} view of the current scene:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


# --------------------------------------------- 在线 Planner：开局/重规划的 PLAN

def build_plan_system_prompt() -> str:
    """在线 PLAN 的 system prompt —— 看当前场景，现场推理出子任务序列。

    与离线建 cache 的 `build_system_prompt` 是**不同任务**：
      离线：给定一条完整 demo 的全部关键帧，做**后验分组**（能看到未来）；
      在线：只看当前这一帧，**预先规划**接下来要做什么（看不到未来）。

    但两者的输出格式与用词约束必须**逐条相同** —— 因为下游的 PerAct 与
    Adapter 都是用离线那套指令训练的。指令一旦落在训练分布外，
    两边会同时受害，而且**不会报任何错**。
    """
    allowed = ", ".join(list(CODEBOOK_ACTIONS) + list(NON_CODEBOOK_ACTIONS))
    return f"""You are a task planner for a Franka robot arm working on a tabletop.

RESPOND WITH A SINGLE JSON OBJECT AND NOTHING ELSE. No reasoning, no prose,
no markdown, no code fences. Any text outside the JSON object is a failure.

You see the CURRENT scene (front view, and a wrist close-up). Decompose the
remaining work into a short ordered sequence of atomic actions the arm should
execute from here.

HARD CONSTRAINTS
1. `action` must be one of: {allowed}
2. `instruction` format: all lowercase, no trailing punctuation,
   {INSTRUCTION_MIN_WORDS}-{INSTRUCTION_MAX_WORDS} words, imperative mood.
   It describes that step alone - never restate the whole task.
   It MUST keep the distinguishing information from the task instruction
   (colour, ordinal, spatial position), otherwise different scene variations
   become indistinguishable to the controller.
3. Start each instruction with the verb prescribed below. These verbs are fixed
   by the downstream controller's training distribution - do not substitute
   synonyms:
{_verb_table()}
4. Plan only what is still left to do, starting from what the images show now.
   Do not include steps that are already finished.
5. Keep the plan short: one atomic action per step, typically 3-8 steps.
   Use `pose-adjust` only when no other action applies.

ACTION DEFINITIONS
{_definitions_block()}

{DISAMBIGUATION}

ON THE REFERENCE SEQUENCE
You are given a reference action sequence from human annotation for this task.

It is a REFERENCE, not a script. Your goal is to produce a plan that actually
completes the task from the scene you are looking at - not to reproduce the
reference verbatim.
  * use it as a prior on **which actions are involved and roughly in what
    order**; follow it when the scene agrees with it;
  * you MAY merge, split or reorder steps, and MAY choose a different
    granularity, when the images call for it;
  * you MAY skip steps that the scene shows are already done (you can be called
    part-way through an episode);
  * BUT when a repeat count is given, the plan MUST contain exactly that many
    repetitions. That count is derived from the scene configuration and is
    reliable - do not try to count objects from the image yourself.

ON THE PHRASINGS LIST
When a list of trained phrasings is given, prefer reusing those exact strings
whenever one of them describes the step you mean. They are the wording the
downstream controller was actually trained on, and they carry the correct object
names and colours for this scene. Write your own wording only when none of them
fits - and then stay as close to their style and vocabulary as you can.

OUTPUT FORMAT - reply with exactly this shape and nothing else:
{{"plan": [
  {{"action": "grasp", "instruction": "grasp the grey jar lid"}},
  {{"action": "lift", "instruction": "lift the grey jar lid"}},
  {{"action": "transfer", "instruction": "move the lid over the red jar"}},
  {{"action": "place", "instruction": "place the lid on the red jar"}}
]}}

Do your reasoning silently. Emit only the JSON object.
"""


def build_plan_user_content(task: str, task_instruction: str,
                            images: list[tuple[str, str]],
                            done: list[dict] | None = None,
                            prior: list[str] | None = None,
                            n_repeat: int | None = None,
                            phrasings: list[str] | None = None) -> list[dict]:
    """images: [(view_name, data_url), ...]；done: 重规划时已完成的段。

    `prior` 是 `Phase_Action_Label.csv` 展开后的参考动作序列，`n_repeat` 是
    由 variation 号确定性推出的重复次数 —— 与离线建 cache 用的是同一套
    （`planner/contract.expand_prior` / `n_repeat_prior`）。

    没有它时实测的两个主要失败形态：
      · stack_blocks 数不清要堆几个（在线规划 8 段 vs 真值 14 段）
      · close_jar 多加一个前置 approach，粒度与训练分布不一致
    先验把「哪些动作、什么顺序、重复几次」直接给定，VLM 只需判断
    「当前场景走到哪一步了」。
    """
    lines = [f"task: {task}", f"task instruction: {task_instruction}"]
    if prior:
        lines.append("reference action sequence (soft prior from human "
                     "annotation, you may deviate): " + " -> ".join(prior))
    if n_repeat and n_repeat > 1:
        lines.append(f"NOTE: this scene repeats the reference unit {n_repeat} "
                     f"times. This count is derived from the variation index "
                     f"and is reliable - your plan must contain exactly "
                     f"{n_repeat} repetitions.")
    if done:
        lines.append("already completed:")
        lines += [f"  - {d['action']}: {d['instruction']}" for d in done]
        lines.append("Plan only the remaining steps.")
    if phrasings:
        # 这些是**控制器真正训练时见过的原文**，取自同一 (task, variation)
        # 的 train episode。实测在线规划注入的指令只有 56.6% 落在训练分布内
        # （模板法 100%），未命中的里既有措辞变体，也有把物体/颜色认错的
        # （"red coffee box"、场景里不存在的颜色）。给出候选清单既示范了
        # 指令该长什么样，也钉住了这个场景里物体的正确名字与颜色。
        lines.append("")
        lines.append("phrasings the controller was trained on, for THIS task "
                     "and THIS scene configuration (they also tell you the "
                     "correct names and colours of the objects in front of you):")
        lines += [f"  - {x}" for x in phrasings]
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for name, url in images:
        content.append({"type": "text", "text": f"{name} view of the current scene:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


# ------------------------------------- 在线 Planner：触发式进度判定 MONITOR

def build_monitor_system_prompt() -> str:
    """四态进度判定。与 `build_online_system_prompt`（选下标）的区别：
    这里允许判定**失败**并要求重做或重规划，而不只是往前走。"""
    return """You are a progress monitor for a robot arm executing a plan.

RESPOND WITH A SINGLE JSON OBJECT AND NOTHING ELSE. No prose, no code fences.

You are given the task, the current plan, which step the controller is on,
how many keyframes it has spent on that step, the gripper state, and camera
images of the CURRENT scene.

Decide one of:
  CONTINUE - the current step is under way and still achievable; keep going.
  NEXT     - the current step's goal is visibly achieved; move to the next step.
  RETRY    - a step failed and must be redone (e.g. the grasp missed, the object
             slipped out of the gripper, the arm is holding nothing).
             Give `index` = the step to go BACK to and redo from. It may be an
             EARLIER step than the current one: if the object was never actually
             grasped, going back to the grasp step is the only thing that helps -
             repeating "place" while holding nothing changes nothing.
             Omit `index` to simply redo the current step.
  REPLAN   - the WORLD changed so that the remaining steps no longer make sense
             (e.g. an object was knocked over or fell off the table, the wrong
             object is now held and must be put back first). The plan will be
             regenerated from the current scene.

🔴 DO NOT use REPLAN just because the arm is in the wrong place, is moving
toward the wrong object, or has not reached the target yet. Those are execution
errors by the controller - the PLAN is still correct, and regenerating it would
produce the same steps and change nothing. In that situation answer CONTINUE
and let the controller keep trying.

WHY STAYING IS EXPENSIVE
The controller only ever attempts the step you name. If you keep answering
CONTINUE while it struggles, it will never attempt the next step and the
episode will run out of time on the first step. Episodes are short - a plan of
N steps has only a few keyframes per step. So:
  * answer NEXT as soon as the current step is essentially done, or when the
    scene shows the arm has effectively moved past it;
  * when the scene is already several steps further along than the controller
    thinks, say so with `index` - you may jump ahead by up to 2 steps at once;
  * reserve CONTINUE for when the current step is clearly still in progress and
    close to succeeding.

Judge from the images, not from the step counter.
Prefer NEXT or CONTINUE. RETRY and REPLAN should be rare.

Output exactly. `index` = which step should be current NOW:
  with NEXT  -> the step to advance to (omit = the next one)
  with RETRY -> the step to go back to and redo (omit = the current one)
  {"decision": "NEXT", "index": 2, "reason": "<at most 12 words>"}
"""


def build_monitor_user_content(task: str, task_instruction: str,
                               plan: list[dict], cur: int, used: int,
                               gripper_open, images: list[tuple[str, str]],
                               stalled: bool = False,
                               quiet: str = "") -> list[dict]:
    lines = [f"task: {task}",
             f"task instruction: {task_instruction}",
             f"gripper: {'open' if (gripper_open is None or gripper_open > 0.5) else 'closed'}",
             f"keyframes spent on the current step: {used}",
             "plan:"]
    for i, s in enumerate(plan):
        mark = "   <-- CURRENT" if i == cur else ""
        lines.append(f"  [{i}] {s['action']}: {s['instruction']}{mark}")
    if stalled or quiet:
        # 给 VLM **新信息**。v1 实测连判 6 次 REPLAN，每次生成几乎相同的计划 ——
        # 因为它每次看到的输入都一样，没有任何理由换个说法。
        sig = []
        if stalled:
            sig.append(f"it has spent {used} keyframes on this step")
        if quiet == "joint_velocity":
            sig.append("the arm's joint velocities are ~0 (it has stopped moving)")
        elif quiet == "pose":
            sig.append("the gripper pose has not changed between keyframes")
        lines.append("")
        lines.append("PROGRESS WARNING: " + "; ".join(sig) + ".")
        lines.append(
            "This means one of: (a) the step is worded in a way the controller "
            "cannot execute -> REPLAN with a different decomposition or "
            "different wording; (b) the step is already done and you have not "
            "noticed -> NEXT; (c) the arm is merely slow -> CONTINUE. "
            "Regenerating an identical plan will not help - if you REPLAN, "
            "change something.")
    content: list[dict] = [{"type": "text", "text": "\n".join(lines)}]
    for name, url in images:
        content.append({"type": "text", "text": f"{name} view of the current scene:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content
