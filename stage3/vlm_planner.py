"""在线 VLM Planner —— 模板法的备选方案（贴近原设计）。

# 和模板法的区别

模板法把子任务序列**开跑前就定死**，只靠夹爪翻转和关键帧预算推进，是开环的：
模型一旦走偏，计划不会跟着调整。在线 VLM 每个关键帧看一眼当前画面，
判断「现在该做第几步」，是闭环的。

# 当前版本的约束（**是阶段性取舍，不是原理限制**）

本版**子任务序列仍来自模板库，VLM 只负责选下标**。这么做只是为了先把闭环
链路跑通、失败模式可枚举。

⚠️ 早先这里写过「码 k_global/k_detail 只能从 cache 里查，VLM 现编不出来」——
**那是错的**。设计（`VLA_Design §3`、`Adapter_Design §4`）里码本来就由
**Stage 2 Adapter** 从 `(front+wrist 观测, 子任务指令)` 预测得到；
`checkpoints/vqap_adapter/best.pth` 已训练完成（val global_top1 0.897），
离线 cache 里的码就是它算的（`scripts/planner_cache.py` 的 codes 步骤）。
所以 VLM 自由生成子任务指令 → Adapter 现场出码，这条路是通的，
缺的只是「评测时实时调 Adapter」这一段接线。

而且 planner 的指令并非自由文本：`planner/prompts.py` 用固定动作词表
（17 个码本动作 + pose-adjust）、固定起始动词表、3–8 词、小写无尾标点
约束输出，`normalize_instruction` 离线在线共用 —— 生成的指令天然落在
Adapter 与 PerAct 的训练分布内。

# 成本与延迟

每个关键帧一次调用，串在 rollout 里。实测评测每局约 26 步，
10 局/任务 × 12 任务 = 120 局 ≈ 3100 次调用。12 个分片并行，
每片约 260 次串行调用。`PlannerClient` 按 sha256(model + messages) 磁盘记忆化，
所以**重跑同一份评测不再付费，且结果逐位可复现**。

# 失败要退化，但**必须留痕**

早先这里写的是「失败即报错，不做静默退化」，理由是：静默退化会让「VLM 方案」
的成绩单里混进模板法产生的步骤，数字说不清是什么。这个理由本身是对的，
但当时的实现把它推到了另一个极端 —— 抛 `PlannerError`，而 YARR 的
`_run_eval_independent` 会把异常继续往上抛，于是**整个分片死掉**。

2026-09-09 实测：一次 `APIConnectionError` 让 12 个分片分别在第 7~24 局
被打死，300 局只跑出 126 局，且 `eval_data.csv` 合并出 0 行。
单局的网络抖动不该让同一任务剩下的二十局陪葬。

现在的做法是**退化 + 留痕**，两者缺一不可：
  - PLAN 失败   → 回落到模板计划把这一局跑完，轨迹里记 `plan_source=
                  "template_fallback"`，`stats["plan_fallback"]` 计数
  - MONITOR 失败 → 退化成 CONTINUE（停在当前段），记 `monitor_fallback`
  - 连续失败 2 次 → 判定 API 断线，本局不再发起调用（否则每次都要走满
                  4 次重试 ×指数退避 ≈ 30 s，实测把速度拖到 25.8 s/局）

关键区别：退化过的局在轨迹里**可识别**，分析时能整局剔除或单独对照，
所以成绩单仍然说得清。静默的是旧实现，不是这一版。
"""

from __future__ import annotations

import json
import os
import re
import threading

from pathlib import Path

from stage3.online_planner import DeterministicPlanner, PlannerError

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 单次决策最多前进几步。v2 试过 2（多段跳），没有收益、反而助长「推太快」，
#: v3 回到 1。
MAX_ADVANCE = 1
#: 送进 VLM 的视角。front 看全局进度，wrist 看有没有抓住东西。
DEFAULT_VIEWS = ("front", "wrist")


class VLMPlanner(DeterministicPlanner):
    """继承确定性状态机，只把「推进与否」的判断换成 VLM。

    继承而不是另起一个类，是为了让退化路径**天然存在**：
    VLM 不可用时直接调用父类的 `observe`，行为与模板法逐字相同。
    """

    def __init__(self, subtasks, task: str, task_instruction: str,
                 client, views=DEFAULT_VIEWS, img_size: int = 224) -> None:
        super().__init__(subtasks)
        self.task = task
        self.task_instruction = task_instruction
        self._client = client
        self._views = tuple(views)
        self._img_size = img_size
        self.stats = {"call": 0, "fail": 0,
                      "advance": 0, "stay": 0, "clamped": 0}
        self.decisions: list[dict] = []

    # ------------------------------------------------------------------
    def observe(self, gripper_open: float | None, frame=None) -> None:
        self.t += 1
        self.used += 1
        if gripper_open is not None:
            self.prev_gripper = float(gripper_open)
        last = len(self.subtasks) - 1
        if self.idx >= last:
            return                                  # 已在最后一段，无需再问
        if not frame:
            raise PlannerError(
                "在线 VLM planner 拿不到相机图像。observation 里应有 "
                f"{'/'.join(self._views)}_rgb —— 检查 eval.yaml 的 rlbench.cameras。")

        try:
            idx = self._ask(gripper_open, frame)
        except Exception as e:                      # 超时 / 限流 / 解析失败
            self.stats["fail"] += 1
            self.decisions.append({"t": self.t, "error": f"{type(e).__name__}: {e}"[:200]})
            # 不退化：静默换规则会让「VLM 方案」的成绩单里混进模板法的步骤。
            raise PlannerError(
                f"在线 VLM planner 调用失败（{type(e).__name__}: {e}）。"
                f"客户端已重试 4 次仍不可用，本局评测中止。") from e

        raw = idx
        idx = min(max(idx, self.idx), min(self.idx + MAX_ADVANCE, last))
        if idx != raw:
            self.stats["clamped"] += 1
        self.stats["advance" if idx > self.idx else "stay"] += 1
        if idx != self.idx:
            self.idx = idx
            self.used = 0

    # ------------------------------------------------------------------
    def _ask(self, gripper_open, frame: dict) -> int:
        from planner.client import image_data_url_from_array
        from planner.prompts import (build_online_system_prompt,
                                     build_online_user_content)
        images = []
        for v in self._views:
            arr = frame.get(v)
            if arr is not None:
                images.append((v, image_data_url_from_array(arr, self._img_size)))
        if not images:
            raise PlannerError("observation 里没有可用的相机图像")
        msgs = [{"role": "system", "content": build_online_system_prompt()},
                {"role": "user",
                 "content": build_online_user_content(
                     self.task, self.task_instruction, self.subtasks,
                     self.idx, gripper_open, images)}]
        self.stats["call"] += 1
        out = self._client.chat(msgs)
        idx, reason = _parse(out["content"])
        self.decisions.append({"t": self.t, "from": self.idx, "to": idx,
                               "reason": reason})
        return idx


def _parse(text: str) -> tuple[int, str]:
    """从模型输出里抠出 index。容忍代码围栏和前后废话。"""
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    try:
        d = json.loads(s)
        return int(d["index"]), str(d.get("reason", ""))[:60]
    except Exception:
        pass
    m = re.search(r'"index"\s*:\s*(\d+)', s) or re.search(r"\b(\d+)\b", s)
    if not m:
        raise PlannerError(f"无法从输出解析 index: {s[:120]!r}")
    return int(m.group(1)), "(parsed from raw text)"


# ---------------------------------------------------------------- 工厂

class VLMPlanFactory:
    """和 `PlanFactory` 一样是模块级类（spawn 子进程要 pickle）。

    client 与 bank 都延迟到子进程里建：`OpenAI` 客户端持有 socket，
    pickle 不过去；bank 是 550 KB 的字典，没必要序列化 12 份。
    """

    kind = "vlm"

    def __init__(self, split: str, model: str | None = None,
                 views=DEFAULT_VIEWS, verbose: bool = True) -> None:
        self.split = split
        self.model = model or os.environ.get("AAVLA_ONLINE_PLANNER_MODEL",
                                             "qwen3.8-max")
        self.views = tuple(views)
        self._verbose = verbose
        self._bank = None
        self._client = None
        self._lock = None
        self.bank                                   # 父进程尽早失败
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise PlannerError(
                "在线 VLM planner 需要 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL。"
                "先 source run/env.sh")

    @property
    def bank(self):
        if self._bank is None:
            from stage3.planners import plan_file
            from stage3.online_planner import PlanBank
            # 计划序列仍取自模板库：VLM 只选下标，不改写计划（见模块 docstring）
            self._bank = PlanBank(plan_file(self.split, "template"))
            if self._verbose:
                print(f"[planner] kind=vlm model={self.model} "
                      f"views={','.join(self.views)} {self._bank.summary()}",
                      flush=True)
        return self._bank

    @property
    def client(self):
        if self._client is None:
            from planner.client import PlannerClient
            # 关掉思考链 + 收紧 max_tokens：答案只有一行 JSON，
            # 而带思考时模型会先吐几百到三千个 token，延迟正比于此。
            # 实测 30s/次 -> 约 1s/次，全量评测从 26 小时降到 1 小时以内。
            self._client = PlannerClient(
                model=self.model, max_tokens=256,
                extra_body={"enable_thinking": False})
        return self._client

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_bank"] = None
        d["_client"] = None                          # 持有 socket，pickle 不过去
        d["_lock"] = None
        return d

    def __call__(self, task: str, episode: int):
        from stage3.online_planner import episode_descriptions
        subs = self.bank.get(task, episode)
        d = episode_descriptions(task, self.split, episode)
        return VLMPlanner(subs, task, d[0] if d else task.replace("_", " "),
                          self.client, self.views)

    def summary(self) -> str:
        return f"VLM({self.model}) over {self.bank.summary()}"


# ============================================================ 真·在线 planner

#: 触发 MONITOR 的心跳间隔（关键帧）。demo 每段中位 1–2 帧，4 帧足以覆盖
#: 正常推进而不至于漏判；调小更灵敏但更贵。
#: 心跳间隔（关键帧）。实测：心跳 4 → 6.4 次调用/局；心跳 2 → 9.5 次/局。
HEARTBEAT = int(os.environ.get("AAVLA_PLANNER_HEARTBEAT", "3"))
#: 单段占用超过这么多关键帧就判定为「停滞」。
#: 🔴 v3 改变了停滞的**处理方式**：v2 是本地强制前进，实测把 NEXT 从 42%
#:    推到 72%，成绩反而从 26.67% 掉到 22.33% —— 盲目往前推不解决问题。
#:    v3 改成「照常问 VLM，但把『已经在这一步耗了 N 帧』作为事实告诉它」，
#:    因为 v1 实测连判 6 次 REPLAN、每次生成几乎相同的计划 —— 它没收到任何
#:    新信息，重规划自然是原样重来。
STALL_LIMIT = 4
#: 关节速度的「停了」阈值（rad/s，取无穷范数）。
VEL_EPS = 0.01
#: 末端位姿的「没动」阈值（米）。
POSE_EPS = 0.005
#: 速度/位姿判定要连续命中几帧才算数 —— 单帧为零可能只是运动规划的间隙。
QUIET_FRAMES = 2
#: 每局的调用上限（成本护栏）。
#: 🔴 实测 200 局：调用/局 中位 13、p90 15、max 22 —— 原来的 12 会在 **52%**
#:    的局上触顶，等于一半的局中途失去 planner。按 p90 留余量取 25。
#: 🔴 HEARTBEAT=1（每个 waypoint 问一次）时**必须同步提高**：中位 episode 有
#:    26 个关键帧，25 的上限会在多数局触顶，等于后半局失去 planner。
MAX_CALLS = int(os.environ.get("AAVLA_PLANNER_MAX_CALLS", "25"))

# ---------------------------------------------------------------- v3.2 开关
# 三项改动各自独立，出问题可以逐项关掉重跑定位，不需要改代码。
#: 指令来源：select = 三层协议（选编号 / 编号+替换 / 模仿句式自由生成）；
#:           free   = v3.1 的自由生成（实测只有 45% 落在训练分布内）
PHRASING_MODE = os.environ.get("AAVLA_PHRASING_MODE", "select")
#: 是否把「已尝试过什么」喂进 PLAN 与 MONITOR 的 prompt。
#: 🔴 默认**关**。实测（B3@40000 · val 120 局）：
#:      开 = v3.2a 26.67%，REPLAN 爆到 173 次
#:      关 = v3.2c 33.33%，REPLAN 降到 6 次
#:    喂执行记忆会让 VLM 反复推翻自己的计划。默认值曾经是「开」，也就是
#:    已知最差的配置 —— 不带环境变量直接跑就会静默拿到它，没有任何报错。
USE_HISTORY = os.environ.get("AAVLA_PLANNER_HISTORY", "0") != "0"
#: 是否启用备选先验（REPLAN / RETRY 用满时换一种分解）
USE_PRIOR_VARIANTS = os.environ.get("AAVLA_PRIOR_VARIANTS", "1") != "0"
#: 每局的关键帧预算，仅用于在 prompt 里告诉 VLM「还剩多少帧」
EPISODE_BUDGET = int(os.environ.get("AAVLA_EPISODE_BUDGET", "26"))
#: 是否按动作语义解释夹爪状态变化（v3.3）
USE_GRIPPER_SEMANTICS = os.environ.get("AAVLA_GRIPPER_SEMANTICS", "1") != "0"

#: 夹爪状态变化在不同原子动作下的含义完全不同 —— 逐条取自
#: `planner/prompts.py` 的 ACTION_DEFINITIONS 原文。
#:
#:   boundary  夹爪开合**就是这一步的完成标志**
#:             grasp「closes, firmly fixing the object」
#:             place「brings the bottom of the object into contact with the surface」
#:
#:   tool      夹爪是**工具或准备姿态**，闭合发生在动作过程中，不代表完成
#:             push 「end effector (usually **closed** or in a specific posture)
#:                    applies horizontal pressure」
#:             transfer「if the gripper **releases at the very end** of a transfer,
#:                    that release ... still counts as transfer」
#:             approach「does NOT involve any physical contact」—— 本不该有变化
#:
#:   hold      定义里明写全程保持抓握，松开多半意味着**脱手**
#:             lift「maintaining a stable grip」· rotate「without loosening」
#:             pull「keeps the grasp unchanged」· wipe「stably grasps」…
#:
#: 🔴 这是 slide_block 归零的直接原因：`approach` 阶段夹爪为「推」而闭合，
#:    被当成边界信号，t=1 就推进，之后一路震荡到 26 步没做成。
GRIPPER_SEMANTICS: dict[str, str] = {
    **{a: "boundary" for a in ("grasp", "place")},
    **{a: "tool" for a in ("approach", "push", "press", "revolve-in", "transfer")},
    **{a: "hold" for a in ("lift", "rotate", "pull", "slide", "wipe", "insert",
                           "hang", "flip-open", "flip-close", "revolve-out")},
    "pose-adjust": "tool",          # 定义即「未持物时的姿态修正」
}


_DWELL_CACHE: dict | None = None


def load_dwell_prior() -> dict:
    """→ {task: {action: 最短停留关键帧数}}，取自 train split 的中位数。

    只用 train，不碰 val/test —— 这是先验，不是标签。
    """
    global _DWELL_CACHE
    if _DWELL_CACHE is not None:
        return _DWELL_CACHE
    import statistics, collections
    path = REPO_ROOT / "aavla_data" / "planner_cache" / "plans_train_template.json"
    out: dict = {}
    try:
        plans = json.loads(path.read_text())["plans"]
        acc = collections.defaultdict(list)
        for key, v in plans.items():
            task = key.split("/")[0]
            for seg in v["subtasks"]:
                acc[(task, seg["action"])].append(int(seg.get("n_keyframes", 1)))
        for (task, action), vals in acc.items():
            med = int(statistics.median(vals))
            if med >= 2:                       # 只记 >1 的，其余默认 1
                out.setdefault(task, {})[action] = med
    except Exception:
        out = {}                               # 先验缺失就退回旧行为
    _DWELL_CACHE = out
    return out


def gripper_class(action: str) -> str:
    """该动作下夹爪变化的含义；未知动作按 boundary 处理（保持旧行为）。"""
    return GRIPPER_SEMANTICS.get(action, "boundary")
#: 同一段最多重试几次。实测 RETRY/局 中位 0、p90 3。
MAX_RETRY = 4
#: 每局最多重规划几次。实测 6% 的局触顶 2；v3 把停滞导向 REPLAN 后需求上升。
MAX_REPLAN = 4
# ------------------------------------------------------------------ v3.5
#: 每局最多「续写计划」几次。
#: 🔴 由来（B4@40000 · val 300 局 · 143 个纯 v3.3 局的轨迹）：
#:      末段的决策 95% 是 NEXT，而末段 NEXT 被 clamp —— target=None，
#:      指令一字不变，PerAct 输入不变，输出必然不变。500/886 次调用白烧。
#:      末段死锁 ≥5 次的局共 40 个，**成功率 0%**；其余局 67%。
#:      而它们中位在 t=9 就走完计划（中位 4 段），之后 17 步无指令可发。
#:    缺的不是更好的段，是**更多的段** —— 88% 的成功局根本没走到末段。
MAX_EXTEND = int(os.environ.get("AAVLA_MAX_EXTEND", "2"))
#: 计划总段数上限，防止反复续写把计划撑爆。
MAX_SEGMENTS = int(os.environ.get("AAVLA_MAX_SEGMENTS", "10"))
#: 是否允许续写。关掉即退化成 v3.4 行为（末段 NEXT 仍为空操作）。
ALLOW_EXTEND = os.environ.get("AAVLA_ALLOW_EXTEND", "1") != "0"
# ------------------------------------------------------------------ v3.6
#: 每段的**最短停留关键帧数**，来自 train split 的真实分段统计。
#: 🔴 由来：sweep 与 reach_and_drag 两个任务在 v3.4/v3.5 上都比模板法跌
#:    18~36 pp，且跌幅跨版本重现（不是噪声，也不是 v3.4/v3.5 的机制造成的
#:    —— EXTEND 在 sweep 上触发 0 次）。逐局对照发现成功与失败的差别精确
#:    地就是「多推进了一段」：
#:        sweep 成功局到达段位 2.2 / 失败局 3.0
#:        drag  成功局到达段位 2.1 / 失败局 3.1
#:    而模板计划带着每段的真实关键帧数（sweep 的 transfer 要待 2 帧），
#:    X3 自己生成的计划里 n_keyframes 一律是 1 —— 这个先验被丢掉了。
#:    这两个任务的完成信号在图像上不可见（扫帚划过灰尘、方块被拖动，
#:    「正在做」和「做完了」看起来一样），VLM 只能猜，必然早推。
#:  规则完全条件式：train 统计里中位 ≥2 的 (task, action) 只有 7 个组合，
#:  其余全部为 1 —— 对它们规则永不触发，行为逐字不变。
MIN_DWELL = os.environ.get("AAVLA_MIN_DWELL", "1") != "0"


class OnlineVLMPlanner:
    """开局现场规划 + 触发式进度判定 + 失败重试/重规划。

    与 `VLMPlanner`（在模板库给的序列里选下标）的根本区别：
    **子任务序列由 VLM 看着画面推理生成，完全不读模板库。** 这样模板库的
    结构性问题（22% 模板只有 1 条源局支撑、8 局回退时指令与场景物体不符、
    预算来自 demo 关键帧数而评测要走 26 步）一个都不会带进来。

    CSV 先验（`Phase_Action_Label.csv`）只作**参考**送进 prompt：
    它给出「涉及哪些动作、大致什么顺序」，以及由 variation 号确定性推出的
    **重复次数**（后者是事实，不是粒度选择，所以仍是硬约束 ——
    实测加上它之后 stack_blocks 的规划一致率从 0% 升到 100%）。
    粒度允许与离线分段不同：目标是把任务做成，不是复刻离线分段。

    码不来自计划，必须配 `--codes adapter`（实时 Adapter）——
    VLM 生成的指令是新的，模板库里没有对应的码可查。
    """

    def __init__(self, task: str, task_instruction: str, client,
                 prior: list[str] | None = None, n_repeat: int | None = None,
                 phrasings: list[str] | None = None,
                 views=DEFAULT_VIEWS, img_size: int = 224,
                 heartbeat: int = HEARTBEAT,
                 fallback: list[dict] | None = None,
                 dwell: dict | None = None) -> None:
        self.task = task
        self.task_instruction = task_instruction
        self._client = client
        # 🔴 API 断线时的兜底计划（模板库的子任务序列）。
        #    没有它时一次 APIConnectionError 会抛 PlannerError，而 YARR 的
        #    _run_eval_independent 直接把异常再抛出去 —— **整个分片死掉**，
        #    该任务剩余的局全部丢失。2026-09-09 实测：12 个分片各自在
        #    7~24 局处被断线打死，300 局只跑出 126 局。
        self._fallback = list(fallback) if fallback else None
        self.plan_source = "vlm"
        #: 连续失败到一定次数就停止再调 VLM —— 否则每次调用都要走满
        #: 4 次重试 ×指数退避 ≈ 30 s，一局能拖到几十分钟（实测 25.8 s/局）。
        self._consec_fail = 0
        self._api_down = False
        # 先验现在是**变体列表**：list[list[str]]。单变体任务自动包一层，
        # 行为与原来逐位相同。走不通时 _next_variant() 换下一种分解。
        pv = prior or []
        if pv and isinstance(pv[0], str):          # 兼容旧的 list[str]
            pv = [list(pv)]
        self._prior_variants: list[list[str]] = [list(v) for v in pv] or [[]]
        self._variant = 0
        self._n_repeat = n_repeat
        # 该 (task, variation) 的真实训练指令。作为软参考进 prompt：
        # 既示范指令该长什么样，也钉住这个场景里物体的正确名字与颜色。
        self._phrasings = list(phrasings or [])
        self._views = tuple(views)
        self._img_size = img_size
        self._heartbeat = heartbeat

        self.subtasks: list[dict] = []
        self.done: list[dict] = []       # 已完成的段，REPLAN 时告诉 VLM
        self.idx = 0
        self.t = 0
        self.used = 0
        self.last_ask = 0
        self.code_epoch = 0              # NEXT/RETRY/REPLAN 时 +1 -> 让 Adapter 重算
        self.retry = 0
        self.n_replan = 0
        self._dwell = dict(dwell or {})   # v3.6：{action: 最短停留帧数}
        self.n_extend = 0                # v3.5：续写过几次
        self.extends: list[dict] = []    # 每次续写追加了什么，供事后判定有效性
        self._plan_exhausted = False     # 续写预算用尽 + 仍在末段 -> 停止再问
        self.prev_gripper: float | None = None
        self.prev_pose = None          # 上一帧的末端位姿（xyz），用于位移判定
        self.quiet = 0                 # 连续「静止」的帧数
        self.history: list[dict] = []
        self.decisions: list[dict] = []
        self.stats = {"plan_call": 0, "monitor_call": 0, "fail": 0,
                      "CONTINUE": 0, "NEXT": 0, "RETRY": 0, "REPLAN": 0,
                      "budget_exhausted": 0, "trigger_flip": 0,
                      "trigger_heartbeat": 0, "trigger_quiet": 0,
                      "trigger_stall": 0, "no_robot_state": 0}

    # ---------------------------------------------------------- 先验变体
    @property
    def prior(self) -> list[str]:
        """当前正在用的那个先验变体。"""
        return self._prior_variants[self._variant]

    def _next_variant(self) -> bool:
        """换下一个先验变体；没有更多变体时返回 False。

        触发时机是「这条路走不通」：REPLAN，或同一段 RETRY 用满。
        实测 slide_block 的两种分解在 train 里分别占 6/10 与 4/10，
        先试频次高的短版本，失败再试长版本。
        """
        if not USE_PRIOR_VARIANTS or self._variant + 1 >= len(self._prior_variants):
            return False
        self._variant += 1
        self.stats["prior_variant_switch"] = self.stats.get("prior_variant_switch", 0) + 1
        return True

    # ---------------------------------------------------------- 对外协议
    @property
    def current(self) -> dict:
        return self.subtasks[min(self.idx, len(self.subtasks) - 1)]

    @property
    def budgets(self) -> list[int]:
        """轨迹落盘要读这个字段；本 planner 不用预算，给个占位。"""
        return [1] * len(self.subtasks)

    def prime(self, gripper_open: float | None, frame=None) -> None:
        """第一次 act()：建立夹爪基线，并**现场规划**出初始子任务序列。"""
        if gripper_open is not None:
            self.prev_gripper = float(gripper_open)
        if not frame:
            raise PlannerError(
                "vlm-plan 需要初始观测来做规划，但没拿到相机图像。"
                f"检查 eval.yaml 的 rlbench.cameras 是否含 {'/'.join(self._views)}。")
        self.subtasks = self._plan(frame)

    def note(self, step: int) -> None:
        st = self.current
        self.history.append({"t": len(self.history), "subtask_index": self.idx,
                             "action": st["action"], "gripper": self.prev_gripper,
                             "instruction": st["instruction"]})

    def observe(self, gripper_open: float | None, frame=None, robot=None) -> None:
        """执行完一个关键帧后判定进度。

        `robot` 是 env 旁路给的机器人低维状态（关节速度 / 末端位姿），
        由 `custom_rlbench_env.extract_obs` 在置空前留下 —— 它们**不在**
        observation 里（那里的 low_dim_state 只有 4 维，与训练一致）。
        """
        self.t += 1
        self.used += 1
        flip = (self.prev_gripper is not None and gripper_open is not None
                and abs(float(gripper_open) - float(self.prev_gripper)) > 0.5)
        if gripper_open is not None:
            self.prev_gripper = float(gripper_open)
        quiet = self._update_motion(robot)

        last0 = len(self.subtasks) - 1
        # ---- 触发判定（本地规则，零成本）----
        # 停滞不再本地强制前进（v2 那样做实测更差），而是**照常问 VLM**，
        # 但把「已经耗了 N 帧」作为事实塞进 prompt，让它有依据换个说法。
        stalled = self.used > STALL_LIMIT
        at_last = self.idx >= last0
        why = ""
        gclass = gripper_class(self.current["action"]) if USE_GRIPPER_SEMANTICS else "boundary"
        if flip and gclass == "tool" and USE_GRIPPER_SEMANTICS:
            # 工具类动作：闭合是准备姿态，不是完成信号 —— 不为它单独发问，
            # 但记账，便于事后确认这条规则拦下了多少次误触发。
            self.stats["flip_suppressed"] = self.stats.get("flip_suppressed", 0) + 1
            flip = False
        if flip and not at_last:
            why = "gripper_flip"
        elif stalled:
            why = "stall"
        elif quiet:
            why = "quiet"                            # 关节速度≈0 或末端没动
        elif self.t - self.last_ask >= self._heartbeat and not at_last:
            why = "heartbeat"
        # 🔴 已在末段时只保留停滞/静止两条触发。实测末段后仍每 3 帧问一次，
        #    连判 4 次 NEXT（已经无处可去），理由还自相矛盾 ——
        #    纯粹浪费调用。停滞/静止仍要问，因为那时可能需要 RETRY/REPLAN。
        if not why:
            return                                   # 不触发就沿用当前段
        if self._api_down:
            # API 已判定断线：本局不再发起调用，沿用当前段把局跑完。
            self.stats["skipped_while_down"] = \
                self.stats.get("skipped_while_down", 0) + 1
            return
        if self._plan_exhausted and at_last:
            # 续写预算用尽且仍停在末段 —— 已经确认没有可执行的动作了，
            # 再问也只会拿回同一个空操作。这正是旧版烧掉 56% 调用的地方。
            self.stats["skipped_exhausted"] = \
                self.stats.get("skipped_exhausted", 0) + 1
            return
        n_calls = self.stats["plan_call"] + self.stats["monitor_call"]
        if n_calls >= MAX_CALLS:
            self.stats["budget_exhausted"] += 1
            return
        if not frame:
            raise PlannerError("vlm-plan 触发了进度判定但拿不到相机图像。")
        self.stats[f"trigger_{why.replace('gripper_flip','flip')}"] = \
            self.stats.get(f"trigger_{why.replace('gripper_flip','flip')}", 0) + 1
        self.last_ask = self.t

        d, tgt, reason = self._monitor(gripper_open, frame,
                                       stalled=stalled, quiet=quiet,
                                       gripper_class=gclass if flip else "",
                                       at_last=at_last)
        self.stats[d] = self.stats.get(d, 0) + 1
        self.decisions.append({"t": self.t, "trigger": why, "decision": d,
                               "idx": self.idx, "target": tgt, "reason": reason})
        self._apply(d, frame, tgt)

    def _update_motion(self, robot) -> str:
        """→ 静止的原因（"joint_velocity" / "pose" / ""）。

        连续 QUIET_FRAMES 帧命中才算数：单帧速度为零很可能只是运动规划的
        间隙，不代表卡住。
        """
        if not robot:
            self.stats["no_robot_state"] += 1
            return ""
        pose = robot.get("gripper_pose")
        xyz = (tuple(float(x) for x in pose[:3])
               if pose is not None and len(pose) >= 3 else None)
        prev, self.prev_pose = self.prev_pose, xyz if xyz else self.prev_pose

        # 🔴 关节在动就一票否决。否则会出现「关节速度 0.5 rad/s 但末端位姿
        #    两帧相同」也被判成静止的情形（单测抓到过）—— 那多半是位姿采样
        #    的时序问题，不是机械臂停了。物理上「停了」必须是关节先停。
        v = robot.get("joint_velocities")
        if v is not None and len(v):
            vmax = max(abs(float(x)) for x in v)
            if vmax >= VEL_EPS:
                self.quiet = 0
                return ""
            why = "joint_velocity"
        else:
            why = ""

        if xyz is not None and prev is not None:
            d = sum((a - b) ** 2 for a, b in zip(xyz, prev)) ** 0.5
            if d < POSE_EPS:
                why = why or "pose"
            else:
                why = ""                             # 末端明显移动了，不算静止
        self.quiet = self.quiet + 1 if why else 0
        return why if self.quiet >= QUIET_FRAMES else ""

    # ---------------------------------------------------------- 决策落地
    def _apply(self, d: str, frame, target: int | None = None) -> None:
        last = len(self.subtasks) - 1
        # 🔴 v3.5：末段的 NEXT 判为 EXTEND，而不是 clamp 成空操作。
        #    NEXT 的字面语义是「这一步做完了，去下一步」——没有下一步时，
        #    忠实的执行就是造一个出来。这是翻译，不是替 VLM 做决定。
        #    旧行为：target=None，指令一字不变 → PerAct 输入不变 → 输出必然
        #    不变 → 死锁到超时。实测末段死锁 ≥5 次的 40 局成功率 0%。
        if d == "NEXT" and self.idx >= last:
            self.stats["next_at_last"] = self.stats.get("next_at_last", 0) + 1
            d = "EXTEND" if ALLOW_EXTEND else "CONTINUE"
        if d == "NEXT" and MIN_DWELL and self._dwell:
            need = self._dwell.get(self.subtasks[self.idx]["action"], 1)
            if self.used < need:
                # 该段的真实分段统计要求至少待 need 帧，而现在还没待够。
                # 这两个任务的完成信号在图像上不可见，VLM 判 NEXT 不可靠 ——
                # 按 CONTINUE 处理（停在本段），并记账。
                self.stats["next_too_early"] = self.stats.get("next_too_early", 0) + 1
                return
        if d == "NEXT":
            if self.idx < last:
                # 允许一次跨多段（上限 MAX_ADVANCE）。v1 每次只能进 1 段，
                # 6 段计划需要 ≥24 个关键帧才走得完，而 episode 只有 26 步 ——
                # 实测 reach_and_drag 15 步只走到第 3 段，从没碰到关键的 slide。
                nxt = self.idx + 1 if target is None else int(target)
                nxt = min(max(nxt, self.idx + 1), min(self.idx + MAX_ADVANCE, last))
                self.done.extend(self.subtasks[self.idx:nxt])
                self.idx = nxt
                self.used = 0
                self.retry = 0
                self.code_epoch += 1
        elif d == "RETRY":
            # 🔴 RETRY 允许**回退**。实测轨迹里 VLM 连判 10 次
            #    “grasp failed, lid is on table”，判断完全正确，但索引单调不减，
            #    指令一直停在「搬运/放置」，而盖子根本没抓住 —— RETRY 变成死胡同，
            #    重试超限后反而往前推，正好是反的。
            #    现在 VLM 可以用 index 指定「退回第几步重做」。
            self.retry += 1
            self.code_epoch += 1
            if target is not None:
                back = min(max(int(target), 0), self.idx)
                if back < self.idx:
                    # 回退时把 done 里对应的段撤掉 —— 它们并没有真的完成
                    del self.done[back:]
                    self.idx = back
                    self.used = 0
                    self.retry = 0
                    return
            if self.retry > MAX_RETRY and self.idx < last:
                self._next_variant()                # 同一段重试到底了，也算走不通
                self.idx += 1                       # 原地重试太多次就放弃这一段
                self.used = 0
                self.retry = 0
        elif d == "REPLAN":
            if self.n_replan >= MAX_REPLAN:
                # 预算用完后再判 REPLAN 就按 CONTINUE 处理并计数。
                # 实测见过一局里连判 6 次 REPLAN（模型一直走向错的物体），
                # 后 4 次全被挡下但白花了调用 —— 这里显式记账，便于事后看清。
                self.stats["replan_blocked"] = self.stats.get("replan_blocked", 0) + 1
                return
            self.n_replan += 1
            self._next_variant()          # 这条分解走不通 -> 换一种再规划
            # 🔴 **不**把走过的段记成「已完成」。实测：grasp 明明失败了
            #    （VLM 自己都说 “lid is on table, not held”），但 idx 已经推到 5，
            #    于是 done 告诉 VLM「approach/grasp/lift/transfer/place 都做完了」，
            #    它就只规划了一步 “rotate” —— 计划直接废掉。
            #    画面才是真相：REPLAN 时让 VLM 从它**看到的场景**重新规划全部剩余动作。
            self.done = []
            self.subtasks = self._plan(frame)
            self.idx = 0
            self.used = 0
            self.retry = 0
            self.code_epoch += 1
        elif d == "EXTEND":
            # 「前面的都做完了，但任务还没完」—— 保留已完成的段，在末尾追加。
            # 与 REPLAN 的区别：REPLAN 丢弃全部进度、idx 归零，会让已经抓住
            # 扫帚、搬到灰尘旁的手臂从「approach the broom handle」重头再来。
            if (not ALLOW_EXTEND or self.n_extend >= MAX_EXTEND
                    or len(self.subtasks) >= MAX_SEGMENTS):
                self.stats["extend_blocked"] = self.stats.get("extend_blocked", 0) + 1
                # 已确认无路可走：本局停止再问，省下剩余的空转调用。
                self._plan_exhausted = True
                return
            self.n_extend += 1
            # 走到末段说明前面的段都被推进过了，据此告诉 VLM「已完成什么」；
            # 它再看当前画面给出**剩余**动作（build_plan_user_content 的 done 参数）。
            self.done.extend(self.subtasks[self.idx:])
            # 续写失败（VLM 答「没有剩余动作」/ 解析失败 / 断线）不是致命错误：
            # 标记为已用尽、保留现有计划继续跑完本局即可。**不能**在这里回落到
            # 模板计划 —— 那会把整份模板计划追加到末尾，与「续写」语义不符。
            try:
                more = self._plan(frame, allow_fallback=False)
            except PlannerError:
                more = []
            if not more:
                self.stats["extend_empty"] = self.stats.get("extend_empty", 0) + 1
                self._plan_exhausted = True
                return
            n0 = len(self.subtasks)
            self.subtasks = self.subtasks + more
            self.idx = n0
            self.used = 0
            self.retry = 0
            self.code_epoch += 1
            self.extends.append({"t": self.t, "n": self.n_extend,
                                 "added": [f"{s['action']}: {s['instruction']}"
                                           for s in more]})
        # CONTINUE：什么都不做

    # ---------------------------------------------------------- VLM 调用
    def _images(self, frame: dict) -> list[tuple[str, str]]:
        from planner.client import image_data_url_from_array
        out = []
        for v in self._views:
            a = frame.get(v)
            if a is not None:
                out.append((v, image_data_url_from_array(a, self._img_size)))
        if not out:
            raise PlannerError("observation 里没有可用的相机图像")
        return out

    def _plan(self, frame: dict, allow_fallback: bool = True) -> list[dict]:
        from planner.prompts import (build_plan_system_prompt,
                                     build_plan_user_content)
        from planner.contract import (use_codebook, normalize_instruction,
                                      NON_CODEBOOK_ACTIONS)
        msgs = [{"role": "system", "content": build_plan_system_prompt()},
                {"role": "user", "content": build_plan_user_content(
                    self.task, self.task_instruction, self._images(frame),
                    done=self.done or None, prior=self.prior,
                    n_repeat=self._n_repeat, phrasings=self._phrasings,
                    history=self.history if USE_HISTORY else None,
                    decisions=self.decisions if USE_HISTORY else None,
                    budget=EPISODE_BUDGET if USE_HISTORY else None)}]
        self.stats["plan_call"] += 1
        try:
            raw = _parse_plan(self._client.chat(msgs)["content"])
        except Exception as e:
            self.stats["fail"] += 1
            if self._fallback and allow_fallback:
                # 断线不该让整个分片陪葬：用模板计划把这一局跑完，并在轨迹里
                # 标记来源，分析时可以单独剔除或对照。
                self.stats["plan_fallback"] = self.stats.get("plan_fallback", 0) + 1
                self.plan_source = "template_fallback"
                self._api_down = True
                print(f"[planner] ⚠ PLAN 调用失败（{type(e).__name__}），"
                      f"回落到模板计划：{self.task}", flush=True)
                self.decisions.append({"t": self.t, "decision": "PLAN",
                                       "source": "template_fallback",
                                       "plan": [f"{s['action']}: {s['instruction']}"
                                                for s in self._fallback]})
                return [dict(s) for s in self._fallback]
            raise PlannerError(f"vlm-plan 规划失败（{type(e).__name__}: {e}）。"
                               f"本局评测中止。") from e
        plan = []
        for st in raw:
            ins, layer = _resolve_phrasing(st["raw"], self._phrasings)
            ins = normalize_instruction(ins)
            # 三层各用了多少次 —— 直接对应「指令是否落在训练分布内」，
            # 事后不必再去逐条比对字符串就能看出协议有没有被遵守。
            self.stats[f"phrasing_{layer}"] = self.stats.get(f"phrasing_{layer}", 0) + 1
            uc = bool(use_codebook(st["action"]))
            if not uc and st["action"] not in NON_CODEBOOK_ACTIONS:
                # 词表外动作：安全降级为不用码本（code_mask=0，注入层恒等），
                # 语言通路照常。这里只记账，不拦截 —— 实测当前 0/755。
                self.stats["off_vocab_action"] = self.stats.get("off_vocab_action", 0) + 1
            plan.append({"action": st["action"], "instruction": ins,
                         "use_codebook": uc,
                         "k_global": -1, "k_detail": [-1] * 9,  # 必须由 Adapter 出
                         "n_keyframes": 1})
        if not plan:
            # 🔴 空计划与「调用失败」同等处理。EXTEND 会在局中调 _plan 问
            #    「还剩什么要做」，而 VLM 完全可能答「没有了」—— 这是合法回答，
            #    不该让整个分片死掉。2026-09-09 实测：v3.5 首跑 9 个分片
            #    被这一行打死，300 局只跑出 119 局。
            self.stats["empty_plan"] = self.stats.get("empty_plan", 0) + 1
            if self._fallback and allow_fallback:
                self.stats["plan_fallback"] = self.stats.get("plan_fallback", 0) + 1
                self.plan_source = "template_fallback"
                return [dict(x) for x in self._fallback]
            raise PlannerError("vlm-plan 生成了空计划")
        self.decisions.append({"t": self.t, "decision": "PLAN",
                               "plan": [f"{s['action']}: {s['instruction']}"
                                        for s in plan]})
        return plan

    def _monitor(self, gripper_open, frame: dict, stalled: bool = False,
                 quiet: str = "", gripper_class: str = "",
                 at_last: bool = False) -> tuple[str, int | None, str]:
        from planner.prompts import (build_monitor_system_prompt,
                                     build_monitor_user_content)
        msgs = [{"role": "system", "content": build_monitor_system_prompt()},
                {"role": "user", "content": build_monitor_user_content(
                    self.task, self.task_instruction, self.subtasks, self.idx,
                    self.used, gripper_open, self._images(frame),
                    stalled=stalled, quiet=quiet, gripper_class=gripper_class,
                    at_last=at_last and ALLOW_EXTEND,
                    history=self.history if USE_HISTORY else None,
                    decisions=self.decisions if USE_HISTORY else None,
                    budget=EPISODE_BUDGET if USE_HISTORY else None)}]
        self.stats["monitor_call"] += 1
        try:
            out = self._client.chat(msgs)
            self._consec_fail = 0
            return _parse_decision(out["content"])
        except Exception as e:
            self.stats["fail"] += 1
            # 进度判定失败不再中止整局：退化成 CONTINUE（停在当前段），
            # 局照常跑完。连续失败两次就判定 API 已断，本局不再发起调用 ——
            # 否则每次都要走满重试+退避，一局能拖几十分钟。
            self._consec_fail += 1
            if self._consec_fail >= 2:
                self._api_down = True
                self.stats["api_down"] = self.stats.get("api_down", 0) + 1
                print(f"[planner] ⚠ MONITOR 连续失败，本局停止调用 VLM："
                      f"{self.task}", flush=True)
            self.stats["monitor_fallback"] = self.stats.get("monitor_fallback", 0) + 1
            return "CONTINUE", None, f"api_error:{type(e).__name__}"


def _resolve_phrasing(item: dict, phrasings: list[str]) -> tuple[str, str]:
    """把一条计划项解析成 (instruction, 使用了哪一层)。

    三层协议（见 planner/prompts.py 的 "ON THE PHRASINGS LIST"）：
      A  {"phrasing_id": 3}                        直接复用训练原文
      B  {"phrasing_id": 3, "substitute": {...}}   复用句式，替换 variation 词
      C  {"instruction": "..."}                    模仿句式自由生成

    候选条目的格式是 "action: instruction"（`_load_phrasings` 如此产出），
    解析时要把 action 前缀切掉。
    """
    pid = item.get("phrasing_id")
    if pid is not None and phrasings:
        try:
            raw = phrasings[int(pid)]
        except (ValueError, TypeError, IndexError):
            raw = None
        if raw is not None:
            ins = raw.split(":", 1)[1].strip() if ":" in raw else raw.strip()
            sub = item.get("substitute") or {}
            if isinstance(sub, dict) and sub:
                for a, b in sub.items():
                    # 整词替换，避免 "red" 命中 "predict" 这类子串
                    ins = re.sub(rf"\b{re.escape(str(a))}\b", str(b), ins,
                                 flags=re.IGNORECASE)
                return ins, "B"
            return ins, "A"
    ins = item.get("instruction")
    if not ins:
        raise PlannerError(
            f"计划项既没有可解析的 phrasing_id 也没有 instruction: {item!r}")
    return str(ins).strip(), "C"


def _parse_plan(text: str) -> list[dict]:
    """与 tools/plan_audit.py 同款容错解析（那边已实测 120/120 成功）。"""
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        d = json.loads(s)
    except Exception:
        d = None
        for m in reversed(list(re.finditer(r"\{", s))):
            try:
                d = json.loads(s[m.start():s.rfind("}") + 1])
                break
            except Exception:
                continue
        if d is None:
            raise PlannerError(f"计划输出里找不到 JSON: {s[:120]!r}")
    plan = d["plan"] if isinstance(d, dict) else d
    # instruction 的解析推迟到 _plan()（那里才拿得到候选清单）
    return [{"action": str(x["action"]).strip(), "raw": x} for x in plan]


_DECISIONS = ("CONTINUE", "NEXT", "RETRY", "REPLAN", "EXTEND")


def _parse_decision(text: str) -> tuple[str, int | None, str]:
    """→ (decision, target_index or None, reason)。

    `index` 让 VLM 判 NEXT 时直接说「现在应该在第几步」，可以一次跨多段 ——
    v1 每次只能进 1 段，长计划在 episode 结束前走不完（见 `_apply`）。
    """
    s = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        d = json.loads(s)
        v = str(d["decision"]).strip().upper()
        if v in _DECISIONS:
            tgt = d.get("index", None)
            return v, (int(tgt) if tgt is not None else None), str(d.get("reason", ""))[:60]
    except Exception:
        pass
    up = s.upper()
    # 顺序有意：REPLAN/RETRY 比 CONTINUE 罕见，先匹配它们避免被子串吞掉
    for v in ("REPLAN", "EXTEND", "RETRY", "NEXT", "CONTINUE"):
        if v in up:
            return v, None, "(parsed from raw text)"
    raise PlannerError(f"无法从输出解析 decision: {s[:120]!r}")


class OnlinePlanFactory:
    """`vlm-plan` 的工厂。**不读模板库** —— 只需任务名、整任务指令与 CSV 先验。

    模块级类 + 延迟加载，理由同 `VLMPlanFactory`（spawn 子进程要 pickle）。
    """

    kind = "vlm-plan"

    def __init__(self, split: str, model: str | None = None,
                 views=DEFAULT_VIEWS, verbose: bool = True) -> None:
        self.split = split
        self.model = model or os.environ.get("AAVLA_ONLINE_PLANNER_MODEL",
                                             "qwen3.8-max")
        self.views = tuple(views)
        self._verbose = verbose
        self._client = None
        self._priors = None
        self._phr = None
        self._bank = None                            # 断线兜底用的模板计划
        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise PlannerError(
                "vlm-plan 需要 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL。"
                "先 source run/env.sh")
        self.priors                                  # 父进程尽早失败

    @property
    def priors(self):
        if self._priors is None:
            from planner.offline import load_prior_variants
            self._priors = load_prior_variants()
            if self._verbose:
                print(f"[planner] kind=vlm-plan model={self.model} "
                      f"views={','.join(self.views)}；"
                      f"CSV 先验 {len(self._priors)} 个任务（仅作参考）",
                      flush=True)
        return self._priors

    @property
    def client(self):
        if self._client is None:
            from planner.client import PlannerClient
            # response_format=json_object：PLAN 是开放式生成，不强制的话模型会
            # 先写几百 token 散文再给答案（实测 3/4 次因此解析失败）。
            self._client = PlannerClient(
                model=self.model, max_tokens=2048,
                extra_body={"enable_thinking": False,
                            "response_format": {"type": "json_object"}})
        return self._client

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_client"] = None                          # 持有 socket
        d["_priors"] = None
        d["_phr"] = None                             # 子进程各自重建
        d["_bank"] = None                            # 550 KB，不必序列化 12 份
        return d

    def __call__(self, task: str, episode: int):
        from stage3.online_planner import (episode_descriptions,
                                           episode_variation)
        from planner.contract import expand_prior, n_repeat_prior
        d = episode_descriptions(task, self.split, episode)
        var = episode_variation(task, self.split, episode)
        # priors[task] 是变体列表；逐个展开（重复次数按 variation 推）后整体传下去
        variants = self.priors.get(task, [[]])
        prior = [(expand_prior(task, var, v) if var is not None else list(v))
                 for v in variants]
        nrep = n_repeat_prior(task, var) if var is not None else None
        return OnlineVLMPlanner(
            task, d[0] if d else task.replace("_", " "), self.client,
            prior=prior, n_repeat=nrep,
            phrasings=self.phrasings(task, var), views=self.views,
            fallback=self.fallback(task, episode),
            dwell=load_dwell_prior().get(task))

    def fallback(self, task: str, episode: int) -> list[dict] | None:
        """API 断线时的兜底：该 (task, episode) 的模板计划。

        取不到就返回 None（回到旧行为：抛 PlannerError）——兜底本身不该
        成为新的失败源。
        """
        try:
            return self.bank.get(task, episode)
        except Exception:
            return None

    @property
    def bank(self):
        if self._bank is None:
            from stage3.planners import plan_file
            from stage3.online_planner import PlanBank
            self._bank = PlanBank(plan_file(self.split, "template"))
        return self._bank

    def phrasings(self, task: str, variation: int | None) -> list[str]:
        """该 (task, variation) 在 train cache 里出现过的全部子任务指令。

        实测每个 (task, variation) 中位只有 15 条、p90 29 条，塞进 prompt 毫无压力。
        找不到该 variation 时退回该任务的全部指令（更宽但仍在分布内）。
        """
        if self._phr is None:
            self._phr = _load_phrasings(self.split)
        d = self._phr.get(task, {})
        got = d.get(str(variation)) if variation is not None else None
        if got:
            return got
        allp = sorted({x for v in d.values() for x in v})
        return allp[:40]                     # 兜底时截断，避免 prompt 过长

    def summary(self) -> str:
        return f"OnlineVLM({self.model})，无模板库"


#: `(task, variation) -> 训练指令清单` 的进程内缓存。
#: 每个评测分片是独立进程，各自建一次；扫 12 个任务 × 100 局约 1 秒。
_PHRASINGS_CACHE: dict[str, dict] = {}


def _load_phrasings(split: str) -> dict:
    """从 train cache 汇总每个 `(task, variation)` 见过的子任务指令。

    🔴 为什么必须用 **train** 而不是当前 split：这些指令是**控制器训练时**
    见过的原文，目的就是让在线生成的指令落回训练分布。实测在线规划的指令
    只有 56.6% 命中训练分布（模板法 100%），是 vlm-plan 落后的主因之一。

    格式：`{task: {variation_str: ["action: instruction", ...]}}`
    """
    key = "train"                         # 永远取 train，与 split 无关
    hit = _PHRASINGS_CACHE.get(key)
    if hit is not None:
        return hit
    import collections
    from stage3.cache_join import PlannerCache
    root = REPO_ROOT / "aavla_data" / "planner_cache" / "train"
    out: dict = collections.defaultdict(lambda: collections.defaultdict(set))
    try:
        cache = PlannerCache(root)
        for task in cache.tasks():
            for e in range(100):
                ep = cache.get(task, "train", e)
                if not ep:
                    continue
                v = str(int(ep["variation"]))
                for sg in ep["segments"]:
                    out[task][v].add(f"{sg['action']}: {sg['instruction']}")
    except Exception as exc:              # cache 缺失不该让评测崩掉
        print(f"[planner] 训练指令清单加载失败（{type(exc).__name__}: {exc}），"
              f"本轮不给候选清单", flush=True)
    doc = {t: {v: sorted(xs) for v, xs in d.items()} for t, d in out.items()}
    _PHRASINGS_CACHE[key] = doc
    return doc
