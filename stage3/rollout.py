"""把在线 Planner 接进 YARR 的 rollout 循环 —— B2/B3 评测的最后一环。

# 接入点的选择

YARR 的 `RolloutGenerator.generator()` 每步从 `obs_history` 造 `prepped_data`
再喂 `agent.act(step, prepped_data, ...)`。要让 B2/B3 拿到子任务指令与码，
只需在 `act` 之前往这个字典里塞四个键。

有两条路：改写 `generator()` 的循环体，或**包装 agent**。这里选后者 ——
`generator()` 的循环里还有 transition 组装、terminal 处理、录像等逻辑，
复制一遍就多一份要同步维护的代码。包装 agent 则只碰 `act()` 这一个方法，
`Stage3RolloutGenerator` 本身只负责「这一局用哪个 planner」，其余全部委托给父类。

# 用哪种 planner 不在这里决定

本模块只认一个 `factory(task, episode) -> planner`，具体是模板法还是
在线 VLM 由 `stage3/planners.py` 决定。这样换 planner 时本文件一行都不用改
（P6 的可回撤性要求）。

# 推进时机

`act()` 拿到的 observation 反映的是**上一个动作执行完之后**的状态。
所以每次 `act()` 里先用当前观测推进状态机（判断上一段是否完成），
再把推进后的子任务注入。夹爪状态取自 `low_dim_state[0]`
（`helpers/utils.py:343` 把 `obs.gripper_open` 放在 robot_state 首位）。

# 轨迹落盘

每局结束把状态机走过的路径 dump 成 json。这不是可选的日志 ——
它是「B3 成绩不好时，到底是 planner 的锅还是模型的锅」的**唯一证据**。
最危险的失败形态是状态机从不推进：B3 会静静退化成「整局只用第一段指令」，
而所有断言都不会报错。只有轨迹能看出来。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from yarr.utils.rollout_generator import RolloutGenerator

from stage3.online_planner import PlannerError


def task_name_of(env) -> str:
    """从 env 反推当前任务名 —— 与 `_independent_env_runner._get_task_name` 同源。"""
    from yarr.utils.process_str import change_case
    if hasattr(env, "_task_class"):
        return change_case(env._task_class.__name__)
    if hasattr(env, "_task_classes"):
        tid = env.active_task_id % len(env._task_classes)
        return change_case(env._task_classes[tid].__name__)
    raise PlannerError("无法从 env 推断任务名")


class TokenCache:
    """子任务指令 → CLIP token 的记忆化。

    `act()` 自己会用 CLIP 编码 token，所以这里只需产出 token，不需要模型。
    唯一指令数很少（train cache 实测 2038 条），记忆化后开销可忽略。
    """

    def __init__(self) -> None:
        self._c: dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        hit = self._c.get(text)
        if hit is None:
            from helpers.clip.core.clip import tokenize
            hit = tokenize([text])[0].numpy()
            self._c[text] = hit
        return hit


class _SubtaskAgent:
    """薄包装：在 `act()` 前推进状态机并注入子任务字段，其余全部透传。"""

    def __init__(self, inner, planner, tokens: TokenCache,
                 with_codes: bool, code_source=None, robot_state=None) -> None:
        self._inner = inner
        self._planner = planner
        self._tokens = tokens
        self._with_codes = with_codes
        self._code_source = code_source        # None=用计划里的码；LiveAdapter=实时预测
        # 取 env 旁路里的机器人低维状态（关节速度 / 末端位姿）。
        # 它们**不在** observation 里 —— custom_rlbench_env.extract_obs 为了让
        # low_dim_state 与训练一致（4 维），把它们置空了；旁路存在 env 对象上。
        self._robot_state = robot_state
        self._first = True
        self._cached_idx = None                # 上次算码时的子任务下标
        self._cached_codes = None              # (k_global, k_detail)
        # 谁需要看画面就把它的视角并进来：
        #   planner._views  —— 在线 VLM planner 要看画面判断进度
        #   code_source.views —— 实时 Adapter 要看画面出码（front+wrist）
        views = set(getattr(planner, "_views", ()) or ())
        views |= set(getattr(code_source, "views", ()) or ())
        self._views = tuple(sorted(views))

    def __getattr__(self, name):          # reset / update_summaries / act_summaries …
        return getattr(self._inner, name)

    def act(self, step, observation, deterministic=False):
        # 先推进：observation 反映的是上一个动作执行后的状态。
        # 第一次调用面对的是初始状态，没有「上一个动作」可评判，
        # 所以只建立夹爪基线（prime）而不推进。
        g = _gripper_open(observation)
        if self._first:
            self._first = False
            # vlm-plan 要在第一帧就现场规划，所以 prime 也得拿到画面。
            # 模板法的 prime 忽略这个参数。
            self._planner.prime(g, _frames(observation, self._views)
                                if self._views else None)
        else:
            # 只有需要看画面的 planner（在线 VLM）才付图像抽取的开销；
            # 模板法是开环的，用不到。
            self._planner.observe(
                g,
                _frames(observation, self._views) if self._views else None,
                robot=self._robot_state() if self._robot_state else None)
        self._planner.note(step)

        st = self._planner.current
        dev = _device_of(observation)
        obs = dict(observation)
        tok = torch.as_tensor(self._tokens(st["instruction"]), device=dev)
        # 与 lang_goal_tokens 同形：(1, timesteps, 77)
        ref = observation.get("lang_goal_tokens")
        obs["subtask_lang_goal_tokens"] = (
            tok.reshape(1, 1, -1).expand_as(ref).contiguous()
            if ref is not None else tok.reshape(1, 1, -1))
        if self._with_codes:
            kg, kd = self._codes_for(st, observation)
            obs["subtask_k_global"] = torch.as_tensor(
                [[kg]], device=dev, dtype=torch.long)
            obs["subtask_k_detail"] = torch.as_tensor(
                [[kd]], device=dev, dtype=torch.long)
            obs["subtask_code_mask"] = torch.as_tensor(
                [[1.0 if st["use_codebook"] else 0.0]], device=dev,
                dtype=torch.float32)
        return self._inner.act(step, obs, deterministic)

    def _codes_for(self, st: dict, observation) -> tuple[int, list[int]]:
        """取当前子任务的码。

        `code_source=None` → 用计划里预存的码（模板库查表，旧行为）。
        `code_source=LiveAdapter` → 子任务**切换时调一次** Adapter，段内复用
        （`VLA_Design §3` 码的生命周期：一个子任务内保持不变；这与训练时
        「段内所有样本共享该段起始帧算出的码」严格对应）。
        """
        if self._code_source is None:
            return int(st["k_global"]), list(st["k_detail"])
        # 🔴 缓存键要带 code_epoch：RETRY 时 idx 不变，但环境已变、
        #    沿用旧码不合理（VLA_Design §3「NEXT/RETRY/REPLAN 均重新调用
        #    Adapter」）。只按 idx 缓存会让 RETRY 拿到过期的码。
        idx = (self._planner.idx, getattr(self._planner, "code_epoch", 0))
        if self._cached_idx != idx or self._cached_codes is None:
            frame = _frames(observation, self._code_source.views)
            self._cached_codes = self._code_source(frame, st["instruction"])
            self._cached_idx = idx
            # 记进轨迹：事后要能分辨「码是查表还是实时算的、算出了什么」
            if self._planner.history:
                self._planner.history[-1]["adapter_codes"] = {
                    "k_global": self._cached_codes[0],
                    "k_detail": self._cached_codes[1],
                    "plan_k_global": int(st["k_global"]),
                }
        return self._cached_codes


def _frames(observation, views) -> dict:
    """从 observation 里取出各视角的当前帧 [C,H,W] uint8。

    `prepped_data` 的每一项形如 (1, timesteps, C, H, W)，最后一个 timestep
    才是**当前**观测（前面的是历史）。取错会让 VLM 看着几步之前的画面做判断。
    """
    out = {}
    for v in views:
        t = observation.get(f"{v}_rgb")
        if t is None:
            continue
        try:
            a = torch.as_tensor(t)
            while a.dim() > 3:
                a = a[-1] if a.shape[0] == 1 and a.dim() == 4 else a[0]
            out[v] = a.detach().cpu().numpy()
        except Exception:
            pass
    return out


def _device_of(observation) -> torch.device:
    for v in observation.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return torch.device("cpu")


def _gripper_open(observation) -> float | None:
    """`low_dim_state` 首位就是 gripper_open（helpers/utils.py:343）。"""
    v = observation.get("low_dim_state")
    if v is None:
        return None
    try:
        return float(torch.as_tensor(v).reshape(-1)[0])
    except Exception:
        return None


class Stage3RolloutGenerator(RolloutGenerator):
    """按 (task, episode) 造 planner，包装 agent 后委托给父类。

    计划由工厂给出而非在线现算，是为了让 B2 与 B3 读到逐字相同的子任务序列 ——
    两臂的差值里只剩「有没有注入码」这一件事。
    """

    def __init__(self, factory, with_codes: bool,
                 trace_dir: str | Path | None = None,
                 code_source=None, verbose: bool = True) -> None:
        self._factory = factory
        self._with_codes = with_codes
        self._code_source = code_source
        self._tokens = TokenCache()
        self._trace_dir = Path(trace_dir) if trace_dir else None
        if self._trace_dir:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            kind = getattr(factory, "kind", "?")
            src = "实时 Adapter" if code_source is not None else "计划查表"
            print(f"[planner] kind={kind}；注入码={with_codes}（码来源：{src}）；"
                  f"轨迹={'→ ' + str(self._trace_dir) if self._trace_dir else '不记录'}",
                  flush=True)

    def generator(self, step_signal, env, agent, episode_length, timesteps,
                  eval, eval_demo_seed: int = 0, record_enabled: bool = False):
        task = task_name_of(env)
        # 缺计划必须硬失败：拿不到子任务指令就等于退化成 B1 的行为，
        # 而 act() 的语言防线只在字段缺失时报错，这里提前给出更清楚的信息。
        planner = self._factory(task, eval_demo_seed)
        wrapped = _SubtaskAgent(agent, planner, self._tokens, self._with_codes,
                                self._code_source,
                                robot_state=lambda: getattr(env, "_planner_state", None))
        self.last_planner = planner          # 供事后分析取 history
        try:
            yield from super().generator(
                step_signal, env, wrapped, episode_length, timesteps,
                eval, eval_demo_seed, record_enabled)
        finally:
            # finally 而不是正常返回后：episode 提前终止（成功/超时/异常）
            # 同样要留下轨迹，否则最该被诊断的那些局恰好没有记录。
            self._dump(task, eval_demo_seed, planner)

    def _dump(self, task: str, episode: int, planner) -> None:
        if not self._trace_dir:
            return
        try:
            n = len(planner.subtasks)
            rec = {
                "task": task, "episode": episode,
                "n_subtasks": n,
                "n_steps": len(planner.history),
                "reached_index": planner.idx,
                "reached_frac": round((planner.idx + 1) / n, 4),
                "budgets": planner.budgets,
                "trace": planner.history,
            }
            # 在线 VLM planner 额外记：调了几次、失败几次、退回几次、
            # 每次决策的理由。没有这些就无法判断「VLM 方案效果不好」
            # 到底是判断错了还是根本没调起来。
            rec["code_source"] = ("adapter" if self._code_source is not None
                                  else "plan")
            if self._code_source is not None:
                rec["adapter_calls"] = getattr(self._code_source, "n_calls", None)
            if hasattr(planner, "stats"):
                rec["vlm_stats"] = planner.stats
                rec["vlm_decisions"] = getattr(planner, "decisions", [])
            (self._trace_dir / f"{task}_ep{episode}.json").write_text(
                json.dumps(rec, ensure_ascii=False))
        except Exception as e:                       # 诊断数据不该拖垮评测
            print(f"[planner] 轨迹落盘失败 {task}/{episode}: {e}", flush=True)
