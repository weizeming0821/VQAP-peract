"""Planner cache 与 PerAct fill_replay 的连接层。

职责：给定 (task, split, episode)，返回该 episode 的分段信息；给定一个绝对帧号的
关键帧，O(1) 查出它属于哪个 segment。

归属规则（VLA_Design §4.3）：
    一个训练样本的子任务 = **其 target 关键帧所属的 segment**。
样本是 `(某帧观测 -> 紧邻的下一个关键帧)` 的链式转移，obs 最多落后一个 segment；
跨边界样本（obs 在 seg_k 末尾、target 是 seg_{k+1} 首帧、指令用 seg_{k+1} 的）
恰好对应部署时 Planner 刚说完 NEXT 的那一刻，语义正确。
样本总数与 baseline 完全相同，三臂对比不因样本量差异被污染。
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "planner_cache" / "train"


class CacheError(RuntimeError):
    pass


class PlannerCache:
    """只读的 cache 视图，供 fill_replay 使用。"""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                 skip_gripper_anomaly: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.meta: dict[str, Any] = {}
        self._eps: dict[tuple[str, str, int], dict] = {}
        self.skipped: dict[str, list[str]] = {"errors": [], "gripper_anomaly": []}

        shards = sorted(glob.glob(str(self.cache_dir / "*.json")))
        if not shards:
            raise CacheError(f"cache 目录为空: {self.cache_dir}")
        metas = []
        for sp in shards:
            d = json.loads(Path(sp).read_text())
            metas.append(json.dumps(d.get("meta", {}), sort_keys=True))
            for ep in d.get("episodes", []):
                key = (ep["task"], ep["split"], int(ep["episode"]))
                if ep.get("errors"):
                    self.skipped["errors"].append(f"{key[0]}/{key[2]}")
                    continue
                # 关键帧上夹爪全程 OPEN 却含 grasp/lift/place -> 该 episode 的
                # 夹爪信号本身异常（实测 10 条，占 0.56%），跳过而非硬报错。
                if skip_gripper_anomaly and self._gripper_anomalous(ep):
                    self.skipped["gripper_anomaly"].append(f"{key[0]}/{key[2]}")
                    continue
                self._eps[key] = ep
        if len(set(metas)) > 1:
            raise CacheError(f"分片间 meta 不一致（{len(set(metas))} 种），"
                             f"说明它们不是同一次配置下生成的，不能混用")
        self.meta = json.loads(metas[0])

    @staticmethod
    def _gripper_anomalous(ep: dict) -> bool:
        grip = ep.get("gripper_at_keypoints") or []
        if not grip or not all(g == 1 for g in grip):
            return False
        holding = {"grasp", "lift", "place", "transfer"}
        return any(s.get("action") in holding for s in ep.get("segments", []))

    # ------------------------------------------------------------------
    def get(self, task: str, split: str, episode: int) -> dict | None:
        return self._eps.get((task, split, int(episode)))

    def require(self, task: str, split: str, episode: int) -> dict:
        ep = self.get(task, split, episode)
        if ep is None:
            raise CacheError(f"cache 中没有 {task}/{split}/{episode}（可能已被跳过）")
        return ep

    def n_episodes(self, task: str | None = None, split: str | None = None) -> int:
        return sum(1 for (t, s, _) in self._eps
                   if (task is None or t == task) and (split is None or s == split))

    def tasks(self) -> list[str]:
        return sorted({t for (t, _, _) in self._eps})

    # ------------------------------------------------------------------
    @staticmethod
    def assert_keypoints(ep: dict, episode_keypoints: list[int],
                         task: str, d_idx: int) -> None:
        """A1（VLA_Design §5.1 C5）：cache 的关键帧必须与 keypoint_discovery 逐位相同。

        不等即抛异常而非记日志 —— 这是本项目「静默数据错误」教训的直接产物。
        """
        if list(episode_keypoints) != list(ep["keypoints"]):
            raise CacheError(
                f"关键帧失配 {task} ep{d_idx}：\n"
                f"  keypoint_discovery -> {list(episode_keypoints)}\n"
                f"  cache              -> {list(ep['keypoints'])}\n"
                f"说明 cache 与当前代码/数据不同源，绝不能继续训练。")

    @staticmethod
    def segment_of_keypoint(ep: dict, keypoint: int) -> dict:
        """绝对帧号的关键帧 -> 它所属的 segment（A2）。"""
        try:
            pos = ep["keypoints"].index(int(keypoint))
        except ValueError:
            raise CacheError(
                f"关键帧 {keypoint} 不在 {ep['task']} ep{ep['episode']} 的 "
                f"keypoints {ep['keypoints']} 中") from None
        seg_idx = ep["keypoint_to_segment"][pos]
        if seg_idx < 0 or seg_idx >= len(ep["segments"]):
            raise CacheError(
                f"{ep['task']} ep{ep['episode']} 的关键帧位置 {pos} 未归属任何 segment")
        return ep["segments"][seg_idx]

    # ------------------------------------------------------------------
    def summary(self) -> str:
        return (f"PlannerCache({self.cache_dir.name}): "
                f"{len(self._eps)} episode / {len(self.tasks())} 任务；"
                f"跳过 error {len(self.skipped['errors'])} 条、"
                f"夹爪异常 {len(self.skipped['gripper_anomaly'])} 条；"
                f"model={self.meta.get('planner_model')} "
                f"views={self.meta.get('views')}")


class TextEmbedCache:
    """CLIP 文本编码的字符串级记忆化。

    PerAct 原实现是**逐样本**调 tokenize + encode_text_with_embeddings
    （launch_utils.py:177）。20 万个样本这么调很贵，而子任务指令的唯一值只有
    约 1762 个语义组，记忆化后额外开销接近 0 —— 设计文档估的「+10% 开销」不成立。
    """

    def __init__(self, clip_model, device) -> None:
        self._clip = clip_model
        self._device = device
        self._cache: dict[str, tuple] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, text: str):
        hit = self._cache.get(text)
        if hit is not None:
            self.hits += 1
            return hit
        import torch
        from helpers.clip.core.clip import tokenize
        tokens = torch.from_numpy(tokenize([text]).numpy()).to(self._device)
        sent, tok = self._clip.encode_text_with_embeddings(tokens)
        out = (sent[0].float().detach().cpu().numpy(),
               tok[0].float().detach().cpu().numpy())
        self._cache[text] = out
        self.misses += 1
        return out

    def stats(self) -> str:
        tot = self.hits + self.misses
        return (f"TextEmbedCache: {self.misses} 条唯一指令 / {tot} 次调用"
                f"（命中率 {self.hits / max(tot, 1):.1%}）")
