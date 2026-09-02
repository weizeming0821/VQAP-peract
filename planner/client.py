"""DashScope（OpenAI 兼容）客户端：重试、限流、磁盘记忆化。

记忆化按 sha256(model + 完整 messages) 落盘，作用有三：
  1. 中断重跑不重复付费；
  2. 离线建 cache 与在线评测共用同一份缓存；
  3. **同一输入永远得到同一输出**——在线 Planner 的决策因此可复现，
     B2/B3 在相同状态下拿到逐字相同的子任务，公平性损失降到最小。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / "planner_cache" / "llm_responses"


class PlannerClient:
    def __init__(self, model: str = "qwen3.8-max",
                 cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 temperature: float = 0.0, max_tokens: int = 8000,
                 max_retries: int = 4, min_interval: float = 0.0,
                 timeout: float = 300.0) -> None:
        from openai import OpenAI
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        base_url = os.environ.get("DASHSCOPE_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL 未设置，"
                               "请先 `source run/env.sh`")
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.stats = {"hit": 0, "miss": 0, "retry": 0, "fail": 0,
                      "prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------ 记忆化
    @staticmethod
    def _key(model: str, messages: list[dict]) -> str:
        blob = json.dumps({"model": model, "messages": messages},
                          sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

    def _cache_path(self, key: str) -> Path:
        sub = self.cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    # ------------------------------------------------------------ 调用
    def chat(self, messages: list[dict], model: str | None = None) -> dict[str, Any]:
        model = model or self.model
        key = self._key(model, messages)
        path = self._cache_path(key)
        if path.is_file():
            with self._lock:
                self.stats["hit"] += 1
            return json.loads(path.read_text())

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if self.min_interval > 0:
                with self._lock:
                    wait = self.min_interval - (time.time() - self._last_call)
                    if wait > 0:
                        time.sleep(wait)
                    self._last_call = time.time()
            try:
                t0 = time.time()
                r = self._client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=self.temperature, max_tokens=self.max_tokens)
                out = {"model": model, "content": r.choices[0].message.content,
                       "prompt_tokens": r.usage.prompt_tokens,
                       "completion_tokens": r.usage.completion_tokens,
                       "seconds": round(time.time() - t0, 2),
                       "finish_reason": r.choices[0].finish_reason}
                path.write_text(json.dumps(out, ensure_ascii=False))
                with self._lock:
                    self.stats["miss"] += 1
                    self.stats["prompt_tokens"] += out["prompt_tokens"]
                    self.stats["completion_tokens"] += out["completion_tokens"]
                return out
            except Exception as exc:            # 网络/限流/服务端错误一律退避重试
                last_exc = exc
                with self._lock:
                    self.stats["retry"] += 1
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt * 2, 30))
        with self._lock:
            self.stats["fail"] += 1
        raise RuntimeError(f"调用失败（重试 {self.max_retries} 次）: "
                           f"{type(last_exc).__name__}: {last_exc}")


# ---------------------------------------------------------------- 图像
_IMG_CACHE: dict[tuple[str, int], str] = {}
_IMG_LOCK = threading.Lock()


def image_data_url(path: str | Path, size: int = 224, quality: int = 88) -> str:
    """本地图片 → data URL。按 (路径, 尺寸) 记忆化，同一帧多次引用只编码一次。"""
    ck = (str(path), size)
    with _IMG_LOCK:
        hit = _IMG_CACHE.get(ck)
    if hit is not None:
        return hit
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    with _IMG_LOCK:
        if len(_IMG_CACHE) < 20000:
            _IMG_CACHE[ck] = url
    return url
