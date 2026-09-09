"""评测时实时调用 Stage 2 Adapter 出码 —— 把评测链路对齐到设计。

# 为什么需要

`VLA_Design §3` 的部署流程是：

    Planner → 子任务指令
        ├─► Adapter(当前观测 front+wrist, 子任务指令) → (k_global, k_detail[9])
        ├─► 码本查表 → (z_g, Z_d)
        └─► CodeInjector → PerAct latents

而此前的评测实现是**查模板库**：码取自同一个 `(task, variation)` 的
**另一条 train episode** 离线算好的值，与机器人当下真正看到的画面无关。
训练侧本来就是 Adapter 出的码（`scripts/planner_cache.py` 的 codes 步骤），
所以这条差异只存在于评测端，补上这一段接线即可对齐。

# 与离线路径的一致性

离线 `cmd_codes` 的输入是「segment 起始帧的 (front, wrist) PNG，uint8 CHW」
加该段指令。这里喂的是 rollout 当前帧的同名相机、同样 uint8 CHW ——
Adapter 的图像塔自己做 `/255 → 双线性缩放到 128 → 再到 224 → ImageNet 归一化`
（`model/module/encoder.py:772`），所以**不要在外面再做任何归一化**。

# 调用时机

每个子任务**起始时调一次，段内不变**（`Adapter_Design` 约束 10、
`VLA_Design §3` 码的生命周期）。段内多个 waypoint 共用同一组码 ——
这与训练时「段内所有样本共享该段起始帧算出的码」严格对应。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "vqap_adapter" / "best.pth"
#: Adapter 图像管线写死的两路相机（与 scripts/planner_cache.py 的 ADAPTER_CAMERAS 一致）
ADAPTER_CAMERAS = ("front", "wrist")


class AdapterError(RuntimeError):
    pass


class LiveAdapter:
    """`(当前帧, 子任务指令) → (k_global, k_detail[9])`。

    模块级类 + 延迟加载：`eval.py` 用 spawn 起子进程，本对象会被 pickle 过去，
    而 torch 模块带 CUDA 句柄，pickle 不了（`PlanFactory` 踩过同一个坑）。
    `__getstate__` 把模型丢掉，子进程首次调用时各自加载。
    """

    views = ADAPTER_CAMERAS

    def __init__(self, ckpt: str | Path = DEFAULT_CKPT,
                 device: str | None = None, verbose: bool = True) -> None:
        self.ckpt = str(ckpt)
        if not Path(self.ckpt).is_file():
            raise AdapterError(
                f"找不到 Adapter 权重 {self.ckpt}。Stage 2 的产物应在 "
                f"checkpoints/vqap_adapter/best.pth")
        self.device = device
        self._verbose = verbose
        self._m = None
        self.n_calls = 0
        self.last_conf = None          # 最近一次 k_global 的 softmax 最大值
        self.last_conf_detail = None   # 9 个 detail 头的平均最大概率

    # ------------------------------------------------------------------
    @property
    def model(self):
        if self._m is None:
            import torch
            from model.adapter import Adapter
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(self.ckpt, map_location="cpu", weights_only=False)
            m = Adapter(ck["model_args"])
            _, unexpected = m.load_state_dict(ck["model"], strict=False)
            if unexpected:
                raise AdapterError(f"Adapter 有多余的权重键: {unexpected[:5]}")
            self._m = m.eval().to(dev)
            self._dev = dev
            if self._verbose:
                print(f"[adapter] 已加载 {Path(self.ckpt).name} "
                      f"(epoch {ck.get('epoch')}, val_top1 "
                      f"{ck.get('best_val_top1', float('nan')):.4f}) -> {dev}",
                      flush=True)
        return self._m

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["_m"] = None                      # 带 CUDA 句柄，pickle 不过去
        return d

    # ------------------------------------------------------------------
    def __call__(self, frame: dict, instruction: str) -> tuple[int, list[int]]:
        import torch
        m = self.model
        imgs = {}
        for cam in ADAPTER_CAMERAS:
            a = frame.get(cam)
            if a is None:
                raise AdapterError(
                    f"实时 Adapter 需要 {cam}_rgb，但 observation 里没有。"
                    f"检查 eval.yaml 的 rlbench.cameras 是否含 "
                    f"{'/'.join(ADAPTER_CAMERAS)}。")
            imgs[cam] = _as_uint8_chw(a).unsqueeze(0).to(self._dev)
        with torch.no_grad():
            out = m(imgs, [instruction])
        gl = out["global_logits"].reshape(-1)
        g = int(gl.argmax(-1))
        d = [int(x) for x in out["detail_logits"].argmax(-1).reshape(-1).tolist()]
        # 🔴 置信度：原来只取 argmax、把分布丢了。码本没见过这个场景时
        #    Adapter 本该「没把握」—— 这是比「任务名白名单」更有原则的
        #    门控依据（不需要测试时知道任务身份）。先采集、暂不据此门控。
        self.last_conf = float(torch.softmax(gl.float(), -1).max())
        dl = out["detail_logits"].reshape(9, -1)
        self.last_conf_detail = float(torch.softmax(dl.float(), -1).max(-1).values.mean())
        # 与离线 cmd_codes 同款硬校验（A4）：越界宁可炸，也不要静默喂个坏索引，
        # 因为 CodebookLookup 的 IndexError 会在更深的地方冒出来，不好定位。
        if not 0 <= g < 36:
            raise AdapterError(f"k_global 越界: {g}")
        if len(d) != 9 or not all(0 <= x < 192 for x in d):
            raise AdapterError(f"k_detail 非法: {d}")
        self.n_calls += 1
        return g, d


def _as_uint8_chw(a):
    """把 observation 里的一帧规整成 Adapter 要的 uint8 [3,H,W]。

    🔴 不做归一化。Adapter 的图像塔自己 `/255 → resize → ImageNet 归一化`；
    在外面再归一化一次会让码彻底失真，而且不会报任何错。
    这里只处理 dtype 与通道顺序，并对「已经被归一化过」的输入显式报错。
    """
    import numpy as np
    import torch
    t = torch.as_tensor(np.asarray(a))
    while t.dim() > 3:                      # (1, timesteps, C, H, W) → 取当前帧
        t = t[-1] if (t.shape[0] == 1 and t.dim() == 4) else t[0]
    if t.dim() != 3:
        raise AdapterError(f"帧的维度不对: {tuple(t.shape)}")
    if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
        t = t.permute(2, 0, 1)              # HWC → CHW
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    t = t[:3]
    if t.dtype == torch.uint8:
        return t.contiguous()
    t = t.float()
    lo, hi = float(t.min()), float(t.max())
    if lo < -0.01:
        raise AdapterError(
            f"帧的数值范围是 [{lo:.2f}, {hi:.2f}]，像是已被归一化到 [-1,1]。"
            f"Adapter 需要原始 uint8/[0,1] 像素 —— 喂归一化过的图不会报错，"
            f"只会静默给出错误的码。")
    if hi <= 1.01:
        t = t * 255.0
    return t.clamp(0, 255).to(torch.uint8).contiguous()
