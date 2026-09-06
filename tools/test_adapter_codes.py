#!/usr/bin/env python
"""实时 Adapter 出码路径的单测。

最关键的一条：**拿 cache 里某个 segment 的起始帧 + 该段指令喂进实时路径，
必须逐位复现 cache 里存的 (k_global, k_detail)**。

离线 `scripts/planner_cache.py cmd_codes` 用的就是同一个 Adapter、同一张图、
同一句指令。若两边对不上，说明我的预处理（dtype / 通道顺序 / 归一化）跟离线
不一致 —— 这种错**不会抛任何异常**，只会让评测时的码悄悄失真，
最后拿到一份说不清是什么的成绩。所以这条必须逐位相等，不能只看"差不多"。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "source" / "peract")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

OK = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("  [OK]   " if cond else "  [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK = OK and bool(cond)


def main() -> int:
    from stage3.adapter_codes import LiveAdapter, AdapterError, _as_uint8_chw
    from stage3.cache_join import PlannerCache
    from PIL import Image

    print("=== 1. 帧规整（不得做归一化）===")
    u8 = torch.randint(0, 255, (3, 128, 128), dtype=torch.uint8)
    check("uint8 CHW 原样透传", torch.equal(_as_uint8_chw(u8), u8))
    hwc = u8.permute(1, 2, 0)
    check("HWC 自动转 CHW", torch.equal(_as_uint8_chw(hwc), u8))
    batched = u8.unsqueeze(0).unsqueeze(0)          # (1, timesteps, C, H, W)
    check("(1,T,C,H,W) 取当前帧", torch.equal(_as_uint8_chw(batched), u8))
    f01 = u8.float() / 255.0
    check("[0,1] 浮点还原回 uint8",
          (_as_uint8_chw(f01).float() - u8.float()).abs().max() <= 1)
    # 🔴 被 PerAct 归一化到 [-1,1] 的图必须报错而不是静默接受
    try:
        _as_uint8_chw(u8.float() / 127.5 - 1.0)
        check("[-1,1] 归一化过的图必须报错", False, "居然接受了")
    except AdapterError:
        check("[-1,1] 归一化过的图必须报错", True)

    print("=== 2. 可 pickle（eval.py 用 spawn 起子进程）===")
    import pickle as pk
    la = LiveAdapter(verbose=False)
    la2 = pk.loads(pk.dumps(la))
    check("LiveAdapter 可 pickle 且不带模型", la2._m is None)

    print("=== 3. 🔴 逐位复现 cache 里的码 ===")
    cache = PlannerCache(REPO_ROOT / "aavla_data" / "planner_cache" / "train")
    root = REPO_ROOT / "aavla_data" / "rlbench" / "train"
    tried = same_g = same_d = 0
    mism = []
    for task in ("close_jar", "open_drawer", "push_buttons", "stack_blocks"):
        for e in range(3):
            ep = cache.get(task, "train", e)
            if ep is None:
                continue
            ep_dir = root / task / "all_variations" / "episodes" / f"episode{e}"
            for seg in ep["segments"][:3]:
                fr = int(seg["start_frame"])
                frame = {}
                for cam in ("front", "wrist"):
                    f = ep_dir / f"{cam}_rgb" / f"{fr}.png"
                    if not f.is_file():
                        frame = {}
                        break
                    a = np.array(Image.open(f).convert("RGB"))
                    frame[cam] = torch.from_numpy(a).permute(2, 0, 1).contiguous()
                if not frame:
                    continue
                g, d = la(frame, seg["instruction"])
                tried += 1
                same_g += (g == int(seg["k_global"]))
                same_d += (d == [int(x) for x in seg["k_detail"]])
                if g != int(seg["k_global"]):
                    mism.append(f"{task}/{e} seg{seg['segment_index']}: "
                                f"实时 {g} vs cache {seg['k_global']}")
    check(f"k_global 逐位一致 {same_g}/{tried}", tried > 0 and same_g == tried,
          "; ".join(mism[:3]))
    check(f"k_detail 逐位一致 {same_d}/{tried}", tried > 0 and same_d == tried)
    print(f"       （共比对 {tried} 个 segment，Adapter 调用 {la.n_calls} 次）")

    print("=== 4. 越界必须硬失败 ===")
    try:
        la({"front": u8}, "grasp the red block")
        check("缺 wrist 时报错", False, "居然没抛异常")
    except AdapterError as e:
        check("缺 wrist 时报错", True)
        print(f"           ↳ {str(e).splitlines()[0][:70]}")

    print()
    print("实时 Adapter 单测:", "PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
