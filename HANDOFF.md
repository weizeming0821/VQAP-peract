# AA VLA 项目交接（2026-09-09 晚）

> **接手会话请先读完这份再动手。** 建议顺序：
> 1. 第一节「一句话现状」—— 知道现在处在什么局面
> 2. 第二节「test 主表」—— 唯一能写进论文的数字
> 3. 第三节「五条已证伪 / 已证实的结论」—— 不读会重复做已经做过的死路
> 4. 第六节「踩过的坑」—— 每条都是真实事故
> 5. 第七节「下一步」
>
> 预检：`python tools/preflight_stage3.py`
> 入口：`cd /data0/xiexiao/VQAP && source run/env.sh`

---

## 一、一句话现状

**test split 已全部跑完。核心主张 `B4 > B2` 在 test 上不成立（−1.67 pp），
而 `B4 < B1` 达到显著（−7.00 pp, p=0.044）。**

也就是说：加码本的臂（B4）在 held-out 测试集上**显著差于**只用整任务指令微调的
baseline（B1），也没有超过无码的子任务指令臂（B2）。val 上曾观察到的
`B4 − B2 = +4.67 pp` **没有复现**。

这不是需要继续调 planner 就能解决的问题 —— 见第三节的证伪链。

---

## 二、test 主表（Seen12 · test split · 25 局/任务 = 300 局）

**全部为 held-out 结果，每格都是完整 300 局，无残缺无回落。**

| 任务 | B0<br>零训练 | B1<br>整任务+微调 | B2<br>子任务指令 | B4<br>v2码注入 | B4·X3 v3.6<br>在线planner |
|---|---:|---:|---:|---:|---:|
| close_jar | 44 | 44 | 28 | 36 | 32 |
| light_bulb_in | 4 | 16 | 12 | 16 | 28 |
| open_drawer | 92 | 76 | 84 | 88 | 72 |
| place_cups | 0 | 0 | 0 | 0 | 0 |
| place_shape_in_shape_sorter | 0 | 8 | 12 | 8 | 20 |
| place_wine_at_rack_location | 44 | 48 | 20 | 16 | 20 |
| push_buttons | 28 | 32 | 24 | 12 | 32 |
| put_groceries_in_cupboard | 20 | 16 | 12 | 0 | 8 |
| reach_and_drag | 92 | 60 | 88 | 88 | 56 |
| slide_block_to_color_target | 64 | 56 | 56 | 48 | 24 |
| stack_blocks | 4 | 40 | 16 | 24 | 12 |
| sweep_to_dustpan_of_size | 64 | 76 | 56 | 52 | 48 |
| **均值** | **38.00** | **39.33** | **34.00** | **32.33** | **29.33** |

### 配对检验（McNemar，同 episode 索引 ⇒ 同物体摆放）

| 对比 | Δ | 翻转局 | p | 判定 |
|---|---:|---:|---:|---|
| B0 → B1（微调的价值） | +1.33 | 84 | 0.744 | ❌ 无差别 |
| B2 → B1（子任务指令的代价） | +5.33 | 112 | 0.156 | ❌ 不显著（方向符合预期） |
| **B2 → B4（核心主张）** | **−1.67** | 63 | 0.615 | ❌ **不成立** |
| **B1 → B4** | **−7.00** | 99 | **0.044** | ✅ **显著变差** |

> 🔴 **口径纪律**：所有臂都必须用同一口径（mean 或 max，不能混）。
> 用户已定：**报 mean 和 max，以 mean 为分析依据**。
> `max-of-3` 相对真值系统性偏高约 +1.3 pp，与要论证的效应量同量级。

---

## 三、五条已经钉死的结论（**不要重复验证**）

### 3.1 评测噪声地板（这是读所有数字的前提）

同权重、同代码重跑 4 次（只有 RRT 随机性）：

```
整体 300 局：单次测量 SD 1.49 pp，两次之差 SD 2.11 pp
单任务 25 局：中位 |Δ| 4.0 pp，|Δ|≥8pp 占 44%，≥12pp 占 12%，≥16pp 占 2%
```

**逐任务 ±12 pp 以内的涨跌一律不可解读。** 之前大量"某任务涨了/跌了"的分析
都是在读噪声。要判断一个改动是否真的有效，必须做**配对 McNemar**，
而不是比较均值。

### 3.2 🔴 子任务指令的语言内容对 PerAct 无影响（`flat` 对照）

`flat` = 把子任务指令**全部换成整任务指令**，码与分段完全不变：

```
B4 模板法（子任务指令）  33.67%  [min 32.33 / max 35.00]   val
B4 flat（整任务指令）    34.33%                            val
                        Δ = −0.67 / +2.00 pp   p = 0.906 / 0.545   ❌ 无差别
```

**PerAct 根本没在用子任务指令里的信息。** 这直接封死了整条 X3 路线 ——
三层措辞协议、指令模板库、变体先验、夹爪语义、末段续写、先验展开，
**优化的都是一个不影响输出的输入**。

这解释了为什么 X3 六个版本、四轮机制改动，每次机制都精确生效、成绩却纹丝不动。

### 3.3 码注入本身是健康的，但可能在加噪

```
tools/b4_health.py --arm B4 --step 40000
  注入幅度 4.78%（判据 ≥3% OK）    参照：v2 初始 16.79%，v1 训练 100k 后 0.0095%
  B4 训练 loss 2.90  vs  无码基线 B2 loss 2.46      ← B4 拟合训练目标反而更差
```

码确实到达了模型（不是 v1 那种被 LAMB 锁死的退化），但**带码的 loss 更高**。
配合 3.4 的门控实验，一致解释是：**注入的码对相当一部分样本是噪声**。

### 3.4 码本覆盖门控能涨点（但只到与 B1 持平）

18 个任务里 5 个不在 `AtomAction_Dataset`（Seen12 val 中占 4 个：
`place_cups` / `place_wine_at_rack_location` / `slide_block_to_color_target` /
`sweep_to_dustpan_of_size`）。对它们把 `subtask_code_mask` 置 0：

```
                  全部12    覆盖8    未覆盖4
B1 基线            35.33    33.50    39.00
B4 原版            33.67    36.50    28.00
B4 门控            36.00    35.50    37.00      ← 未覆盖组 +9.00 pp
```

> ⚠️ **一个必须避免的误读**：门控**不改动**覆盖 8 任务，所以那三次的覆盖 8 分数
> （39.50 / 33.50 / 33.50）是**同一个量的三次独立测量**，极差 6.0 pp。
> 我曾把第三次抽样误读成"门控导致覆盖组下跌 3 pp"，这是错的。

开关：`AAVLA_CODE_GATE_OFF="任务1,任务2,..."`（默认关）。
**用户明确：只作分析用，不进最终架构。**

### 3.5 🔴 Adapter 置信度**不能**用来判断码本覆盖

采集 1673 次带置信度的码调用（免费，`--planner template --codes adapter`）：

```
              n     中位     p25    p50    p75
覆盖 8      1156   0.907    0.581  0.908  0.943
未覆盖 4     517   0.678    0.332  0.678  0.924

最佳阈值 τ=0.20 的二分准确率 71.5%，随机基线 69.1%  →  只比瞎猜好 2.4 pp
```

逐任务更清楚：`sweep`（未覆盖）0.907、`slide_block`（未覆盖）0.823 —— 和覆盖任务
一样高；而 `light_bulb`（覆盖）0.604、`put_groceries`（覆盖）0.703 反而低。

**Adapter 在没见过的场景上是"自信地错"，它不知道自己不知道。**
所以置信度门控不能替代任务白名单。这条路已排除。

---

## 四、X3 在线 planner 的完整历史（**建议就地定版，不要再迭代**）

### 版本序列与实测

```
B3@40000 · val · 10 局/任务（120 局）
  v3.0  26.67    v3.1  29.17    v3.2a  26.67    v3.2c  33.33    v3.3  34.17

B4@40000 · val · 25 局/任务（300 局）· 同 11 任务口径（X1 同口径 32.36）
  v3.3  27.64    v3.4  27.64    v3.5  29.45

B4@40000 · test · 25 局/任务（300 局）
  v3.6  29.33     （split 难度校正后 ≈ 26.17，见下）
```

> 🔴 **v3.3 在 B3、v3.4/3.5 在 B4，两段序列不能连读。**
> v3.6 在 test、其余在 val，跨 split 比较必须校正
> （校正量 = [(B0test−B0val)+(B1test−B1val)]/2，逐任务从 −8 到 +16，波动很大）。

### 各版本改了什么、验证结果如何

| 版本 | 改动 | 机制验证 | 成绩 |
|---|---|---|---|
| v3.2c | 关掉执行记忆 | REPLAN 173 → 6 | +6.66（唯一有机制支撑的一次） |
| v3.3 | 夹爪语义分类（按动作分 boundary/tool/hold） | 压制 186 次误触发 ✅ | 无差别 |
| v3.4 | RETRY 位置回退 | 触发 157/300 局，移动中位 0.446 m ✅ | **Δ=+0.00, p=1.000** |
| v3.5 | 末段禁用 NEXT + EXTEND 续写 | 末段空转 500 → **0** ✅ | +1.82（噪声内） |
| v3.6 | 先验展开（transfer → transfer,transfer） | 段数 4 → 5 ✅ | 未修好目标任务 |

**每一次机制都精确生效，成绩都没动** —— 与 3.2 的 `flat` 结论完全一致。

### v3.6 的一个未诊断回归

`open_drawer` 在此前所有版本都是 88–96，v3.6 掉到 66（校正后），远超噪声上限。
但 `open_drawer` 的先验是 `[approach, grasp, pull]`，训练统计全是 1 帧，
**先验展开对它不该有任何作用**。可能来自 `grasp` 段被展开（`close_jar` 的 grasp
中位是 2），或 EXTEND 在 test 上触发方式不同。**未诊断，轨迹都在，可免费查。**

### 环境变量（全部可逐字回退）

```bash
AAVLA_PHRASING_MODE=select    # 三层指令协议；free = v3.1 自由生成
AAVLA_PLANNER_HISTORY=0       # 执行记忆，默认关（开 = v3.2a，实测最差）
AAVLA_PRIOR_VARIANTS=1        # 备选先验
AAVLA_GRIPPER_SEMANTICS=1     # v3.3 夹爪语义
AAVLA_RETRY_ROLLBACK=0        # v3.4 位置回退，默认关（实测效果为零）
AAVLA_ALLOW_EXTEND=1          # v3.5 末段续写
AAVLA_MAX_EXTEND=2  AAVLA_MAX_SEGMENTS=10
AAVLA_EXPAND_DWELL=1          # v3.6 先验展开（与 MIN_DWELL 互斥）
AAVLA_MIN_DWELL=1             # v3.6 硬拦截版本（EXPAND_DWELL 开时自动失效）
AAVLA_CODE_GATE_OFF=""        # 码本覆盖门控，默认关
```

---

## 五、数据与代码地图

### 结果归档（**逐局分数是唯一可做配对分析的东西，务必保留**）

```
result/p7/versions/
  _perep/              全部历史运行的逐局分数（B0/B1/B2/B3/B4 各 ckpt）
  rep2/                B4@20000/40000/60000 与 B1@40000 的第二次测量（噪声标定用）
  B0_test/  B1_test/  B2_template_test/  B4_template_test/     ← test 主表
  B4_x3_v33/  B4_x3_v34_rollback/  B4_x3_v35/  B4_x3_v36_test/  ← X3 各版本
  B4_gated/            码本门控实验
  B4_flat/             flat 对照（子任务指令 → 整任务指令）
  B4_conf/             Adapter 置信度采集（1673 次调用）
  v3.3/                v3.3-on-B3 的完整轨迹
  _failed_*/           失败运行的存档（API 断线 / CLIP 离线 / 空计划）
```

每个目录含 `per_episode.json`（`{"task|episode": {"score", "goal"}}`）、
分片日志、以及 `traces/`（planner 逐帧轨迹）。

> 🔴 **分片日志和 traces 会被下一轮运行覆盖。** 跑完立刻归档，否则无法做配对分析。
> 曾因此丢掉 v3.2c 的全部逐局数据，导致无法归因。

### 本阶段改动的文件（全部已 commit 并推送）

```
stage3/vlm_planner.py      X3 v3.3–v3.6 全部机制 + 断线韧性 + 停留先验
stage3/rollout.py          v3.4 位置回退 + gap 仪表 + 码本门控开关
stage3/adapter_codes.py    Adapter 置信度记录（softmax max）
planner/prompts.py         EXTEND 决策 + 末段 LAST-STEP NOTE
model/module/encoder.py    CLIP 本地快照解析（断网依赖）
run/env.sh                 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE

tools/test_rollback_v34.py       19 项
tools/test_planner_resilience.py 15 项
tools/test_vlm_plan_v35.py       24 项
tools/test_vlm_plan_v36.py       17 项
```

相关提交：`22f172e`(v3.3) `28da219`(v3.4) `089548b`(断线韧性) `99f24e1`(v3.5)
`20bd6eb`(空计划) `757e0cd`(v3.6-A) `e6d18ca`(v3.6-B) `9bd1155`(CLIP离线+置信度)

---

## 六、🔴 踩过的坑（本轮新增，旧坑见 git 历史中的上一版 HANDOFF）

### 1. CLIP 每次评测都联网校验 —— 打死过两次运行

```
Error: Failed to load CLIP text tower from: openai/clip-vit-base-patch16
Error: HTTPSConnectionPool(host='huggingface.co', port=443)
```

v3.5 丢了整个 `close_jar`（275/300 局）；v3.6-test 12 个分片死 6 个（150/300 局）。
模型本地早已缓存 1.2 GB，纯粹是解析路径要联网。

**不能只靠 `HF_HUB_OFFLINE` + `use_safetensors=True`** —— 该组合仍会查
`/api/models/...` 并在离线模式抛 `OfflineModeIsEnabled`；回退到默认解析又会命中
`pytorch_model.bin`，被 transformers 的 CVE-2025-32434 检查拒绝（torch<2.6）。
**只有直接给本地快照目录才彻底断网**（`model/module/encoder.py::_resolve_clip_path`）。

### 2. VLM API 断线会杀掉整个分片

一次 `APIConnectionError` → `PlannerError` → YARR 把异常继续上抛 → **整个分片死掉**，
该任务剩余的局全丢。12 个分片分别在第 7~24 局被打死，300 局只跑出 126 局。

已改成**退化 + 留痕**（`P12`）：PLAN 失败回落模板计划并记
`plan_source="template_fallback"`；MONITOR 失败退化成 CONTINUE；连续失败 2 次判定
断线、本局停止调用（否则每次走满 4 重试 × 指数退避 ≈ 30 s，把速度拖到 25.8 s/局）。

**分析时必须先剔除 `plan_source == "template_fallback"` 的局。**
v3.6-test 第二次跑有 180/300 局回落，整体数字完全不可用。

### 3. EXTEND 把 `_plan` 引入局中路径，空计划会杀分片

v3.5 首跑 300 局只出 119 局，9 个分片死于 `PlannerError: vlm-plan 生成了空计划`。
EXTEND 会在局中问 VLM"还剩什么要做"，而 VLM 完全可能答"没有了" —— 这是合法回答。
那句 `raise` 落在断线兜底的 `try` 之外。已修（`20bd6eb`）。

### 4. 同一臂的两个评测并发会互撞 `eval_data.csv`

路径是 `checkpoints/stage3_main/<arm>/seed0/eval_data.csv`，只按臂名分。
**同臂并发必然互相覆盖**；不同臂并发是安全的（B0/B1 test 就是这么并行的）。

### 5. `--shards` 上限就是任务数

`n = min(a.shards, len(tasks))`。12 个任务时给再多 GPU 也只有 12 个分片，
**加卡不会更快**（实测 GPU 利用率只有 0~34%，瓶颈是 CPU/仿真器）。

### 6. GPU 预留器会正确拒绝，别绕过它

B1 test 第一次被拒：GPU7 扣掉给别人预留的 8 GB 后只剩 4.1 GB，放不下 6 个分片
（每个 4715 MiB）。这是护栏在工作，改成串行排队即可。

---

## 七、下一步

### 用户设定的优先级

```
确定 X3 最终方案  >  改进 B4  >  其他
```

但按第三节的证伪链，**X3 已无继续投入的价值**（`flat` 证明语言通路无效）。
建议向用户说明后就地定版，把优先级实质转到 B4。

### 立刻可做（免费）

1. **诊断 v3.6 的 `open_drawer` 回归**（轨迹在 `B4_x3_v36_test/traces/`）
2. **B1 test 重复 2 次**（免费，用户已同意并行安排）
3. B3 的 test（补齐主表；B3 是 v1 注入层，与 B4 对照）

### 需要讨论后再做

**带码本门控重训 B4**（~10 h GPU，4 卡）。依据：
- 门控在**评测时**就能让未覆盖组涨 9 pp（3.4）
- B4 训练 loss 高于无码基线（3.3）—— 注入层正在拟合无意义的码
- 训练期就带门控，注入层不必再去拟合那些噪声

这是目前唯一还有净收益空间、且有三条独立证据支撑的方向。

### 用户的最终目标与现实差距

用户希望 **X3 + B4 超越 B1 十个百分点**。当前 test 上：

```
B1  39.33          目标 ≈ 49.3
B4  32.33          差距 −17.0 pp
B4·X3 v3.6 29.33   差距 −20.0 pp
```

**必须如实告知：按现有证据，这个目标不是靠调 planner 能达到的。**
拆子任务本身先扣分（test 上 B2 34.00 < B1 39.33），码本没有补回来
（B4 32.33 < B2 34.00）。

---

## 八、用户设定的工作规则（必须遵守）

```
1. 设计文档仅供参考，不是约束
2. **分步规划，执行前必须先与用户确认** —— 曾因未经同意就执行被明确批评
3. 有异议 / 发现缺陷 / 有更好建议要提出讨论
4. **每次都要报告改了哪些文件、加了哪些文件**
5. 显卡：盘子上限 4 张（自己在跑的也计入）；空闲显存够就可与别人共用，
   共用卡预留 8 GB；**绝不 kill 别人的进程**；用完即释放
6. 目录约定：权重 `checkpoints/`，结果 `result/`，日志 `log/`；不用 `runs/`
7. 磁盘空间由用户负责，只需报告是否不够
8. VLM 成本 ¥0.0139/次，300 局约 ¥33–40。planner 迭代先用 120 局探路
9. 汇总口径：**报 mean 和 max，以 mean 为分析依据**；两个臂必须同口径
10. 噪声涨跌可忽略；X3 定版后 B4 会重复 3 次实验
```

---

## 九、机器与环境

```
conda env    aavla（py3.10, torch 2.4.1+cu121, numpy 1.26）
入口         cd /data0/xiexiao/VQAP && source run/env.sh
GPU          8 × RTX 5880 Ada（49 GB），共享机器，约 100 个用户
             别人长期占 GPU1(23 GB) 和 GPU7(37 GB)
CPU          208 核；12 分片时 load ≈ 40，磁盘 util ≈ 45%
replay       aavla_data/replay/seen12/... 165,773 样本 / 202 GB / 三臂共享只读
数据         aavla_data/rlbench/{train,val,test}/ 各 18 任务 × 25 局
             （val 和 test 每任务只有 25 局，这是样本量的硬上限）
显卡看门     python tools/gpu_reserver.py status
```

---

## 十、设计文档索引

| 文档 | 内容 |
|---|---|
| `Exp_Design.md` | 实验设计 + **第七节：实测结果汇总（已更新到本轮）** |
| `VLA_Design.md` | Stage 3 集成 / Planner / cache |
| `Adapter_Design.md` | Stage 2 Adapter 结构；§5 码向量注入的接口契约 |
| `VQAP_Design.md` | Stage 0/1 码本预训练 |
