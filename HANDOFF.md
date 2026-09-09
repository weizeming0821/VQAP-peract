# AA VLA 项目交接（2026-09-09 · 新机 + Seen18/UnSeen 改版）

> **接手会话请先读完这份再动手。** 建议顺序：
> 1. 第一节「一句话现状」
> 2. 第二节「这轮改了什么口径」—— 不读会拿旧划分的结论去解释新数据
> 3. 第三节「已经钉死的结论」—— 不读会重复做已经做过的死路
> 4. 第五节「还没做的」+ 第六节「踩过的坑」
>
> 预检：`source run/env.sh && python tools/preflight_stage3.py`
> 入口：`cd /root/autodl-tmp/VQAP && source run/env.sh`

---

## 一、一句话现状

**整仓已迁到新机并按新划分（Seen18 / UnSeen8）重建完毕，四臂权重全部归零到 step 0，
等待第一次公平等 K 的训练。**

上一轮在**旧划分**下跑完的 test 主表结论是负的（`B4 < B1`，p=0.044 显著变差），
但那轮三臂从未满足等 K 约束。**核心主张 `码注入臂 > B2` 还没有在公平条件下测过一次** ——
这正是本轮要做的事。

---

## 二、这轮改了什么口径（🔴 最容易搞错的地方）

### 2.1 任务划分整个换了

| | 旧（已作废） | 新（当前） |
|---|---|---|
| Seen | Seen12（官方 18 里挑 12 个微调） | **Seen18**（官方 18 全集，全部微调） |
| UnSeen | UnSeen6（官方 18 里剩下的 6 个） | **UnSeen8**（从 10 个候选里筛，全部在官方 18 **之外**） |
| UnSeen 的含义 | 未参与 Stage 3 微调，**但主干预训练见过** | **主干从未见过**（不在 `peract_600k` 的 21 个预训练任务里） |
| replay | `seen12`，165,773 样本 / 203 GB | **`seen18`，239,871 样本 / 292.6 GB** |

换的理由是实测出来的：`peract_600k` 的 `config.yaml` 显示主干在 600k 预训练中见过
全部 21 个任务（官方 18 + 3 个），旧 UnSeen6 只能声称「未参与 Stage 3 微调」，
**撑不起论文要的「码本帮助模型泛化到未见任务」**。

UnSeen 分两层（【声明 B′】，**分列报告，不合并求均值**）：

```
Tier-A（主结果，6 个）close_drawer  pick_up_cup  open_jar
                      phone_on_base  lamp_on      basketball_in_hoop
    主干没见过，但码本在 AtomAction 69 任务里见过其原子片段
    —— 这正是被检验的假设（原子知识跨任务迁移），不是数据污染

Tier-B（对照，2 个）  take_money_out_safe  take_lid_off_saucepan
    码本也没见过。用来排除「码本恰好见过该任务」这一替代解释

备选（2 个）          press_switch  put_knife_on_chopping_board
```

**定案规则事先冻结（【声明E】）**：先剔 E1 天花板为 0 的任务，再剔 B0 零样本
sr ≥ 90% 的任务。**只用 B0 与天花板，必须在看到 B1/B2/B3 任何结果之前完成。**
定案结果填进 `stage3/tasks.py` 的 `UNSEEN8_FINAL`（现在是 `None`，读它会当场报错）。

### 2.2 🔴 臂的编号：文档的 B3 ≠ 代码的 B3

```
设计文档（Exp_Design / VLA_Design）里的 B3 = 「码注入臂」这个角色
代码（stage3/arms.py）里：
    B3 = v1 注入层（FiLM + 门控 cross-attn）
    B4 = v2 注入层（纯残差相加）      ← 实际充当文档里的「B3」角色
```

v1 旧机实测被 LAMB 锁死（注入幅度 0.0095%，等于没注入），**本轮不训 B3**。
用户已定：**训 B1 / B2 / B4 三臂，K = 20001**。写结果时注明「文档 B3 = 实现 B4」。

### 2.3 判读口径

- 报 **mean 和 max，以 mean 为分析依据**；两个臂必须同口径，不能混
- 目标不定死具体数字，**以 `B3 > B2 > B1` 显著为准**（用户 2026-09-09 定）
- `B2 − B1` 是 Planner 的功劳，**不计入 VQAP 贡献**；`B3 − B2` 才是码本的净贡献

---

## 三、已经钉死的结论（**不要重复验证**）

### 3.1 评测噪声地板（读任何数字的前提）

同权重、同代码重跑 4 次（只有 RRT 随机性）：

```
整体 300 局：单次测量 SD 1.49 pp，两次之差 SD 2.11 pp
单任务 25 局：中位 |Δ| 4.0 pp，|Δ|≥8pp 占 44%，≥12pp 占 12%，≥16pp 占 2%
```

**逐任务 ±12 pp 以内的涨跌一律不可解读。** 判断改动是否有效必须做**配对 McNemar**
（同 episode 索引 ⇒ 同物体摆放），不能比较均值。

> 这是在旧划分下标定的，但它量的是仿真器的随机性，与任务划分无关，新机仍适用。
> 新机换了 GPU 与 torch 版本，**首次拿到新主表后应重标一次**。

### 3.2 🔴 子任务指令的语言内容对 PerAct 无影响（`flat` 对照）

把子任务指令**全部换成整任务指令**，码与分段完全不变：

```
B4 模板法（子任务指令）  33.67%  [min 32.33 / max 35.00]
B4 flat（整任务指令）    34.33%
                        Δ = −0.67 / +2.00 pp   p = 0.906 / 0.545   ❌ 无差别
```

**PerAct 根本没在用子任务指令里的措辞。** 这封死了在线 planner 的**措辞**优化路线 ——
三层措辞协议、指令模板库、变体先验、夹爪语义，优化的都是一个不影响输出的输入。
这解释了 X3 六个版本、四轮机制改动，每次机制都精确生效、成绩却纹丝不动。

> ⚠️ **不要把这条推广成「planner 无用」。** planner 还决定**分段时机**与**取哪个码**，
> 这两条都会实打实改变输出。而且 UnSeen 没有 train cache ⇒ 没有模板 ⇒
> **vlm-plan 是 UnSeen 上唯一可用的 planner**。

### 3.3 码注入本身是健康的，但可能在加噪

```
tools/b4_health.py --arm B4 --step 40000
  注入幅度 4.78%（判据 ≥3% OK）   参照：v2 初始 16.79%，v1 训练 100k 后 0.0095%
  B4 训练 loss 2.90  vs  无码基线 B2 loss 2.46      ← 带码反而更难拟合训练目标
```

### 3.4 🔴 码携带的是任务身份，不是动作语义（2026-09-09 直接在码本索引上算出）

```
I(k_global; task  ) = 1.93 bit        ← 码更像「任务指纹」
I(k_global; action) = 1.20 bit
I(k_global; action | task) = 1.10 bit    （H(action) = 2.97 bit）

细节码 9 个槽在样本内恒等：56,342 / 56,496 = 99.7%
grasp (n=13074) 用了 32/36 个码，众数只占 18.5%；approach 22.1%
```

**任务身份是 PerAct 从图像和语言里已经拿到的信息**，所以码在 Seen 上是冗余噪声
（与 3.3 的 loss 升高完全一致），在 UnSeen 上是一个没见过的任务的指纹、等于无意义。
细节码支路 99.7% 退化成常量，cross-attention 实际没在工作。

> 复现：`python3` 读 `data/atomaction_codebook_index.json` 的 `records`，
> 按 `k_global` / `action` / `task` 算互信息即可，秒级。

### 3.5 Adapter 置信度**不能**用来判断码本覆盖

1673 次带置信度的码调用，最佳阈值 τ=0.20 的二分准确率 71.5%，随机基线 69.1%
—— 只比瞎猜好 2.4 pp。`sweep`（未覆盖）0.907、`slide_block`（未覆盖）0.823 都很高；
`light_bulb`（覆盖）0.604 反而低。**Adapter 在没见过的场景上是「自信地错」。**
置信度门控不能替代任务白名单，这条路已排除。

### 3.6 码本覆盖门控能涨点（旧划分实测）

18 个任务里 5 个不在 `AtomAction_Dataset`。对它们把 `subtask_code_mask` 置 0：

```
                  全部12    覆盖8    未覆盖4
B1 基线            35.33    33.50    39.00
B4 原版            33.67    36.50    28.00
B4 门控            36.00    35.50    37.00      ← 未覆盖组 +9.00 pp
```

> ⚠️ 门控**不改动**覆盖 8 任务，那三次的覆盖组分数（39.50/33.50/33.50）是**同一个量的
> 三次独立测量**，极差 6.0 pp —— 不可读成「门控让覆盖组下跌」。

开关：`AAVLA_CODE_GATE_OFF="任务1,任务2,..."`（默认关）。

---

## 四、新机与环境

```
路径      /root/autodl-tmp/VQAP        （旧机是 /data0/xiexiao/VQAP，已不存在）
conda     /root/autodl-tmp/envs/aavla  （prefix 式，env 建在数据盘；根分区只有 30 G）
          python 3.10.21 / torch 2.8.0+cu128 / numpy 1.26.4 / pytorch3d 0.7.9
GPU       4 × RTX 5090（32.6 GB, sm_120）—— **独占，没有别的用户**
          🔴 sm_120 必须 torch ≥ 2.7+cu128，旧机的 2.4.1+cu121 跑不了
CPU/RAM   100 核 / 754 GB
磁盘      /root/autodl-tmp 750 GB，已用 87%，**可用约 102 GB**
CoppeliaSim /root/autodl-tmp/CoppeliaSim (Edu V4_1_0_Ubuntu20_04)
网络      走本地代理（HTTP_PROXY=127.0.0.1:6478）；github / huggingface 可达
          ⚠️ git 用 HTTPS，**SSH 密钥在本机不存在**，`git fetch origin` 会 Permission denied
```

**与旧机的差异会改变排期**：卡从 8 张变 4 张、核从 208 变 100。
评测分片数受 CPU 与显存双重限制，不要照抄旧机的 12 分片。

### 环境等价性已验收（G1，删旧工件之前跑完）

`open_drawer` 全量 3,785 样本逐字段比对，报告存 `result/p5_train/g1_open_drawer_verify.txt`：

- **34 个几何/结构字段 + 4 个字符串字段全部逐位相同**（含 RGB、点云、相机内外参、
  `trans_action_indicies`、`rot_grip_action_indicies` 与全部码字段）
- `reward` 仅在 1,125 个 `terminal=-1` 的 `add_final` 填充帧上不同 —— 两边都是未初始化
  内存，`is_valid_transition` 永久排除，训练中永不被采样
- 4 个语言嵌入字段有 fp16 末位差异：**这是物理下限，不是损坏**。CLIP RN50 以 fp16 前向，
  量级 9 处 1 ULP 就是 7.8e−3。实测差异按指令确定性（同一句指令的全部样本 ULP 倍数完全
  相同），新环境自身连跑两次逐位相同。判据已修订为 `1−cos ≤ 3e−5` 且相对误差 ≤ 1e−2
- **对模型的实际影响**：同一份权重、内容配对样本，total_loss 1.176208 → 1.176080，
  相对差 1.1e−4 = 单批标准差的万分之 1.7

---

## 五、当前状态与还没做的

### 已就绪 ✅

| 项 | 状态 |
|---|---|
| RLBench 数据 | train/val 各 18 任务 × 25 或 100 局；**test 28 任务 × 25 局**（Seen18 + UnSeen 候选 10） |
| Planner cache | `aavla_data/planner_cache/train/` 18 任务，1792 可用 episode |
| 共享 replay | `aavla_data/replay/seen18/multi/PERACT_BC/seed0`，239,871 样本 / 292.6 GB，`signature=ea5a9e86…`，验收全过 |
| 码本 / Adapter | 已预训练冻结；`checkpoints/vqap_pretrain/stage1/` |
| 四臂 iteration-0 | B1/B2/B3/B4 均已从官方权重装好，**weights 目录只剩 step 0** |
| B0 | 官方 `peract_600k` 600k 权重就位 |
| CLIP | RN50 在 `~/.cache/clip/`；ViT-B/16 已下载到 HF 缓存，**并已转出 `model.safetensors`** 让 `_resolve_clip_path` 直连快照 |
| 预检 | `tools/preflight_stage3.py` 判据已改到新机/新划分，全绿 |

### 还没做 ⬜

1. **E1 天花板（`replay_ceiling`）在代码里根本不存在** —— 全仓搜不到实现。
   它是 UnSeen 定案【声明E】的第一道筛，也是 E2/E3 归一化的分母，**必须先写**。
2. **UnSeen8 尚未定案** —— `stage3/tasks.py::UNSEEN8_FINAL` 是 `None`，
   要跑完 E1 天花板 + B0 零样本摸底才能按冻结规则筛出。
3. **三臂训练** B1 / B2 / B4，K=20001，等 K 等 batch 等种子。
4. **主评测** 四臂 × 28 任务 × 25 局。UnSeen 侧 B2/B4 必须走 `--planner vlm-plan --codes adapter`。
5. **噪声地板重标** —— 新机新 torch，3.1 的数字要复测一次。

### 🔴 没迁过来的东西

`result/p7/versions/`（旧机全部逐局分数与 planner 轨迹）**不在本机**。
旧 test 结果无法再做配对复核，只剩 `Exp_Design 第七节` 里的汇总数字。
**本轮所有运行跑完必须立刻归档逐局分数**，否则重蹈覆辙。

旧的 Seen12 ckpt 已归档到 `checkpoints/_seen12_archive/`（10 GB，与 replay 同盘，
**没有释放空间**）。确认不再需要后可以删。

---

## 六、🔴 踩过的坑

### 1. 旧 ckpt 会让重训静默失效

`offline_train_runner.py` 用 `existing_weights[-1]`（最大步数）恢复。
B1 曾留着 40000 步的旧权重、B3 留着 70000 —— 直接开训会从旧权重续，
而那是 **Seen12 replay 训出来的**，正是新设计要消除的混源。
已全部归档，四臂 weights 目录现在只剩 `0`。**换数据集重训前务必检查这一点。**

### 2. eval_data.csv 的表头不会自己更新

YARR 的 CSVWriter 只在**文件不存在**时写表头。换一批任务再评，列数变了而表头不变，
新行会与旧表头**逐列错位**；列数恰好相同时错位悄无声息，读出来是另一个任务的成绩。
已加 `scripts/stage3_eval.py::rotate_eval_csv_if_schema_changed`，在 `cmd_run` 里
**只轮转一次**（分片是并发起的，每片各转会互相抢文件）。

### 3. CLIP 每次评测都联网校验 —— 旧机打死过两次运行

v3.5 丢了整个 `close_jar`（275/300 局）；v3.6-test 12 个分片死 6 个（150/300 局）。
**不能只靠 `HF_HUB_OFFLINE` + `use_safetensors=True`** —— 该组合仍会查
`/api/models/...` 并在离线模式抛 `OfflineModeIsEnabled`；回退到默认解析会命中
`pytorch_model.bin`，被 transformers 的 CVE-2025-32434 检查拒绝（torch<2.6）。
**只有直接给本地快照目录才彻底断网**（`model/module/encoder.py::_resolve_clip_path`，
靠 `model.safetensors` 认目录 —— 本机的那份是从 `.bin` 转出来的，已逐张量核对一致）。

### 4. VLM API 断线会杀掉整个分片

一次 `APIConnectionError` → `PlannerError` → YARR 上抛 → 整个分片死掉。
已改成退化 + 留痕：PLAN 失败回落模板计划并记 `plan_source="template_fallback"`，
MONITOR 失败退化成 CONTINUE，连续失败 2 次判定断线、本局停止调用。
**分析时必须先剔除 `plan_source == "template_fallback"` 的局。**

### 5. 同一臂的两个评测并发会互撞 eval_data.csv

路径 `checkpoints/stage3_main/<arm>/seed0/eval_data.csv` 只按臂名分。
**同臂并发必然互相覆盖**；不同臂并发是安全的。

### 6. `--shards` 上限就是任务数

`n = min(a.shards, len(tasks))`。给再多卡也只有 len(tasks) 个分片。
瓶颈是 CPU/仿真器，不是 GPU。

### 7. 并发跑多个 stage3_eval 必须用 `--display-base` 错开号段

`pick_displays` 只扫「现在有没有 lock」，两个父进程同时扫会挑到同一批号，
分片互相把对方的 X server 带走。旧机上曾因此把 B3 训练打死在 step 15600。

### 8. 游离文件会破坏 RLBench 的数据集断言

`.ipynb_checkpoints` 混进 `front_rgb/` 会让该目录条目数与其余相机不等，
`get_stored_demos` 抛 `RuntimeError: Broken dataset assumption`。
**任何浏览过数据目录的操作之后、正式重建之前，先跑一次隐藏文件扫描** ——
`audit_report.json` 只数 episode，发现不了这类缺陷。

---

## 七、下一步

```
Phase 1  实现 E1 天花板                （与 Phase 2 并行，主要吃 CPU/仿真）
Phase 2  Seen18 三臂重训 B1/B2/B4       两波 × 2 卡，K=20001
Phase 3  UnSeen 定案（E1 + B0 摸底 → 冻结 8 个）
Phase 4  主评测 四臂 × 28 任务 × 25 局
Phase 5  改进（见下）
```

### 改进的方向（按证据强弱）

用户关心的两条，待深入讨论：

1. **更有效且轻量、保持即插即用的码注入方式** —— 现有 v1 门控被 LAMB 锁死、
   v2 纯残差虽健康但在加噪（3.3）。根因是 3.4：码内容本身不携带可迁移的动作语义。
2. **用 prompt 让 planner 自己决定要不要调码本** —— 替代硬编码的任务白名单门控。
   planner 本就有这个权力，且这是唯一能推广到 UnSeen 的门控形式
   （白名单需要事先知道任务在不在码本里，UnSeen 上做不到）。
   ⚠️ 注意 3.5：Adapter 置信度不能作为判据，planner 得靠别的信号。

其余候选：关掉细节码支路（99.7% 退化成常量）、补 dense0 输出上的 FiLM
（FiLM 的 β 被 SpatialSoftmax3D 完全抵消）、训练期带码本覆盖门控。

### X3 在线 planner

`flat`（3.2）证明措辞通路无效，**Seen 上继续迭代没有价值，建议就地定版**。
但 UnSeen 没有模板，vlm-plan 是唯一可用路径 —— X3 的工作应重新定位成
「保证在 UnSeen 上稳定可用」（断线韧性、空计划、分片存活），而不是继续优化措辞。

版本史与全部环境变量见 `Exp_Design 第七节`（历史结果，已封存）。

---

## 八、用户设定的工作规则（必须遵守）

```
1. 设计文档仅供参考，不是约束
2. **分步规划，执行前必须先与用户确认** —— 曾因未经同意就执行被明确批评
3. 有异议 / 发现缺陷 / 有更好建议要提出讨论
4. **每次都要报告改了哪些文件、加了哪些文件**
5. 显卡：本机 4 张独占。用完即释放
6. 目录约定：权重 `checkpoints/`，结果 `result/`，日志 `log/`；不用 `runs/`
7. 磁盘空间由用户负责，只需报告是否不够
8. VLM 成本 ¥0.0139/次，300 局约 ¥33–40。planner 迭代先用 120 局探路
9. 汇总口径：**报 mean 和 max，以 mean 为分析依据**；两个臂必须同口径
10. 噪声涨跌可忽略；定版后会重复 3 次实验
```

---

## 九、设计文档索引

| 文档 | 内容 |
|---|---|
| `Exp_Design.md` | 实验设计（Seen18/UnSeen8）+ **第七节：旧划分的历史结果，已封存** |
| `VLA_Design.md` | 任务划分 §2.4 / Stage 3 集成 / Planner / cache |
| `Adapter_Design.md` | Stage 2 Adapter 结构；§5 码向量注入的接口契约 |
| `VQAP_Design.md` | Stage 0/1 码本预训练 |
| `stage3/tasks.py` | **任务划分的唯一真源**（代码侧） |
| `stage3/arms.py` | 四臂定义与 replay 字段隔离 |
