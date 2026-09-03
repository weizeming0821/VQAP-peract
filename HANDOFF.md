# AA VLA 项目交接（P5 训练起点）

> 给接手会话：**先把这份读完，再跑 `python tools/preflight_stage3.py`**。
> 预检覆盖环境/数据/cache/权重/配置/代码不变式/单测/replay 九类前提，PASS 才动手。

---

## 一、这个项目在做什么

验证一个假设：**把「原子动作码本」注入 VLA，能否提升机器人操作的成功率与组合泛化。**

方法叫 VQAP，baseline 是 PerAct（CoRL 2022，RLBench 18 任务）。链路三段：

```
Stage 0/1  码本预训练（AtomAction_Dataset：69 任务 / 56,496 phase / 17 原子）  ✅ 已完成，冻结
           → 双码本 Kg=36 / Kd=192，d_code=512
Stage 2    Adapter：(图像, 子任务指令) → 码索引                                 ✅ 已完成，冻结
           → val 全局码 top-1 89.69%
Planner    VLM 把任务拆成原子动作子任务                                        ✅ 离线 cache 已建
Stage 3    把码注入 PerAct，训练四臂并评测                                      ⬅️ **你从这里开始**
```

### 四臂：逐级只多一样东西，差值才能干净归因

```
B0 ──(+微调)──> B1 ──(+子任务指令)──> B2 ──(+码注入)──> B3
```

| 差值 | 归因 |
|---|---|
| **B0 本身** | 验证评测管线正确性（**E0**，所有结论的前提门禁） |
| B1 − B0 | 微调本身的效果 |
| B2 − B1 | Planner 的功劳，**不计入 VQAP 贡献** |
| **B3 − B2** | **码本的功劳 —— 全文核心主张** |

**没有 B2 就无法区分「增益来自码本」还是「来自任务被拆短」。**

最终目标是两个实验：**E2**（Seen12 四臂对比）、**E3**（UnSeen6 四臂对比）。

---

## 二、当前进度

| 阶段 | 状态 | 关键产物 |
|---|---|---|
| P0 环境 | ✅ | conda env `aavla`；B0 冒烟 **90.0%** vs 官方 92.0% |
| P1 数据 | ✅ | `data_rlbench/` 2700 episode / 128 GB，六项审计全过 |
| P2 D1 诊断 | ✅ | 码一致率 **61.6%**（随机 2.8%、域内上界 89.7%） |
| P3 Planner cache | ✅ | `planner_cache/train/` 1800 ep，结构通过率 99.94%，花费约 310 元 |
| P4 Stage 3 集成 | ✅ | 三臂冒烟 loss 全降；可训参数 2,117,917 / 2,550,685 |
| **P5 训练** | ⬅️ **待启动** | 框架已就绪，六个 bug 已修，预检 PASS |
| P6 在线 Planner | ⬜ | B2/B3 的**评测**需要它 |
| P7 全量评测 | ⬜ | 4 臂 × 18 任务 × 25 局 = 1800 rollouts |
| P8 结果聚合 | ⬜ | E0 / E2 / E3 |

---

## 三、🔴 五条必须知道的事（踩过的坑）

### 1. 所有需要渲染的命令都必须 `xvfb-run -a`
训练每 `log_freq` 步调 `update_summaries → visualise_voxel → pyrender.OffscreenRenderer`，
无头环境下抛 `NoSuchDisplayException` 并**终止训练**。数据生成、评测同理。

### 2. GPU 与磁盘都在被其他用户消耗，且波动很大
共享机器。实测同一天内空闲卡从 6 张变成 2 张，磁盘从 1.5 T 掉到 547 G。

> 🔴 **三臂必须用相同的 `ddp.num_devices`** —— 有效 batch 不同就成了混淆项。
> 启动前用预检看空闲卡数，选一个**三臂都能保证**的值，并用 `CUDA_VISIBLE_DEVICES` 锁定。

### 3. BUG 1 的教训：静默退化比崩溃危险得多
`act()` 曾经不传码 → B3 训练用码、评测不用码 → 跑出来就是 B2 的成绩，**不报错**。
已修，并加了「装配了码本却拿不到码就 `raise`」的硬防线 + 回归测试 `tools/test_act_codes.py`。
**改任何与码相关的代码后，务必重跑这个测试。**

### 4. `fill_multi_task_replay` 的子进程崩溃不会传播到父进程
它用 `Process` 逐任务 spawn，子进程只在自己的 stderr 打 traceback，父进程照常 `join`。
`scripts/build_replay.py` 里已加「样本数低于预期一半就拒绝写完成标记」的硬校验。
**任何包装这类多进程调用的脚本都要加类似校验。**

### 5. `purge_replay_on_shutdown` 默认 True
YARR 训练一结束就删掉整个 replay。已改为 False —— 这是三臂共享同一份 176 GB replay 的前提。

---

## 四、下一步：P5 训练

### 前置
```bash
cd /data0/xiexiao/VQAP && source run/env.sh
python tools/preflight_stage3.py          # 必须 PASS
```

### Step 1 · 构建共享 replay（约 176 GB / 30 分钟，只需一次）
```bash
CUDA_VISIBLE_DEVICES=6,7 python scripts/build_replay.py
```
完成后会写 `replay/seen12/_BUILD_COMPLETE.json`，重跑自动跳过。
预期约 **141,300 样本**（P1 用关键帧解析独立算出，可交叉验证）。

### Step 2 · 训练（顺序 B1 → B3 → B2）
```bash
# 先按空闲卡数改 conf/stage3.yaml 的 ddp.num_devices，三臂保持一致
cd source/peract
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 nohup xvfb-run -a python train.py \
    --config-name=stage3 stage3.arm=B1 > /data0/xiexiao/VQAP/log/train_B1.out 2>&1 &
```
B1 打头是因为它**不含任何新组件** —— 若 B1 就崩，锅在冻结边界/LR/数据，而非方法。

进度：`OfflineTrainRunner` 每 100 步打一行，无进度条：
`Train Step 000100 | Loss: 2.1 | Sample time: | Step time:`

### Step 3 · 选 K（best checkpoint）
官方三段式协议，**不要塞进训练循环**：
```
train.py                        每 2500 步存 ckpt（num_weights_to_keep=25）
eval.py eval_type=missing       在 val 上评所有 ckpt → eval_data.csv
eval.py eval_type=best          用最佳 ckpt 在 test 上出最终数字
```

> ⚠️ **B2/B3 的 val 评测需要在线 Planner（P6），现在做不了。**
> 决定：**K 由 B1 的 val 曲线选定，三臂共用**。这不是将就 ——
> `Exp_Design` 的公平性硬约束本来就要求三臂同 K，从唯一不含新组件的 B1 选是最中性的。

### 关键配置（`source/peract/conf/stage3.yaml`）
| 项 | 值 | 为什么 |
|---|---|---|
| `pos_encoding_with_lang` | **False** | 官方 ckpt 如此，仓库默认 True，**不改会权重加载失配** |
| `aug_rpy` | `[0,0,0]` | 对齐官方 ckpt |
| `load_existing_weights` | **True** | 否则 YARR 的断点恢复不生效 |
| `task_uniform` | True | 拉平 `stack_blocks` 占 Seen12 样本 35% 的失衡 |
| `lr` | **1e-4** | 官方 5e-4 是从零训 600k 步的值；从收敛点微调降一个量级 |
| `batch_size` | 4/卡 | P4 实测：torch 峰值 26.1 G；batch 6 是 38.5 G，加 DDP 梯度桶易 OOM |
| `training_iterations` | 50001 | 名义上限，按 val 曲线早停 |

### 🚦 唯一的硬门禁：B1@2k 的 val 成功率 ≥ B0
低于 B0 说明 **LR 偏大**，降到 3e-5 重训 B1（约 1 h 代价）。
另有一条自校验：**B1@step-0 的成功率必须等于 B0**（权重逐位相同）。不等说明训练管线在动权重之前就出了问题。

---

## 五、代码地图

```
model/code_injector.py        CodeInjector（FiLM + cross-attn，432,768 参数）+ CodebookLookup
stage3/arms.py               四臂定义 + 字段白名单 + GuardedSample（越权访问抛异常）
stage3/cache_join.py         PlannerCache（cache 查表）+ TextEmbedCache（CLIP 编码记忆化）
planner/                     契约 / prompt / DashScope 客户端 / 离线分段 / 三道质量门禁
scripts/planner_cache.py     离线 cache 的唯一入口（build/repair/backfill/codes/gates）
scripts/build_replay.py      独立构建共享 replay
scripts/init_from_official.py 把 peract_600k 装成各臂的 iteration-0 检查点
tools/preflight_stage3.py    ⭐ 训练前全面预检
tools/test_*.py              5 套单测（改代码后必跑）
source/peract|RLBench|YARR   第三方源码，**各自有独立 git**，改动在其自身历史里
```

**版本管理**：主仓库 https://github.com/weizeming0821/VQAP-peract （SSH deploy key 已配）。
第三方改动看 `cd source/<repo> && git log`。`result/` `logs/` `replay/` `planner_cache/`
`data_rlbench/` `source/` 均不入库。

---

## 六、⚠️ 已知风险与待解决问题

| # | 问题 | 现状 |
|---|---|---|
| 1 | **B3 的评测需要在线 Planner（P6）** | `act()` 的码接口已打通（走 `observation` 字典），P6 只需往里塞 `subtask_k_global` / `subtask_k_detail` / `subtask_code_mask` 三个键。**架构完备，缺数据来源** |
| 2 | **长程任务「鬼打墙」** | 在线 Planner 必须有确定性兜底：子任务索引单调不减、每段关键帧预算（可从 cache 统计）、RETRY 上限、停滞检测、`episode_length=25` 硬上限 |
| 3 | **`code_mask=0` 样本仅 1.06%** | 注入层关断路径的**正确性由 bit-exact 单测保证**（数学结构，非训练），该比例只影响训练动态完整性 |
| 4 | **指令措辞只影响码预测 0.7 个点**（D1 实测） | 说明码几乎全由图像决定。而 PerAct 本来就看得见图像 —— **码能提供的增量信息可能有限**，这是 B3−B2 的最大不确定性 |
| 5 | **B2 可能「拿到子任务指令却用不上」** | 能适应「整任务→子任务」分布变化的只有 `lang_preprocess` 一层线性（0.066 M）。若 B2 ≈ B1，**不得直接下结论说子任务分割无用**，更可能是容量瓶颈 |
| 6 | **UnSeen6 信号薄** | 官方成绩里 6 个任务只有 2 个落在有信号区间（2 个触天花板、2 个在地板）。E3 结论会偏薄，须同时报 B0 作参照 |
| 7 | `reach_and_drag` ep64 的 cache 无法生成 | JSON 连续 3 轮解析失败，已在 `PlannerCache` 里跳过 |
| 8 | 7 条 episode 夹爪信号异常 | 关键帧上夹爪全程 OPEN 却含 grasp/lift/place，已跳过。注意另有 3 条 `turn_tap` 也是全程 OPEN 但**合理**（用张开的夹爪拨水龙头） |
| 9 | 细节码 9 槽在样本内恒等 | `Adapter_Design §7 R1` 已记录并接受。报告细节码指标时须注明只有 1 个自由度 |

---

## 七、如果结果不及预期

**D1 诊断已备好红灯预案**：用 VQAP 编码器的 oracle 码（`result/p2_d1_diagnosis/agreement.json` 里已算出）
在 RLBench segment 上微调 Adapter —— 标签自动生成、不动码本、成本几小时。
D1 同时给出了这条预案的**精确收益上界**：把码一致率从 61.6% 抬向域内的 89.7%。

**但先别急着上全量评测**（1800 rollouts、2–4 天）。若 mini-eval 上三臂差异都在噪声内，
先回头定位是哪一环没起作用 —— 是码没信息（看 D1）、注入没生效（看 `test_act_codes`）、
还是语言瓶颈（看 B2−B1）。

---

## 八、设计文档

`VLA_Design.md`（Stage 3 集成 / Planner / cache）· `Adapter_Design.md`（注入契约 §5）·
`VQAP_Design.md`（码本）· `Exp_Design.md`（实验协议与公平性约束）。

⚠️ 文档是**参考而非最终方案**，且有若干处已被实测推翻，例如：
- 「归一化后 short 指令与 RLBench 指令完全同分布」→ 实测词数中位数 4 vs 6，交集才是 3–8 词
- 「共享 replay 额外开销 +10%」→ 文本编码记忆化后接近 0
- 「replay 120–360 GB」→ 精算 176 GB
- 「R1 码本语义错位是最高风险」→ D1 实测后下调到 🟡

以实测为准，`result/*/README.md` 里记录了每个阶段的实测结论。
