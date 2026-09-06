# AA VLA 项目交接（2026-09-07）

> **接手会话请先读完这份,再动手。** 顺序建议：
> 1. 第二节「当前状态」—— 知道已经测出了什么
> 2. 第三节「核心发现」—— 知道为什么现在要改注入层
> 3. 第四节「踩过的坑」—— 每一条都是真实事故,不读会重蹈
> 4. 第七节「下一步」—— 知道现在该干什么
>
> 预检：`python tools/preflight_stage3.py`（覆盖环境/数据/cache/权重/配置/代码不变式/单测/replay）

---

## 一、这个项目在做什么

验证一个假设：**把「原子动作码本」注入 VLA,能否提升机器人操作的成功率与组合泛化。**

方法叫 VQAP,baseline 是 PerAct（CoRL 2022,RLBench）。链路：

```
Stage 0/1  码本预训练  → 双码本 Kg=36 / Kd=192, d_code=512          ✅ 完成,冻结
Stage 2    Adapter：(front+wrist 观测, 子任务指令) → 码索引          ✅ 完成,冻结
           checkpoints/vqap_adapter/best.pth  val global_top1 89.69%
Planner    把任务拆成原子动作子任务                                  ✅ 离线 cache 已建
Stage 3    把码注入 PerAct,训练四臂并评测                            ⬅️ 当前阶段
```

### 四臂设计

```
B0 ──(+微调)──> B1 ──(+子任务指令)──> B2 ──(+码注入)──> B3
                                    B4 = B3 但换成 v2 注入层（新增,见第六节）
```

| 差值 | 归因 |
|---|---|
| B0 本身 | 验证评测管线正确性（E0,所有结论的前提门禁） |
| B1 − B0 | 微调本身的效果 |
| B2 − B1 | Planner 的功劳,**不计入 VQAP 贡献** |
| **B3 − B2** | **码本的功劳 —— 全文核心主张** |

**没有 B2 就无法区分「增益来自码本」还是「来自任务被拆短」。B2 目前仍未训练。**

---

## 二、当前状态

### 2.1 实测结果（**全部是 val,test 一次都没跑**）

> 口径：Seen12 · **val** split · 每任务 25 局 = 300 局 · `template + plan codes`。
> 评测非确定性（RRT 随机）,12 任务均值单次 σ ≈ 1.5 pp,差值 σ ≈ 2.1 pp。
> **相差 3 pp 以内不应认为有差异。**

| 任务 | B0 | B1@10000（峰值） | B1@40000 | B3@20000 | B3@40000 |
|---|---:|---:|---:|---:|---:|
| close_jar | 32 | 48 | 56 | 32 | 40 |
| light_bulb_in | 8 | 12 | 8 | 20 | 20 |
| open_drawer | 72 | 72 | 76 | 80 | 84 |
| place_cups | 0 | 4 | 0 | 0 | 0 |
| place_shape_in_shape_sorter | 8 | 12 | 12 | 8 | 0 |
| place_wine_at_rack_location | 32 | 44 | 40 | 36 | 40 |
| push_buttons | 24 | 28 | 16 | 20 | 20 |
| put_groceries_in_cupboard | 4 | 8 | 8 | 0 | 0 |
| reach_and_drag | 84 | 96 | 72 | 72 | 80 |
| slide_block_to_color_target | 64 | 96 | 56 | 44 | 64 |
| stack_blocks | 8 | 28 | 20 | 16 | 16 |
| sweep_to_dustpan_of_size | 44 | 76 | 60 | 36 | 60 |
| **均值** | **31.67** | **43.67** | **35.33** | **30.33** | **35.33** |

**B1 完整 val 曲线（300 局/点）**
```
step:   0     2500   5000   7500  10000  12500  15000  17500  20000  ...  40000
      29.7   39.7   39.0   42.0   43.7   39.7   39.7   40.7   34.3       35.33
                             └峰值┘                            └谷底┘
```
10000 步后进入下行,20000 步跌到 34.3（较峰值 −9.4 pp,>3σ,真实退化）。

**B3 完整 val 曲线（300 局/点）**
```
step: 20000  40000  60000  80000  100000
      30.33  35.33  36.00  35.00  35.00
             └────── 完全饱和,再训无增益 ──────┘
```

### 2.2 各臂状态

| 臂 | 训练 | 评测 | 备注 |
|---|---|---|---|
| **B0** | 零训练（官方 `peract_600k`） | ✅ val 31.67% | test 未跑 |
| **B1** | ✅ 40000 步 | ✅ val 0–20000 + 40000 | test 未跑；22500–37500 **不需要评**（用户已确认） |
| **B2** | ❌ 只到 step 400 被机器卡死 | ❌ | **需重跑**,约 10 h |
| **B3** | ✅ 100000 步（41 个 ckpt） | ✅ val 五点 | test 未跑 |
| **B4** | ❌ 代码就绪,未训 | ❌ | v2 注入层已实现 + 单测通过 |

### 2.3 test 的状态

```
❌ 任何臂、任何 ckpt 都没有跑过 test split
✅ test 的子任务计划已生成 aavla_data/planner_cache/plans_test_template.json
   （300 条,variation 精确命中 292、回退 8、缺失 0）
```

> ⚠️ 文献里 PerAct 论文报告的 ~43.7% 是**他们的 test split**,与我们的 val 不可直接比。
> 我们的 B0 在自己的 val 上是 31.67%,系统性低约 12 pp,但逐任务排序一致 → 判定为
> 数据差异而非管线错误（E0 门禁通过）。

### 2.4 评测方案对比（B3@40000 · val）

| 方案 | 局数 | 均值 | 说明 |
|---|---:|---:|---|
| **X1 `template + plan`** | 300 | **35.33%** | ← 当前最好,零成本,与全部历史数据同口径 |
| X2 `template + adapter` | 300 | 34.33% | 实时 Adapter 出码,−1.00 pp（噪声内） |
| X3-v1 `vlm-plan + adapter` | 120 | 26.67% | |
| X3-v2 `vlm-plan + adapter` | 300 | 22.33% | 推进更快版,反而更差 |
| X3-v3.1 | — | **未测成** | 三次启动都被机器负载挡住 |

**结论：test 用 X1。** 在线 planner 三版都输给模板法。

---

## 三、核心发现：码本贡献为零（三条独立证据）

| 证据 | 观测 |
|---|---|
| **权重侧** | `code_injector.gate` 范数 100000 步只从 0.0036 长到 0.0061（1.7×）,注入对 latents 的改变量 **0.0095%** |
| **行为侧** | 把码来源从「模板库查表」换成「实时 Adapter 看当前画面预测」,35.33% → 34.33%（**−1.00 pp,噪声内**） |
| **端到端** | `B1@40000 = B3@40000 = 35.33%`；两臂逐任务相关系数 **r=+0.961**,失败模式高度一致 |

> ⚠️ 措辞纠正：均值相同是**巧合**,不是「同一个模型」。逐任务差的绝对值均值 6.67 pp,
> 范围 [−16, +12],恰好抵消。但每任务 25 局的差值 SE ≈ 13.5 pp,所有差都在 1.2σ 内 ——
> 数据完全兼容于「两臂表现相同」。要分辨这些差异需要每任务约 100 局。

### 四条已量化的根因

```
① 全局码与语言高度冗余     H(k_global)=3.941 bit
                          H(k_global | 子任务指令)=0.910 bit    → 指令已解释 77%
② 细节码 82.8% 退化        9 槽位完全相同的段 5652/6824
③ 门控被 LAMB 锁死         ‖Δp‖ ≡ lr·‖p‖（见第四节第 1 条）
④ FiLM 的 β 被抵消         SpatialSoftmax3D 对空间常数偏置不变
```

**③ 是本阶段要修的（B4）。① 是天花板 —— 即使门全开,增量信息也不到 1 bit。**

---

## 四、🔴 踩过的坑（每一条都是真实事故）

### 1. LAMB 的信任比会锁死零初始化张量 —— 本项目最大的一个坑

`source/peract/helpers/optim/lamb.py:105-122`：

```python
trust_ratio = ‖p‖ / ‖adam_step‖
p -= lr · trust_ratio · adam_step
⇒ ‖Δp‖ ≡ lr · ‖p‖          # 步长正比于参数自身范数！
```

零初始化张量只有第一步自由（`‖p‖=0` → `trust_ratio=1`）,之后被锁进每步至多长 `lr` 的
倍增。`gate`（128 维）第一步只走 `lr·3.162·√128 = 3.6e-3`,之后 100000 步只长到 6.1e-3。

**判据：任何零初始化的可训张量在 LAMB 下都要警惕。** B4 的 v2 注入层用小随机初始化
（`w_g` std=0.01、`w_o` std=0.02）绕开了这个问题。

### 2. `xvfb-run -a` 会撞号,能打死别的进程

两步（扫空号 → 起 Xvfb）之间没有锁。并发分片会挑中同一个号,先退出的在 trap 里
kill 自己的 Xvfb 并删 lock,**把另一个分片的 X server 一起带走**（`XIO: fatal IO error`）。
**曾因此打死一次训练。**

已修：`scripts/stage3_eval.py:pick_displays()` 父进程一次性分配互不相交的号,用
`xvfb-run -n <num>`。并发跑多个评测时**必须用 `--display-base` 错开号段**（130/160/190）。

### 3. DDP master port 写死,第二个训练起不来

`conf/stage3.yaml` 的 `master_port: 29500`。第二个作业报 `EADDRINUSE`,而且错误埋在
`mp.spawn` 子进程里,父进程只吐一大段 `ProcessRaisedException` —— **白等了 35 分钟才发现**。

已修：`train.py:_free_port()` 在 spawn 前探测,被占自动换并告警。

### 4. 合并结果会「塌列」和「静默残缺」

- **塌列**：12 分片各跑 1 任务时 YARR 走**单任务** env 分支,列名不带任务后缀
  （`_independent_env_runner.py:272`）,12 份 CSV 列名全一样,外连接塌成一列。
- **静默残缺**：列是按分片任务名补的,**分片没产出数据时列照样存在**,只是值为空 ——
  实测 `put_groceries_in_cupboard` 整列缺失,而退出码 0、列数 12、均值却是拿 11 个任务算的。

已修：`_merge_shards` 补任务后缀 + **逐任务查值**；残缺时 run 以非零码退出。

### 5. `eval_data.csv` 会被不同配置互相覆盖

它是单一数据流、按 step 存行。不同 planner/codes 跑同一 step 会互相覆盖 ——
实测 CSV 里 step 2500 的 13.33 其实是 vlm 那轮的,把 template 的 15.83 覆盖了。

已修：每轮跑完立刻归档带完整配置标签的 JSON 到 `result/p7/`。**读结果一律读这些 JSON,
不要读 `eval_data.csv`。**

### 6. 分片日志会被下一轮截断

原来固定叫 `B3.eval.sh6.out`,下一轮开跑就 truncate,**想回查分片失败原因时现场已经没了**。
已修：日志名带 ckpt/planner/codes。

### 7. `pkill -f` / `os.kill` 循环会杀掉自己

`pkill -f "xxx"` 会匹配到**自己这条 bash 命令行**（命令文本里含该字符串）。
**已自杀三次。** 正确做法：按 argv 精确匹配 + 要求 `argv[0]` 是 python,并抽成独立脚本
文件（不内联,否则脚本文本本身又会被匹配到）。见
`/tmp/.../scratchpad/kill_watch.py` 的写法。

### 8. `ps -o user=` 会把用户名截断到 8 字符

`weizeming` → `weizemin`,拿它和 `$USER` 比**永远不相等**。
已修：`tools/gpu_reserver.py` 改用 `os.getuid()` 比较。

### 9. DDP 真正占卡的是 `mp.spawn` 子进程

它们的 cmdline 是 `python -c from multiprocessing.spawn import spawn_main; ...`,
**不含 "train.py"**。按关键词匹配判归属必然漏判。已修：只按 uid 判。

### 10. 闭包 pickle 不了（踩过两次）

`eval.py:215` 用 spawn 起子进程,`Stage3RolloutGenerator`（连同它持有的工厂）会被
pickle；`replay_dataset._seal` 那次是 DataLoader 的 spawn worker。
**凡是要跨进程的可调用对象,一律模块级类。**

### 11. 共享机器会被打穿,DataLoader 饿死会让 DDP 静默卡死

实测 load average 226、sda 读队列深度 251 时：B2 的 `Sample time` 从 0.0004 跳到 0.145,
随后两个 rank 都停在 `futex_wait_queue`,75 秒 CPU 时间只涨 1 秒 —— **完全卡死,不报错**。

**开训前必看：load average < 40、sda 队列深度 < 20。** 否则大概率重演。
卡死后父进程的 SIGTERM 带不走子进程,要按 PID 逐个 SIGTERM + SIGKILL。

### 12. 模板法的推进规则曾有一个致命 bug

`MIN_BUDGET = 2` 让每个单关键帧段被强行占住两帧,误差逐段累积。离线比对：
**逐关键帧分段一致率只有 20.9%,79% 的帧落后真值 1–2 段**。B3@2500 因此只有 10.0%。
修成 `MIN_BUDGET=1` + 累计边界后升到 92.5%,成绩 10.0% → 15.8%。

**`tools/test_online_planner.py` 第 10 组是这条的回归防线（一致率 ≥85%）。**

### 13. 其它

- 训练**不需要** X（voxel summary 已改为 `PERACT_VOXEL_SUMMARY=1` 才开,默认关）；评测需要。
- 评测的 tensorboard 默认关（`YARR_EVAL_TENSORBOARD=1` 才开）—— 曾产生 19 GB tfevents 把盘写满。
- 长任务一律 `setsid`,否则会话重启会连坐杀掉。
- 高负载时 `setsid nohup ... &` 可能根本没执行到（日志文件时间戳不变即是证据）,
  启动后**必须验证日志被 truncate 了**。

---

## 五、代码地图

### 本阶段新增/大改的文件（都还没 commit）

| 文件 | 作用 |
|---|---|
| `scripts/stage3_eval.py` | **评测唯一入口**。子命令 `setup-b0/bench/run/plans/prune/report/trace`。分片、分卡预检、display 分配、结果自归档、硬校验 |
| `stage3/planners.py` | planner 的**唯一选择点**。`template / flat / oracle / vlm / vlm-plan` |
| `stage3/online_planner.py` | 模板库 + 计划预生成 + `DeterministicPlanner`（累计边界推进） |
| `stage3/vlm_planner.py` | 在线 VLM planner：`VLMPlanner`（选下标）+ `OnlineVLMPlanner`（现场规划,v3.1） |
| `stage3/adapter_codes.py` | `LiveAdapter` —— 评测时实时调 Stage 2 Adapter 出码 |
| `stage3/rollout.py` | `Stage3RolloutGenerator` + `_SubtaskAgent`,轨迹落盘 |
| `stage3/replay_dataset.py` | replay 作为**不可变工件**：路径解析、6 道校验、密封 |
| `model/code_injector.py` | `CodeInjector`（v1）+ **`CodeInjectorV2`（v2,B4 用）** |
| `tools/test_online_planner.py` | 51 条断言,含分段一致率回归 |
| `tools/test_vlm_plan.py` | vlm-plan 的触发规则/四态决策/护栏 |
| `tools/test_injector_v2.py` | **18 条断言,重点验注入幅度 ≥5%** |
| `tools/test_adapter_codes.py` | 实时 Adapter 与离线 cache 逐位复现（35/35） |
| `tools/gpu_reserver.py` | 显卡看门：`watch/status/release/hold/unhold`,盘子上限 4 |

### 结果与日志的位置

```
result/p7/*.json          ← 🔴 唯一权威的结果来源（带完整配置标签）
result/p6_<planner>/traces/<arm>_<split>_<ckpt>/   逐局 planner 轨迹
checkpoints/stage3_main/<arm>/seed0/weights/<step>/
log/p5/                   训练与评测日志
Exp_Design.md 第七节       结果汇总表
```

---

## 六、B4 的 v2 注入层（已实现,未训练）

### 设计

```python
h   = latents + M(z_g) * mask          # 全局码：ln → mlp(256) → w_g(std=0.01),逐通道相加
out = h + cross_attn(h, Z_d) * mask    # 细节码：cross-attn 残差,**无门控**
```

**没有 gate、没有 FiLM,两个投影都是正常尺度初始化** —— 第四节第 1 条那个坑从源头消失。

### 单测实测的注入幅度

```
v1 初始注入   0.0000%      （训 100000 步后也只有 0.0095%）
v2 初始注入  16.79%
   全局码支路   6.33%
   细节码支路  15.47%
   换全局码 → 输出改变  7.16%
   换细节码 → 输出改变 21.09%
```

### 已确认的设计决定（用户拍板）

```
w_g std = 0.01          全局码初始注入约 16%
w_o std = 0.02          保持不变（v1 原值）
hidden  = 256           保持不变
warmup  不加            保留现有训练框架
细节码  保留 cross-attn,残差相加 out = h + o
```

### 关于 `code_mask=0`

**只在 `action == pose-adjust` 时发生,实测 112/10323 = 1.08%。**
所以「mask=0 时逐位等价」**不是安全网**（98.9% 的样本走 mask=1）,
它的价值是**单测锚点** —— 抓「乘/加写反」这类实现 bug。

### 如何启用

```yaml
# conf/stage3.yaml
stage3:
  injector: v1     # v1=B3 那版（默认）；v2=B4
```

`launch_utils.py` 按此选类。**v1 保留是为了 B3 的既有 checkpoint 仍能加载**（两版参数名
不同,混用会报错而不是静默加载错）。

### ⚠️ B4 还缺一步：臂定义

`stage3/arms.py` 目前只有 B0–B3。B4 需要**独立的 checkpoint 目录**,不能覆盖 B3 的 41 个 ckpt。
两个办法：

- **(a) 新增 `B4` 臂**（= B3 的字段权限 + `injector=v2`）—— 约 15 行,**推荐**。
  归档文件名、`stage3_eval.py --arm`、结果表都靠臂名区分,混用一定出乱子。
- (b) 只改 `framework.logdir` 指到别处,`arm` 仍写 B3 —— 省事但结果目录会出现
  「arm=B3 却是 v2」的歧义。

**这一步尚未做,接手时先定。**

---

## 七、下一步

### 立刻可做（不占 GPU）

1. **新增 B4 臂**（第六节的 (a)）
2. 读 `result/p7/*.json` 熟悉结果口径

### 等机器负载 < 40 后（当前 181,是唯一硬阻塞）

```bash
# 两组并行,4 卡（DDP 端口冲突已修,可安全并行）
cd source/peract && source ../../run/env.sh

# B2：子任务指令,无码
CUDA_VISIBLE_DEVICES=a,b setsid nohup python train.py --config-name=stage3 \
    stage3.arm=B2 framework.training_iterations=40001 \
    framework.num_weights_to_keep=30 > ../../log/p5/B2.trainN.out 2>&1 &

# B4：子任务指令 + v2 注入
CUDA_VISIBLE_DEVICES=c,d setsid nohup python train.py --config-name=stage3 \
    stage3.arm=B4 stage3.injector=v2 framework.training_iterations=40001 \
    framework.num_weights_to_keep=30 > ../../log/p5/B4.train1.out 2>&1 &
```

**启动后必须验证**：日志出现 `Resuming training from iteration 0`、
`Stage 3 注入层已挂载(v2)`、`冻结边界：可训参数 2,5xx,xxx`,且 `Train Step` 在推进。

### 训练完成后

```bash
# val 选点（零成本）
python scripts/stage3_eval.py run --arm B4 --ckpt 40000 --split val \
    --episodes 25 --shards 12 --gpu a,b --stagger 20 --display-base 130

# test（方案已定：template + plan）
python scripts/stage3_eval.py run --arm B4 --ckpt <best> --split test \
    --episodes 25 --shards 12 --gpu a,b --stagger 20 --display-base 130
```

### B4 的判据（**提前定死,避免事后找理由**）

```
B4@40000 > B1@40000(35.33%) + 3 pp   →  注入机制确实是瓶颈,方向对
B4 ≈ B1                              →  **注入形式不是瓶颈**,问题在码本身携带的信息
                                         （0.91 bit 增量 + 细节码 82.8% 退化）
B4 < B1 − 3 pp                       →  注入有害,回退并重新设计
```

---

## 八、待决策 / 未解决的问题

| # | 问题 | 现状 |
|---|---|---|
| 1 | **B4 臂定义用 (a) 还是 (b)** | 未定,建议 (a) |
| 2 | **B2 的预期** | 用户希望 B2 < B1 且 B2 < B3（证明「需要专门模块」）。但按现有证据预测 **B2 ≈ 35.3%** —— 若真的明显偏低,反而会推翻「码完全惰性」的结论,是重要发现。**不应朝期望方向调,如实报告** |
| 3 | **test 的 ckpt 选择** | 用户定：B3/B4 可跨 ckpt 在 test 上选最优；B1 固定用与 B3 相同的 step（40000）。这是有意让给 B3/B4 的不对称,已接受 |
| 4 | **在线 planner 是否继续** | 三版都输给模板法（26.67 / 22.33 / v3.1 未测成）。诊断：注入指令落在训练分布内的比例 **模板法 100% vs 在线 56.6%**（r=+0.29,重要但非唯一原因）。v3.1 已实现候选指令清单等修复,**未验证** |
| 5 | `slide_block_to_color_target` | CSV 先验（5 段）、离线真值（2.7 段）、cache 质量（16.3% pose-adjust,断层第一）三方对不上。**用户已决定暂不处理** |
| 6 | `place_cups` / `put_groceries_in_cupboard` | 所有臂上几乎恒为 0,合计压低均值约 8 pp。与「独占原子样本稀缺」一致（`VLA_Design §7 R2`） |
| 7 | 细节码 82.8% 退化 | 是 Stage 0/1 的问题,修它要回到码本预训练。**用户已决定暂不讨论** |
| 8 | 未 commit | 全部改动都在工作区,**没有任何 commit**。用户明确要求过「不要打 git 原子提交」 |

---

## 九、用户设定的工作规则（必须遵守）

```
1. 设计文档仅供参考,不是约束
2. **分步规划,执行前必须先与用户确认** —— 曾因未经同意就执行被明确批评
3. 有异议/发现缺陷/有更好建议要提出讨论
4. **每次都要报告改了哪些文件、加了哪些文件**
5. 显卡：盘子上限 **4 张**（自己在跑的卡也计入）；只占空闲 > 30 GB 的卡；
   **绝不 kill 别人的进程,不影响别人的作业**
6. 目录约定：权重 `checkpoints/`,结果 `result/`,日志 `log/`；**不用 `runs/`**
7. 磁盘空间由用户负责,只需报告是否不够
8. VLM 调用有成本（实测 ¥0.0139/次,300 局约 ¥40）—— planner 迭代一律先用 120 局
```

---

## 十、设计文档索引

| 文档 | 内容 |
|---|---|
| `Exp_Design.md` | 实验设计 + **第七节：实测结果汇总** |
| `VLA_Design.md` | Stage 3 集成 / Planner / cache |
| `Adapter_Design.md` | Stage 2 Adapter 结构；**§5 码向量注入的接口契约** |
| `VQAP_Design.md` | Stage 0/1 码本预训练 |

---

## 附：机器与环境

```
conda env       aavla（py3.10, torch 2.4.1+cu121, numpy 1.26）
入口            cd source/peract && source ../../run/env.sh
GPU             8 × RTX 5880 Ada（49 GB）,共享机器,96 个用户
replay          aavla_data/replay/seen12/... 165,773 样本 / 202 GB / 三臂共享只读
磁盘            /data0 约 168 GB 可用（28T/29T）
显卡看门        python tools/gpu_reserver.py watch --interval 45
                （盘子 4 张；要用卡时 hold --gpus a,b,用完 unhold）
```
