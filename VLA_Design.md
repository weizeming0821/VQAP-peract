# VQAP–PerAct 系统设计（Stage 3）

> **本文档定位**：描述如何把预训练好的 **VQAP 双码本 + Adapter** 接入 **PerAct**，构建「Planner 子任务规划 → Adapter 预测码 → 码向量注入 PerAct」的系统，并在 RLBench 18 任务上训练与评测。
>
> **配套文档**
> - [VQAP_Design.md](VQAP_Design.md) —— Stage 0/1：数据集、双码本模型结构、预训练、损失、超参数
> - [Adapter_Design.md](Adapter_Design.md) —— Stage 2：Adapter 结构与训练；§5 定义了码向量注入 PerAct 的**接口契约与注入点**
> - [Exp_Design.md](Exp_Design.md) —— 实验执行计划
>
> **本文档覆盖范围**：§1 系统总览 · §2 任务集与数据 · §3 Planner 设计 · §4 离线 Cache 设计与生成 · §5 Cache 质量门禁 · §6 在线 Planner · §7 已知风险与待定项。
> **§8 Stage 3 模型集成与训练方案**、**§9 实验方案** 待后续讨论轮次补入。

---

## 一、系统总览

### 1.1 三阶段链路

```
[Stage 0/1] 码本预训练（AtomAction_Dataset：69 任务 / 56,496 phase / 17 原子）
   动作编码器 + 双码本 NSVQ + FM 重构解码器 + VASA
   → codebook.pth（Kg=36, Kd=192, N_detail=9, d_code=512）        ✅ 已完成，冻结

[Stage 2] Adapter（14.2 M 可训 + 冻结 DINOv2 ViT-B/14 与 CLIP ViT-B/16 文本塔）
   (front + wrist 图像, 子任务指令) → (k_global, k_detail[9])      ✅ 已完成，冻结

[Planner] 子任务规划                                                🆕 本文档 §3–§6
   离线：对 RLBench demo 逐 episode 调用，产出带子任务边界的训练 cache
   在线：评测时规则触发 + VLM 判定阶段切换

[Stage 3] PerAct 集成                                               🆕 §8（待补）
   冻结：PerAct 主干、CLIP RN50、码本、Adapter
   可训：注入层（FiLM + cross-attn）、PerAct 动作头、lang_preprocess
```

### 1.2 部署时序

```
任务指令 + 初始观测
   └─► [Planner] 初始规划 → 子任务序列
        └─► 每个子任务（= 一个原子动作）：
             ├─ [Adapter]（冻结，子任务起始时调用一次）→ k_g, k_d[9]
             ├─ [码本查表]（冻结）→ z_g ∈ R^512, Z_d ∈ R^{9×512}
             │    ※ use_codebook=false → 注入层输出恒等
             └─ [PerAct + 注入] 逐关键帧执行，段内复用同一组码向量
        └─► [Planner] 阶段判定 → CONTINUE / NEXT / RETRY / REPLAN
```

**码的生命周期**：一个子任务内保持不变。`NEXT` / `RETRY` / `REPLAN` 均视为新的子任务起始点，**重新调用 Adapter**（`RETRY` 时环境状态已变，沿用旧码不合理）；`CONTINUE` 不重新调用。

---

## 二、任务集与数据

### 2.1 RLBench 版本与任务集

| 项 | 取值 |
|---|---|
| RLBench | **`MohitShridhar/RLBench`（PerAct fork）** |
| 任务集 | **PerAct 论文的 18 个任务** |
| 预训练权重 | 官方 `peract_600k`（`source/peract/peract_600k.zip`） |
| 数据生成 | `source/RLBench/rlbench/dataset_generator.py`（官方脚本） |

> **⚠️ 必须使用 PerAct fork**：官方 18 任务中的 `slide_block_to_color_target`（4 variations）、`sweep_to_dustpan_of_size`（2）、`place_wine_at_rack_location`（3）**只存在于 fork**，上游 `stepjam/RLBench` 中只有单 variation 的同族任务（`slide_block_to_target` / `sweep_to_dustpan` / `stack_wine`，均为 `variation_count()=1`）。二者**不是改名关系**——官方版本正是靠多 variation 做语言 grounding，换成单 variation 版会削掉这层难度。

### 2.2 ⚠️ 官方 ckpt 的实际训练范围

`peract_600k` 的 `config.yaml` 实测显示它训练在 **21 个任务**上（论文 18 个 + `change_channel` / `set_clock_to_time` / `put_rubbish_in_color_bin`），600k 步。

**连带后果：UnSeen 必须从这 21 个任务之外选取**（2026-09 改版，见 §2.4）。

> **【声明 A′｜UnSeen 的定义】** UnSeen 指 **PerAct 主干从未见过的任务**——既不在 `peract_600k` 的 21 任务预训练集中，也不在 Stage 3 微调集中。所有臂在 UnSeen 上**零训练**。
>
> **【声明 B′｜码本预训练范围】** 码本与 Adapter 在 `AtomAction_Dataset`（69 任务 / 17 原子）上预训练。UnSeen 的 **Tier-A 6 个在**该 69 任务内（码本见过其原子片段），**Tier-B 2 个不在**。Tier-A 检验原子知识的跨任务迁移（主结果），Tier-B 排除「码本恰好见过该任务」这一替代解释（对照）。

> 旧【声明 A】（「UnSeen 指未参与 Stage 3 微调的任务」）随本次改版**作废**：它承认主干已见过 UnSeen 的完整轨迹，主张只能停在「组合泛化」，撑不起论文要的「码本帮助模型泛化到未见任务」。

### 2.3 ⚠️ 官方 ckpt 与仓库默认配置的差异

Stage 3 必须按官方 ckpt 的配置构建模型，否则权重加载失配：

| 项 | 官方 ckpt | `conf/method/PERACT_BC.yaml` 默认 | 处置 |
|---|---|---|---|
| `pos_encoding_with_lang` | **false** | True | **必须改为 false**（架构不同，否则加载错误） |
| `transform_augmentation.aug_rpy` | `[0, 0, 0]` | `[0, 0, 45]` | 按需对齐，三臂一致即可 |
| `replay.batch_size` | 16 | 8 | 三臂一致即可 |
| `training_iterations` | 600001 | 40000 | Stage 3 另定 K |

### 2.4 任务划分（Seen 18 / UnSeen 8）

> **2026-09 改版**：原划分是在官方 18 任务**内部**按「是否参与 Stage 3 微调」切成 Seen12 / UnSeen6。
> 实测 `peract_600k` 的 `config.yaml` 后确认：主干在 600k 预训练中见过全部 **21** 个任务（含官方 18 个全部），
> 因此原 UnSeen6 只能声称「未参与 Stage 3 微调」，撑不起「未见过的任务」这一主张——【声明A】已如实标注过该短板，
> §7 T1 也把「是否重新划分」列为待定。本次改版把 UnSeen 整体移出官方 18 任务之外，兑现 T1 的待办。

**UnSeen 的检验目标**：**码本能否让策略在主干从未见过的任务上获得增益**。判读以 **B3 − B2** 为核心（码本净贡献，已扣除子任务分割的功劳）。

| | 任务 | 主干见过 | 码本见过 | Stage 3 微调 |
|---|---|---|---|---|
| **Seen 18** | 官方 PerAct 18 任务全集 | ✅ 600k 预训练 | 13/18 | ✅ |
| **UnSeen · Tier-A**（主结果） | `close_drawer`, `pick_up_cup`, `open_jar`, `phone_on_base`, `lamp_on`, `basketball_in_hoop` | ❌ | ✅ 该任务的原子片段 | ❌ 零训练 |
| **UnSeen · Tier-B**（对照） | `take_money_out_safe`, `take_lid_off_saucepan` | ❌ | ❌ | ❌ 零训练 |
| **候选备选** | `press_switch`, `put_knife_on_chopping_board` | ❌ | ✅ | ❌ 零训练 |

**Tier-A 检验原子知识的跨任务迁移；Tier-B 排除「码本恰好见过该任务」这一替代解释。两层分列报告，不合并求均值。**

UnSeen 逐任务详情：

| Tier | 任务 | 变体 | 原子序列 | 码本样本 | 与 Seen18 的关系 |
|---|---|---|---|---|---|
| A | `close_drawer` | 3 | approach→push | 900 | 同物体反向（`open_drawer` 官方 92%） |
| A | `pick_up_cup` | 5 | approach→grasp→lift | 1500 | 新物体，颜色 grounding，3 相 |
| A | `open_jar` | 5 | grasp→rotate→lift→transfer→place | 2500 | 同物体反向（`close_jar` 官方 56%） |
| A | `phone_on_base` | 1 | approach→grasp→lift→transfer→place | 500 | 新物体，标准 pick-place |
| A | `lamp_on` | 1 | approach→press | 200 | 新物体，2 相极短 |
| A | `basketball_in_hoop` | 1 | approach→grasp→lift→transfer | 400 | 新物体，目标容差大 |
| B | `take_money_out_safe` | 3 | 待标注 | — | 同物体反向（`put_money_in_safe` 官方 32%） |
| B | `take_lid_off_saucepan` | 1 | 待标注 | — | 新物体，3 相极短 |

**五条硬约束的核验结果**：

| # | 约束 | 结果 |
|---|---|---|
| ① | UnSeen ∩ `peract_600k` 的 21 个预训练任务 = ∅（主干确实没见过） | ✅ 已核 `config.yaml` |
| ② | UnSeen 与 21 任务无**近重复**（同物体同动作、仅换措辞） | ✅ 已剔除 `put_rubbish_in_bin`(↔`put_rubbish_in_color_bin`)、`sweep_to_dustpan`(↔`sweep_to_dustpan_of_size`)、`change_clock`(↔`set_clock_to_time`)、`push_button`(↔`push_buttons`) |
| ③ | UnSeen 原子 ⊆ Seen18 的 13 种原子（保证是组合泛化，而非「没学过这个动作」的平凡失败） | ✅ Tier-A 已核；⬜ Tier-B 待数据生成后补标 |
| ④ | Tier-A ⊆ AtomAction 69 任务；Tier-B ∩ AtomAction 69 = ∅ | ✅ 已核 `dataset_metadata.json` |
| ⑤ | Seen18 覆盖 18 任务的全部 13 种原子 | ✅ 取全集，天然成立 |

> 划分改动后必须重跑本核验。**最终 8 个从 10 个候选中按事先冻结的规则筛出**（先剔 E1 天花板为 0 者，再剔 B0 零样本 sr≥90% 者；筛选不看 B2/B3 任何结果），见 `Exp_Design` 声明E。

### 2.5 数据规模与划分

| 数据集 | split | 规模 | 状态 | 是否建 cache | 用途 |
|---|---|---|---|---|---|
| **Seen 18** | train | 18 × 100 = **1800 ep** | ✅ 已生成 | ✅ 18 任务全部已建 | Stage 3 训练，**全部 1800 条都读** |
| | val | 18 × 25 = 450 ep | ✅ 已生成 | 🔶 抽 ~50 条 | cache 质量抽检 + checkpoint 选型（选型走真实 rollout，不读 cache） |
| | test | 18 × 25 = 450 ep | ✅ 已生成 | ❌ 不建 | E0 / E1 / E2 的最终数字 |
| **UnSeen 候选 10** | test | 10 × 25 = **250 ep**（≈11.7 GB） | ⬜ 待生成 | ❌ 不建 | E1 天花板 + E3 零样本评测 |
| | train / val | — | ❌ **不生成** | — | UnSeen 不参与任何训练，见下注 |

> **⚠️ UnSeen 不建 train 的连带后果**：评测 B2/B3 需要子任务指令，而 §6 的确定性模板 planner 的模板**取自 train split 的离线 cache** —— UnSeen 没有 train cache 就没有模板。
> **已决定改用 VLM 在线 planner**（具体方案在 B2/B3 重新训练时确定）。若最终仍回退到模板 planner，则须为 UnSeen 各补 ~25 条 train demo **专用于建模板库**（绝不进 replay，由下述断言硬隔离）。
> 注意该决定**只影响评测**：Stage 3 训练读的是离线 cache，18 个任务的 cache 已全部建好。

**⚠️ 防泄漏硬约束**：训练启动时**断言**：读到的 `task` 集合 ⊆ 配置声明集合（**必须恰好是 Seen18，UnSeen 任一任务出现即失败**），且 episode 数 == 预期数（Seen18 → 恰好 1800）。必须是**启动即失败**式，不能只写日志。

> 依据：`Exp_Design.md` 风险 11 记录过一次真实事故——数据集子集过滤的索引错位导致**静默跨 episode 取数据**，不报任何错。过滤类逻辑必须有硬校验。

---

## 三、Planner 设计

### 3.1 职责边界

| Planner 负责 | 不负责 |
|---|---|
| 把任务分解为有序子任务序列 | 任务级成功/失败/超时判定（RLBench 仿真判） |
| 判断子任务何时切换 / 重试 / 重规划 | 关键帧提取（由 PerAct 的 `keypoint_discovery` 负责） |
| 审核候选关键帧、剔除脏数据 | 修改关键帧编号（**硬禁止**，见 §4.2） |
| 从 17 类白名单中选定 `action`、生成 `instruction` | — |

**子任务定义**：一个子任务 = 一个原子动作 = 一段**连续的 PerAct 关键帧**。

### 3.2 指令格式：一条 canonical 字符串同时喂两个消费方

Adapter 与 PerAct **吃同一条子任务指令字符串**，不能各来一套。

```
canonical 格式：全小写、无句尾标点、3~8 词祈使句
例：  "grasp the jar lid"      "approach the red jar lid"      "push the lid down"
```

两条 tokenization 路径均可接受该格式：

| | Adapter | PerAct |
|---|---|---|
| 文本塔 | CLIP **ViT-B/16** text tower | CLIP **RN50** |
| 分词器 | `helpers/clip/core/simple_tokenizer.py`（强制 `.lower()`） | **同一个** |
| 预训练指令分布 | AtomAction short：`"Grasp the jar lid."` | RLBench 原生：`"close the red jar"` |
| 归一化 | `Adapter_Design §4.3.2`：**去尾句号** | 无需处理 |
| 结果 | `"Grasp the jar lid."` → `"grasp the jar lid"` ✅ | 同分布 ✅ |

**唯一的实质分布偏移**：PerAct 预训练见的是**整任务**指令，我们喂**子任务**指令。由 `Adapter_Design §5.4` 把 `lang_preprocess`（`Linear(512→128)`，65 K，语言进入冻结主干的唯一门户）设为**可训**来吸收。

**指令内容要求**：必须保留任务指令中的区分性信息（颜色、序号、方位）——官方 18 任务中有多个靠 variation 做语言 grounding（`slide_block_to_color_target` 4 色、`place_wine_at_rack_location` 3 个架位、`sweep_to_dustpan_of_size` 2 种尺寸），丢掉这些信息会让子任务指令无法区分 variation。

### 3.3 `use_codebook` 规则

| 条件 | 取值 |
|---|---|
| `action` ∈ 17 类原子动作白名单 | **true** |
| 其余全部（含 `pose-adjust`、任何白名单外标签） | **false** |

白名单：`approach, grasp, lift, place, push, pull, press, rotate, slide, insert, hang, wipe, flip-open, flip-close, revolve-in, revolve-out, transfer`

`use_codebook=false` 时注入层输出**恒等**（`Adapter_Design §5.3` 红线 3：必须有单元测试断言此时输出与原版 PerAct **bit-exact**）。

> `hang` / `insert` 在 AtomAction 中样本稀缺（`hang` 仅 100 条且只来自 1 个任务、`insert` 283 条），但确属白名单，照常调用码本。该风险见 §7 R2，留作后续消融点。

---

## 四、离线 Cache 设计

### 4.1 设计目标

Cache 的唯一目的：为 Stage 3 训练提供**带子任务边界的干净数据**，同时喂饱三个消费方，并避免在训练循环中实时调用 Planner API。

```
                  ┌─► Adapter     子任务起始帧 (front, wrist) + 子任务指令 → (k_g, k_d[9])
cache 的一条 segment ─┼─► PerAct      观测 + 子任务指令（替换原整任务指令）
                  └─► 注入模块    查表得到的 (z_g, Z_d)
```

### 4.2 核心结构：子任务 = PerAct 关键帧的连续区间

**分割候选必须来自 PerAct 自己的 `keypoint_discovery`**，不得使用其它关键帧算法。

理由：PerAct 的训练样本是 `(某帧观测 → 下一个关键帧位姿)`（见 §4.3）。若子任务边界用另一套算法的帧号，会出现"边界落在两个 PerAct 关键帧之间"的情况，样本无法干净归属子任务。用同一套关键帧则**边界天然是关键帧边界，零对齐误差**。

```
demo → keypoint_discovery(heuristic) → [kp_0, kp_1, kp_2, kp_3, kp_4, kp_5]
                                        └─seg_0─┘  └seg_1┘  └───seg_2───┘
                                         approach    grasp        lift
```

> **🔴 硬约束**：Planner **不得新增、删除或移动关键帧编号**，只能把给定关键帧归组。关键帧列表必须与 Stage 3 训练时 `keypoint_discovery(demo)` 的输出**逐位相同**，否则本节的同构性即失效（校验项 C5，§5.1）。

### 4.3 PerAct 训练样本结构与子任务归属规则

`agents/peract_bc/launch_utils.py::fill_replay` / `_add_keypoints_to_replay` 的实际行为：

```python
episode_keypoints = keypoint_discovery(demo)
for i in range(0, len(demo)-1, demo_augmentation_every_n):   # 每 10 帧一个起始帧
    obs = demo[i]
    for keypoint in [k for k in episode_keypoints if k > i]:
        replay.add(obs → action(demo[keypoint]), lang=description)
        obs = demo[keypoint]        # obs 前进到该关键帧
```

三条性质：

1. **样本是链式相邻转移**：`(某帧 → 紧邻的下一个关键帧)`，`obs` 与 `target` 永远相邻。
2. **语言是逐样本存储的**（`obs_dict['lang_goal_emb']` / `lang_token_embs` 由 `description` 现算）。换成子任务指令**不需要改 PerAct 结构**，只需逐样本传不同的 `description`。
3. `demo_augmentation_every_n=10` 决定样本总量。**三臂沿用同一默认值 10**，对比即公平。

**归属规则**：

> 一个训练样本的子任务 = **其 target 关键帧所属的 segment**。

- `obs` 最多落后一个 segment；跨边界样本（`obs` 在 seg_k 末尾、`target` 是 seg_{k+1} 首帧、指令用 seg_{k+1} 的）**恰好对应部署时 Planner 刚说完 `NEXT` 的那一刻**，语义正确。
- **样本总数与 baseline 完全相同**，三臂对比不因样本量差异被污染。

### 4.4 码向量的计算时机

每个 `(episode, segment)` 预计算**一组**码：

```
Adapter 输入 = 该 segment 起始帧的 (front, wrist) 图像 + 该 segment 指令
            → (k_global, k_detail[9])，写入 cache
```

segment 内所有样本共享同一组码，与部署时「切换时调用一次、段内保持不变」严格一致。开销 ≈ 1800 episode × ~5 段 ≈ 9000 次 Adapter 前向，可忽略。**不在训练 step 中实时计算**——Adapter 冻结、输入固定，实时计算是纯浪费。

### 4.5 `terminal` / `reward` 保持原样

`launch_utils.py:172-173` 中 `terminal = (k == len(episode_keypoints)-1)` 是 **episode 级**的。引入子任务后是否要改成段级？

**实测结论：不改。** `qattention_peract_bc_agent.py::update()` 只读取
`trans_action_indicies` / `rot_grip_action_indicies` / `gripper_pose` / `ignore_collisions` / `lang_goal_emb` / `lang_token_embs` / `low_dim_state` / obs、pcd。
`terminal` 与 `reward` **只被写入 replay，从不被读取**——PerAct BC 是纯行为克隆，损失只有 trans / rot / grip / collision 四个 CE，没有任何 TD 项。两字段是死数据。

### 4.6 Replay Buffer 字段扩展

`create_replay` 的 `extra_replay_elements` 需新增三个字段：

```python
ReplayElement('k_global',     (),    np.int32)
ReplayElement('k_detail',     (9,),  np.int32)
ReplayElement('use_codebook', (),    np.bool)
```

### 4.7 Cache Schema

```json
{
  "meta": {
    "schema_version": "vqap_peract_cache_v1",
    "rlbench_fork": "MohitShridhar/RLBench", "rlbench_commit": "...",
    "keypoint_method": "heuristic",
    "task_set": "peract_official_18",
    "planner_model": "...", "planner_prompt_sha256": "...",
    "instruction_format": "lowercase_no_punct_3to8w",
    "adapter_ckpt_sha256": "...", "codebook_ckpt_sha256": "..."
  },
  "episodes": [{
    "task": "close_jar", "variation": 3, "episode": 7, "split": "train",
    "keypoints": [12, 31, 47, 68, 95, 141],
    "task_instruction": "close the red jar",
    "segments": [{
      "segment_index": 0,
      "action": "grasp",
      "instruction": "grasp the red jar lid",
      "keypoint_indices": [0, 1],
      "start_frame": 0,
      "k_global": 12,
      "k_detail": [3, 88, 41, 7, 155, 20, 91, 3, 62],
      "k_global_oracle": 12, "oracle_agree": true,
      "use_codebook": true
    }],
    "prior_alignment": "exact",
    "prior_deviation_reason": null,
    "anomalies": [],
    "episode_quality": "ok"
  }]
}
```

- `start_frame` = 该 segment 的起始帧（= 上一段末关键帧；首段为 0），Adapter 的图像输入取自此帧。
- `k_global_oracle` / `oracle_agree` 为诊断字段，见 §7 D1。

### 4.8 生成流程

```
① RLBench demo（官方 dataset_generator.py 生成，或官方预生成数据集）
② keypoint_discovery(demo, method='heuristic') → 候选关键帧列表
③ 渲染各关键帧的 front / wrist 图像 + 夹爪状态
④ 查 Phase_Action_Label.csv 得到该 (task, variation) 的参考动作序列
⑤ 调用 Planner（每 episode 一次）→ segments + 审核结论
⑥ 对每个 segment 调用冻结 Adapter → (k_global, k_detail)
⑦ 按 §3.3 规则写入 use_codebook
⑧ 写出 cache JSON → 过 §5 的三道质量门禁
```

**API 调用量**：1800（train 全量）+ ~50（val 抽检）≈ **1850 次**。

### 4.9 Phase_Action_Label.csv：**软**先验

CSV 提供每个 `(task, variation)` 的人工标注动作序列，作为 Planner 的参考。

**当前覆盖情况**：官方 18 任务中 **15 个已覆盖**；3 个缺失（`slide_block_to_color_target` / `sweep_to_dustpan_of_size` / `place_wine_at_rack_location`，因任务名随 fork 变更）。这 3 个可用同族旧任务的序列作为初始近似：

| 缺失任务 | 同族旧任务参考序列 |
|---|---|
| `slide_block_to_color_target` | `approach → press` |
| `sweep_to_dustpan_of_size` | `approach → grasp → transfer → wipe` |
| `place_wine_at_rack_location` | `approach → grasp → transfer → insert` |

> ⚠️ 这 3 条是近似，**需在实际数据上核对后补入 CSV**。

**为何是软先验**：PerAct 关键帧由「夹爪翻转 + 速度趋零」触发，CSV 的 phase 由人工按语义标注，**两者数量和语义都不保证对应**（即使段数相同，逐段语义也可能错位）。因此 Planner 允许偏离，但必须显式声明：

```json
"prior_alignment": "exact" | "modified" | "not_applicable",
"prior_deviation_reason": "先验第3段 lift 在本 episode 中无独立关键帧，已并入 transfer"
```

所有 `modified` 的 episode 进入 §5.3 的重点抽检队列。

### 4.10 离线 Planner Prompt

**System**

```
你是 RLBench 机械臂演示轨迹的动作分段审核员。
给定一条演示的候选关键帧（已由 PerAct 的关键帧算法自动提取），
把它们归组为若干个连续的原子动作段。

硬约束：
1. 不得新增、删除或修改关键帧编号，只能把给定的关键帧分配到动作段。
2. 每个关键帧必须且只能属于一个段；段按时间顺序连续，不得交叉或留空。
3. action 必须取自以下 17 类白名单（另允许 pose-adjust 用于姿态微调段）：
   approach, grasp, lift, place, push, pull, press, rotate, slide,
   insert, hang, wipe, flip-open, flip-close, revolve-in, revolve-out, transfer
4. instruction 格式：全小写、无句尾标点、3~8 词祈使句，风格对齐 RLBench
   官方指令（如 "close the red jar"）。必须保留任务指令中的区分性信息
   （颜色、序号、方位），否则无法区分同一任务的不同 variation。

动作定义（务必按此口径判定，与码本训练时的人工标注标准一致）：
  approach : 空载接近目标，夹爪尚未接触物体
  grasp    : 夹爪由开变闭并与目标物体建立接触
  lift     : 已持物，竖直方向抬离原位置
  transfer : 已持物，水平方向移动至目标区域
  place    : 已持物，下放并释放
  press    : 以末端或持有物向下/向前施压触发目标
  pull     : 沿直线方向拉动铰接或滑动部件
  push     : 沿直线方向推动物体
  rotate   : 绕轴旋转目标物体
  slide    : 使物体在平面上滑移
  insert   : 将持有物沿轴向插入目标孔位
  hang     : 将持有物挂置于支撑结构上
  wipe     : 以持有工具做往复清扫
  flip-open / flip-close : 翻转打开 / 合上盖状结构
  revolve-in / revolve-out : 绕铰链向内 / 向外转动门状结构

关于参考序列：会给你一个来自人工标注的「期望动作序列」作为参考。但关键帧
算法的粒度与它不一定一致——数量可能不同，同一段的语义也可能不对齐。
请以实际图像为准：
  - 能对齐        → prior_alignment = "exact"
  - 需合并/拆分/改标签 → "modified"，并在 prior_deviation_reason 说明
  - 不要为了凑参考序列而强行分段。

输出严格 JSON，不得有额外文字。
```

**User**

```
任务：{task_name}   variation：{variation_index}
官方任务指令：{variation_descriptions}
参考动作序列（人工标注先验，仅供参考）：{Phase_Action_Label.csv 查表结果}

候选关键帧共 {M} 个（按时间顺序）：
  [0] frame=12  gripper=open    <front 图> <wrist 图>
  [1] frame=31  gripper=closed  <front 图> <wrist 图>
  ...
初始场景（frame=0）：<front 图>

请把这 {M} 个关键帧归组为动作段。
```

**输出 schema**

```json
{
  "segments": [
    {"segment_index": 0, "action": "grasp",
     "instruction": "grasp the red jar lid",
     "keypoint_indices": [0, 1], "confidence": "high"}
  ],
  "prior_alignment": "modified",
  "prior_deviation_reason": "...",
  "anomalies": [{"type": "redundant_keypoint", "keypoint_index": 3, "detail": "..."}],
  "episode_quality": "ok"
}
```

`episode_quality ∈ {ok, suspect, reject}`；`reject` 的 episode 从 Stage 3 训练集剔除。

---

## 五、Cache 质量门禁

Planner 输出**不得直接投入训练**，必须依次通过三道关。

### 5.1 第一道 · 自动结构校验（全量，任一项失败即 reject）

| # | 检查项 |
|---|---|
| C1 | `keypoint_indices` 构成 `[0, M)` 的**完整无重叠划分**（并集 == 全集，两两不交） |
| C2 | `segments` 按 `segment_index` 递增，`keypoint_indices` 时间有序、不交叉 |
| C3 | `action` ∈ 17 类白名单 ∪ `{pose-adjust}` |
| C4 | `instruction` 满足 canonical 格式（全小写 / 无尾标点 / 词数 ∈ [3,8]） |
| C5 | **关键帧列表与 `keypoint_discovery(demo)` 的实际输出逐位相同**（防止 Planner 改帧号或代码版本漂移） |
| C6 | `k_global ∈ [0,36)`；`k_detail` 长度 9 且各元素 ∈ [0,192) |
| C7 | `use_codebook` 与 §3.3 规则一致 |

### 5.2 第二道 · 统计一致性体检（全量，产出报告供人工判读）

| 指标 | 健康信号 | 异常信号与含义 |
|---|---|---|
| 每 `(task, variation)` 的 segment 数分布 | 集中在 1~2 个值 | 方差大 → Planner 判定不稳定 |
| `prior_alignment` 中 `modified` 占比 | 低 | 偏高 → 关键帧与 CSV 粒度系统性错配，需回头审视 §4.2 假设 |
| 每 segment 的关键帧数分布 | 多为 1~3 | 大量 0 或 >5 → 归组粒度有问题 |
| 同 `(task, variation)` 内指令的唯一值数 | 少数几种 | 过多 → 措辞漂移，会污染 Adapter 输入分布 |
| `action` 序列 vs CSV 先验的编辑距离 | 中位数 ≤ 1 | 偏大 → 语义口径漂移 |
| **同 `(task, variation, segment_index)` 内 `k_global` 的众数占比** | > 80% | 接近均匀 → Adapter 在新域上输出近乎随机（§7 D1 红灯） |
| **`oracle_agree` 比例** | 高 | 低 → §7 R1 的语义错位坐实 |
| `suspect` / `reject` 占比 | < 5% | 偏高 → prompt 或关键帧算法有问题 |

### 5.3 第三道 · 人工抽检

- **分层抽样**：每个 task 至少 3 条；`prior_alignment == "modified"` 与 `episode_quality == "suspect"` 的**全部**进入候选池。规模约 60~100 条 episode。
- **判读材料**：每条 episode 渲染为一张对照图——关键帧图像横排，下方标注 Planner 给出的 segment 归组与指令。
- **回填**：`human_check ∈ {pass, fail, borderline}`。
- **放行门槛**：抽检通过率 **≥ 95%** 才允许进入 Stage 3 训练；否则修正 prompt 后重跑。

三道关的产物（结构校验报告 + 统计报告 + 抽检记录）随 cache 一并归档，作为数据可信度凭证。

---

## 六、在线 Planner（评测阶段）

### 6.1 与离线的关系

| | 离线（建 cache） | 在线（评测） |
|---|---|---|
| 输入 | 完整录制的 demo，可见全部未来帧 | 当前 rollout 的实时观测，无未来信息 |
| 边界确定方式 | 后验归组（确定性可复核） | 因果判定（需 VLM） |
| 指令来源 | 本节同一套格式规则生成 | **同一套格式规则**（同一常量字符串写入两侧 prompt） |
| 调用频率 | 每 episode 一次 | 触发条件命中时（~5–15 次/episode） |

**🔴 一致性硬约束**：在线输出的子任务指令必须与离线 cache **同格式、同分布**。保证方式：两侧共用同一段「指令格式规则」prompt 文本与同一套后处理代码路径。

### 6.2 触发条件

PerAct 逐关键帧执行（一个 episode 通常只有几到十几个关键帧），故触发条件按**关键帧**计，不按控制步数：

| 触发条件 | 初值 | 说明 |
|---|---|---|
| 夹爪状态翻转 | — | 最可靠信号，标志抓取/释放完成 |
| 连续 N 个关键帧目标位姿变化很小 | N = 2 | 判定卡住 |
| 已执行关键帧数 > 该子任务训练期望次数 × 富余系数 | 2× | 期望次数由 cache 统计得出 |
| 每 K 个关键帧强制检查 | K = 3 | 兜底 |

> 所有参数为**初值，待实测校准**。

### 6.3 判定与状态机

调用 VLM（多模态），传入 `front_rgb` + 结构化 Task Dashboard 文本，输出：

```json
{"phase_decision": "CONTINUE|NEXT|RETRY|REPLAN",
 "phase_decision_reason": "...",
 "current_subtask_status": "in_progress|looks_complete|stuck",
 "updated_plan": {"current_subtask": "...", "upcoming_subtasks": ["..."]}}
```

**Task Dashboard 结构**

```
## Task
"{task_instruction}"

## Progress
Completed: N subtask(s)
  ✓ {completed_subtask_1}
  ...

## Current Subtask
  "{current_subtask_instruction}"
  Keyframes executed: N
  Gripper: HOLDING object / EMPTY
  Trigger: gripper_flip / pose_stall / budget_exceeded / periodic

## Upcoming Subtasks
  1. {next_subtask}
  ...

## Image
Examine the attached front-view image and decide the next phase.
```

**`Upcoming Subtasks` 的来源**

| 任务类型 | 来源 |
|---|---|
| Seen 18 | 由该 `(task, variation)` 的离线 cache 统计出的典型子任务序列作为候选模板 |
| UnSeen 8 | 由 VLM 依据任务指令 + 17 类白名单自主规划（**没有 train cache 就没有模板**，见 §2.5 注；泛化测试本就不该提供模板） |

**Adapter 调用规则**：`NEXT` / `RETRY` / `REPLAN` → 重新调用 Adapter 预测码；`CONTINUE` → 复用当前码。

### 6.4 职责分界

| Planner 负责 | RLBench 仿真负责 |
|---|---|
| 初始分解为子任务序列 | 判定 episode 成功（reward > 0） |
| 执行中判断是否切换子任务 | 判定 episode 失败（超时、物体掉落） |
| 异常时建议重试或重规划 | 提供 ground truth 成功/失败信号 |

---

## 七、已知风险与待定项

### R1 🔴 码本语义错位（最大风险）

**风险形式**：Adapter 学的映射是「(AtomAction phase 起始帧, AtomAction short 指令) → 该 phase 轨迹经 VQAP 编码器得到的码」。Stage 3 喂给它的是「(PerAct demo 的 segment 起始帧, Planner 生成的指令)」。三处可能错位：

| 错位来源 | 说明 | 严重度 |
|---|---|---|
| **时间边界** | AtomAction 的 `grasp` phase 由人工核验分割定义；我们的 `grasp` segment 由 PerAct 关键帧归组定义。**同名不同区间** | 🔴 高 |
| **图像域** | AtomAction 256×256 vs PerAct 数据 128×128；且官方 18 任务中有 **5 个不在 AtomAction 中**（见 R2） | 🟡 中（`Adapter_Design §4.3.1` 的「统一降到 128 再升 224」管线已针对此设计） |
| **指令措辞** | AtomAction short 指令 vs Planner 生成指令 | 🟡 中 |

**缓解手段（已纳入设计）**：
1. §4.10 的 prompt 中给出 17 类动作的**操作性定义**，口径对齐 AtomAction 当初的人工标注标准。
2. §5.2 的统计体检监控「同语义组内码的众数占比」——**一致性比正确性更关键**：只要码是 (图像, 指令) 的确定性函数且对同类场景稳定，它就携带子任务信息，注入层能学会利用。

**红灯预案**：若一致性很差，可用 D1 的 oracle 码在 RLBench segment 上**微调 Adapter**（标签可自动生成，不动码本，成本远低于重训 Stage 0/1）。

### D1 诊断实验：码本语义错位的量化（⬜ 待详细设计）

**关键可行性**：RLBench demo 的 `low_dim_obs.pkl` 含 VQAP 编码器所需的全部字段（`gripper_pose` / `joint_positions` / `joint_velocities` / `joint_forces` / `gripper_joint_positions` / `gripper_touch_forces` / `gripper_open`）。因此对任意 segment 可**直接跑冻结的 VQAP 编码器**得到 oracle 码：

```
segment 轨迹 → AtomAction traj_stats 归一化 → VQAP 编码器 → argmin → (k_g*, k_d*)   [oracle]
segment 起始帧 + 指令 → 冻结 Adapter        → (k_g^, k_d^)                          [部署可得]
```

两者一致率即为 R1 的直接量化。

**三种码来源的取舍**：

| 方案 | 训练时 | 部署时 | 评价 |
|---|---|---|---|
| **(a) Adapter 预测**（**采用**） | Adapter(图像, 指令) | Adapter(图像, 指令) | **训练/部署零 gap** |
| (b) VQAP 编码器 oracle | Encoder(segment 轨迹) | ❌ 部署时无未来轨迹 | 训练/部署 gap 大 |
| (c) oracle 训练 + Adapter 部署 | Encoder | Adapter | gap 最大 |

**设计采用 (a)；(b) 仅作诊断量**（写入 cache 的 `k_global_oracle` / `oracle_agree` 字段），不进训练。

> 建议在正式生成 1800 条 cache **之前**先在 2~3 个任务 × 各 20 条 demo 上跑一次该诊断——它是唯一能低成本提前证伪整条码注入路线的实验。详细设计待后续轮次补入。

### R2 5 个部署任务不在 Adapter 训练集中

官方 18 任务中，**13 个**在 `AtomAction_Dataset`（70 任务）中，**5 个不在**：

| | 任务 |
|---|---|
| 在 AtomAction 中（13） | open_drawer, meat_off_grill, turn_tap, put_item_in_drawer, close_jar, reach_and_drag, stack_blocks, light_bulb_in, put_money_in_safe, put_groceries_in_cupboard, place_shape_in_shape_sorter, push_buttons, insert_onto_square_peg |
| **不在（5）** | `slide_block_to_color_target`, `sweep_to_dustpan_of_size`, `place_wine_at_rack_location`, `stack_cups`, `place_cups` |

**叠加风险**：`place_cups`（独占原子 `hang`）与 `place_wine_at_rack_location`（独占原子 `insert`）恰是"独占原子任务"，而 `hang` 在 AtomAction 全库仅 **100 条且只来自 1 个任务**、`insert` 仅 **283 条**。**最稀缺的原子，其部署载体任务恰好没见过。**

**处置**：记录风险，不为此调整训练数据。按 §3.3 仍照常调用码本，先跑出端到端结果，再决定是否对这些段关闭码注入——这本身构成一个现成的消融点。

**2026-09 补充｜UnSeen 侧的覆盖情况**：新 UnSeen 8 个任务中，**Tier-A 6 个全部在** AtomAction 69 任务内（码本见过其原子片段），**Tier-B 2 个全部不在**——这不是风险，而是 §2.4 刻意设计的两层对照（见【声明 B′】）。

### R3 码标签高度任务特异

`Adapter_Design §7 R3` 实测：跨任务迁移（用其它任务同 action 的众数预测本任务）的 `k_global` 命中率仅 **21.3%**，低于仅靠 action 的 25.0%；而 `(action, task)` 众数达 54.4%，逼近完整指令组的 56.8%。

**推论**：文本侧信号几乎全部来自"是哪个任务"，而非"是哪个原子"——与组合泛化的诉求相反。真正的杠杆在图像侧（冻结 DINOv2 的跨任务泛化能力）。

### R4 细节码 9 槽位在样本内恒等

`Adapter_Design §7 R1` 实测：全库 56,496 条记录，样本内 9 个细节槽位的平均唯一索引数 **1.00 / 9**（细节支路 cross-attention 退化为均匀平均池化）。

**影响**：`Adapter_Design §5.3` 中「让 9 个细节码获得空间分工」的 cross-attention 注入失去意义（$Z_d$ 的 9 行完全相同）；细节码消融的结论也会失去意义。

**处置**：接受现状，先跑通到 Stage 3 拿端到端结果。报告细节码指标时必须注明只有 1 个自由度。

### T1 ✅ 已解决（2026-09）：UnSeen 的信号强度不足

**原问题**：以官方 `peract_600k` 在 test 集上的成绩为参照，旧 UnSeen6 中只有 2 个任务（`put_item_in_drawer` 60%、`put_money_in_safe` 32%）落在有信号区间，其余 2 个触天花板（`turn_tap` 96%、`meat_off_grill` 92%）、2 个在地板（`insert_onto_square_peg` 8%、`stack_cups` 4%）。结论建立在 2 个任务上，偏薄。

| | 地板（≤8%） | 可用 | 天花板（≥90%） | 均值 |
|---|---|---|---|---|
| 旧 Seen 12 | 4 | **6** | 2 | 43.7% |
| 旧 UnSeen 6 | 2 | **2** | 2 | 48.7% |

**处置**：本条原文即写着「待定：是否重新优化 Seen/UnSeen 划分」。§2.4 的 2026-09 改版执行了这一待办——UnSeen 整体移出官方 18 任务，改从 21 个预训练任务之外选取，并按难度梯度挑选（避开天花板与地板），同时用【声明E】的事先冻结规则从 10 个候选中筛出最终 8 个。

**残留风险**：新 UnSeen 是主干真正没见过的任务，四臂**全部趴在地板**的可能性显著存在。三条缓解：
1. 8 个中有 3 个是「同物体反向」任务（`close_drawer` / `open_jar` / `take_money_out_safe`），与 Seen 共享物体与场景，是防地板的保险；
2. 定案前先跑 **E1 天花板** + **B0 零样本摸底**，剔掉本就执行不了的任务；
3. 10 个候选全部生成数据，主结果报定案的 8 个，另 2 个进附录——无论筛选结果如何都不必补生成。

### T2 待补：Phase_Action_Label.csv 的 3 个缺失任务

`slide_block_to_color_target` / `sweep_to_dustpan_of_size` / `place_wine_at_rack_location` 需在实际数据上核对后补入 CSV（§4.9 给出了同族任务的初始近似）。

> 遗留约束：`Phase_Action_Label.csv` 的先验查表存在「未标注 variation 时静默回退到该 task 第一条 variation」的机制。**今后任何新增 variation 都必须同步标注**，否则会无声地用错先验。建议在 cache 生成时加一条「命中回退路径即告警」的日志。

### T3 待实测：PerAct 关键帧数与 CSV 先验段数的匹配程度

目前无实测数据。若普遍是「关键帧数 ≫ 先验段数」，归组自然；若反过来（关键帧比段还少），说明两套粒度不兼容，§4.2 的方案需回炉。

**建议**：正式跑 1800 次 API 之前，先取 2~3 个任务各 5 条 demo 跑一次 `keypoint_discovery`，统计关键帧数分布并与 CSV 段数对照。成本极低、价值很高。

---

## 八、Stage 3 模型集成与训练方案

### 8.1 起点与配置对齐

| 项 | 取值 |
|---|---|
| 起点权重 | 官方 `peract_600k`（`ckpts/multi/PERACT_BC/seed0/weights/600000/QAttentionAgent_layer0.pt`） |
| **`pos_encoding_with_lang`** | **必须设 `false`** —— 官方 ckpt 用的是 false，仓库默认是 True，架构不同会导致权重加载失配 |
| `lang_fusion_type` | `seq`（官方） |
| 其余结构超参 | 对齐官方 ckpt 的 `config.yaml`（见 §2.3） |

> ⚠️ **语言是 bag-of-words**：`pos_encoding_with_lang=false` 时位置编码只加在体素网格上，语言 token **拿不到任何位置编码**（源码注释中作者自述为疏漏）。对本项目影响有限（子任务指令靠实词区分而非词序），但记录备查。

### 8.2 参数量实测与冻结 / 可训边界

从官方 ckpt 逐模块实测（127 个张量，state_dict 合计 39.25 M，其中 6.024 M 是 `SpatialSoftmax3D` 的 `register_buffer` 位置网格，**非可学习参数**）：

| 模块 | 参数量 | Stage 3 状态 |
|---|---|---|
| `layers`（6 层 latent self-attn） | 25.209 M | ❄️ 冻结 |
| `cross_attend_blocks` | 3.235 M | ❄️ 冻结 |
| `latents`（2048×512） | 1.049 M | ❄️ 冻结 |
| `pos_encoding`（20³×128 = 1,024,000） | 1.024 M | ❄️ 冻结 |
| `patchify` | 0.512 M | ❄️ 冻结 |
| `decoder_cross_attn` | 0.083 M | ❄️ 冻结 |
| `input_preprocess` / `proprio_preprocess` | 0.001 M | ❄️ 冻结 |
| CLIP RN50 / 码本 / Adapter | — | ❄️ 冻结 |
| **`up0`** | **1.536 M** | ✅ 可训 |
| **`dense0`** | **0.262 M** | ✅ 可训 |
| **`final`** | **0.221 M** | ✅ 可训 |
| **`lang_preprocess`**（`Linear(512→128)`） | **0.066 M** | ✅ 可训 |
| **`dense1`** | **0.016 M** | ✅ 可训 |
| **`rot_grip_collision_ff`** | **0.014 M** | ✅ 可训 |
| **`trans_decoder`** | **0.002 M** | ✅ 可训 |
| **注入层**（仅 B3） | **≈ 0.43 M** | ✅ 可训 |

**可学习参数总计 33.23 M；可训 = 2.117 M（B1/B2，占 6.4%）或 2.55 M（B3）。**

`lang_preprocess` 必须可训：整任务指令 → 子任务指令是语言输入分布的实质改变，而它是语言进入冻结主干的**唯一门户**。

### 8.3 注入点的精确位置

`latents` 是冻结主干之后的第一个张量，也是**唯一同时通向平移与旋转两个动作分支**的张量。其真实形状为 **`[B, 128, 20, 20, 20]`**（`input_dim_before_seq = im_channels × 2 = 128`；源码中 `# [B,20,20,20,64]` 的注释已过时）。

```python
latents = self.decoder_cross_attn(ins, context=x)
latents = latents[:, l.shape[1]:]                      # 裁掉语言部分 → [B,8000,128]
latents = latents.view(b, 20, 20, 20, 128)
latents = rearrange(latents, 'b ... d -> b d ...')     # [B,128,20,20,20]

# ★ 注入点：latents = CodeInjector(latents, z_g, Z_d, code_mask)

feats.extend([self.ss1(latents), self.global_maxp(latents)])   # → dense0 → 旋转/夹爪/碰撞
u0 = self.up0(latents)                                          # → final → trans_decoder → 平移
```

> 🔴 **必须插在 `rearrange` 之后、`feats.extend` 之前**。插在 `feats.extend` 之后会漏掉旋转/夹爪/碰撞分支。

### 8.4 注入模块

结构沿用 `Adapter_Design.md §5.3`：

```
(a) 全局码 → FiLM（通道维）
    c_g   = SiLU(W₂ · SiLU(W₁ · LN(z_g))) ∈ R^256
    [γ,β] = W_film · c_g ∈ R^{2×128}
    h     = latents ⊙ (1+γ)[:,:,None,None,None] + β[...]

(b) 细节码 → cross-attention（空间维：8000 体素 token × 128 维）
    S = Z_d + slot_embed ∈ R^{9×512}
    Q = W_q · LN(h_tok) ∈ R^{8000×128};   K = W_k·S,  V = W_v·S ∈ R^{9×128}
    O = W_o( softmax(QKᵀ/√d_h) · V ),     H = 4 头,  d_h = 32
    latents' = h + g ⊙ O,                 g ∈ R^128 可学习门
```

**三条实现红线**

1. `W_film` 的 weight 与 bias 全部 **zeros-init** → 初始 γ = β = 0。
2. ⚠️ **`W_o` 用 small-normal 初始化，只把 `g` 置零**。若 `W_o = 0` 则 `O ≡ 0`，而 `∂L/∂g ∝ O`，`g` 的梯度恒为零，**这条支路会永久失活**。
3. `code_mask = 0` 时强制 γ = β = 0 且 `g ⊙ O = 0`。必须有单元测试断言此时输出与冻结的原版 PerAct **bit-exact**。

**已知通路强弱差异**：`SpatialSoftmax3D` 对每通道在空间维做 softmax，通道级 FiLM 的 β 在空间上是常数、**在 softmax 中被完全抵消**，γ 只起逐通道温度作用。码进入旋转分支的完整路径是 `global_maxp(latents)`（128 维，完整）+ `ss1`（384 维，仅 γ 的温度效应）+ `ss_final(u)` / `global_maxp(u)`（256 维，经 `up0`/`final` 的卷积与 LReLU 后完整）。**若实验出现「平移明显改善、旋转几乎不动」，第一嫌疑就在这里**，届时可补一个 `dense0` 输出上的 FiLM（0.13 M，随时可加）。

细节码消融只需关掉 (b) 支路，不影响 (a)。

### 8.5 四臂定义

| 臂 | 语言条件 | 码注入 | 可训参数 | 训练 |
|---|---|---|---|---|
| **B0｜官方原样** | 整任务指令 | 无 | — | **零训练**，仅评测 |
| **B1｜baseline** | 整任务指令 | 无 | 2.117 M | 微调 K 步 |
| **B2｜planner-only** | **子任务指令** | 无 | 2.117 M | 微调 K 步 |
| **B3｜ours** | 子任务指令 | FiLM + cross-attn | **2.55 M** | 微调 K 步 |

**B0 的作用**：评测未经任何改动的官方权重，用于**验证评测管线的正确性**——若 B0 复现不出已发表成绩（§2.2 的成绩表），说明问题出在评测环境/协议，而非方法。近乎零成本，且同时作为「微调是否有帮助」的参照基线。

**为何必须有 B2**：B3 相对 B1 多拿了「任务被拆成子任务」这一额外信息。没有 B2 就无法区分增益来自码本还是来自子任务分割。**只有 B3 > B2 才能证明码本本身有效**；B2 > B1 是 Planner 的功劳，不计入 VQAP 的贡献。

**公平性硬约束**（B1/B2/B3）：同 K 步、同数据、同 batch size、同 LR schedule、同种子、同评测协议。论文报告各臂参数量。

### 8.6 共享 Replay Buffer

**决策：三臂共用同一份 replay，各臂按需读取字段。**

理由：
- **磁盘**：**实测**单样本 1.25 MB（含 subtask 字段）。Seen12（1200 demo）实测 165,773 样本 / **203 GB**；改版后的 **Seen18（1800 demo）为 239,871 样本 / 292.8 GB**（逐任务数由 `keyframe_stats.json` + planner cache 独立复算，公式已在 4 个任务上与实测逐位核对）。建三份不可接受。共享方案的额外开销仅 **+10%**（多一套子任务语言嵌入）。
  > `replay_capacity` 为 300,000，Seen18 的 239,871 仍有 20% 余量，无需调整。
- **公平性**：三臂在同一种子下看到**逐条相同、顺序相同**的样本，比重建三次强得多。

#### 8.6.1 字段命名规约（防混淆）

> **规约：所有 Stage 3 新增字段一律加 `subtask_` 前缀。凡带该前缀的字段，B0 / B1 一律不得读取。**

| 字段 | 形状 / 类型 | 内容 | 消费方 |
|---|---|---|---|
| `lang_goal_emb` | `(1024,)` f32 | **整任务**指令的 CLIP 句向量（PerAct 原生字段，语义不变） | B0, B1 |
| `lang_token_embs` | `(77, 512)` f32 | **整任务**指令的 CLIP token 向量 | B0, B1 |
| `lang_goal` | `(1,)` object | 整任务指令原文（调试用） | B0, B1 |
| `subtask_lang_goal_emb` | `(1024,)` f32 | **子任务**指令的 CLIP 句向量 | B2, B3 |
| `subtask_lang_token_embs` | `(77, 512)` f32 | **子任务**指令的 CLIP token 向量 | B2, B3 |
| `subtask_lang_goal` | `(1,)` object | 子任务指令原文（调试用） | B2, B3 |
| `subtask_k_global` | `()` i32 | 全局码索引 ∈ [0, 36) | B3 |
| `subtask_k_detail` | `(9,)` i32 | 细节码索引，各 ∈ [0, 192) | B3 |
| `subtask_code_mask` | `()` bool | 由 cache 的 `use_codebook` 写入 | B3 |
| `subtask_index` | `()` i32 | segment 序号（诊断/分析用） | 分析 |
| `subtask_action` | `()` str | 原子动作标签（诊断/分析用） | 分析 |

**PerAct 原生字段一律不改名、不改语义**——`lang_goal_emb` 永远是整任务指令，避免"同名不同义"这类最难排查的错误。

#### 8.6.2 隔离机制（硬保证）

仅靠命名约定不够，需要运行时强制：

> 每个臂声明一个**允许读取的字段白名单**；`replay_sample` 经一层薄包装暴露给 agent，**访问白名单外的键直接抛异常**。

```
B0 / B1 白名单：PerAct 原生字段全集（不含任何 subtask_* 字段）
B2      白名单：PerAct 原生字段 − {lang_goal_emb, lang_token_embs, lang_goal}
                 + {subtask_lang_goal_emb, subtask_lang_token_embs, subtask_lang_goal}
B3      白名单：B2 白名单 + {subtask_k_global, subtask_k_detail, subtask_code_mask}
```

这样「B1 误读子任务指令」「B2 误读码」这类静默错误在第一次访问时即崩溃，不会悄悄训出一个说不清是什么的模型。

> 依据：`Exp_Design.md` 风险 11 记录过一次真实事故——数据过滤的索引错位导致**静默跨 episode 取数据**，不报任何错。凡是"多套数据共存、按配置选用"的结构，都必须有硬隔离。

#### 8.6.3 构建流程

```
① 读 cache（§4.7）→ 得到每个 episode 的 keypoints / segments / 指令 / 码
② fill_replay 逐 demo 遍历（沿用 PerAct 原生逻辑，keypoint_discovery 结果须与 cache 逐位一致）
③ 每生成一个样本：
     - 按 §4.3 归属规则定位其 target 关键帧所属 segment
     - 写入整任务语言嵌入（PerAct 原生路径）
     - 写入该 segment 的子任务语言嵌入 + 码 + code_mask + 诊断字段
④ 断言：任一样本的 subtask_* 字段不得为空；subtask_index 必须落在该 episode 的 segment 范围内
```

### 8.7 训练配方

| 项 | 取值 | 说明 |
|---|---|---|
| 损失 | **完全沿用 PerAct 原生**：平移体素 CE + 3 轴旋转 CE + 夹爪 CE + 碰撞 CE | **不新增任何损失项** |
| 优化器 | LAMB（官方） | |
| `num_warmup_steps` | 3000（官方） | |
| `lr_scheduler` | false（官方） | |
| **LR** | **先用官方 5e-4 起步，由 10k 探针确定最终值** | 从收敛 ckpt 微调，5e-4 可能偏大；一旦定值，三臂必须相同 |
| **K（训练步数）** | **先跑 10k 探针，据学习曲线定** | 三臂必须一致；论文报告学习曲线，以堵"没训到收敛"的质疑 |
| `batch_size` | 16（对齐官方 ckpt） | 三臂一致 |
| `demo_augmentation_every_n` | 10（默认） | 三臂一致 |
| SE3 增广 | 照常开启 | 码是旋转不变的语义条件，与增广无冲突 |
| **Code-dropout** | **关闭（p = 0）** | 见下 |

#### 关于关闭 Code-dropout

**理由**：部署时 Adapter 恒定运行、码恒可用；训练时码也恒可用。**不存在 train/test 不一致**，因而无需用 dropout 制造"无码"场景，关闭可少一个超参。

**天然的无码暴露仍然存在**：`subtask_code_mask = 0` 的样本（`pose-adjust` 及白名单外动作，§3.3）在训练中自然出现，注入层仍会见到 `code_mask = 0` 的情况，`code_mask` 的硬关断路径不会失去训练信号。

> ⚠️ **需实测**：Seen18 中 `pose-adjust` 只出现在 `put_groceries_in_cupboard` 与 `reach_and_drag`，占比可能很低。cache 生成后应统计 `code_mask = 0` 的样本比例；若接近 0，则 `code_mask` 关断路径实际未被训练到，需在单元测试（§8.4 红线 3）之外额外留意。
>
> **残余风险**：若日后需要"码本可选"的部署模式（例如对 R2 中的稀缺原子任务动态关码），届时需重新引入 code-dropout 并重训。

### 8.8 风险与待定

#### ⚠️ P1 B2 可能"拿到子任务指令却用不上"

语言在 PerAct 中的通路：

```
CLIP RN50 ❄️ → lang_preprocess ✅（可训，单层 Linear 512→128，无非线性，0.066 M）
   → 与体素 token 拼接 → Perceiver ❄️（全冻结）→ decoder_cross_attn ❄️
```

**能适应语言分布变化的只有 `lang_preprocess` 这一个单层线性映射**，其后全部冻结。从"整任务指令"到"子任务指令"是实质的分布改变，单层线性能否承载存疑。

**处置：先采 (a)**，即保持单层 Linear，记录为已知瓶颈。若 10k 探针显示 **B2 ≈ B1**，**不得直接下结论说"子任务分割无用"**——更可能是此处的容量瓶颈。备选：

| 选项 | 做法 | 代价 |
|---|---|---|
| **(a) 保持现状**（**采用**） | 单层 Linear 可训 | 若 B2≈B1，中间臂失去区分力 |
| (b) 同时解冻 `decoder_cross_attn` | +0.083 M，语言读出侧亦可适应 | 偏离"只训动作头"的干净边界 |
| (c) `lang_preprocess` 扩为残差 MLP | 初始化为精确复现原 Linear；三臂共享，不破坏公平性 | 偏离最小改动原则 |

#### P2 存储需求可能达 120–360 GB

replay 使用 `use_disk: True`。样本数取决于 demo 长度与关键帧数，目前无实测。

**建议**：正式训练前先对 1 个任务填一次 replay，实测单样本大小与样本数，外推总量并确认磁盘容量。

#### P3 UnSeen 上的地板风险（原「灾难性遗忘」，2026-09 改写）

在 Seen18 上微调动作头会使其向这 18 个任务特化，而新 UnSeen 是主干**从未见过**的任务——三臂在其上的绝对值可能很低甚至趋零。三臂受影响相同，**不损害对比公平性**，但会削弱 E3 的判读力。

缓解见 §7 T1（同物体反向任务作保险 + 定案前的天花板/摸底筛选）。**结果中必须同时报 B0 作为参照**：B0 未经 Stage 3 微调，是「微调是否损害了跨任务泛化」的唯一参照系。

#### P4 评测管线依赖在线 Planner

`source/peract/eval.py` 需接入 §6 的在线 Planner 才能评测 B2/B3；B0/B1 不需要（用整任务指令）。**这是 Exp1/Exp2 的关键路径依赖。**

---

## 九、实验方案

见 [Exp_Design.md](Exp_Design.md)。要点：

- **四臂对比** B0（官方原样）/ B1（baseline）/ B2（planner-only）/ B3（ours），定义见 §8.5。
- **P0 实验**：E0 评测管线校验（前提门禁）· E1 天花板基线 · E2 **Seen18** 四臂对比 · E3 **UnSeen8 零样本**四臂对比（Tier-A/B 分列）· E7 原子动作动机实验。
- **一次评测、多重视图**：5 次评测运行（`b0_official` / `b1_seed0` / `b2_seed0` / `b3_seed0` / `replay_ceiling`）× 18 任务 × 25 局，各实验是对同一份 rollout 记录的聚合视图，B0 不重复评测。
- **统计口径**：单种子，不报 mean±std，逐任务分列 + 天花板归一化。
- **视频录制**：所有评测脚本统一保留；录制与落盘解耦（episode 结束后按成败决定是否写盘），避免体积膨胀。
