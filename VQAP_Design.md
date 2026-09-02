# VQAP 设计文档（数据集 / 双码本模型 / 预训练）

> **本文档定位**：专门记录数据集、VQAP 双码本模型结构、预训练（Stage 0/1）、损失、配置与超参数。**VLA 集成部分已拆分至配套文档** [VLA_Design.md](VLA_Design.md)（如何把这里预训练好的码本接入 π₀ VLA）。两份文档配套使用：VLA_Design.md 复用的码本、编码器、动作白名单、NSVQ 查表机制等接口均在本文档定义，对接契约见 VLA_Design.md §〇。
>
> **版本说明**：本文档按当前代码实现对齐，涵盖 AtomAction-NSVQ 编码链路、双码本量化（NSVQ 机制、防坍缩策略、Perplexity 监控）、各训练阶段模块输出、整体模块连接与两阶段训练工作流。关键实现说明：
> - **ChannelAttention**：乘法残差门控，$\hat{X}' = \hat{X}(1+w)$，$w = \sigma(\cdot) \in [0,1]$，缩放系数 $\in[1,2]$，只放大不抑制
> - **NSVQ 码本**：以 `nn.Parameter(requires_grad=True)` 形式存在，无 EMA/承诺损失；训练时 `codebook_indices` 必须为空（由 argmin 决定），eval 时必须显式传入；`AtomAction_NSVQ.forward` 不含码本索引参数，推理查表通过 `GlobalCodebookModule.lookup_codewords()` / `DetailCodebookModule.lookup_codewords()` 完成
> - **VisualFlowMatchingHead**：条件向量仅由流时间 $\tau$ 构成（$c = \text{MLP}(e_\tau)$），不含全局语义码；视觉信息通过 $F_{diff}$ 的 cross-attention 注入
> - **VisualSelfAttentionLayer**：attention 子层含残差，FFN 子层当前**不含残差**（$\text{output} = \text{FFN}(\text{LN}(x))$）

---

## 一、全局超参数与维度约定

| 符号 | 含义 | 推荐值（对齐 `config/model.yaml`） |
|---|---|---|
| $d$ | 动作序列经 FFN 投影后每时间步的隐藏维度 | **512** |
| $C$ | Transformer Encoder 隐藏维度（与 $d$ 一致） | **512** |
| $d_{ff}$ | Transformer FFN 内部维度 | **2048** |
| $H$ | 注意力头数 | **8** |
| $d_{head}$ | 每头维度 $= C / H$ | **64** |
| $K_g$ | 全局语义码本大小 | **36** |
| $K_d$ | 细节码本大小 | **192** |
| $d_{code}$ | 码本向量维度（与 $C$ 同为 512） | **512** |
| $N_{detail}$ | 细节码数量 | **9** |
| $d_{dino}$ | DINOv2 特征维度（ViT-B/14） | **768** |
| $T$ | batch 内最长轨迹长度（动态，batch 内统一） | 动态 |

> **关键维度关系**：$d_{code}=C=512$，因此 TransformerEncoder 输出可直接送入 NSVQ 量化器（$512\to512$），无需额外降/升维。GlobalCodebookModule / DetailCodebookModule 内部仍保留 `Linear(512, 512)` 投影层（代码实际存在），在当前配置下近似等价映射。

---

## 二、数据预处理流水线

### 2.0 整体数据流

```
RLBench 仿真环境
  └─ traj_generator_segmentation（自动采集 + 人工分割标注）
       └─ AtomAction_Dataset/
            └─ {task}/variation{N}/episodes/{ep}/phase_{i}/
                 ├─ low_dim_obs.pkl          # 低维观测序列（每帧一个 Observation 对象）
                 ├─ {view}_rgb/              # 各视角 RGB 图像序列（PNG）
                 └─ phase_metadata.json
  └─ DEMO_SEG_CHECK_v3.csv（人工核验的分割标注）
       └─ scripts/create_atomaction_dataset_re.py
            └─ AtomAction_Dataset_re/        # 按动作类别重组
                 └─ {action}/{task}/Variation{N}/Phase_XXX/
  └─ data/dataset.py（AtomActionDataset）
      ├─ 读取 low_dim_obs.pkl → 原始 trajectory_data
      ├─ 调用 data/utils.py::normalize
      │    └─ 读取 dataset_metadata.json 的 traj_stats
      │       → 位置/速度/力矩/接触力按分位数线性归一化
      │       → 关节角/夹爪关节按 Z-score
      │       → gripper_pose: 位置 3 维保留 + 四元数 4 维转 6D
       └─ 调用 ViewSelector（DINOv2）→ selected_views
  └─ data/utils.py（AtomActionDataset_collate_fn）
       └─ 动态 padding + mask → batch
    └─ 模型输入构造
      ├─ 四路特征投影 → trajectory_features [B, T, 512]
      ├─ ChannelEncoder → channel_encoded_features [B, T, 512]
      └─ TransformerEncoder → z [B, T, 512]
```

---

### 2.1 原始数据采集与分割

**RLBench 采集**：通过 `traj_generator_segmentation` 模块在 RLBench 仿真环境中批量采集专家演示轨迹，每个 episode 包含完整任务执行过程，每帧记录：

- 低维观测（存储为 `low_dim_obs.pkl`，每个元素为 `Observation` 对象）：
  - `gripper_pose`：末端执行器位姿，$[p_x, p_y, p_z, q_x, q_y, q_z, q_w] \in \mathbb{R}^7$（位置 + 四元数）
  - `joint_positions`：7 关节角度 $\in \mathbb{R}^7$
  - `joint_velocities`：7 关节速度 $\in \mathbb{R}^7$
  - `joint_forces`：7 关节力矩 $\in \mathbb{R}^7$
  - `gripper_joint_positions`：夹爪关节角度 $\in \mathbb{R}^2$
  - `gripper_touch_forces`：夹爪接触力 $\in \mathbb{R}^6$
  - `gripper_open`：夹爪开合状态 $\in \{0, 1\}$（标量）
- 五视角 RGB 图像序列（`front`, `left_shoulder`, `right_shoulder`, `overhead`, `wrist`），每视角存储为独立目录 `{view}_rgb/`，文件名为帧编号（如 `0.png`, `1.png`, ...）

**分割标注**：每个 episode 按关键帧切割为若干 phase（原子动作片段），切割边界由人工核验写入 `DEMO_SEG_CHECK_v3.csv`。CSV 格式为 `Task | Variation | Phase_0_action | Phase_1_action | ...`，每个 phase 对应一个原子动作标签（如 `grasp`、`lift`、`place` 等）。

---

### 2.2 数据集重组（AtomAction_Dataset_re）

`scripts/create_atomaction_dataset_re.py` 读取原始数据集和标注 CSV，将所有 phase 按 **action → task → variation → phase** 的四级目录结构重新组织：

```
AtomAction_Dataset_re/
  {action}/
    action_metadata.json            # 动作级统计（所有任务的 phase 总数等）
    {task}/
      task_metadata.json            # 任务级统计（各 variation phase 数量等）
      Variation{N}/
        variation_metadata.json     # variation 级元数据（来源 episode、phase 索引等）
        Phase_000/ ... Phase_NNN/
          low_dim_obs.pkl           # 从原始 episode 直接复制
          {view}_rgb/               # 各视角图像
          phase_metadata.json       # 本 phase 的来源信息与统计量
```

**关键处理逻辑**：
- 标签归一化：`LABEL_NORMALIZATION` 字典修正拼写错误（如 `"appraoch"→"approach"`）
- 每个原始 episode 中可能有多个同类动作（如多次 `grasp`），每次出现均作为独立样本
- phase 按 `requested_episode` 数值→帧编号字典序排列，保证索引稳定性
- 校验：标注 phase 数量与数据集 phase 数量一致，episode 目录与 metadata 同步

---

### 2.3 数据集索引与加载（AtomActionDataset）

`data/dataset.py` 中的 `AtomActionDataset` 遍历重组后数据集，将每个有效 phase 注册为一个样本：

**样本有效性条件**：
1. `low_dim_obs.pkl` 文件存在
2. 至少有一个视角的 RGB 图像目录非空

**加载的训练动作白名单**由 `config/global.yaml` 中的 `train_actions` 字段控制（必须为非空列表，且每个动作在数据集中存在）。数据集共包含 18 种动作类型，各类型的训练策略如下：

| 动作类型 | 训练状态 | 备注 |
|---|---|---|
| `approach`、`transfer`、`flip-open`、`flip-close`、`grasp`、`hang`、`insert`、`lift`、`place`、`press`、`pull`、`push`、`revolve-in`、`revolve-out`、`rotate`、`slide`、`wipe` | **参与训练**（17 个，对齐 `config/global.yaml::train_actions`） | 覆盖主要原子操作语义 |
| `pose-adjust` | **永不参与训练** | 废弃动作，数据质量不可靠，排除在白名单之外 |

> `train_actions` 当前包含 17 个动作（`approach` 与 `transfer` 直接纳入），非文档早期草稿的"15 核心 + 2 暂定"。

`__getitem__` 的输出结构：

| 字段 | 类型 | 说明 |
|---|---|---|
| `Action` | `str` | 原子动作标签 |
| `Task` | `str` | 任务名称 |
| `Variation` | `str` | Variation 名称 |
| `trajectory_data` | `Dict[str, List[Any]]` | 已完成字段级归一化的帧序列，长度=轨迹帧数；`gripper_pose` 由原始 7 维变为 9 维 |
| `trajectory_length` | `int` | 轨迹帧数 $T$ |
| `selected_views` | `List[Dict]` | top-k 视角结果列表 |

---

### 2.4 视角选择（ViewSelector / DINOv2）

`data/view_select.py` 中的 `ViewSelector` 对每个 phase 的五个视角做信息量评分，返回变化量最大的 top-k 个视角：

**评分方式**：对每个视角的起始帧和结尾帧分别提取 DINOv2 全局 CLS 特征，取余弦相似度的相反数作为变化分数：

$$
\text{score}^v = 1 - \cos(f_s^v, f_e^v) = 1 - \frac{f_s^v \cdot f_e^v}{\|f_s^v\|\|f_e^v\|}
$$

分数越高表示视觉变化越明显，选取 top-k 个视角。

**图像预处理流程**（`build_dinov2_transform`）：

```
PIL.Image → Resize(224) → CenterCrop(224) → ToTensor()
  → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  → Tensor [3, 224, 224]
```

**缓存机制**：视角选择结果会缓存在 phase 目录的 `view_selection_cache` 文件中，避免重复推理，缓存版本号 `VIEW_SELECTION_CACHE_VERSION=1` 用于失效检测。

每个视角结果包含：
- `best_view`：视角名称（`str`）
- `best_start_image`：起始帧张量 $[3, 224, 224]$
- `best_end_image`：结尾帧张量 $[3, 224, 224]$
- `best_score`：变化分数（0 维标量，dtype 由 `global.yaml` 的 `tensor_dtype` 控制）

> **⚠️ 设计说明**：数据加载阶段的视角评分 DINOv2（用于选视角）与 VASA 模块的 DINOv2（用于视觉编码）是两套独立实例。前者是离线冻结的官方预训练模型（`dinov2_vits14_reg`，ViT-**S**/14，`config/global.yaml::view_selector`），只用于选视角并缓存结果；top-$k$ 当前配置为 $k=1$（`config/global.yaml::atomactiondataset.top_k`），即仅选变化量最大的**单个**视角，多视角加权融合逻辑在 $k=1$ 时退化为直通。后者是 `model/module/encoder.py::DINO_FeatureExtractor` 中单独加载的官方 DINOv2 backbone（`dinov2_vitb14`，ViT-**B**/14），现已支持通过超参数选择导出 CLS 特征或 `x_norm_patchtokens`。当选择 patch 模式时，模块会直接输出 `[N, P, feature_dim]` 的 patch 序列，不再做额外池化；`ImageEncoder` 再按视角权重将多视角结果融合为起始帧特征、结尾帧特征和视角权重（$k>1$ 才有融合，$k=1$ 时退化为直通）。二者职责不同，不共享同一个模型实例。

---

### 2.5 Batch 整理与动态 Padding（AtomActionDataset_collate_fn）

`data/utils.py` 中的 `AtomActionDataset_collate_fn` 将不等长轨迹 padding 至 batch 内最大长度：

**输出结构**：

| 字段 | 形状 | 说明 |
|---|---|---|
| `Action` / `Task` / `Variation` | `List[str]`，长度 $B$ | 标签列表 |
| `trajectory_data[field]` | `[B, T_{max}, D_{field}]` | 各低维字段，padding 位置补零 |
| `trajectory_length` | `[B]`，`torch.long` | 各样本实际帧数 |
| `trajectory_mask` | `[B, T_{max}]`，`torch.bool` | `True` = 有效帧，`False` = padding |
| `selected_views` | `List[List[Dict]]`，长度 $B$ | 保留每样本原始视角结果 |

`__getitem__` 输出并进入 collate 的字段维度：`joint_positions`→$7$，`gripper_pose`→$9$（位置 3 维 + 旋转 6 维），`joint_velocities`→$7$，`joint_forces`→$7$，`gripper_joint_positions`→$2$，`gripper_touch_forces`→$6$，`gripper_open`→$1$

> 完整的批次字段规格、各字段的归一化说明以及 `selected_views` 结构详见 §2.8。

---

### 2.6 动作特征构造（模型输入预处理）

从 batch 的 `trajectory_data` 中取所有已在 `AtomActionDataset.__getitem__` 阶段完成归一化的低维字段，再构造动作编码器输入。各字段的归一化方式由分布特点决定：

**各字段归一化策略总览**

| 字段 | 原始维度 | 归一化方法 | 说明 |
|---|---|---|---|
| `gripper_pose` → 位置 $p$ | 3 | 分位数线性归一化（$q_{0.01}\mapsto -1$，$q_{0.99}\mapsto 1$） | 空间语义保留，分布有限但受任务影响 |
| `gripper_pose` → 四元数 $q$ | 4 | **转 6D，不做统计归一化** | 旋转矩阵元素天然 $\in [-1,1]$，统计归一化破坏几何约束 |
| `joint_positions` | 7 | Z-score | 关节角分布近似正态，有已知物理上下界 |
| `joint_velocities` | 7 | 分位数线性归一化（$q_{0.01}\mapsto -1$，$q_{0.99}\mapsto 1$） | 重尾分布（静止时 $\approx 0$，运动峰值大） |
| `joint_forces` | 7 | 分位数线性归一化（$q_{0.01}\mapsto -1$，$q_{0.99}\mapsto 1$） | 极度重尾（接触瞬间峰值显著） |
| `gripper_joint_positions` | 2 | Z-score | 范围小，近似均匀分布 |
| `gripper_touch_forces` | 6 | 分位数线性归一化（$q_{0.01}\mapsto -1$，$q_{0.99}\mapsto 1$） | 接触力，重尾 |
| `gripper_open` | 1 | **Embedding(2, 64)**，跳过统计归一化 | 二值状态，不适合统计归一化 |

所有统计量均在训练集全局（跨所有 action 类别）计算，仅使用 `trajectory_mask=True` 的有效帧，不按 action 分组。
为与 openpi 和 Imagine2Act 对齐，分位数归一化仅做线性映射，不对超出 $q_{0.01}$ / $q_{0.99}$ 区间的值额外做 clamp。

**步骤一：连续字段归一化**

Z-score 归一化（用于 `joint_positions`、`gripper_joint_positions`）：

$$
\tilde{x} = \frac{x - \mu}{\sigma + \epsilon}, \quad \epsilon = 10^{-6}
$$

分位数归一化（用于位置、`joint_velocities`、`joint_forces`、`gripper_touch_forces`）：

$$
\tilde{x} = \frac{x - q_{0.01}}{q_{0.99} - q_{0.01} + \epsilon} \cdot 2 - 1
$$

其中 $q_{0.01}$、$q_{0.99}$ 为训练集该字段的第 1/99 百分位数。该线性映射使 $q_{0.01}$、$q_{0.99}$ 分别对应 $-1$、$1$，但对超出该区间的值不额外截断，因此输出在极端情况下可以越过 $[-1,1]$。

**步骤二：四元数 → 6D 旋转表示**

取旋转矩阵 $R = \text{quat2mat}(q_t^{ee}) \in \mathbb{R}^{3 \times 3}$ 的前两列并展平：

$$
r_t^{ee,6d} = [R_{:,0};\; R_{:,1}] \in \mathbb{R}^6
$$

6D 表示避免了四元数的双覆盖和欧拉角的万向锁问题，且旋转矩阵元素天然 $\in [-1,1]$，无需再做统计归一化。

**步骤三：拼接连续特征**

$$
x_t^{cont} = [\tilde{p}_t^{ee};\; r_t^{ee,6d};\; \tilde{j}_t^{pos};\; \tilde{j}_t^{vel};\; \tilde{j}_t^{frc};\; \tilde{g}_t^{jp};\; \tilde{g}_t^{tf}] \in \mathbb{R}^{3+6+7+7+7+2+6} = \mathbb{R}^{38}
$$

**步骤四：夹爪状态嵌入**

$$
g_t = \text{round}(\text{gripper\_open}[b, t]) \in \{0, 1\}, \quad e_g = \text{Embedding}(2, 64)[g_t] \in \mathbb{R}^{64}
$$

**步骤五：分组 FFN 投影与拼接**

将连续特征按语义分为三组投影（各组 2 层 MLP，激活 GELU），保证维度和为 448，与夹爪嵌入 64 维合计 512：

$$
\phi_{ee}([\tilde{p}_t;\; r_t^{6d}]) \in \mathbb{R}^9 \to \mathbb{R}^{192}, \quad
\phi_{body}([\tilde{j}^{pos};\; \tilde{j}^{vel};\; \tilde{j}^{frc}]) \in \mathbb{R}^{21} \to \mathbb{R}^{192}, \quad
\phi_{gm}([\tilde{g}^{jp};\; \tilde{g}^{tf}]) \in \mathbb{R}^8 \to \mathbb{R}^{64}
$$

$$
x_t = [\phi_{ee}(\cdot);\; \phi_{body}(\cdot);\; \phi_{gm}(\cdot);\; \phi_{go}(e_g)] \in \mathbb{R}^{192+192+64+64} = \mathbb{R}^{512}
$$

最终得到整段 batch 输入 $X \in \mathbb{R}^{B \times T_{max} \times 512}$，配合 `trajectory_mask` 作为 padding mask 输入编码器。

---

### 2.7 训练集归一化统计量计算

训练开始前需离线扫描整个训练集，对各字段分别计算归一化所需统计量，实现为 `data/utils.py` 中的 `compute_and_save_statistics` 函数。

**各字段统计量需求**

| 字段 | 需要统计量 | 算法 |
|---|---|---|
| `gripper_pos`（位置前3维） | $q_{0.01}$，$q_{0.99}$ | 在线直方图（5000 bins）估计分位数 |
| `joint_positions` | $\mu$，$\sigma$ | Welford 在线算法 |
| `joint_velocities` | $q_{0.01}$，$q_{0.99}$ | 在线直方图 |
| `joint_forces` | $q_{0.01}$，$q_{0.99}$ | 在线直方图 |
| `gripper_joint_positions` | $\mu$，$\sigma$ | Welford 在线算法 |
| `gripper_touch_forces` | $q_{0.01}$，$q_{0.99}$ | 在线直方图 |

所有统计量**仅使用训练集**计算，按各字段的每个维度独立统计（不跨维度合并），只统计 `trajectory_mask=True` 的有效帧。

**实现要点**

- 采用在线流式计算（每次读取一个 sample），避免将全量数据加载到内存
- Z-score 字段使用 Welford 算法（$O(1)$ 内存更新均值和方差）
- 分位数字段使用固定区间直方图（对每维预设值域范围，$\pm 5\sigma$ 或物理上下界，5000 bins）

**统计量保存位置**：数据集根目录 `dataset_metadata.json` 的 `traj_stats` 字段；`data/utils.py::normalize` 在 `__getitem__` 阶段直接读取该字段。格式示例：

```json
{
  "traj_stats": {
    "gripper_pose": {
      "mean": [7 values],
      "std": [7 values],
      "q01": [7 values],
      "q99": [7 values]
    },
    "joint_positions": {
      "mean": [7 values],
      "std": [7 values]
    },
    "joint_velocities": {
      "mean": [7 values],
      "std": [7 values],
      "q01": [7 values],
      "q99": [7 values]
    },
    "joint_forces": {
      "mean": [7 values],
      "std": [7 values],
      "q01": [7 values],
      "q99": [7 values]
    },
    "gripper_joint_positions": {
      "mean": [2 values],
      "std": [2 values]
    },
    "gripper_touch_forces": {
      "mean": [6 values],
      "std": [6 values],
      "q01": [6 values],
      "q99": [6 values]
    }
  }
}
```

> **实现说明**：`gripper_pose` 的统计量在 metadata 中保留完整 7 维，但实际归一化时仅使用前 3 维位置的 $q_{0.01}$ / $q_{0.99}$；后 4 维四元数直接转成 6D，不参与统计归一化。

> **⚠️ 注意**：统计量全局（跨所有 action 类别）计算，不按 action 分组。推理时使用完全相同的统计量。若跨任务迁移，需使用目标任务训练集重新计算或使用联合统计量。

---

### 2.8 DataLoader 输出数据包结构与维度

`AtomActionDataset_collate_fn`（`data/utils.py`）将 `AtomActionDataset.__getitem__` 返回的样本列表整理为以下 batch 字典，供模型直接消费。

#### 顶层字段

| 键 | 类型 | 形状 / 长度 | 说明 |
|---|---|---|---|
| `Action` | `List[str]` | $B$ | 每样本的原子动作标签 |
| `Task` | `List[str]` | $B$ | 每样本的任务名称 |
| `Variation` | `List[str]` | $B$ | 每样本的 variation 名称 |
| `trajectory_data` | `Dict[str, torch.Tensor]` | — | 各低维字段的 padded 张量，见下表 |
| `trajectory_length` | `torch.LongTensor` | $[B]$ | 每样本实际帧数（未 padding 长度） |
| `trajectory_mask` | `torch.BoolTensor` | $[B,\, T_{max}]$ | `True` = 有效帧，`False` = padding |
| `selected_views` | `List[List[Dict]]` | $B$ | 每样本 top-k 视角结果列表，见下表 |

#### `trajectory_data` 各字段维度

所有字段已在 `__getitem__` 阶段完成归一化，collate 仅做 padding（padding 位置补零）。

| 字段名 | 形状 | 归一化方式 | 说明 |
|---|---|---|---|
| `gripper_pose` | $[B,\, T_{max},\, 9]$ | 位置 3 维分位数归一化；四元数转 6D，不做统计归一化 | 原始 7 维（位置 3 + 四元数 4）→ 9 维（位置 3 + 旋转 6D） |
| `joint_positions` | $[B,\, T_{max},\, 7]$ | Z-score | 7 关节角度 |
| `joint_velocities` | $[B,\, T_{max},\, 7]$ | 分位数归一化（$q_{0.01}\mapsto -1$，$q_{0.99}\mapsto 1$） | 7 关节速度 |
| `joint_forces` | $[B,\, T_{max},\, 7]$ | 分位数归一化 | 7 关节力矩 |
| `gripper_joint_positions` | $[B,\, T_{max},\, 2]$ | Z-score | 夹爪关节角度 |
| `gripper_touch_forces` | $[B,\, T_{max},\, 6]$ | 分位数归一化 | 夹爪接触力 |
| `gripper_open` | $[B,\, T_{max},\, 1]$ | 不做统计归一化，保留原始 $\{0, 1\}$ 值 | 夹爪开合状态（二值） |

> `trajectory_mask[b, t] = True` 当且仅当 `t < trajectory_length[b]`，模型在计算注意力和损失时应屏蔽 `False` 位置。

#### `selected_views` 各元素结构

`selected_views[b]` 为长度 $\le$ `top_k` 的列表，每个元素为：

| 键 | 类型 | 形状 | 说明 |
|---|---|---|---|
| `best_view` | `str` | — | 视角名称（`front` / `left_shoulder` / `right_shoulder` / `overhead` / `wrist`） |
| `best_start_image` | `torch.Tensor` | $[3,\, 224,\, 224]$ | 该视角起始帧，经 DINOv2 标准预处理（Resize→CenterCrop→ToTensor→Normalize） |
| `best_end_image` | `torch.Tensor` | $[3,\, 224,\, 224]$ | 该视角结尾帧，预处理同上 |
| `best_score` | `torch.Tensor` | 标量（0 维） | 变化分数 $= 1 - \cos(f_s^v, f_e^v)$，越大表示视觉变化越明显；dtype 由 `global.yaml::tensor_dtype` 控制 |

#### 数据流示意

```
AtomActionDataset.__getitem__(i)
  ├─ Action / Task / Variation: str
  ├─ trajectory_data: Dict[field → List[frame_value]]   # 已归一化，长度=T_i
  ├─ trajectory_length: int                              # = T_i
  └─ selected_views: List[Dict]                          # 长度 ≤ top_k

AtomActionDataset_collate_fn(batch)                      # batch size = B
  ├─ Action / Task / Variation: List[str] (B,)
  ├─ trajectory_data[field]: Tensor [B, T_max, D_field]  # padding 位置补零
  ├─ trajectory_length: LongTensor [B]
  ├─ trajectory_mask: BoolTensor [B, T_max]
  └─ selected_views: List[List[Dict]] (B,)
```

---

## 三、AtomAction-NSVQ 完整设计

当前实现落点：

- `model/module/atomaction_nsvq.py`：四路动作特征投影与模块编排
- `model/module/encoder.py`：`ChannelEncoder`、`TransformerEncoder`
- `model/module/utils.py`：`ChannelAttention`、`MultiHeadAttention`、`RotaryPositionEncoding1D`
- `config/model.yaml`：分支投影、`channel_encoder`、`transformer_encoder`、RoPE 配置

### 3.1 动作输入与特征投影

每时间步原始动作字段（来自 `low_dim_obs.pkl`）：

$$
a_t^{raw} = \bigl[p_t^{ee} \in \mathbb{R}^3,\; q_t^{ee} \in \mathbb{R}^4,\; j_t^{pos} \in \mathbb{R}^7,\; j_t^{vel} \in \mathbb{R}^7,\; j_t^{frc} \in \mathbb{R}^7,\; g_t^{jp} \in \mathbb{R}^2,\; g_t^{tf} \in \mathbb{R}^6,\; g_t \in \{0,1\}\bigr]
$$

**预处理步骤：**

1. **位置分位数归一化**：$\tilde{p}_t = \dfrac{p_t^{ee} - q^{pos}_{0.01}}{q^{pos}_{0.99} - q^{pos}_{0.01} + \epsilon} \cdot 2 - 1 \in \mathbb{R}^3$，其中 $q^{pos}_{0.01}$、$q^{pos}_{0.99}$ 分别映射到 $-1$、$1$，不对区间外样本额外截断
2. **四元数 → 6D 旋转**：$r_t^{6d} = [R_{:,0}; R_{:,1}] \in \mathbb{R}^6$，$R=\text{quat2mat}(q_t^{ee})$，元素天然 $\in[-1,1]$，不做统计归一化
3. **关节角/夹爪关节 Z-score**：$\tilde{j}_t^{pos} = (j_t^{pos}-\mu^{jp})/(\sigma^{jp}+\epsilon)$，$\tilde{g}_t^{jp}$ 同理
4. **速度、力矩、接触力分位数归一化**：$\tilde{j}_t^{vel},\;\tilde{j}_t^{frc},\;\tilde{g}_t^{tf}$ 均按 $q_{0.01}\mapsto -1$、$q_{0.99}\mapsto 1$ 线性映射，不额外 clamp
5. **夹爪嵌入**：$e_g = \text{Embedding}(2, 64)[g_t] \in \mathbb{R}^{64}$

所有统计量（$q_{0.01}$，$q_{0.99}$，$\mu$，$\sigma$）由 `compute_and_save_statistics` 离线计算并写入数据集根目录 `dataset_metadata.json` 的 `traj_stats` 字段。

**连续特征拼接：**

$$
x_t^{cont} = [\tilde{p}_t;\; r_t^{6d};\; \tilde{j}_t^{pos};\; \tilde{j}_t^{vel};\; \tilde{j}_t^{frc};\; \tilde{g}_t^{jp};\; \tilde{g}_t^{tf}] \in \mathbb{R}^{38}
$$

**分组 FFN 投影（各组独立 2 层 MLP，激活函数 GELU）：**

$$
\phi_{ee}: \mathbb{R}^9 \to \mathbb{R}^{192}, \quad \text{Linear}(9 \to 384) \to \text{GELU} \to \text{Linear}(384 \to 192)
$$

$$
\phi_{body}: \mathbb{R}^{21} \to \mathbb{R}^{192}, \quad \text{Linear}(21 \to 384) \to \text{GELU} \to \text{Linear}(384 \to 192)
$$

$$
\phi_{gm}: \mathbb{R}^8 \to \mathbb{R}^{64}, \quad \text{Linear}(8 \to 64) \to \text{GELU} \to \text{Linear}(64 \to 64)
$$

$$
\phi_{go}: \mathbb{R}^{64} \to \mathbb{R}^{64}, \quad \text{Linear}(64 \to 64)
$$

**拼接：**

$$
x_t = [\phi_{ee}([\tilde{p}_t; r_t^{6d}]);\; \phi_{body}([\tilde{j}_t^{pos}; \tilde{j}_t^{vel}; \tilde{j}_t^{frc}]);\; \phi_{gm}([\tilde{g}_t^{jp}; \tilde{g}_t^{tf}]);\; \phi_{go}(e_g)] \in \mathbb{R}^{192+192+64+64} = \mathbb{R}^{512}
$$

整段输入：$X = [x_1, \dots, x_T] \in \mathbb{R}^{T \times d}$，padding mask $m \in \{0,1\}^T$

> **语义分组依据**：末端执行器 EE（位置+旋转）= 操作意图的核心，分配 192 维；关节体态（角度+速度+力矩）= 机器人内部状态，分配 192 维；夹爪机械量（关节角+接触力）= 抓取精细状态，分配 64 维；夹爪开合嵌入 = 离散语义，独立 64 维。四组之和恰好等于 $d=512$。

---

### 3.2 ChannelEncoder（通道编码）

拼接后的 `trajectory_features` 不直接进入时序 Transformer，而是先经过 `ChannelEncoder`。当前实现中，`ChannelEncoder` 定义在 `model/module/encoder.py`，由两个逐时间步子模块串联组成：

1. `ChannelAttention`：SE 风格通道门控，定义在 `model/module/utils.py`
2. `output_ffn`：逐时间步 FFN，结构为 `Linear(512 -> 512) -> LayerNorm(512) -> GELU -> Linear(512 -> 512)`

为避免 padding 零向量经过带 bias 的线性层后变成非零值，`trajectory_mask` 会在进入 `ChannelEncoder` 前、经过 `ChannelAttention` 后，以及 FFN 输出后重复施加。

设 $M = m.\text{unsqueeze}(-1) \in \{0,1\}^{B \times T \times 1}$，前向计算可写为：

$$
X_0 = X \odot M
$$

$$
X_{ca} = \text{ChannelAttention}(X_0) \odot M
$$

$$
X_{ce} = \text{FFN}(X_{ca}) \odot M
$$

其中 `ChannelAttention` 的内部计算为（对应 `model/module/utils.py::ChannelAttention.forward`）：

$$
\hat{X} = X_0.\text{reshape}(B \cdot T,\; 512)
$$

$$
w = \sigma\!\bigl(\text{Linear}(64 \to 512)\bigl(\text{GELU}\bigl(\text{Linear}(512 \to 64)(\hat{X})\bigr)\bigr)\bigr) \in [0,1]^{B \cdot T \times 512}
$$

$$
\hat{X}' = \hat{X} + \hat{X} \odot w = \hat{X} \odot (1 + w)
$$

$$
X_{ca} = \hat{X}'.\text{reshape}(B,\; T,\; 512)
$$

> **实现说明**：当前 `channel_gate` 结构为 `Linear(512→64) → GELU → Linear(64→512) → Sigmoid`，门控值 $w \in [0,1]$，乘法残差使缩放系数 $(1+w) \in [1,\, 2]$，即各通道只能被放大，不能被抑制。这是当前代码的实际行为，后续如需支持抑制可去掉残差项或改用 Tanh 门控。

**维度流：**

$$
X \in \mathbb{R}^{B \times T \times 512} \xrightarrow{\text{ChannelEncoder}} X_{ce} \in \mathbb{R}^{B \times T \times 512}
$$

**ChannelEncoder 配置：**

| 组件 | 规格 |
|---|---|
| 输入维度 | 512 |
| `ChannelAttention` 瓶颈维度 | 64（`model.yaml::channel_encoder.bottleneck_dim`） |
| `ChannelAttention` 门控结构 | `Linear(512→64) → GELU → Linear(64→512) → Sigmoid`，门控 $w \in [0,1]$ |
| `ChannelAttention` 残差形式 | $\hat{X}' = \hat{X} + \hat{X} \odot w = \hat{X}(1+w)$，缩放系数 $\in [1,2]$，只放大不抑制 |
| 输出 FFN | `Linear(512 -> 512) -> LayerNorm(512) -> GELU -> Linear(512 -> 512)` |
| mask 处理 | 输入前与每个子模块输出后都重新乘 `trajectory_mask` |
| 输出维度 | 512 |

---

### 3.3 Transformer Encoder

`ChannelEncoder` 的输出 `channel_encoded_features` 再进入 `TransformerEncoder`。当前实现位于 `model/module/encoder.py`，配置由 `config/model.yaml::transformer_encoder` 与 `config/model.yaml::rope` 控制。

RoPE 由 `model/module/utils.py::RotaryPositionEncoding1D` 提供，并在多头注意力中作用于每个 head 的 q、k。注意力计算时使用 `trajectory_mask` 屏蔽 key 侧 padding 位置；每层残差输出后再次乘 mask，保证 padding 位置保持为零。

**Transformer Encoder 配置（当前实现）：**

| 组件 | 规格 |
|---|---|
| 层数 | 4 |
| 隐藏维度 $C$ | 512 |
| 注意力头数 $H$ | 8 |
| 每头维度 $d_{head}$ | 64 |
| FFN 维度 $d_{ff}$ | 2048 |
| 激活函数 | GELU |
| 归一化 | Pre-LN，支持 `layernorm`（默认）或 `rmsnorm`，由 `norm_type` 配置项控制 |
| Dropout | 0.1（训练），推理关闭 |
| RoPE | `theta = 10000.0`，`max_seq_len = 2048` |
| 注意力方式 | 双向全序列注意力，padding 位置通过 mask 置 $-\infty$ |

**维度流：**

$$
trajectory\_features \in \mathbb{R}^{B \times T \times 512}
\xrightarrow{\text{ChannelEncoder}}
channel\_encoded\_features \in \mathbb{R}^{B \times T \times 512}
\xrightarrow{\text{RoPE + 4-layer BiTransformer}}
z \in \mathbb{R}^{B \times T \times 512}
$$

每层计算（以第 $\ell$ 层为例）：

$$
z' = z^{(\ell-1)} + \text{MHA}(\text{LN}(z^{(\ell-1)}), m) \in \mathbb{R}^{T \times 512}
$$

$$
z^{(\ell)} = z' + \text{FFN}(\text{LN}(z')) \in \mathbb{R}^{T \times 512}
$$

其中 $\text{MHA}$ 的注意力权重对 padding 位置（$m_t=0$）置为 $-\infty$，softmax 后这些位置权重为 0。

---

### 3.4 双码本量化

#### 3.4.1 全局语义支路

**带掩码平均池化：**

$$
\bar{z}^{global} = \frac{\sum_{t=1}^{T} m_t \cdot z_t}{\sum_{t=1}^{T} m_t} \in \mathbb{R}^{512}
$$

（代码对应 `masked_average_pool` 输出 `pooled_global_features`）

**投影到码本空间：**

$$
h_g = \text{Linear}(512 \to d_{code})(\bar{z}^{global}) \in \mathbb{R}^{d_{code}} = \mathbb{R}^{512}
$$

其中 $d_{code}=512$（`config/model.yaml::global_codebook.codebook_dim`）。当前 $d_{code}=C=512$，因此该投影为等价变换；代码中该层 `Linear(512, 512)` 实际存在。

（代码对应 `projected_global_features`，亦即输出键 `global_feature`）

**NSVQ 量化（见 3.4.3）：**

$$
f_q^{global} = Q_g(h_g) \in \mathbb{R}^{d_{code}} = \mathbb{R}^{512}, \quad Q_g \in \mathbb{R}^{K_g \times d_{code}} = \mathbb{R}^{36 \times 512}
$$

（代码对应 `global_codeword`）

#### 3.4.2 细节支路

**$N_{detail}=9$ 个可学习查询（位置编码区分）：**

$$
Q_{learn} = \text{LearnableEmbed}(9, 512) \in \mathbb{R}^{9 \times 512}
$$

**Cross-Attention（4 头，每头维度 128，含 RoPE）：**

$$
Z_q^{detail} = \text{CrossAttn}(Q=Q_{learn}, K=z, V=z;\; \text{mask}=m,\; \text{RoPE}) \in \mathbb{R}^{9 \times 512}
$$

Cross-Attention 配置：4 个头，每头维度 $512/4=128$；key/value 侧的 padding 位置通过 `trajectory_mask` 屏蔽；RoPE 按 `max(N_{detail}, T)` 统一缓存后切片，同时作用于 query（长度 $N_{detail}$）和 key（长度 $T$）两侧，使不同查询槽位具有位置区分能力。

**投影（$d_{code}=512$，等价变换，代码实际存在）：**

$$
H_d^{(n)} = \text{Linear}(512 \to d_{code})(Z_q^{detail,n}) \in \mathbb{R}^{d_{code}} = \mathbb{R}^{512}, \quad n = 1, \dots, 9
$$

**NSVQ 量化：**

$$
f_{q,n}^{detail} = Q_d(H_d^{(n)}) \in \mathbb{R}^{d_{code}} = \mathbb{R}^{512}, \quad Q_d \in \mathbb{R}^{K_d \times d_{code}} = \mathbb{R}^{192 \times 512}
$$

> $K_d=192$ 来自 `config/model.yaml::detail_codebook.codebook_size`。

#### 3.4.3 NSVQ 机制详述

NSVQ（Noise-Substituted Vector Quantization，参考 LAPA 实现）以**噪声替换量化残差**的方式让梯度绕过不可微的 argmin 操作，无需 EMA 更新或承诺损失。

**码本存在形式与初始化：**

$$
\text{codebooks} = \texttt{nn.Parameter}(\texttt{requires\_grad=True}) \in \mathbb{R}^{K \times d_{code}}
$$

其中 $d_{code}=512$（`config/model.yaml::global_codebook.codebook_dim` / `detail_codebook.codebook_dim`）。初始化为 $\text{Uniform}(-1/K,\; 1/K)$，缩小初始值域，减少码本间早期碰撞。码本作为普通可学习参数，通过反传直接更新。

**训练前向：**

$$
k^* = \arg\min_j \|h - e_j\|_2^2, \quad j = 1, \dots, K \quad \text{（硬 argmin，不可微）}
$$

$$
r = h - e_{k^*}, \quad \xi \sim \mathcal{N}(0, I) \quad \text{（量化残差与同维随机噪声）}
$$

$$
\text{vq\_error} = \frac{\|r\|_2}{\|\xi\|_2 + \varepsilon} \cdot \xi \quad \text{（将噪声缩放至与残差等范数）}
$$

$$
q = h + \text{vq\_error} \quad \text{（对 } e_{k^*} \text{ 的可微近似，用于后续计算）}
$$

**梯度流分析：**
- 对编码器输出 $h$：$q = h + \text{vq\_error}$ 直接传导；$r = h - e_{k^*}$ 中 $h$ 的梯度也通过 $\|r\|_2$ 回传
- 对码本 $e_{k^*}$：$r = h - e_{k^*}$，而 $e_{k^*} = \texttt{codebooks}[k^*]$，`requires_grad=True` 保证梯度可达
- **无需承诺损失（commitment loss）或 EMA**，码本直接由端到端梯度优化

**推理前向（硬量化，支持外部索引直查）：**

若未提供外部索引，则推理分支仍按最近邻硬量化执行：

$$
k^* = \arg\min_j \|h - e_j\|_2^2, \quad f_q = e_{k^*} \quad \text{（直接取码本向量，不加噪声）}
$$

若外部模块已经预测出离散码索引，则当前实现允许在 eval 模式下直接将 `codebook_indices` 输入量化器，跳过距离计算与 argmin，直接查表：

$$
\hat{k} \in \{0, 1, \dots, K-1\}, \quad f_q = e_{\hat{k}}
$$

其中 `codebook_indices` 必须是整型，且满足值域约束 $[0, K)$；训练态若显式传入该参数会报错，避免与 NSVQ 噪声近似训练路径混用。

**Perplexity 监控：**

$$
\text{ppl} = \exp\!\left(-\sum_{k=1}^{K} p_k \log p_k\right), \quad p_k = \frac{n_k}{\sum_j n_j}
$$

其中 $n_k$ 为当前 batch 中被分配至码 $k$ 的样本数。目标：$\text{ppl} > K/2$；若 $\text{ppl} < K/4$ 则提示码本坍缩，需触发 §3.4.4 中的替换机制。

#### 3.4.4 码本防坍缩（replace_unused_codebooks）

维护整型计数器 `codebooks_used ∈ ℤ^K`（初始为 0），每次前向传播后原地更新 `codebooks_used[k*] += 1`。替换函数 `replace_unused_codebooks(used_steps)` 由训练循环按 epoch 调用（全局码本与细节码本各自独立维护），其中 `used_steps` 为本统计窗口（当前 epoch）的真实前向步数：

$$
\mathcal{U} = \left\{k \;\middle|\; \frac{\texttt{codebooks\_used}[k]}{\texttt{used\_steps}} < \theta_{\text{discard}}\right\} \quad \text{（低利用率"死码"集合）}
$$

对每个死码 $k \in \mathcal{U}$，从活跃集合 $\mathcal{A} = [K] \setminus \mathcal{U}$ 中随机采样后加微小噪声覆盖：

$$
e_k \leftarrow e_{\text{sampled}} + \varepsilon_{\text{noise}} \cdot \mathcal{N}(0, I), \quad k \in \mathcal{U}
$$

替换后清零计数器：`codebooks_used[:] = 0`。若 $|\mathcal{A}| = 0$（全部码均为死码），则对整个码本施加扰动而非替换。

| 参数 | 推荐值 | 说明 |
|---|---|---|
| 触发周期 | 每 epoch 判定 | 由 trainer 的 `_maybe_replace_dead_codebooks` 控制：perplexity 阈值或固定间隔 `replace_interval_epochs`（默认 20）触发，统计窗口为该 epoch（§5.5） |
| $\theta_{\text{discard}}$ | 0.1 | 平均使用率（`codebooks_used / used_steps`）低于 10% 视为死码 |
| $\varepsilon_{\text{noise}}$ | 1e-3（`config/model.yaml::nsvq.replace_noise_scale`） | 防止替换后与活跃码完全重叠 |

---

### 3.5 信息分离约束

$$
L_{sep} = \left|\cos\!\left(\bar{z}^{global}, \; \frac{1}{9}\sum_{n=1}^{9} Z_q^{detail,n}\right)\right|
$$

其中 $\bar{z}^{global} \in \mathbb{R}^{512}$ 为全局支路 masked-avg-pool 输出（§3.4.1 中的 `pooled_global_features`），$Z_q^{detail,n} \in \mathbb{R}^{512}$ 为细节支路 cross-attention 输出的第 $n$ 个 query 特征（`detail_query_features`）。由于 $d_{code}=C=512$，两个特征天然同维，无需额外投影即可直接计算余弦相似度。

> **实现状态**：`AtomAction_NSVQ.forward` 的输出字典已暴露 `pooled_global_features`（$[B, C]$）与 `detail_query_features`（$[B, N_{detail}, C]$）；`utils/loss_func.py::compute_separation_loss` 取细节 query 的均值与全局特征算 `|cos|`，得到 `loss_sep`。该损失已并入 $L_{AP}$（见 §3.6.5 / §5.1），权重 $\lambda_{sep}$ 由 `config/train.yaml::loss.lambda_sep` 控制（当前 0.01）。

---

### 3.6 Flow Matching 重构解码器（训练专用）

#### 3.6.0 模块定位与作用边界

Flow Matching 重构解码器是一个**纯训练期模块**，唯一作用是：以量化后的码（全局码 $f_q^{global}$、细节码 $F_q^{detail}$）为条件，通过 Flow Matching 目标重构专家末端执行器轨迹，使重构梯度回传到 $\{$动作编码器、全局/细节支路投影、双码本$\}$，**迫使离散码保留足够的轨迹信息**。它本质上扮演 VQ-VAE 中解码器的角色，只是把“直接回归重构”换成了“流匹配重构”。

**作用边界（重要）：**
- **不涉及任何推理 / 采样 / ODE 积分**。该解码器仅在阶段一、阶段二的码本训练中运行；每次前向只采样一个流时间 $\tau$、做一次速度场回归（外加夹爪 BCE），**不沿 ODE 多步积分、不自回归、不生成轨迹**。
- **为何用 Flow Matching 而非直接 MSE 回归**：同一组离散码对应的动作轨迹是多模态的（同一“抓取”码可对应多条合法轨迹）。直接回归会把多模态目标平均成模糊轨迹，弱化码的判别性；流匹配以“噪声→数据”的条件速度场建模整个条件分布，提供更强、更细的重构梯度，从而逼迫码本携带更丰富的信息。该建模方式与 openpi / $\pi_0$ 的动作流匹配一致。

#### 3.6.1 重构目标与 Flow Matching 概率路径

**重构目标（连续部分）：** 归一化后的末端执行器位姿轨迹

$$
a_1 = [\tilde{p}_t;\; r_t^{6d}] \in \mathbb{R}^{T \times 9}, \quad (\text{位置 3 维} + \text{6D 旋转 6 维})
$$

其中 $\tilde{p}_t$、$r_t^{6d}$ 与 §3.1 编码器输入使用**完全相同的归一化**（位置分位数归一化、四元数转 6D）。夹爪 $g_t \in \{0,1\}$ **不纳入流匹配变量**，由独立的 BCE 头处理（见 §3.6.4–3.6.5）。

**线性（最优传输）概率路径：** 约定 $\tau=1$ 为干净数据、$\tau=0$ 为纯噪声。每个样本采样一个流时间 $\tau^{(b)} \sim \mathcal{U}(0,1)$，并采样同维高斯噪声 $\epsilon \sim \mathcal{N}(0, I) \in \mathbb{R}^{T \times 9}$，构造加噪轨迹与目标速度场：

$$
a^\tau = \tau \, a_1 + (1-\tau)\,\epsilon \in \mathbb{R}^{T \times 9}
$$

$$
u = \frac{d\,a^\tau}{d\tau} = a_1 - \epsilon \in \mathbb{R}^{T \times 9} \quad \text{（沿线性路径为常向量，即速度场回归目标）}
$$

解码器以 $a^\tau$ 与流时间 $\tau$ 为输入、以量化码为条件，预测速度场 $\hat{v} \approx u$。$\tau$ 默认每样本采一个并在时间维共享，亦可改用偏向高噪声端的 Beta 分布以强化低 $\tau$ 区域的重构。

> **与 openpi 的约定对齐说明**：openpi / $\pi_0$ 同样采用 $A^\tau = \tau A_1 + (1-\tau)\epsilon$ 的线性路径并回归条件速度场（方向约定可能取 $\epsilon - A_1$，与本文符号相差一个负号，二者等价）。本质差异仅在于：openpi 在**推理**时从纯噪声出发用 Euler 法积分该速度场来**生成**动作；而本模块只做**训练期**的速度场回归，**不做任何积分**。

#### 3.6.2 条件注入架构

**输入投影：** 加噪轨迹先投影到主干维度

$$
h^{(0)} = \text{Linear}(9 \to 512)(a^\tau) \in \mathbb{R}^{T \times 512}
$$

序列位置由归一化 RoPE（§6.5）在注意力内部注入，padding 帧由 `trajectory_mask` 屏蔽。

**时间步嵌入与全局条件融合（AtomAction FM Head 专用）：**

$$
e_\tau = \text{SinusoidalEmbed}(\tau, d_{te}) \in \mathbb{R}^{d_{te}}
$$

$$
c = \text{MLP}([f_q^{global};\; e_\tau]) \in \mathbb{R}^{512}
$$

其中 $c$ 为整条轨迹共享的条件向量，对应当前 Flow Matching 步的噪声层级（流时间 $\tau$）与全局动作语义；$d_{te}$ 为时间步嵌入维度（由 `flow_matching_head.time_embed_dim` 配置，当前为 256）。`condition_fusion` MLP 首层维度为 `global_code_dim(512) + time_embed_dim(256) = 768 → 512`（`config/model.yaml::flow_matching_head.global_code_dim` / `condition_hidden_dim`）。

> 注：此处的条件融合方式仅适用于 AtomAction_NSVQ 的 `FlowMatchingHead`（全局码 + 时间）。VASA 侧的 `VisualFlowMatchingHead` 条件向量仅含时间嵌入（$c = \text{MLP}(e_\tau)$，无全局码），视觉语义改由 $F_{diff}$ 通过 cross-attention 注入，详见 §4.2。

**全局码 + 时间通过 AdaLN-gated 调制每个主干子层：**

对每个解码层的每个残差子层 $s \in \{\text{SelfAttn},\ \text{FFN}\}$（有 Cross-Attn 的层还包含 $s=\text{CrossAttn}$），由条件向量 $c$ 通过**各子层独立的** `AdaptiveModulation` 产生调制参数：

$$
[\gamma_{s}^{(\ell)}, \beta_{s}^{(\ell)}, g_{s}^{(\ell)}] = \text{Linear}_s^{(\ell)}(512 \to 1536)(c), \quad \ell = 1, \dots, 8
$$

其中每个 `AdaptiveModulation` 包含一个独立的 $\text{Linear}(C \to 3C)$（即 $512 \to 1536$），权重与 bias 均 `zeros_init`；各子层互不共享参数。

定义

$$
\operatorname{AdaRMSNorm}(x; c) = (1 + \gamma) \odot \text{RMSNorm}(x) + \beta
$$

$$
x \leftarrow x + g_{s}^{(\ell)} \odot \text{Sublayer}_{s}(\operatorname{AdaRMSNorm}(x; c))
$$

其中 $g_{s}^{(\ell)} \in \mathbb{R}^{512}$ 为逐通道残差门控；其作用不是直接改写主干状态，而是控制当前子层输出写回主干的强度。训练初期将调制层线性映射做 `zeros_init`，可得到 $\gamma=\beta=g=0$，于是网络从”近似恒等映射 + 关闭残差更新”状态开始优化。

**细节码通过 Cross-Attention（每两层一次，即第 2、4、6、8 层）：**

$$
F_q^{detail} = [f_{q,1}^{detail}, \dots, f_{q,9}^{detail}] \in \mathbb{R}^{9 \times d_{code}} = \mathbb{R}^{9 \times 512}
$$

$$
\hat{F}_q^{detail} = \text{Linear}(d_{code} \to 512)(F_q^{detail}) \in \mathbb{R}^{9 \times 512}
$$

（当前 $d_{code}=512$，因此 `detail_projection` 为 `Linear(512, 512)`，近似等价映射。）

$$
x \leftarrow x + g_{ca}^{(\ell)} \odot \text{CrossAttn}(Q=\operatorname{AdaRMSNorm}(x; c), K=\hat{F}_q^{detail}, V=\hat{F}_q^{detail}), \quad \ell \in \{2, 4, 6, 8\}
$$

其中 $g_{ca}^{(\ell)}$ 同样由条件向量 $c$ 生成，用于限制细节码注入强度；这样全局语义负责控制“该往哪里更新”，细节码负责补充“具体如何更新”。

#### 3.6.3 解码器完整配置（8 层主干 + 分支）

| 组件 | 规格 |
|---|---|
| 输入 | 加噪 ee 轨迹 $a^\tau \in \mathbb{R}^{T \times 9}$ + 流时间 $\tau$（夹爪不加噪、不作为输入） |
| 输入投影 | $\text{Linear}(9 \to 512)$，将 $a^\tau$ 投影到主干隐藏维 |
| 主干层数 | 8 层双向注意力（重构任务可用双向） |
| 主干隐藏维 | 512 |
| 注意力头数 | 8，每头 64 |
| FFN 维度 | 2048 |
| 归一化与全局条件注入 | RMSNorm + AdaRMSNorm-gated（每个子层独立生成 scale / shift / gate，由条件 $c$ 产生） |
| 细节条件注入 | 第 2、4、6、8 层 Cross-Attn + gated residual（K/V = 升维后的细节码 $\hat{F}_q^{detail}$） |
| 序列位置编码 | 归一化 RoPE（见 §6.5） |
| 位置支路（速度场头） | 主干后 2 层 Attn + 2 层 MLP，输出 $\hat{v}^{pos} \in \mathbb{R}^{T \times 3}$ |
| 旋转支路（速度场头） | 主干后 2 层 Attn + 2 层 MLP，输出 $\hat{v}^{rot} \in \mathbb{R}^{T \times 6}$ |
| 夹爪支路（BCE 头） | 复用位置支路中间特征，2 层 MLP，输出 logit $\hat{g} \in \mathbb{R}^{T \times 1}$（非速度场） |

#### 3.6.4 输出头与维度流

主干输出 $h^{(8)}$ 后接三个并行头。**位置 / 旋转头预测 Flow Matching 速度场，夹爪头直接预测二值状态 logit**：

$$
a^\tau \in \mathbb{R}^{T \times 9} \xrightarrow{\text{Linear}(9 \to 512)} h^{(0)} \in \mathbb{R}^{T \times 512} \xrightarrow{\text{8-layer decoder (cond. on } c,\ \hat{F}_q^{detail})} h^{(8)} \in \mathbb{R}^{T \times 512}
$$

$$
\xrightarrow{\text{position head}} \hat{v}_t^{pos} \in \mathbb{R}^{T \times 3} \qquad \xrightarrow{\text{rotation head}} \hat{v}_t^{rot} \in \mathbb{R}^{T \times 6} \qquad \xrightarrow{\text{gripper head}} \hat{g}_t \in \mathbb{R}^{T \times 1}
$$

> **速度场 vs. 状态 logit（关键区别）**：
> - $\hat{v}_t^{pos}$、$\hat{v}_t^{rot}$ 是 Flow Matching 的**速度场**，回归目标为 $u = a_1 - \epsilon$ 的对应分量（见 §3.6.5），**不是动作本身**；
> - $\hat{g}_t$ 是夹爪开合的**二值 logit**：夹爪未被加噪、不属于流匹配变量 $a^\tau$，该头从被条件调制的主干隐状态**直接预测当前帧夹爪状态**（BCE）。这是有意为之——二值离散量不适合连续速度场建模，且训练专用、无需积分，单步分类即可；高噪声（$\tau \to 0$）时输入近乎纯噪声，夹爪预测主要依赖码本条件，正好迫使码携带夹爪信息。
>
> 因此流匹配变量维度为 9（pos 3 + rot 6），夹爪是独立的辅助分类输出，合计重构 $3+6+1=10$ 维 ee 动作。6D 旋转速度场在 $\mathbb{R}^6$ 中回归即可，无需在训练损失中做正交化（仅当真要还原旋转矩阵时才施加 Gram-Schmidt）。

#### 3.6.5 重构损失（$L_{AP}$）

只对有效时间步（$m_t=1$）计算。位置 / 旋转用速度场 MSE，目标为 §3.6.1 的 $u = a_1 - \epsilon$ 的对应分量（$u_t^{pos} = \tilde{p}_t - \epsilon_t^{pos}$，$u_t^{rot} = r_t^{6d} - \epsilon_t^{rot}$）：

$$
L_{FM}^{pos} = \frac{1}{\sum m_t} \sum_{t=1}^{T} m_t \cdot \|\hat{v}_t^{pos} - u_t^{pos}\|_2^2
$$

$$
L_{FM}^{rot} = \frac{1}{\sum m_t} \sum_{t=1}^{T} m_t \cdot \|\hat{v}_t^{rot} - u_t^{rot}\|_2^2
$$

$$
L_{BCE}^{grip} = \frac{1}{\sum m_t} \sum_{t=1}^{T} m_t \cdot \text{BCE}\bigl(\sigma(\hat{g}_t),\; g_t\bigr)
$$

$$
L_{AP} = L_{FM}^{pos} + \lambda_{rot} L_{FM}^{rot} + \lambda_{grip} L_{BCE}^{grip} + \lambda_{sep} L_{sep}
$$

推荐：$\lambda_{rot}=1.0$，$\lambda_{grip}=1.0$，$\lambda_{sep}=0.01$

> **无承诺损失项**：NSVQ 以噪声替换近似量化残差，码本由端到端梯度直接更新（§3.4.3），因此 $L_{AP}$ **不含** commitment / VQ 损失项，与 §5.2 的总损失定义一致。

---

### 3.7 AtomAction-NSVQ 各阶段输出汇总

下表列出 AtomAction-NSVQ 模块在三个阶段的输出张量及其用途：

| 输出张量 | 形状 | 阶段一（联合建模） | 阶段二（增强 $L_{future}$） | VQAP-VLA 推理 |
|---|---|---|---|---|
| `z` | $[B,\, T,\, 512]$ | Transformer 输出，传入两条量化支路 | 同左 | 不运行编码器 |
| `f_q_global` | $[B,\, d_{code}] = [B,\, 512]$ | NSVQ 训练近似量化向量，传入 FM 解码器 | 额外传入 VASA FiLM 支路，贡献 $L_{future}$ | 由 VLA 预测 $k_g^*$ 后，经 `GlobalCodebookModule.lookup_codewords()` 直接查表得到 |
| `k_g` | $[B]$ | 全局码索引（用于 perplexity 监控、后续 VLA 伪标签） | 同左 | VLA 的预测目标 |
| `F_q_detail` | $[B,\, N_{detail},\, d_{code}] = [B,\, 9,\, 512]$ | 细节量化向量，传入 FM 解码器 Cross-Attn | 码本**冻结**，向量参与解码但不更新 | 由 VLA 预测 $K_d^*$ 后，经 `DetailCodebookModule.lookup_codewords()` 直接查表得到 |
| `K_d` | $[B,\, N_{detail}] = [B,\, 9]$ | 细节码索引矩阵（perplexity 监控、伪标签） | 同左（不反传） | VLA 的预测目标 |
| `perplexity_g/d` | 标量 | 码本利用率监控，每步记录 | 同左 | — |
| 损失 | — | $L_{AP} + L_{AG} + \lambda_{future}(e) L_{future}$（$\lambda_{future}$ 由 0 线性升至 0.1） | $L_{AP} + \lambda_{future}^{high} L_{future}$（$L_{AG}$ 停用） | 不计算损失 |

**训练 vs 推理的关键差异：**
- **训练时**：`f_q_global` / `F_q_detail` 为 NSVQ 噪声近似值（$q = h + \text{vq\_error}$），含随机成分
- **推理时**：`f_q_global` / `F_q_detail` 为硬量化码本向量 $e_{k^*}$，确定性输出；量化器直接查表，不再执行 argmin

> **当前代码接口对齐**：`AtomAction_NSVQ.forward` 在 eval 模式下不接受外部码本索引（当前签名无此参数）。VQAP-VLA 推理阶段的查表操作通过直接调用 `GlobalCodebookModule.lookup_codewords(codebook_indices)` 和 `DetailCodebookModule.lookup_codewords(codebook_indices)` 完成，绕过动作编码器，仅执行码本向量查找；`NSVQQuantizer.forward` 在 eval 模式下要求必须显式传入 `codebook_indices`，否则抛错。

---

## 四、VASA 完整设计

> **前置说明——视角选择属于数据集构建阶段，不属于 VASA 模块：**
> 视角选择已在 §2.4 中完成，使用**冻结**的预训练 DINOv2 对五个视角的起始帧与结尾帧提取全局 CLS 特征，计算余弦相似度，选出变化量最大的 top-$k$ 视角，结果缓存到磁盘。VASA 在训练时直接读取已选好的 $k$ 个视角的起始帧 $\{I_s^v\}_{v=1}^k$ 与结尾帧 $\{I_e^v\}_{v=1}^k$（以及对应的变化分数 $\{\text{score}^v\}$），**不再重新执行视角评分**。

### 4.1 多视角图像特征提取

当前代码中，VASA 的入口就是 `model/module/encoder.py::ImageEncoder`。它直接读取 batch 级 `selected_views`，对每个视角的起始帧 / 结尾帧分别调用官方 DINOv2 backbone 提取图像特征，再按数据集构建阶段缓存的 `best_score` 做加权融合，最终得到融合后的起始帧特征、结尾帧特征与视角权重 $w$。`DINO_FeatureExtractor` 当前支持两种输出模式：

1. `feature_type=cls`：直接输出 CLS 特征；
2. `feature_type=patch`：直接输出 `x_norm_patchtokens`，保留 `[N, P, 768]` 的 patch 序列，不做额外池化。

**DINOv2 图像特征提取：**

$$
c_s^v = \text{DINO}_{\text{cls}}(I_s^v) \in \mathbb{R}^{768}, \quad c_e^v = \text{DINO}_{\text{cls}}(I_e^v) \in \mathbb{R}^{768}
$$

$$
C_s^v = \text{DINO}_{\text{patch}}(I_s^v) = x\_\text{norm\_patchtokens}(I_s^v) \in \mathbb{R}^{P \times 768}, \quad C_e^v = \text{DINO}_{\text{patch}}(I_e^v) = x\_\text{norm\_patchtokens}(I_e^v) \in \mathbb{R}^{P \times 768}
$$

其中 `feature_type=cls` 时输出单个 CLS 向量，`feature_type=patch` 时直接输出 patch token 序列。所有起始帧 / 结尾帧 / 视角共享同一个 DINOv2 编码器实例（参数共享），当前 `model.yaml` 默认配置已切到 patch 特征。

**多视角加权融合（显式区分 $k=1$ 与 $k>1$）：**

当 top-$k > 1$ 时，以数据集构建阶段保存的变化分数 $\text{score}^v = 1 - \cos(f_s^v, f_e^v)$ 作为权重，对 $k$ 个视角的 CLS 特征做加权平均：

$$
w^v = \frac{\text{score}^v}{\sum_{j=1}^k \text{score}^j}, \quad \sum_{v=1}^k w^v = 1
$$

当 `feature_type=cls` 时：

$$
c_s = \sum_{v=1}^k w^v c_s^v \in \mathbb{R}^{768}, \quad c_e = \sum_{v=1}^k w^v c_e^v \in \mathbb{R}^{768}
$$

当 `feature_type=patch` 时：

$$
C_s = \sum_{v=1}^k w^v C_s^v \in \mathbb{R}^{P \times 768}, \quad C_e = \sum_{v=1}^k w^v C_e^v \in \mathbb{R}^{P \times 768}
$$

当 $k=1$ 时，不做加权融合，直接取唯一视角特征：

$$
w^1 = 1, \quad c_s = c_s^1, \quad c_e = c_e^1
$$

若采用 patch 模式，则对应退化为 $C_s = C_s^1, C_e = C_e^1$。这样实现时无需额外的 mask 分支，也避免对长度为 1 的视角列表做无意义的重复堆叠。后续 §4.2、§4.3 中，若使用单向量视觉条件则继续记作 $c_s$、$c_e$；若使用 patch 序列视觉源，则记作 $C_s$、$C_e$。至此，VASA 可以视为从共享视觉入口分出两条下游支路：

1. 未来帧潜空间预测支路：使用 AtomAction_NSVQ 的全局语义码与 $c_s$ 预测 $c_e$。
2. Action Grounding 支路：以结尾帧特征为 query、初始帧特征为 key/value，先做一次 cross-attention，再显式叠加始末帧差分项，最后经两层 self-attention 提取图像差分特征序列 `img_diff_features`，再将其送入视觉侧 Flow Matching Head。

> **两处 DINOv2 的区别总结：**
> | | §2.4 视角选择 | §4.1 VASA 特征提取 |
> |---|---|---|
> | 所在阶段 | 数据集构建（离线，缓存） | 训练前向（在线） |
> | 参数状态 | **冻结**，不参与梯度 | 独立官方 DINOv2 实例，可通过 VASA 损失反传更新 |
> | 输出用途 | 计算余弦相似度，选视角 | CLS 特征 $c_s/c_e$，输入 VASA 两条支路 |
> | 输入 | 五视角所有图像（选角时） | 已选 top-$k$ 视角图像（训练时） |

---

### 4.2 Action Grounding（视觉-动作对齐支路）

Action Grounding 的核心目标是：先从起始 / 结尾观测中提取**视觉交互后的 patch 级变化特征序列**，再把这组变化特征作为视觉侧 Flow Matching Head 的条件输入。主方案优先保持路径最短、语义最直接，同时尽可能复用 AtomAction_NSVQ 中已经实现好的 Flow Matching 主干与输出分支。

> **方案选型与信息上限（重要前提）**
>
> 本支路的唯一作用是让 $L_{AG}$ 的向量场回归梯度穿过视觉交互编码器与 ImageEncoder 回传到 DINOv2，**使视觉特征逐渐对动作敏感**。是否值得引入"细节条件"，取决于视觉侧到底有多少动作信息可供榨取，这是一个硬上限：
> - **§4.1 当前默认配置取 DINOv2 的 patch token 序列**（起始帧、结尾帧各 1 组 $P \times 768$ 特征），同时仍保留 CLS-only 回退模式。若退回 CLS-only 设定，则 Action Grounding 能 ground 的信息上限 = 这两个全局向量所含内容；两帧之间不可见的量（速度曲线、接触力时序、中间轨迹形状）在物理上不可恢复，应继续由 AtomAction 动作侧的 $L_{AP}$ 承载，视觉侧只 ground 可见的几何 / 空间变化。
> - 因此存在一个**决定性变量**：视觉源是 **CLS-only** 还是 **patch token**。
>   - **CLS-only**（现状）→ 2 个全局向量只够支撑单条件，应选**方案1**；强行复刻 AtomAction 的 9 细节 query 会因源信息不足而坍缩（§6.1 的细节码坍缩在此更严重），属于伪结构。
>   - **patch token**（每帧 16×16=256 个）→ 空间细节真实存在，多 query 交叉注意力才有意义，且能驱动 **patch 特征**而非仅 CLS。注意 CLS 相关 ≠ patch 相关；下游 VLA（VLA_Design.md §3 感知层）恰恰吃 patch token，因此"让 patch 特征动作相关"才同时服务本目标与下游。
>
> **当前选型**：`ImageEncoder` 默认采用 `feature_type=patch`，直接输出多视角加权后的 patch 序列；视觉交互编码器则以结尾帧 patch 序列为 query、起始帧 patch 序列为 key/value，先做一次 cross-attention，再显式加上始末帧差分项，最后经过两层 self-attention，输出图像差分特征序列 $F_{diff}$。CLS-only 模式保留为轻量回退 / 消融设置，此时序列长度退化为 1；**方案2（patch 源 + 结构化非量化条件 + $L_{align}$）** 仍作为追求"强相关"的备选增强，详见 §4.2.2。

这里不再把视觉交互编码器做成纯 cross-attention 堆叠，而是采用更直接的差分建模：**以结尾帧图像特征为 query，以初始帧图像特征为 key/value，先做一次 cross-attention；随后显式加入始末帧差分项 $c_e-c_s$；最后再用两层 self-attention 整理差分特征序列**。由于输入是图像 patch 特征而不是一维动作轨迹，当前实现**不额外叠加 1D RoPE**。

先将结尾帧特征与起始帧特征分别投影到与动作主干一致的隐藏维度。若采用 patch 模式，则 $T=P$；若采用 CLS 模式，则 $T=1$：

$$
Q_e = W_q X_e \in \mathbb{R}^{B \times T \times 512}, \quad KV_s = W_{kv} X_s \in \mathbb{R}^{B \times T \times 512}
$$

同时将始末帧原始差分投影到同一隐藏维度：

$$
\Delta_{img} = W_{\Delta}(X_e - X_s) \in \mathbb{R}^{B \times T \times 512}
$$

视觉交互编码器的完整前向（与 `VisualTransformerEncoder.forward` 对应）：

$$
F_{cross} = \text{CrossAttention}\!\bigl(\text{LN}(Q_e),\; \text{LN}(KV_s),\; \text{LN}(KV_s)\bigr) \in \mathbb{R}^{B \times T \times 512}
$$

$$
F_{diff}^{(0)} = \text{dropout}(F_{cross}) + \Delta_{img} \in \mathbb{R}^{B \times T \times 512}
$$

$$
F_{diff} = \text{SelfAttnBlock}_2\!\bigl(\text{SelfAttnBlock}_1\!\bigl(F_{diff}^{(0)}\bigr)\bigr) \in \mathbb{R}^{B \times T \times 512}
$$

**实现细节（与代码对齐）：**
- cross-attention Q 与 K/V **分别**经独立的 Pre-LayerNorm（`cross_attention_query_norm`、`cross_attention_key_value_norm`），对齐代码中 `self.cross_attention_query_norm(query_tokens)` 与 `self.cross_attention_key_value_norm(key_value_tokens)`；
- $\Delta_{img}$ 加在 cross-attention 输出之后（含 dropout），$Q_e$ 本身不再作为残差写回；
- key/value mask 全为 `True`（图像 patch 无 padding），无需掩蔽；
- `SelfAttnBlock` 为 `VisualSelfAttentionLayer`，其 attention 子层包含残差连接，**当前实现的 FFN 子层不含残差**（即输出 = $\text{FFN}(\text{LN}(x))$，而非 $x + \text{FFN}(\text{LN}(x))$）；当前代码行为如此，后续如有需要可补上 FFN 残差。

其中 $F_{diff}$ 表示”结尾观测在起始观测上下文中的图像差分特征序列”，在代码中对应输出键 `img_diff_features`。

**序列条件视觉 FM Head（主方案）：**

视觉侧 Flow Matching Head（`VisualFlowMatchingHead`）的输入接口定义为：

$$
\hat{y}_{AG} = \text{FlowMatchingHead}_{vis}(a^\tau, \tau, F_{diff}, m)
$$

其中 $F_{diff} \in \mathbb{R}^{B \times T_{vis} \times 512}$ 是唯一视觉条件来源，不再拆成 `global_codeword + detail_codewords` 两路接口。与 §3.6 的 AtomAction FM Head 相比，关键差异在于条件向量的构成：

- **AtomAction FM Head**：$c = \text{MLP}([f_q^{global};\, e_\tau]) \in \mathbb{R}^{512}$（全局码 + 时间）
- **视觉 FM Head**：$c = \text{MLP}(e_\tau) \in \mathbb{R}^{512}$（**仅时间**，不含全局码）；视觉语义由 $F_{diff}$ 通过 cross-attention 注入（对应 `condition_injection_layers`）

当前阶段先把接口固定为”视觉序列条件”，后续视觉 FM Head 可直接消费这组 token，而不是要求视觉交互编码器先压缩成单向量。这样可以优先保证三点：

1. 先用 cross-attention 抽取”终帧对起始帧”的对齐变化；
2. 再显式保留始末帧差分残差，不丢失直接变化量；
3. 最后仅用两层 self-attention 在差分序列内部做整理，保留 patch 级空间细节。

**独立 Flow Matching Head：**

Action Grounding 使用的仍是独立的 `VisualFlowMatchingHead`。它在主干结构和输出头设计上尽量复用 §3.6 的实现思路，但视觉条件接口改为接收序列 $F_{diff}$；当前版本优先完成视觉交互编码器本体，后续再将这组视觉 token 接入视觉 FM Head。

Action Grounding 的监督保持为动作向量场回归：

$$
L_{AG} = L_{FM,img}^{pos} + \lambda_{rot} L_{FM,img}^{rot} + \lambda_{grip} L_{BCE,img}^{grip}
$$

这一支路中，$c_s$ 与 $c_e$ **不做 detach**。因此 $L_{AG}$ 的梯度会穿过视觉交互编码器与 ImageEncoder 回传到 DINOv2，使视觉特征逐渐对动作更敏感。

> **提醒：结构化条件方案作为备选增强**
> 若后续实验发现“单个图像差分序列 `img_diff_features` 作为唯一视觉条件来源”表达能力不足，可以再升级到结构化条件方案：在保留当前主路径的基础上，额外引入少量视觉 detail tokens 或 learnable visual queries，为视觉 FM Head 增加显式的细节条件注入。该方案保留为增强选项，但当前版本不作为主路径。

---

### 4.3 未来帧潜空间预测支路（全局码 FiLM）

这一路的目标不是直接生成图像，而是在 DINOv2 的潜空间中预测结尾帧 **patch 特征**。当前 `ImageEncoder` 默认 `feature_type=patch`，起始帧与结尾帧均输出 patch token 序列：$C_s, C_e \in \mathbb{R}^{P \times 768}$。输入为 AtomAction_NSVQ 输出的全局语义码 `global_codeword` 与起始帧 patch 序列 $C_s$；输出为结尾帧 patch 潜向量预测 $\hat{C}_e \in \mathbb{R}^{P \times 768}$。

> **为什么选 patch 而非 CLS？** 下游 VLA 消费的是 patch token，让全局码对齐 patch 级视觉结局既服务于本阶段训练目标，也为下游 patch 特征的动作语义打下基础。CLS-only 作为轻量回退 / 消融设置保留。

**FiLM 调制起始帧 patch 序列：**

全局语义码升维至与 patch 特征对齐，FiLM 参数 **broadcast** 至所有 patch 位置：

$$
[\gamma, \beta] = \text{Linear}(d_{code} \to 768 \times 2)(f_q^{global}) \in \mathbb{R}^{768} \times \mathbb{R}^{768}
$$

其中 $d_{code}=512$（`config/model.yaml::future_predictor.global_code_dim`），`film_linear` 为 `Linear(512 → 1536)`。同一组 $(\gamma, \beta)$ 独立作用于每个 patch，不引入额外的 patch 间交互。

**多层 FFN 预测未来帧 patch 潜向量：**

$$
\hat{C}_e = \text{FFN}_{future}(H_{film}) \in \mathbb{R}^{P \times 768}
$$

推荐使用 3 层 MLP（逐 patch 独立作用，参数共享）：

`768 -> 3072 -> 3072 -> 768`

**损失：逐 patch cosine 对齐**

对预测结果与目标逐 patch 独立计算余弦相似度，再按 patch 数平均：

$$
L_{future} = \frac{1}{P}\sum_{p=1}^{P} \left(1 - \cos\!\left(\hat{C}_e[p],\; \text{sg}\!\left(C_e[p]\right)\right)\right)
$$

这里对目标侧 $C_e$ 使用 stop-gradient，使该支路主要承担”全局语义码 + 起始观测 $\to$ 未来语义状态”的建模任务，而不是驱动视觉编码器本身漂移。逐 patch 计算相比先 mean-pool 再计算余弦能够保留更细粒度的空间对齐梯度；视觉特征与动作的对齐主要由 §4.2 的 Action Grounding 支路负责。

> **细节码不注入此支路。** $L_{future}$ 的职责是让全局码”知道”任务的视觉结局，细节码（子轨迹时序片段）不应越权参与视觉外观预测，避免梯度来源混乱。细节码对视觉的约束通过 $L_{AG}$ 支路的 FM 解码器 cross-attention 实现。

---

## 五、总损失与训练工作流

### 5.0 VQAP 预训练总览

**数据集**：AtomAction_Dataset，~58,000 条 phase 级别数据（RLBench 多任务专家演示，每条为一个原子动作片段轨迹）。训练仅使用白名单内的动作类型：15 种核心动作 + `approach`/`transfer`（暂定）；`pose-adjust` 永久排除（详见 §2.3）

**训练定位**：VQAP 训练是面向 VLA 模型的**码本预训练阶段**，核心目标是构建含有原子动作语义的离散双码本（全局 $K_g=36$、细节 $K_d$（当前 $K_d=192$））。VLA 推理时直接复用该码本，不重新训练 VQAP。

**VASA 的辅助角色**：VASA 并非独立目标，而是服务于码本训练的两条辅助支路：

| 支路 | 损失 | 目的 |
|---|---|---|
| Action Grounding | $L_{AG}$ | 通过始末帧图像差分特征预测轨迹，使 DINOv2 特征对动作变化敏感；为 $L_{future}$ 提供动作相关的高质量视觉目标 |
| Future Prediction | $L_{future}$ | 以全局码 + 起始帧预测末帧 patch 特征，给全局码注入高维视觉语义，弥补从低维噪声轨迹数据单独学习码本的不足 |

**训练规模**：总计 $E_0 + E_1$ epoch（`config/train.yaml::stage`），当前 $E_0=150$（Stage 0）、$E_1=100$（Stage 1），batch=$B$（`data.batch_size`，当前 $B=64$）；每个 epoch 评估并更新 best 权重，`latest.pth` / `codebook.pth` 按 `save_every_epochs`（当前 2）节流，并在每个 stage 末尾强制保存。

---

### 5.1 各损失含义与监督信号

#### $L_{AP}$（AtomAction Reconstruction Loss，动作重构损失）

以双码本量化向量（全局码 $f_q^{global} \in \mathbb{R}^{d_{code}}$、细节码 $F_q^{detail} \in \mathbb{R}^{N_{detail} \times d_{code}}$）为条件，通过 8 层 AdaLN-gated Flow Matching 解码器（§3.6）重构专家轨迹。是双码本信息压缩的核心约束，驱动整条动作编码链路。

| 子损失 | 公式 | 监督信号来源 |
|---|---|---|
| $L_{FM}^{pos}$ | $\frac{1}{\sum m_t}\sum_t m_t \|\hat{v}_t^{pos} - u_t^{pos}\|_2^2$ | EE 位置速度场 $u^{pos} = \tilde{p} - \epsilon^{pos}$，来自 `gripper_pose` 前 3 维 |
| $L_{FM}^{rot}$ | $\frac{1}{\sum m_t}\sum_t m_t \|\hat{v}_t^{rot} - u_t^{rot}\|_2^2$ | EE 旋转速度场 $u^{rot} = r^{6d} - \epsilon^{rot}$，来自四元数转 6D 旋转 |
| $L_{BCE}^{grip}$ | $\frac{1}{\sum m_t}\sum_t m_t \,\text{BCE}(\sigma(\hat{g}_t), g_t)$ | 夹爪开合 $g_t \in \{0,1\}$，来自 `gripper_open`（不加噪，直接分类） |
| $L_{sep}$ | $\lvert\cos(\bar{z}^{global},\, \frac{1}{9}\sum_n Z_q^{detail,n})\rvert$ | 无外部标签；约束全局与细节 pre-projection 512 维特征的表示正交性 |

$$L_{AP} = L_{FM}^{pos} + \lambda_{rot} L_{FM}^{rot} + \lambda_{grip} L_{BCE}^{grip} + \lambda_{sep} L_{sep}$$

**梯度流**：经 FM 解码器 → 双码本 → 支路投影 → TransformerEncoder → ChannelEncoder → 动作特征投影。

---

#### $L_{AG}$（Action Grounding Loss，视觉-动作对齐损失）

以视觉交互编码器输出的图像差分特征 $F_{diff} \in \mathbb{R}^{B \times P \times 512}$ 为唯一条件，通过独立的视觉 FM Head 重构专家轨迹（输出头结构与 $L_{AP}$ 一致）。

$$L_{AG} = L_{FM,img}^{pos} + \lambda_{rot} L_{FM,img}^{rot} + \lambda_{grip} L_{BCE,img}^{grip}$$

**关键设计**：$C_s$、$C_e$ 不做 detach，$L_{AG}$ 梯度直接穿过视觉交互编码器 → ImageEncoder → DINOv2 LoRA，是唯一显式驱动 DINOv2 更新的路径。**Stage 1 停用**（DINOv2 已冻结）。

---

#### $L_{future}$（Future Patch Prediction Loss，未来帧潜空间对齐损失）

以全局码 $f_q^{global}$ 经 FiLM 调制起始帧 patch 特征 $C_s$ 为输入，通过逐 patch 共享参数的 3 层 FFN 预测结尾帧 patch 特征 $\hat{C}_e \in \mathbb{R}^{P \times 768}$：

$$L_{future} = \frac{1}{P}\sum_{p=1}^{P}\left(1 - \cos\!\left(\hat{C}_e[p],\; \text{sg}(C_e[p])\right)\right)$$

**关键设计**：目标侧 $C_e$ 使用 stop-gradient；Stage 1 中 DINOv2 完全冻结，$C_s$ 亦无梯度回传至视觉编码器。梯度**仅流向** FiLM 参数、FFN_future 以及全局量化路径（$f_q^{global}$ 经 NSVQ 可微近似）。全局码被迫携带”任务视觉结果”信息，使其不仅编码动作模式，还隐含目标空间状态。

---

#### 训练时监督信号汇总

| 监督来源 | 数据形状 | 用于哪些损失 | 是否在线采样 |
|---|---|---|---|
| EE 位置速度场 $u^{pos} = \tilde{p} - \epsilon^{pos}$ | $[B, T, 3]$ | $L_{FM}^{pos}$（$L_{AP}$、$L_{AG}$） | $\epsilon$ 在线采样，$\tilde{p}$ 来自数据集 |
| EE 旋转速度场 $u^{rot} = r^{6d} - \epsilon^{rot}$ | $[B, T, 6]$ | $L_{FM}^{rot}$（$L_{AP}$、$L_{AG}$） | $\epsilon$ 在线采样，$r^{6d}$ 来自数据集 |
| 夹爪开合状态 $g_t \in \{0,1\}$ | $[B, T, 1]$ | $L_{BCE}^{grip}$（$L_{AP}$、$L_{AG}$） | 数据集，不加噪 |
| 结尾帧 DINOv2 patch 特征 $C_e$（stop-grad） | $[B, P, 768]$ | $L_{future}$ | DINOv2 前向在线提取 |
| 流时间 $\tau \sim \mathcal{U}(0,1)$ | $[B]$ | 所有 FM 损失（构造加噪轨迹 $a^\tau$） | 每步在线采样 |
| 噪声 $\epsilon \sim \mathcal{N}(0,I)$ | $[B, T, 9]$ | $L_{FM}^{pos}$、$L_{FM}^{rot}$ | 每步在线采样 |

---

### 5.2 模块连接与数据流

```
DataLoader
  ├─ trajectory_data [B, T_max, D]
  ├─ trajectory_mask  [B, T_max]
  └─ selected_views   [B, top_k, Dict]
        │
        ├─ 动作特征投影 (§3.1) ─────────────► trajectory_features [B, T, 512]
        │                                              │
        │                              ChannelEncoder (§3.2)
        │                                              │
        │                           channel_encoded [B, T, 512]
        │                                              │
        │                          TransformerEncoder (§3.3)
        │                                              │
        │                                     z [B, T, 512]
        │                                    ╱           ╲
        │              全局支路 (§3.4.1)              细节支路 (§3.4.2)
        │         masked_avg_pool                    LearnableQuery×CrossAttn
        │              │                                      │
        │         h_g [B,d_code]                    H_d [B,N_detail,d_code]
        │           NSVQ Q_g                           NSVQ Q_d
        │              │                                      │
        │    f_q_global [B,d_code]               F_q_detail [B,N_detail,d_code]
        │    k_g [B]                              K_d [B,9]
        │         │                    │
        │         └──── 信息分离约束 (§3.5) ──► L_sep
        │         │                    │
        │         └──── FM 解码器 (§3.6) ─────► L_AP
        │
        └─ selected_views → ImageEncoder → C_s, C_e, w [B, P, 768], [B, P, 768], [B, K]
                    │                    （patch 模式默认；CLS 模式退化为 [B,768]）
                  VASA (§四)
               ╱               ╲
            Action Grounding（视觉条件 FM）    Future Prediction（使用 f_q_global）
              │                         │
       visual interaction encoder            FiLM(C_s, f_q_global) broadcast per patch
              │                         │
     img_diff_features [B, P, 512]        FFN_future (per patch, params shared)
              │                         │
              │                     Ĉ_e [B, P, 768]
              │                         │ per-patch cos vs sg(C_e [B, P, 768])
     sequence-conditioned FlowMatchingHead_vis   │
              │                         │
            L_AG                    L_future
```

---

### 5.3 总损失

$$
L_{total} = \lambda_{AP} L_{AP} + \lambda_{AG} L_{AG} + \lambda_{future}(e) \cdot L_{future}
$$

（Stage 1 中 $\lambda_{AG} = 0$，即停用 $L_{AG}$）

$$
L_{AP} = L_{FM}^{pos} + \lambda_{rot} L_{FM}^{rot} + \lambda_{grip} L_{BCE}^{grip} + \lambda_{sep} L_{sep}
$$

$$
L_{AG} = L_{FM,img}^{pos} + \lambda_{rot} L_{FM,img}^{rot} + \lambda_{grip} L_{BCE,img}^{grip}
$$

$$
L_{future} = \frac{1}{P}\sum_{p=1}^{P}\left(1 - \cos\!\left(\hat{C}_e[p],\; \text{sg}(C_e[p])\right)\right)
$$

**固定权重**：$\lambda_{AP} = 1.0$，$\lambda_{AG} = 1.0$，$\lambda_{rot} = 1.0$，$\lambda_{grip} = 1.0$，$\lambda_{sep} = 0.01$

**$\lambda_{future}$ 按 epoch 线性调度**（$e$ 为当前 epoch）：

$$
\lambda_{future}(e) =
\begin{cases}
\dfrac{\lambda_{max}^{s0} \cdot e}{E_0} & 1 \le e \le E_0 \quad\text{（Stage 0，$0 \to \lambda_{max}^{s0}$）}\\[10pt]
\lambda_{max}^{s0} + \dfrac{(\lambda_{max}^{s1} - \lambda_{max}^{s0}) \cdot (e - E_0)}{E_{ramp}} & E_0 < e \le E_0 + E_{ramp} \quad\text{（Stage 1 前 $E_{ramp}$ epoch，爬升）}\\[10pt]
\lambda_{max}^{s1} & E_0 + E_{ramp} < e \le E_0 + E_1 \quad\text{（Stage 1 剩余 epoch，固定）}
\end{cases}
$$

$L_{future}$ 权重先低后高的原因：训练早期 $f_q^{global}$ 尚无稳定动作语义，此时强推”全局码预测未来帧”会把优化重心引偏到视觉外观拟合；Stage 0 以极小权重缓慢引入，保证 $L_{AP}$ 主导码本建立；Stage 1 前半段在码本和 DINOv2 均稳定后快速提升权重，后 $E_1$ 中固定为 $\lambda_{max}^{s1}=1.0$，让模型在全损失下充分收敛。代入当前值：epoch 1–150 从 $0\to0.1$，epoch 151–170 从 $0.1\to1.0$，epoch 171–250 固定 $1.0$（$E_0=150$, $E_1=100$, $E_{ramp}=20$）。

---

### 5.4 两阶段训练方案

#### Stage 0 — 联合预热（Epoch 1–$E_0$，当前 $E_0=150$）

**运行损失**：$L_{AP} + L_{AG} + \lambda_{future}(e) \cdot L_{future}$，$\lambda_{future}$ 从 0 线性爬升至 0.1

| 模块 | 状态 |
|---|---|
| 动作特征投影、ChannelEncoder、TransformerEncoder | 可训练 |
| 全局码本 $Q_g$（$K_g \times d_{code}$） | 可训练 |
| 细节码本 $Q_d$（$K_d \times d_{code}$） | 可训练 |
| AtomAction-NSVQ FM 重构解码器 | 可训练 |
| VASA 视觉交互编码器 + 视觉 FM Head | 可训练 |
| 未来帧 FiLM + FFN 预测器 | 可训练 |
| DINOv2（LoRA 微调，rank=8，融合 `attn.qkv`，全部 block） | 可训练（独立 lr=5e-5） |

**DINOv2 LoRA 配置**（`config/train.yaml::lora`，由 `_apply_dinov2_lora` 经 PEFT 注入到 `vasa.image_encoder.feature_extractor.backbone`）：

| 参数 | 值 |
|---|---|
| 应用范围 | backbone 内所有匹配 `target_modules` 的层（DINOv2 全部 block） |
| 目标模块 | `attn.qkv`（融合的 Q/K/V projection，DINOv2 ViT 用单个 qkv linear） |
| rank | 8 |
| alpha | 16（scale = 2.0） |
| dropout | 0.1 |
| bias | none |
| lr | 5e-5（独立参数组，约为主干 1/6） |

**Stage 0 结束时**：将 LoRA 权重 merge 进 DINOv2 base model，然后冻结 DINOv2 全部参数。

---

#### Stage 1 — 视觉语义注入（Epoch $E_0+1$–$E_0+E_1$，当前 151–250）

**运行损失**：$L_{AP} + \lambda_{future}(e) \cdot L_{future}$，$\lambda_{future}$ 从 $\lambda_{max}^{s0}=0.1$ 经 $E_{ramp}=20$ epoch 爬升至 $\lambda_{max}^{s1}=1.0$，之后固定

| 模块 | 状态 |
|---|---|
| 动作特征投影、ChannelEncoder、TransformerEncoder | 可训练 |
| 全局码本 $Q_g$、细节码本 $Q_d$ | 可训练 |
| AtomAction-NSVQ FM 重构解码器 | 可训练 |
| 未来帧 FiLM + FFN 预测器 | 可训练 |
| DINOv2（含已 merge 的 LoRA） | **完全冻结** |
| VASA 视觉交互编码器 + 视觉 FM Head | **不参与训练**（$L_{AG}$ 停用） |

Stage 1 中 DINOv2 冻结后，$C_s$ 和 $C_e$ 均为固定表示，$L_{future}$ 的梯度**只更新** FiLM、FFN_future 和 $f_q^{global}$ 路径，不影响任何视觉编码器参数。

---

### 5.5 超参数配置与训练监控

#### 优化器与学习率

| 参数组 | Stage 0 lr | Stage 1 lr |
|---|---|---|
| AtomAction-NSVQ 全部参数 | 3e-4 | 1e-4 |
| VASA 视觉交互编码器、视觉 FM Head、FiLM、FFN_future | 3e-4 | 1e-4（仅 FiLM、FFN_future） |
| DINOv2 LoRA | 5e-5 | 冻结 |

| 超参数 | 值 |
|---|---|
| Optimizer | AdamW |
| $\beta_1$ | 0.9 |
| $\beta_2$（AtomAction-NSVQ + VASA 主干） | 0.99 |
| $\beta_2$（DINOv2 LoRA） | 0.99 |
| weight_decay（矩阵参数，`ndim ≥ 2`） | 1e-3 |
| weight_decay（`ndim < 2`、`.bias`、含 `norm`/`embedding`/`codebooks` 的参数） | 0（见 `_should_skip_weight_decay`） |
| $\varepsilon$ | 1e-8 |
| Scheduler | `LambdaLR`：线性 warmup + cosine decay，每阶段独立创建，`scheduler.step()` 按 epoch 调用 |
| min_lr_ratio | 0.05（cosine 衰减下限；Stage 0 末：1.5e-5；Stage 1 末：5e-6） |
| Warmup（Stage 0 起始） | **2 epoch** 线性（epoch 0 → 50% peak，epoch 1 → peak） |
| Warmup（Stage 1 起始） | **2 epoch** 线性（epoch 0 → 50% peak = 5e-5，epoch 1 → peak = 1e-4） |
| Gradient clip norm | 1.0 |
| Batch size | 64（`data.batch_size`） |
| Precision | bf16（默认）；可配 fp16（启用 `GradScaler`）或 fp32（`runtime.precision`） |

#### 码本死码检测与替换

死码替换在**每个 epoch 结束后**评估一次（`_maybe_replace_dead_codebooks`），统计窗口为当前 epoch 的真实 batch 数 `used_steps`；触发后调用 `_reset_codebook_usage_counters` 清零计数器，避免窗口残留：

- **主要策略（perplexity 触发）**：用当前 epoch 的 `perplexity_g` / `perplexity_d`，若 `perplexity_g < K_g/4`（当前 $36/4=9$）或 `perplexity_d < K_d/4$（当前 $192/4=48$），即触发 `replace_unused_codebooks(used_steps=...)`（全局、细节各自独立）。阈值来自 `config/train.yaml::codebook.perplexity_g_threshold` / `perplexity_d_threshold`
- **固定间隔（可选）**：`(epoch + 1) % replace_interval_epochs == 0` 时也触发；`replace_interval_epochs` 默认 20，设为 `inf` 则只保留 perplexity 触发
- **DDP 同步**：替换前对 `codebooks_used` 做 `all_reduce(AVG)`，替换仅在 rank 0 执行，随后将替换数与新码本 `broadcast` 到所有 rank

#### 权重保存策略

best 判定**每个 epoch 都执行**（与 latest 节流解耦，避免漏掉非保存间隔上的最优）；`latest.pth` / `codebook.pth` 按 `save_every_epochs`（默认 2）节流，并在每个 stage 的最后一个 epoch 强制保存（保证 stage0 末状态可续训）。所有文件采用先写 `.tmp` 再原子替换的方式写入，并按当前 stage 落到 `stage{N}/` 子目录：

| 文件 | 更新条件 | 目录 |
|---|---|---|
| `latest.pth` | `save_every_epochs` 间隔或 stage 末尾 | `stage{N}/` |
| `codebook.pth` | 与 `latest.pth` 同频（仅码本权重，供 VLA 迁移） | `stage{N}/` |
| `best_lap.pth` | 当前 $L_{AP}$ < 历史最优 $L_{AP}$（每 epoch 判定） | `stage{N}/` |
| `best_ltotal.pth` | 当前 $L_{total}$ < 历史最优 $L_{total}$（每 epoch 判定） | `stage{N}/` |

Stage 0 → Stage 1 切换时重置历史最优值（两阶段 $L_{total}$ 因 $L_{AG}$ 有无不可比）。

#### 训练监控指标

| 指标 | 健康参考值 | 说明 |
|---|---|---|
| `perplexity_g` | > 18（$K_g/2$） | 全局码本利用率，过低说明码坍缩 |
| `perplexity_d` | > $K_d/2$（当前 192/2 = **96**） | 细节码本利用率 |
| `L_AP`（三子损失分开记录） | 稳定下降 | 动作重构主损失 |
| `L_AG` | Stage 0 后期趋于平稳 | 视觉-动作对齐（Stage 0 专用） |
| `L_future` | 随 $\lambda$ 抬升后开始下降 | 全局码视觉语义质量 |
| `L_sep` | 趋近于 0 | 全局/细节特征正交性 |
| 码本替换率 | 训练后期 < 5% | 过高说明码本仍不稳定 |

#### 后续调整策略

> **暂不启用，预留供训练后参考**。根据实际训练信号按需选用。

| 训练信号 | 触发条件 | 调整动作 |
|---|---|---|
| 码本坍缩 | `perplexity_g < 9` 连续 4+ epoch | 降低 AtomAction-NSVQ lr（3e-4 → 2e-4）；缩短 replace 触发间隔（2 → 1 epoch） |
| $L_{AP}$ 不收敛 | Stage 0 前 20 epoch $L_{AP}$ 无明显下降 | 提高 main lr（3e-4 → 5e-4）；检查 FM decoder 梯度 |
| DINOv2 LoRA 过拟合 | $L_{AG}$ 下降但 val 不改善 | 降低 LoRA lr（5e-5 → 2e-5）；增大 LoRA dropout（0.1 → 0.2） |
| Stage 1 $L_{future}$ 振荡 | 切换后 $L_{future}$ 连续 5 epoch 剧烈波动 | 延长 Stage 1 warmup（2 → 4 epoch）；降低 $\lambda_{future}$ 最终值（1.0 → 0.5） |
| 整体过拟合 | 训练/验证 $L_{AP}$ gap > 0.5 | weight_decay 1e-3 → 5e-3；考虑在 Stage 0 增加轨迹增广 |
| 显存不足（无法 bf16） | GPU OOM | 退回 fp16 + GradScaler；或减半 batch size + 2× 梯度累积 |

---

### 5.6 训练 Pipeline 实现

#### 文件组织

```
scripts/
  train_vqap.py       ← 唯一训练脚本（VQAPTrainer 类 + 入口）
train_vqap.sh         ← torchrun 启动快捷脚本（仓库根目录）
run_tensorboard.sh    ← 拉起 TensorBoard
utils/
  init_logger_tensorboard.py  ← init_logger() / init_tensorboard() / finish_tensorboard() 工具函数
  loss_func.py        ← compute_atomaction_reconstruction_loss / compute_action_grounding_loss /
                        compute_future_patch_loss / compute_future_weight_schedule 等
```

训练编排逻辑（stage 切换、checkpoint、dead code 替换、断点续训）集中在 `train_vqap.py` 的 `VQAPTrainer` 内；各损失项的具体计算下沉到 `utils/loss_func.py`，由 `_compute_loss` 调用组装。

---

#### LR 调度器（make_lr_lambda）

两个阶段复用同一函数，以 epoch 为单位，`scheduler.step()` 在每个 epoch 结束后调用一次：

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def make_lr_lambda(warmup_epochs: int, total_epochs: int, min_ratio: float = 0.05):
    """线性 warmup + cosine decay。两阶段均使用此函数，仅参数不同。"""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return lr_lambda

# Stage 0（$E_0$ epoch，warmup_epochs=2）
scheduler = LambdaLR(optimizer, make_lr_lambda(warmup_epochs=2, total_epochs=E_0))

# Stage 1（$E_1$ epoch，warmup_epochs=2，在 _setup_stage1() 中重建）
scheduler = LambdaLR(optimizer, make_lr_lambda(warmup_epochs=2, total_epochs=E_1))
```

LR 关键节点（以主体参数组为例）：

| epoch（阶段内） | Stage 0 lr（base=3e-4） | Stage 1 lr（base=1e-4） |
|---|---|---|
| 0 | 1.5e-4（warmup 50%） | **5e-5**（warmup 50%，平滑切换） |
| 1 | **3e-4**（peak） | **1e-4**（peak） |
| 中段 | ~1.6e-4 | ~5.2e-5 |
| 末 epoch | 1.5e-5（5% peak） | 5e-6（5% peak） |

---

#### VQAPTrainer 类方法结构

```
VQAPTrainer
 ├── __init__()                     配置读取（model/train/global 三份 YAML）→ DDP 初始化 →
 │                                  种子/精度/cuda 设置 → ckpt 目录 + resume 路径解析 →
 │                                  logger / TensorBoard → 数据集 → 模型（Stage 0 挂 DINOv2 LoRA）→
 │                                  恢复 model 权重（若 resume）→ DDP 包装 → 优化器/调度器（含 resume 恢复）
 ├── _init_distributed()            读取 WORLD_SIZE/RANK/LOCAL_RANK，set_device + init_process_group(nccl)
 ├── _resolve_resume_path()         仅当 --resume-stage 给出时生效；--resume-path > stage{N}/latest.pth
 ├── _load_resume_state()           torch.load(map_location=cpu)，校验 ckpt.stage == --resume-stage
 ├── _validate_resume_config()      续训配置一致性校验（见“Stage-Aware 断点续训”）
 ├── _init_dataset()                AtomActionDataset + DistributedSampler + DataLoader（collate_fn）
 ├── _init_model(apply_lora)        构建 VQAP，apply_lora 时 _apply_dinov2_lora() 仅挂 backbone
 ├── _apply_dinov2_lora(model)      PEFT LoraConfig（target=attn.qkv）→ get_peft_model 包 backbone
 ├── _freeze_stage1_modules(model)  冻结 backbone / visual_transformer_encoder / flow_matching_head；
 │                                  保留 future_predictor + atomaction_nsvq 可训练
 ├── _wrap_model_for_ddp()          多卡时包 DDP（broadcast_buffers=False, find_unused_parameters=True）
 ├── _init_optimizer(stage)         参数分组（main / lora；β₂ 分离、wd 分组）→ AdamW（详见 §5.5）
 ├── _init_scheduler(stage)         LambdaLR（make_lr_lambda，每阶段独立，scheduler.step() 按 epoch）
 ├── _setup_stage1(epoch)           merge_and_unload LoRA → 冻结 → 重建 optimizer/scheduler → 重置 best
 ├── train()                        外层 epoch 循环（阶段切换、死码替换、checkpoint、日志）
 ├── _train_epoch(epoch)            单 epoch 前向/反向/梯度裁剪；返回 epoch 均值指标 + used_steps
 ├── _compute_loss(batch, epoch)    调 utils.loss_func 组装全部损失（§5.3 + §5.4 stage 条件）
 ├── _maybe_replace_dead_codebooks(epoch, metrics, used_steps)  perplexity / 固定间隔触发判定
 ├── _replace_dead_codebooks(used_steps)  all_reduce 计数 → rank0 替换 → broadcast 新码本
 ├── _reset_codebook_usage_counters()     清零 global/detail quantizer.codebooks_used
 ├── _save_checkpoint(epoch, metrics, save_latest)  best 每 epoch 判定 + latest/codebook 节流
 ├── _atomic_torch_save(payload, path)     先写 .tmp 再 rename
 └── _log_epoch_summary(epoch, metrics, time)  logger + TensorBoard 标量
```

> 备注：`λ_future` 不是 trainer 方法，而由 `utils.loss_func.compute_future_weight_schedule(loss_cfg, stage_cfg, epoch)` 计算；perplexity 不单独聚合，而是作为普通指标随 `_train_epoch` 内的 `all_reduce(SUM)` 统一跨卡平均。

---

#### DDP 初始化

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")

# 内存与计算优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

DDP 包装（`_wrap_model_for_ddp`，仅多卡时执行；单卡/CPU 保持原模型）：

```python
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=False,        # 码本等 buffer 由替换逻辑显式 broadcast，无需每步同步
    find_unused_parameters=True,    # Stage 0/1 有效参数集不同
    gradient_as_bucket_view=True,   # 减少显存碎片（openpi 技巧）
)
```

> 当前实现 Stage 0/1 均保持 `find_unused_parameters=True`；Stage 1 死码替换时由 `_replace_dead_codebooks` 显式 `broadcast` 码本，配合 `broadcast_buffers=False` 避免每步同步开销。

---

#### DINOv2 双实例隔离

VQAP 运行中同时存在两个独立的 DINOv2 实例：

| 实例 | 所属模块 | 权重来源 | 训练期间状态 |
|---|---|---|---|
| **实例 A** | `AtomActionDataset → View_Selector` | `torch.hub` 官方权重 | 永久冻结；视角评分结果已缓存到磁盘，训练时不调用 |
| **实例 B** | `VQAP → VASA → DINO_FeatureExtractor` | `torch.hub` 官方权重（各自独立加载） | Stage 0：LoRA 微调；Stage 1：merge + 冻结 |

两者通过 `torch.hub.load()` 各自独立创建，持有不同内存对象，LoRA 梯度更新只影响实例 B 的内存副本，**不影响实例 A，也不写回 hub cache 文件**。

续训时 `self.model.load_state_dict(resume_state["model"], strict=True)` 只写入 VQAP 权重（在 DDP 包装前调用），ViewSelector 对象不会被触及。

---

#### Stage-Aware 断点续训（关键细节）

续训由命令行显式开启：仅当给出 `--resume-stage {0,1}` 时才进入续训分支；`--resume-path` 可选，必须与 `--resume-stage` 同时使用。路径优先级：`--resume-path` > `stage{N}/latest.pth`（`_resolve_resume_path`）。`_load_resume_state` 加载后会校验 checkpoint 内 `stage` 字段与 `--resume-stage` 一致，防止指错文件。

checkpoint 保存 `stage` 字段，加载时按 stage 分支构建模型：

```
续训 Stage 0（--resume-stage 0，ckpt["stage"] == 0）：
  1. _init_model(apply_lora=True)          ── 构建 VQAP 并向 backbone 注册 LoRA 结构
  2. model.load_state_dict(ckpt["model"])  ── 覆盖含 LoRA key 的权重
  3. _reset_codebook_usage_counters()      ── 清零码本使用计数
  4. _wrap_model_for_ddp()
  5. 重建 optimizer / scheduler 后恢复其 state_dict；恢复 global_step / best_lap / best_ltotal

续训 Stage 1（--resume-stage 1，ckpt["stage"] == 1）：
  1. _init_model(apply_lora=False)         ── 构建 VQAP（无 LoRA，基于已 merge 的 backbone）
  2. model.load_state_dict(ckpt["model"])  ── 覆盖 merged DINOv2 权重
  3. _freeze_stage1_modules()              ── 冻结视觉对齐支路（不再恢复 LoRA 结构）
  4. _wrap_model_for_ddp()
  5. 重建 Stage 1 optimizer / scheduler 后恢复其 state_dict；恢复 global_step / best_*
```

**续训配置一致性校验（`_validate_resume_config`）**，防止手改 YAML 破坏 stage 语义：

- 已进入 Stage 1 后，`stage0_epochs` 不可更改（否则 λ_future 调度与全局 epoch 语义错位）——延长训练只能增大 `stage1_epochs`；
- 当前所在 stage 的 epoch 数只能持平或增大，不能改小；
- 安全网：若 `start_epoch >= total_epochs`（手改导致无可训练 epoch）直接报错。

> 若需从 Stage 0 末尾续训并自动切入 Stage 1，使用 `--resume-stage 0` 即可：`train()` 外层循环检测到 `epoch >= stage0_epochs` 时自动调用 `_setup_stage1()`。

---

#### Stage 0 → Stage 1 切换（_setup_stage1）

对应 §5.4 两阶段训练方案，切换发生在 epoch $E_0+1$（当前 $150+1=151$）起始：

```
0. （多卡）dist.barrier() → 临时 _unwrap_ddp_model()
1. backbone.merge_and_unload()  ── 将 LoRA 权重合并进 VASA backbone，移除 adapter 结构
                                  （切换前断言 backbone 必须是 PeftModel）
2. _freeze_stage1_modules()     ── 冻结 backbone / visual_transformer_encoder / flow_matching_head；
                                  保留 future_predictor + atomaction_nsvq 可训练
3. _wrap_model_for_ddp() → current_stage = 1
4. 重建 Stage 1 优化器（仅一个 main 参数组，无 lora 组；lr=stage1_main=1e-4）
5. 重建 Stage 1 scheduler（`LambdaLR(make_lr_lambda(warmup_epochs=2, total_epochs=stage1_epochs))`，epoch 0 → 5e-5，epoch 1 → 1e-4）
6. 重置 best_lap = best_ltotal = +∞（§5.5：两阶段 L_total 不可比）
7. tb_writer.add_text("stage_transition", "...", epoch+1)  ── 在 TEXT 面板标注切换点
8. （多卡）dist.barrier()
```

> Stage 1 中被冻结的 backbone / visual_transformer_encoder / flow_matching_head 仍会参与前向，因此 `_train_epoch` 在 Stage 1 额外把它们切到 `.eval()`，避免 dropout 等训练态扰动。

---

#### 训练循环（train / _train_epoch）

```
train():
  for epoch in [start_epoch, total_epochs):
    if current_stage == 0 and epoch >= stage0_epochs:
        _setup_stage1(epoch)                 ← 自动切换（含从 stage0 末尾续训的情形）

    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)             ← 每 epoch 必须调用，确保 DDP 数据不重复

    epoch_metrics, used_steps = _train_epoch(epoch)   ← 指标均值 + 本 epoch batch 数
    scheduler.step()                          ← 每 epoch 结束调用一次，LambdaLR 更新 lr

    # 死码替换：每个 epoch 判定一次（perplexity 或固定间隔触发），随后清零计数窗口
    replace_metrics = _maybe_replace_dead_codebooks(epoch, epoch_metrics, used_steps)
    epoch_metrics.update(replace_metrics)
    _reset_codebook_usage_counters()

    if rank == 0 and tb_writer:               ← codebook/perplexity_{g,d} 与 replaced_{g,d}
        tb_writer.add_scalar("codebook/perplexity_g", epoch_metrics["perplexity_g"], step)
        ...

    # latest/codebook 按间隔节流，并在每个 stage 末尾强制保存；best 判定在 _save_checkpoint 内每 epoch 执行
    save_latest = ((epoch+1) % save_every_epochs == 0
                   or (epoch+1) == stage0_epochs
                   or (epoch+1) == total_epochs)
    _save_checkpoint(epoch, epoch_metrics, save_latest=save_latest)
    _log_epoch_summary(epoch, epoch_metrics, time)

_train_epoch(epoch):
  model.train()
  if current_stage == 1:                      ← 冻结的视觉支路切 eval，避免训练态扰动
      backbone.eval(); visual_transformer_encoder.eval(); flow_matching_head.eval()
  for batch in dataloader:
    batch → device
    optimizer.zero_grad(set_to_none=True)
    with autocast(dtype=bf16/fp16) or nullcontext():   ← 由 runtime.precision 决定
        loss_outputs = _compute_loss(batch, epoch)
    if grad_scaler.is_enabled():              ← fp16 路径：scale → unscale → clip → step → update
        grad_scaler.scale(loss_total).backward(); grad_scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        grad_scaler.step(optimizer); grad_scaler.update()
    else:                                     ← bf16/fp32 常规反传
        loss_total.backward()
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    累加 EPOCH_METRIC_KEYS 各项；global_step += 1
  # 跨卡聚合：对 [各指标和, num_steps] 做 all_reduce(SUM)，以全局总 batch 数为分母求均值
  epoch_metrics = local_sums / global_batch_count       ← 含 loss_*、perplexity_g/d 等
  epoch_metrics["grad_norm"] = 末批次 grad_norm（多卡 all_reduce 后 /world_size）
  epoch_metrics["lambda_future"] = compute_future_weight_schedule(...)   ← 各卡相同，无需聚合
  return epoch_metrics, num_steps
```

> `EPOCH_METRIC_KEYS` 统一了被累加并跨卡平均的指标：`loss_total / loss_ap / loss_fm_pos_ap / loss_fm_rot_ap / loss_bce_grip_ap / loss_sep / loss_ag / loss_fm_pos_ag / loss_fm_rot_ag / loss_bce_grip_ag / loss_future / perplexity_g / perplexity_d`。`grad_norm` 取末批次值，`lambda_future` 为 epoch 的确定性函数。

#### 损失计算（_compute_loss）

与 §5.3 总损失、§5.4 stage 条件直接对应：

模型 `forward` 一次返回三组输出：`shared_outputs`（速度场/夹爪目标与 `trajectory_mask`）、`atomaction_outputs`（AP 支路预测 + perplexity + pre-projection 特征）、`vasa_outputs`（AG 支路预测 + 未来帧特征）。损失由 `utils.loss_func` 的三个函数组装：

```python
def _compute_loss(self, batch, epoch):
    outputs = self.model(trajectory_data, trajectory_mask, selected_views)
    shared, atom, vasa = outputs["shared_outputs"], outputs["atomaction_outputs"], outputs["vasa_outputs"]

    # L_AP（§5.1，全程启用；compute_atomaction_reconstruction_loss 内含 loss_sep）
    atom_loss = compute_atomaction_reconstruction_loss(shared, atom, loss_cfg)
    # L_future（§5.1，全程；目标 end_img_features 在损失函数内 detach）
    loss_future = compute_future_patch_loss(vasa["pred_end_patch_features"], vasa["end_img_features"])
    lambda_future = compute_future_weight_schedule(loss_cfg, stage_cfg, epoch)

    loss_total = lambda_ap * atom_loss["loss_ap"] + lambda_future * loss_future

    # L_AG（§5.1，仅 Stage 0；Stage 1 各 AG 子项置零张量并入字典，方便日志统一记录）
    if self.current_stage == 0:
        ag_loss = compute_action_grounding_loss(shared, vasa, loss_cfg)
        loss_total = loss_total + lambda_ag * ag_loss["loss_ag"]
    else:
        ag_loss = {k: zero for k in ("loss_fm_pos_ag", "loss_fm_rot_ag", "loss_bce_grip_ag", "loss_ag")}

    return {**atom_loss, **ag_loss,
            "loss_future": loss_future, "lambda_future": lambda_future, "loss_total": loss_total,
            "perplexity_g": atom["global_perplexity"], "perplexity_d": atom["detail_perplexity"]}
```

要点：
- `loss_ap` 已在 `compute_atomaction_reconstruction_loss` 内合成 `loss_fm_pos_ap + λ_rot·loss_fm_rot_ap + λ_grip·loss_bce_grip_ap + λ_sep·loss_sep`（§3.6.5），trainer 不再单独加 `loss_sep`；
- `loss_future` 的目标 `end_img_features` 在 `compute_future_patch_loss` 内部 `.detach()`（对应 §5.1 的 stop-gradient）；
- Stage 1 不计算 AG 支路损失，但仍把零张量写入返回字典，使 `EPOCH_METRIC_KEYS` 在两个 stage 维度一致。

---

#### Checkpoint 设计

在 rank 0 执行**原子写**（先写 `.tmp` 再 `replace`，防止断电截断），落到 `stage{N}/` 子目录。best 每 epoch 判定，`latest.pth` / `codebook.pth` 按 `save_every_epochs` 与 stage 末尾节流（§5.5）：

| 文件 | 触发条件 | 用途 |
|---|---|---|
| `latest.pth` | `save_latest` 为真（间隔或 stage 末尾） | 断点续训 |
| `best_lap.pth` | $L_{AP}$ 创最优（每 epoch 判定） | 最佳动作重构权重 |
| `best_ltotal.pth` | $L_{total}$ 创最优（每 epoch 判定） | 最佳综合权重 |
| **`codebook.pth`** | 与 `latest.pth` 同频 | **仅码本权重，供 VLA 迁移（详见 [VLA_Design.md](VLA_Design.md) §〇.1）** |

> 三类文件都不需要写时（非保存间隔且 best 未刷新），`_save_checkpoint` 提前返回，省去全模型 `state_dict` 拷贝。

完整 checkpoint payload：`epoch`（= epoch+1）、`global_step`、`stage`、`model`（完整 state_dict）、`optimizer`、`scheduler`、`best_lap`、`best_ltotal`、`model_args`、`train_args`、`global_args`、`tb_log_dir`。

`codebook.pth` 结构（直接取码本模块与量化器的 `state_dict`，而非按 key 过滤）：

```python
{
    "epoch": epoch + 1,
    "global_step": self.global_step,
    "stage": self.current_stage,
    "perplexity_g": metrics["perplexity_g"],
    "perplexity_d": metrics["perplexity_d"],
    "model_args": self.model_args,
    "global_codebook_module": model.atomaction_nsvq.global_codebook_module.state_dict(),
    "detail_codebook_module": model.atomaction_nsvq.detail_codebook_module.state_dict(),
    "global_codebook": model.atomaction_nsvq.global_codebook_module.quantizer.state_dict(),
    "detail_codebook": model.atomaction_nsvq.detail_codebook_module.quantizer.state_dict(),
}
```

VLA 侧迁移时可加载 `global_codebook` / `detail_codebook`（量化器内含码本 `nn.Parameter`）或整模块 `*_codebook_module`，按 [VLA_Design.md](VLA_Design.md) 集成需求选用。

---

#### 日志系统

```
utils/init_logger_tensorboard.py
  ├── init_logger(rank, exp_name, log_dir, is_resume) → logging.Logger
  │     rank 0：文件（DEBUG）+ 控制台（INFO）
  │     rank n>0：仅独立日志文件（多卡异常调试用）
  │     续训：append 模式；新训练：write 模式
  │     文件路径：log/vqap_{exp_name}[_resume][_rank{n}].log
  └── init_tensorboard(rank, exp_name, cfg, ckpt_dir, is_resume) → SummaryWriter
        rank 0 执行；TensorBoard 日志存于 {ckpt_dir}/tensorboard/{exp_name}/
        续训时复用同一 log_dir，step 从 checkpoint 恢复，曲线自然连续
        训练超参数以 YAML 格式写入 TEXT 面板
```

TensorBoard 监控指标（x 轴 step = epoch + 1，由 `_log_epoch_summary` 与 `train()` 写入）：

| 类别（面板） | 指标 | 频率 |
|---|---|---|
| 损失总览 `loss` | `loss/total`, `loss/ap`, `loss/sep`, `loss/ag`, `loss/future` | 每 epoch（epoch 内均值） |
| AP 分解 `loss_ap` | `loss_ap/fm_pos`, `loss_ap/fm_rot`, `loss_ap/bce_grip` | 每 epoch |
| AG 分解 `loss_ag` | `loss_ag/fm_pos`, `loss_ag/fm_rot`, `loss_ag/bce_grip` | 每 epoch |
| 训练状态 `train_state` | `train_state/grad_norm`, `train_state/lambda_future`, `train_state/stage` | 每 epoch |
| 学习率 `lr` | `lr/main`，`lr/lora`（Stage 0） | 每 epoch |
| 码本健康度 `codebook` | `codebook/perplexity_g`, `codebook/perplexity_d`, `codebook/replaced_g`, `codebook/replaced_d` | 每 epoch |
| Best | `best/lap_s{stage}`, `best/ltotal_s{stage}` | best 刷新时 |

---

#### 启动方式

仓库根目录提供 `train_vqap.sh`（torchrun 启动）与 `run_tensorboard.sh`（拉起 TensorBoard）。当前 `train_vqap.sh` 内容：

```bash
CUDA_VISIBLE_DEVICES="1,5" torchrun --nproc_per_node=2 scripts/train_vqap.py
```

`scripts/train_vqap.py` 的命令行参数（`parse_args`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `config/train.yaml` | 训练配置 |
| `--model-config` | `config/model.yaml` | 模型结构配置 |
| `--global-config` | `config/global.yaml` | 数据集/全局配置 |
| `--exp-name` | None | 覆盖 `train.yaml` 中的 `exp_name` |
| `--resume-stage` | None | 续训阶段 `{0,1}`；省略则全新训练 |
| `--resume-path` | None | 显式 checkpoint 路径，必须与 `--resume-stage` 同用 |
| `--disable-tensorboard` | False | 即使 yaml 启用也强制关闭 TensorBoard |

用法示例：

```bash
# 新训练（2 卡）
CUDA_VISIBLE_DEVICES="1,5" torchrun --nproc_per_node=2 scripts/train_vqap.py --exp-name vqap_run1

# 从 Stage 0 最新 checkpoint 续训（自动在 epoch >= stage0_epochs 时切入 Stage 1）
torchrun --nproc_per_node=2 scripts/train_vqap.py --exp-name vqap_run1 --resume-stage 0

# 从指定 Stage 1 checkpoint 续训
torchrun --nproc_per_node=2 scripts/train_vqap.py --exp-name vqap_run1 \
    --resume-stage 1 --resume-path checkpoints/vqap_run1/stage1/best_lap.pth
```

---

## 六、问题诊断与优化建议

### 6.1 【严重】细节码坍缩风险

**问题**：9 个 learnable queries 经同一个 cross-attention 后，若无正交约束，可能收敛到相似的向量，导致细节码退化为同一个码的重复采样。

**优化方案**（三选一）：

方案A - 正交初始化 + 多样性损失：

$$
L_{div} = -\frac{1}{\binom{9}{2}} \sum_{i<j} \|f_{q,i}^{detail} - f_{q,j}^{detail}\|_2
$$

方案B - 强制分组语义（推荐）：将 9 个查询分配预设语义槽（如位置轨迹、手腕姿态、速度曲线、接触时机等），通过约束 cross-attention 的 query 位置编码来区分。

方案C - Perplexity 监控 + replace_unused_codebooks：$\text{ppl} = \exp(-\sum_k p_k \log p_k)$，训练中监控是否 perplexity 过低；配合 §3.4.4 的死码替换机制（对细节码本独立维护 `codebooks_used_d`），周期性将低利用率细节码替换为活跃码加噪声。

### 6.2 【中等】全局码本大小 K=36 的合理性

**分析**：RLBench 中的原子动作类型（Grasp、Lift、Place、Push、Pull、Rotate、Insert、Open、Close 等）约 10–15 类，但考虑不同方向、力度的变体，K=36 合理。但需注意：

- 若训练数据中原子动作类别不均匀，部分码会饥饿（欠利用）
- 建议在阶段一训练后，用聚类分析（K-means on $z_q^{global}$）验证 K=36 的合理性，按实际聚类数调整

### 6.3 【中等】未来帧预测分支过早主导训练

**问题**：训练早期若 $L_{future}$ 权重过大，模型会倾向于直接拟合“起始帧 $\to$ 结尾帧”的视觉外观变化，而此时全局码与动作语义尚未稳定建立对应关系，容易让未来帧分支反客为主。

**建议**：

1. 先用 $L_{AP} + L_{AG}$ 建立离散动作语义与视觉变化特征之间的对应关系。
2. 将 $\lambda_{future}$ 在预热阶段保持较低值（如 0.05）。
3. 待 `perplexity_g`、`perplexity_d` 与 $L_{AG}$ 趋于稳定后，再把 $\lambda_{future}$ 线性提高到 0.5 或 1.0。

这样可以避免一开始就在“未来帧图像与动作尚未对齐”的条件下强推潜空间预测，降低优化难度。

### 6.4 【中等】非通用型动作舍弃策略

**原始标注**："（不确定）"

**建议**：Approach 和 Transfer 动作不应完全丢弃，而应：
1. 不放入码本训练数据（因为确实难以通用化）
2. 但在 VQAP-VLA 推理时，这些动作可由 VLA 策略直接生成（不经过码本），即**码本覆盖有界原子动作，开放性动作由 VLA 连续输出**。

### 6.5 【轻微】RoPE 在变长序列下的位置编码一致性

**问题**：不同 batch 的 $T_{batch}$ 不同，RoPE 的绝对位置会变化。对于动作语义，相对位置（动作在轨迹中的相对进度）比绝对位置更重要。

**建议**：将时间步归一化到 $[0, 1]$，再映射为 RoPE 角度，即：

$$
\theta_t = \frac{t-1}{T-1}, \quad \text{position\_id}(t) = \lfloor \theta_t \cdot T_{max} \rfloor
$$

### 6.6 【轻微】解码器中 FiLM vs AdaLN 选择

**决策**：Flow Matching 解码器采用 **AdaLN-gated** 作为全局条件注入方式；细节码仍在偶数层通过 Cross-Attention 注入。

**优于纯 FiLM 的原因（简要）：**
- AdaLN-gated 额外提供逐通道 gate，控制的是“残差更新写回主干的强度”，而不只是像 FiLM 那样改写子层输入分布；在 8 层主干下更不容易出现层层累积的过调制。
- 调制层采用 `zeros_init` 后，训练初期满足 $\gamma=\beta=g=0$，残差分支近似关闭，网络从近似恒等映射出发；相比每层直接 FiLM，优化更稳定。
- 条件作用在归一化后的特征上，且默认尺度为 $(1+\gamma)$ 而非直接用 $\gamma$ 缩放，能够减少对隐藏状态统计量的持续漂移。

**实现约束**：不再采用“奇数层 FiLM、偶数层 Cross-Attn”的折中方案；当前版本统一为“所有主干子层使用 AdaLN-gated，偶数层额外插入细节码 Cross-Attn”。

---

## 七、模型参数量实测


### 7.1 AtomAction_NSVQ 参数量明细

| 子模块 | 参数量 |
|---|---|
| ee_projector（`Linear(9→384)→GELU→Linear(384→192)`） | 77.76 K |
| body_projector（`Linear(21→384)→GELU→Linear(384→192)`） | 82.37 K |
| gripper_mech_projector（`Linear(8→64)→GELU→Linear(64→64)`） | 4.74 K |
| gripper_open_embedding（`Embedding(2, 64)`） | 128 |
| gripper_open_projector（`Linear(64→64)`） | 4.16 K |
| ChannelEncoder（ChannelAttention + output_ffn，512→512） | 592.45 K |
| TransformerEncoder（4 层 BiTransformer，C=512, H=8, FFN=2048） | 12.61 M |
| GlobalCodebookModule（pool→Linear(512→$d_{code}$) + NSVQ $K_g$=36, $d_{code}$=512） | ~0.28 M |
| DetailCodebookModule（9 learnable queries + CrossAttn + projection + NSVQ $K_d$=192, $d_{code}$=512） | ~1.54 M |
| FlowMatchingHead（8-layer AdaLN-gated decoder + position/rotation/gripper heads） | 60.83 M |
| **AtomAction_NSVQ 合计** | **75.59 M** |

### 7.2 VASA 参数量明细

| 子模块 | 参数量 |
|---|---|
| ImageEncoder（DINOv2 ViT-B/14，feature_type=patch） | 86.58 M |
| VisualTransformerEncoder（1 cross-attn + 2 self-attn, C=512） | 8.54 M |
| VisualFlowMatchingHead（8-layer conditional decoder + position/rotation/gripper heads） | 68.18 M |
| FutureFramePredictor（FiLM + 3-layer FFN, 768→3072→3072→768） | 14.56 M |
| **VASA 合计** | **177.86 M** |

### 7.3 汇总

| 统计口径 | 参数量 |
|---|---|
| **VQAP 总计** | **253.45 M** |
| ├─ AtomAction_NSVQ | 75.59 M |
| └─ VASA | 177.86 M |
| 　　├─ DINOv2 ViT-B/14 backbone（独立） | 86.58 M |
| 　　└─ VASA 其余模块 | 91.28 M |
| VQAP 不含 DINOv2 | 166.87 M |
| VQAP 不含 VASA（仅 AtomAction_NSVQ） | 75.59 M |
| VLA 骨干（12层, d=1024, pi0.5 规模） | ~350 M（参考值，非本项目实测） |



---

## 八、实验与监控建议

| 指标 | 频率 | 目标 / 告警线 |
|---|---|---|
| `perplexity_g`（全局码本利用率） | 每 1000 步 | 目标 $> K_g/2 = 18$；告警 $< K_g/4 = 9$ |
| `perplexity_d`（细节码本利用率） | 每 1000 步 | 目标 $> K_d/2$（当前 96）；告警 $< K_d/4$（当前 48） |
| `replaced_codebooks_g/d`（每次替换的死码数量） | 每次替换 | 应随训练递减；末期接近 0 |
| $L_{FM}^{pos}$、$L_{FM}^{rot}$、$L_{BCE}^{grip}$（分项重构误差） | 每步 | 分别监控各模态收敛速度 |
| $L_{future}$（未来帧潜空间 cosine 对齐损失） | 每步 | 目标收敛至 < 0.3（即 cosine sim > 0.7） |
| $L_{AG}$（视觉 FM 重构损失） | 每步 | 监控视觉-动作对齐收敛速度 |
| 细节码成对距离（每 batch 中 9 个码的均值 $L_2$） | 每 1000 步 | 应 > 0；趋近 0 提示细节码坍缩 |
| 全局码本 t-SNE 可视化 | 每 10k 步 | 验证不同原子动作类别的分离性 |

> **码本聚类验证**（阶段一末尾建议执行一次）：对全量训练集的 $z_q^{global}$ 做 K-means（$K=36$），计算聚类内方差与间方差比（如 Davies-Bouldin index），验证 $K_g = 36$ 的合理性；若实际聚类数明显少于 36，考虑减小 $K_g$。

---

## 九、参数初始化方案

### 9.1 设计原则

1. **GELU 激活前用 Kaiming normal**（`mode='fan_in', nonlinearity='relu'`）：GELU 是 ReLU 的平滑近似，Kaiming 初始化保证激活层前后特征方差稳定
2. **残差分支末层 zeros_init**：残差连接的最后一个 Linear 权重和 bias 初始化为 0，使训练初期该分支输出为零、网络行为等价于恒等映射，避免深层梯度异常
3. **AdaRMSNorm-gated 调制层 zeros_init**：调制头输出 $[\gamma, \beta, g]$，其 Linear 权重和 bias 全部初始化为 0；此时 $\gamma=\beta=g=0$，AdaRMSNorm 退化为普通 `RMSNorm`，且残差更新 $x \leftarrow x + g \odot \Delta x$ 初始关闭，确保从近似恒等映射出发
4. **SE 门控末层 bias=−4**：`sigmoid(−4) ≈ 0.018`，ChannelAttention 初始缩放系数 $(1+w) \approx 1$，接近恒等，防止训练初期通道被随机放大
5. **码本初始化已在 §3.4.3 定义**（`Uniform(−1/K, 1/K)`），不在此重复

### 9.2 统一基础初始化

整个模型先以 `model.apply(_init_weights)` 递归应用以下规则，再由各模块对特殊层二次覆盖：

```python
def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.RMSNorm):   # Flow Matching 解码器专用
        nn.init.ones_(m.weight)        # RMSNorm 无 bias，只初始化 weight
```

`xavier_uniform_` 适合不接激活函数的线性层（Attention 投影、输出层）；接 GELU 的 FFN 输入层在各模块中二次覆盖为 Kaiming normal。Encoder 使用 `nn.LayerNorm`，Flow Matching 解码器使用 `nn.RMSNorm`，基础规则均已覆盖。

---

### 9.3 动作特征投影（§3.1）

**分组 FFN（$\phi_{ee}$ / $\phi_{body}$ / $\phi_{gm}$ / $\phi_{go}$）：**

| 层 | 初始化 | 说明 |
|---|---|---|
| 输入 Linear（接 GELU） | `kaiming_normal_(mode='fan_in', nonlinearity='relu')` | 二次覆盖 |
| 输出 Linear | `xavier_uniform_()` | 基础规则覆盖 |
| 所有 bias | `zeros_()` | 基础规则覆盖 |

**夹爪状态嵌入 `Embedding(2, 64)`：** `normal_(mean=0, std=0.02)`，基础规则覆盖

---

### 9.4 ChannelEncoder（§3.2）

**ChannelAttention（SE 门控）——二次覆盖：**

```python
# Linear(512→64)：接 GELU，改为 Kaiming
nn.init.kaiming_normal_(channel_attn.fc1.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(channel_attn.fc1.bias)

# Linear(64→512)：门控输出层，bias=−4 使初始 w ≈ 0，缩放系数 (1+w) ≈ 1
nn.init.zeros_(channel_attn.fc2.weight)
nn.init.constant_(channel_attn.fc2.bias, -4.0)
```

**`output_ffn` 残差末层——二次覆盖：**

```python
# output_ffn 结构为 Linear→LayerNorm→GELU→Linear
# 第一个 Linear(512→512) 接 GELU，改为 Kaiming
nn.init.kaiming_normal_(channel_enc.output_ffn[0].weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(channel_enc.output_ffn[0].bias)

# 最后一个 Linear(512→512) 为残差末层，zeros_init
nn.init.zeros_(channel_enc.output_ffn[-1].weight)
nn.init.zeros_(channel_enc.output_ffn[-1].bias)
```

---

### 9.5 Transformer Encoder（§3.3）

基础规则已覆盖所有 Attention（Q/K/V/Out）投影和 FFN 输出层（`xavier_uniform_`）及 LayerNorm；对接 GELU 的 FFN 输入层二次覆盖：

```python
for layer in transformer_encoder.layers:
    # FFN Linear(512→2048) 接 GELU
    nn.init.kaiming_normal_(layer.ffn.fc1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.zeros_(layer.ffn.fc1.bias)
    # FFN Linear(2048→512) 为残差末层，基础规则的 xavier_uniform_ 保留
```

---

### 9.6 双码本量化（§3.4）

**全局/细节支路投影 `Linear(C→d_code)`（即 `Linear(512→512)`）：** `xavier_uniform_()`，基础规则覆盖。当前 $d_{code}=C=512$，该层为等价变换，梯度仍通过该层正常回传。

**细节支路可学习查询 `LearnableEmbed(9, 512)`（`nn.Parameter`）：**

$$
Q_{learn} \sim \mathcal{N}(0,\; 0.02^2 I) \in \mathbb{R}^{9 \times 512}
$$

小方差正态初始化，防止 CrossAttn 早期饱和；若采用 §6.1 方案 A 的正交约束，可在 `normal_` 后叠加 `nn.init.orthogonal_` 并乘以 0.02 的缩放系数。

**全局/细节码本（`nn.Parameter`）：** 已在 §3.4.3 定义为 `Uniform(−1/K, 1/K)`，不重复

---

### 9.7 Flow Matching 解码器（§3.6）

**时间步嵌入与全局条件融合 MLP：** 首层 `kaiming_normal_()`，末层 `xavier_uniform_()`，基础规则覆盖

**主干 8 层 Transformer（自注意力 + FFN）：** 同 §9.5，FFN 输入层 Kaiming，其余 `xavier_uniform_`

**AdaRMSNorm-gated 调制层 `Linear(512→1536)`（输出 $\gamma$ / $\beta$ / $g$ 各 512 维）——二次覆盖：**

```python
for mod_linear in decoder.adaln_modulation_layers:   # SelfAttn / FFN / CrossAttn 子层各一组
    nn.init.zeros_(mod_linear.weight)
    nn.init.zeros_(mod_linear.bias)          # γ = β = g = 0
```

**细节码升维 `Linear(d_code→512)`（当前 `Linear(512→512)`，等价变换）：** `xavier_uniform_()`，基础规则覆盖

**输出头末层——二次覆盖：**

```python
# position / rotation head 最后一层：small normal，使初始速度场接近零
nn.init.normal_(pos_head.final_linear.weight, std=0.01)
nn.init.zeros_(pos_head.final_linear.bias)
nn.init.normal_(rot_head.final_linear.weight, std=0.01)
nn.init.zeros_(rot_head.final_linear.bias)

# gripper head 最后一层：zeros_init → sigmoid(0) = 0.5，对称初始化
nn.init.zeros_(grip_head.final_linear.weight)
nn.init.zeros_(grip_head.final_linear.bias)
```

---

### 9.8 VASA 模块（§四）

**结尾帧 query 投影层 $W_q$、初始帧 key/value 投影层 $W_{kv}$ 与差分投影层 $W_{\Delta}$（$768 \to 512$）：** `xavier_uniform_()`，基础规则覆盖

**视觉交互编码器：** 结构固定为“1 层 cross-attention + 差分残差 + 2 层 self-attention”。初始化方式与 §9.5 相同；即 Attention 投影采用 `xavier_uniform_()`，FFN 第一层采用 `kaiming_normal_()`，LayerNorm / RMSNorm 采用单位初始化。该模块的接口定义为输出图像差分特征序列 $F_{diff} \in \mathbb{R}^{B \times T \times 512}$；其中 patch 模式下 $T=P$，CLS 模式下 $T=1$。当前实现不额外使用 1D RoPE。

**视觉条件接口：**

- 当前阶段先固定视觉交互编码器输出为序列 $F_{diff}$，不再在编码器末端做平均池化
- 后续视觉 FM Head 若需要额外的时间步条件融合，再对 `img_diff_features` 做投影 / 交互即可；对应新增层的初始化仍沿用 §9.7 的原则

也就是说，当前主方案不再额外引入视觉侧的 `detail_codewords` 或对应的 detail queries，而是将序列 $F_{diff}$（代码中对应 `img_diff_features`）作为视觉 FM Head 的唯一视觉条件来源。

**Action Grounding 的独立 Flow Matching Head：** 主干注意力层、FFN、AdaLN 调制层、位置 / 旋转 / 夹爪输出头均沿用 §9.7 的初始化原则；但该视觉版本不包含 `detail_projection` 和细节 Cross-Attention 分支的参数。

> **备选增强**：如果后续启用结构化条件方案，再新增视觉 detail queries，并按 `normal_(mean=0, std=0.02)` 初始化即可；当前主方案默认不启用。

**未来帧 FiLM `Linear(d_code→768×2)`（$\gamma$ / $\beta$ 各 768 维，当前 `Linear(512→1536)`）——采用近恒等初始化：**

```python
nn.init.zeros_(future_predictor.film_linear.weight)
future_predictor.film_linear.bias.data[:768].zero_()
future_predictor.film_linear.bias.data[768:].zero_()
```

这样在训练初期有 $\gamma \approx 0, \beta \approx 0$，FiLM 调制近似退化为逐 patch 的 `LN(C_s[p])`，不会过早强行扭曲输入特征。

**未来帧 FFN `Linear(768→3072→3072→768)`：**

```python
# 前两层接 GELU，采用 Kaiming
nn.init.kaiming_normal_(future_predictor.ffn.fc1.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(future_predictor.ffn.fc1.bias)
nn.init.kaiming_normal_(future_predictor.ffn.fc2.weight, mode='fan_in', nonlinearity='relu')
nn.init.zeros_(future_predictor.ffn.fc2.bias)

# 输出层采用 small normal，避免初始输出塌为全零向量
nn.init.normal_(future_predictor.ffn.fc3.weight, std=0.01)
nn.init.zeros_(future_predictor.ffn.fc3.bias)
```

**DINOv2 ViT-B/14 官方 backbone：** 直接加载官方预训练权重，不做额外随机初始化；训练时是否冻结，由 §5.3 的阶段性策略决定。当前文档版本不再引入 LoRA 专用初始化。

---

*本文档（VQAP：数据集 / 双码本 / 预训练）与配套的 [VLA_Design.md](VLA_Design.md)（VLA 集成）一起使用。所有维度数值为推荐初始配置，应根据实际 GPU 显存、任务规模和消融实验结果调整。*
