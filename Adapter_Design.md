# VQAP 设计文档 v2 —— 双码本 + Adapter + 插件式集成

> **本文档定位**：在 [VQAP_Design.md](VQAP_Design.md)（v1，Stage 0/1 码本预训练的模型结构与损失）的基础上，新增 **Stage 2 Adapter** —— 把"观测 + 子任务指令 → 码本索引"变成一个可监督、可独立训练、与 baseline 完全解耦的模块；并给出码向量注入 baseline 的接口契约。
>
> **与既有文档的关系**
> - `VQAP_Design.md`（v1）：Stage 0/1 的模型结构、损失、维度约定 —— **本版继续沿用，未作结构性改动**，只在 §3 列增量与体检结论。

> **实施状态**（2026-08-29）
> - **Stage 0/1（§3）✅ 已训完**：epoch 250，$ppl_g$ 17.78 / $ppl_d$ 51.21。
> - **Stage 2（§4）✅ 已训完**：标签导出、划分、模型、训练脚本全部落地；best epoch 94，val 全局码 top-1 **89.69%**（§4.9）。
> - **Stage 3（§5）⬜ 注入点与注入方式已定稿，未实现**；完整训练方案见 `VLA_Design.md`（待重写）。
> - **Planner（§6）⬜ TBD**。
> - **§7 记录全部已知风险与实测结论**，其中 R1 / R2 / R8–R10 为未解决项。

---

## 〇、本版决策与待定项

### 0.1 已锁定的决策

| # | 决策 | 落点 |
|---|---|---|
| 1 | **VQAP 保持不变**。Stage 0/1 已训完并冻结，Adapter 在其之上顺序训练，标签静态 | §3、§4.1 |
| 2 | Adapter 采用**轻量 Transformer**，输入 = 子任务指令 token + 双相机图像 token，输出 = 双码本索引 | §4.3、§4.4 |
| 3 | Adapter 固定使用 **`front` + `wrist`** 两路相机 | §4.3.1 |
| 4 | Adapter 的 tokenizer 使用**独立冻结模块**：图像 = DINOv2 ViT-B/14，文本 = CLIP ViT-B/16 text tower。**不复用** VASA 的 DINOv2 实例，也不复用 PerAct 的 CLIP RN50 | §4.3 |
| 5 | Adapter 与 baseline **完全解耦**：只消费 `(图像, 指令)`，不读取 baseline 的任何内部张量 | §4.1 |
| 6 | **保留双码本**（$K_g{=}36$ 全局 + $K_d{=}192$ 细节 + NSVQ + FM 重构解码器）。细节码保留注入路径，其消融放入 `Exp_Design.md` | §3.1、§3.2.1、§5.3 |
| 7 | Adapter **不输入本体状态**，只吃图像与指令 | §4.3.3、§7 |
| 8 | **不引入码空间辅助损失**，训练目标只有分类损失 | §4.5 |
| 9 | **不做原子级掩码**。`code_atom_prior` 及"空掩码降级"机制**整体删除**，Adapter 在全部 36 路上自由预测 | §0.2 |
| 10 | **Adapter 每个子任务触发一次**。一个子任务 = 一个原子动作；子任务内 baseline 可执行多个 waypoint，全部注入**同一组码向量** | §1.3 |
| 11 | 图像输入管线**固定为**「任意来源 → 降采样到 128×128 → resize 224×224」，训练与部署走同一条管线 | §4.3.1 |
| 12 | 指令格式**固定为 `short` 形式**（去尾句号）；Adapter 训练与 planner 输出使用同一格式 | §4.3.2 |
| 13 | 码向量注入 baseline：**全局码走 FiLM，细节码走 cross-attention**，作用于同一个注入点 | §5.3 |
| 14 | 标签导出使用 **`checkpoints/vqap_pretrain/stage1/latest.pth`**，并以 sha256 锁定 | §4.2 |

### 0.2 实施期新增决策（2026-08 落地时确定）

| # | 决策 | 理由 | 落点 |
|---|---|---|---|
| 15 | 指令 **不截断**，保留 CLIP 全部 77 位（原设计 $T_{txt}{=}20$） | 截断需处理位置边界；padding 由可见性矩阵屏蔽，代价仅注意力从 158² 增到 215² | §4.3.2 |
| 16 | **不用模态类型嵌入** | 与图像/文本各自投影层的 bias 数学冗余，query 本身是自由参数 | §4.4.1 |
| 17 | 注意力用**可见性矩阵**而非全可见 | 观测段不被码预测任务污染；全局码只由观测决定，细节码以全局码为条件 | §4.4.1.1 |
| 18 | 为 Adapter **新写** `AdapterAttention` / `AdapterEncoderLayer`，不改动共享的 `MultiHeadAttention` / `VisualSelfAttentionLayer` | 共享类被 VQAP 已训模型的推理路径依赖，改它的风险 > 消除重复的收益 | §4.4.1.1 |
| 19 | `encode_codebook_indices` **内联**编码前缀，不改 `forward`（原设计为提取方法重构） | 同上；重复由「dropout 置零后两者索引逐位相同」的测试守护 | §4.2.1 |
| 20 | 划分**不做 Seen/UnSeen 任务级留出**，只 train/val | Adapter 与码本一样是预训练产物，在全部 69 任务上学习 | §4.8.1 |
| 21 | **单卡训练**，$\lambda_d{=}1.0$ | 实测 2.1 min/epoch，DDP 复杂度与收益不成比例 | §4.9.1 |
| 22 | `best.pth` **不含优化器状态**（56.9 MB），`latest.pth` 含（170.7 MB） | 前者是部署产物（插件 `vqap_adapter.pth` 的来源），后者供续训 | §4.9.4 |


---

## 一、系统总览

### 1.1 三阶段链路

```
┌─ Stage 0/1：码本预训练（AtomAction_Dataset：69 任务 / 56,496 phase / 17 原子）─┐
│  动作编码器 + 双码本(NSVQ) + FM 重构解码器 + VASA(AG / Future)                 │
│  结构与损失沿用 v1；✅ 已训完（epoch 250 = 150 + 100）                          │
└──────────────────────────────┬────────────────────────────────────────────────┘
                               │ 冻结编码器 + 双码本 + 解码器
                               ▼
┌─ Stage 2：Adapter（本版新增，顺序训练，标签静态）────────────────────────────┐
│  ① 一次性导出标签表：phase → (k_g, k_d[9])                                    │
│  ② 独立训练轻量 Transformer Adapter（14.21 M 可训）                            │
│  ✅ 已训完（100 epoch / 3.4 h 单卡；best epoch 94，val top-1 89.69%）           │
└──────────────────────────────┬────────────────────────────────────────────────┘
                               │ 冻结 Adapter
                               ▼
┌─ Stage 3：baseline 集成（PerAct，18 RLBench 任务）───────────────────────────┐
│  仅训练：注入层（FiLM + cross-attn）+ PerAct 动作头 + lang_preprocess          │
│  冻结：PerAct 主干、CLIP RN50、码本、Adapter                                  │
└───────────────────────────────────────────────────────────────────────────────┘
```

**插件产物**：`atomaction_codebook.pth`（码本 + 索引契约）与 `vqap_adapter.pth`（Adapter 权重 + 输入规格）。二者构成"即插即用插件"的全部内容。

### 1.2 部件职责与冻结状态

| 部件 | 职责 | Stage 0/1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| 动作编码器（v1 §3.1–3.3） | 轨迹 → $z$ | 训练 | ❄️（产标签） | 不运行 |
| 双码本 $C_g,C_d$ | 离散动作语义字典 | 训练 | ❄️ | ❄️（查表） |
| FM 重构解码器 | 迫使码携带动作内容 | 训练 | 不运行 | 不运行 |
| VASA（AG / Future） | 给码注入视觉语义 | 训练 | 不运行 | 不运行 |
| **Adapter** | 观测 + 指令 → 码索引 | — | **训练** | ❄️ |
| DINOv2 ViT-B/14（Adapter 图像塔） | 视觉 token 化 | — | ❄️ | ❄️ |
| CLIP ViT-B/16 text（Adapter 文本塔） | 指令 token 化 | — | ❄️ | ❄️ |
| **Planner（VLM API）** | 子任务规划 / 阶段切换 / 错误恢复 | — | — | 不训练（TBD，§6） |
| **注入层**（FiLM + cross-attn） | 码向量 → 动作特征调制 | — | — | **训练** |
| PerAct 主干 | 感知与表征 | — | — | ❄️ |
| PerAct 动作头 + `lang_preprocess` | 解码可执行动作 | — | — | **训练** |

### 1.3 部署时序

```
任务指令 + 初始观测
   │
   ▼
[Planner]  拆分子任务列表 → {instruction, atom_action, use_codebook}
   │
   ▼
┌───────────── 每个子任务（= 一个原子动作）────────────────────────────────┐
│                                                                          │
│  子任务起始观测 (front, wrist)  +  子任务指令                              │
│        │                                                                 │
│        ▼                                                                 │
│  [Adapter]（冻结，本子任务内只调用一次）→ k_g, k_d[9]                      │
│        │                                                                 │
│        ▼                                                                 │
│  [码本查表]（冻结）→ z_g ∈ R^512,  Z_d ∈ R^{9×512}                        │
│        │                                                                 │
│        │  ※ use_codebook=false → code_mask=0 → 注入层输出恒等             │
│        ▼                                                                 │
│  ┌── waypoint 1 ──┐  ┌── waypoint 2 ──┐   ...   ┌── waypoint k ──┐        │
│  │ PerAct + 注入  │  │ PerAct + 注入  │         │ PerAct + 注入  │        │
│  │ (z_g, Z_d)     │  │ 同一组码向量   │         │ 同一组码向量   │        │
│  └────────────────┘  └────────────────┘         └────────────────┘        │
└──────────────────────────────────────────────────────────────────────────┘
   │
   ▼
[Planner] 阶段判定 → 下一个子任务 / 重试 / 重规划（TBD，§6）
```

**码的生命周期**：在一个子任务内**保持不变**。Adapter 只在子任务起始被调用一次，其输入观测正是该子任务的起始帧 —— 与 §4.2 的标签构造方式（用 phase 起始帧配 phase 的码）严格对应。


---

## 二、维度与超参约定

### 2.1 沿用 v1

| 符号 | 含义 | 值 |
|---|---|---|
| $K_g$ | 全局码本大小 | **36** |
| $K_d$ | 细节码本大小 | **192** |
| $N_{detail}$ | 细节码数量 | **9** |
| $d_{code}$ | 码向量维度 | **512** |
| $C$ | 动作编码器隐藏维 | **512** |
| $d_{dino}$ | DINOv2 ViT-B/14 特征维 | **768** |

### 2.2 本版新增

| 符号 | 含义 | 值 |
|---|---|---|
| $d_{ada}$ | Adapter 隐藏维 | **512** |
| $L_{ada}$ | Adapter 层数 | **4** |
| $H_{ada}$ | Adapter 注意力头数 | **8** |
| $P_{cam}$ | 每相机视觉 token 数（16×16 patch 池化到 8×8） | **64** |
| $N_{cam}$ | 相机数（front, wrist） | **2** |
| $T_{txt}$ | 指令 token 数（CLIP 上下文长度，**不截断**） | **77** |
| $N_{query}$ | 查询 token 数（1 global + 9 detail） | **10** |
| $S_{ada}$ | Adapter 序列长度 $=N_{cam}P_{cam}+T_{txt}+N_{query}$ | **215** |
| $d_{clip}$ | CLIP ViT-B/16 文本 token 维 | **512** |
| $H_{img}$ | Adapter 图像管线中间分辨率 | **128** |
| $H_{in}$ | Adapter 图像塔输入分辨率 | **224** |
| $\lambda_d$ | 细节码分类损失权重 | **1.0** |
| $\epsilon$ | label smoothing | **0.05** |

---

## 三、Stage 0/1：码本预训练

### 3.1 结构与损失：完全沿用 v1

不作任何结构性改动，包括：四路动作特征投影、ChannelEncoder、TransformerEncoder（4 层）、全局支路（masked-avg-pool + NSVQ $Q_g$）、细节支路（9 learnable query + cross-attn + NSVQ $Q_d$）、NSVQ 噪声替换与死码替换、信息分离约束 $L_{sep}$、FM 重构解码器（8 层，AdaRMSNorm 条件 + layer 2/4/6/8 细节码 cross-attn）、VASA 的 Action Grounding 与 Future Prediction 两支路、两阶段调度与 DINOv2 LoRA 策略。

$$L_{total}=\lambda_{AP}L_{AP}+\lambda_{AG}L_{AG}+\lambda_{future}(e)L_{future}$$
$$L_{AP}=L^{pos}_{FM}+\lambda_{rot}L^{rot}_{FM}+\lambda_{grip}L^{grip}_{BCE}+\lambda_{sep}L_{sep}$$

配置见 `config/train.yaml`、`config/model.yaml`。

### 3.2 已训产物与体检结论

**产物**：`checkpoints/vqap_pretrain/stage1/{latest,best_lap,best_ltotal}.pth` + `codebook.pth`

| 项 | 值 |
|---|---|
| epoch | 250（Stage 0 的 150 + Stage 1 的 100，完整调度） |
| global_step | 55,250 |
| $L_{AP}$ 最优 / $L_{total}$ 最优 | 0.0632 / 0.1078 |
| 全局码困惑度 $ppl_g$ | **17.78 / 36**（阈值 9.0 ✅） |
| 细节码困惑度 $ppl_d$ | **51.21 / 192**（阈值 48.0 ✅） |


### 3.3 checkpoint 选择

标签导出、Stage 3 查表、Adapter 训练三处**必须使用同一个 checkpoint**，并以 sha256 锁定。

**选定 `stage1/latest.pth`**。理由：NSVQ 的死码替换在 Stage 1 内于 epoch 159 / 179 / 199 / 219 / 239 触发过 5 次（`replace_interval_epochs: 20`），中途保存的 `best_*.pth` 的码本可能在其后被整体替换过，用它导出的索引与最终码本不保证同源；`latest.pth` 是完整调度末态，唯一且无歧义。


---

## 四、Stage 2：Adapter

### 4.1 定位与设计原则

**定位**：Adapter 是**冻结编码器的摊还（amortization）** —— 它预测"如果能看到接下来那段真实轨迹，VQAP 编码器会给出哪个索引"。

四条设计原则：

1. **顺序训练，标签静态。** Stage 0/1 完全收敛并冻结后再训 Adapter。理由：同步训练时 argmin 标签是移动靶，且 `replace_unused_codebooks` 每次触发都会**整体重标**（被替换死码上的全部样本标签瞬间失效）。静态标签使 Adapter 训练降级为一个普通有监督分类任务，可独立、反复迭代而不触碰 VQAP。
2. **与 baseline 完全解耦。** Adapter 只消费 `(图像, 指令)`，不读取 baseline 的任何内部张量。插件接口收敛为 `(obs, instruction) → codeword`；换 baseline 只需重写注入层，Adapter 无需重训。
3. **轻量且自足。** 冻结的视觉/文本塔（DINOv2 + CLIP，官方权重，独立实例）+ 14.21 M 可训 Transformer（实测 14,209,252）。不使用 VLM：train_actions 白名单内只有 **572 个唯一 `(task, variation, source_phase_index)` 语义组合**（586 是含 `pose-adjust` 的 18 动作口径），大模型会记忆而非泛化；且语义推理职责由 planner 承担，Adapter 只做"感知 → 码"的细粒度映射。
4. **不做任何门控。** Adapter 无条件输出索引；是否启用码本注入由 planner 决定（§6，TBD）。

### 4.2 标签导出

#### 4.2.1 代码增量（✅ 已实现）

原实现的 NSVQ 在 eval 分支只支持"按给定索引查表"，**不支持"由特征求索引"** —— [nsvq.py](model/module/nsvq.py) 的 `NSVQQuantizer.forward` 在 `not self.training and codebook_indices is None` 时直接抛 `ValueError`。因此导出必须走**新增**的硬量化路径，而不是复用 `forward`：

| 文件 | 新增内容 | 性质 |
|---|---|---|
| [model/module/nsvq.py](model/module/nsvq.py) | `NSVQQuantizer.quantize_hard(inputs) -> (codeword, index)`：`compute_codebook_distances` + `argmin` + `lookup_codewords`，`@torch.no_grad()`，不做噪声替换、不更新使用计数、不算困惑度 | 纯新增 |
| 同上 | `GlobalCodebookModule.encode_indices(feat, mask)` | 纯新增 |
| 同上 | `DetailCodebookModule.encode_indices(feat, mask)` | 纯新增 |
| [model/module/atomaction_nsvq.py](model/module/atomaction_nsvq.py) | `encode_codebook_indices(trajectory_data, mask)`：**内联**编码前缀（四路投影 → ChannelEncoder → TransformerEncoder）+ 两个 `encode_indices` | 纯新增 |

**`forward` 一行未动**（决策 19）。原设计要把编码前缀抽成公共方法，实施时改为在 `encode_codebook_indices` 里内联那 20 行 —— Stage 0/1 已永久冻结，`forward` 不会再改，"实现漂移"的前提不成立，而改动已训模型的推理路径有实际风险。

重复的那段代码由一条**等价性断言**守护：把全模型 dropout 置 0 并保持 `train()` 模式（`forward` 的特征路径要求 `training=True`），`forward` 与 `encode_codebook_indices` 输出的索引必须**逐位相同**。成立的原因是 NSVQ 的噪声只加在输出上（$q=h+\text{vq\_error}$），`argmin` 取自距离矩阵、不受影响。实测通过，同时锁死了 9 个细节槽位的顺序契约。

#### 4.2.2 导出流程

脚本：[data/import_codebook_index.py](data/import_codebook_index.py)（✅ 已实现，全量导出约 30 min）

```
python data/import_codebook_index.py \
    --checkpoint checkpoints/vqap_pretrain/stage1/latest.pth \
    --output data/atomaction_codebook_index.json
```

```
① 用与训练完全相同的配置构建 AtomActionDataset
   （config/global.yaml::atomactiondataset，同一 dataset_root / views / top_k / view_selector）
② DataLoader：shuffle=False、同一个 AtomActionDataset_collate_fn（顺序遍历，结果可复现）
③ 只构建 AtomAction_NSVQ 子模块（省掉 VASA 的 DINOv2）：取 ckpt 内嵌的 model_args 重建
   → 与当前 config/model.yaml 交叉校验（不一致报错）
   → 按 "atomaction_nsvq." 前缀过滤 ckpt 权重、断言 key 集合完全相等 → strict 加载 → eval()
   → 全程 float32（硬量化取 argmin，fp32 不在近邻边界翻码，见 §7 R5）
④ 逐 batch 调用 atomaction_nsvq.encode_codebook_indices(trajectory_data, trajectory_mask)
     k_g       = argmin_j ||h_g − e_j^{(g)}||²                [B]
     k_{d,1..9} = argmin_j ||H_d^{(n)} − e_j^{(d)}||²          [B, 9]
⑤ 回查 phase_metadata.json（start/end/keyframe 帧号、front/wrist 起始帧图像路径）
   与 variation_metadata.json（phase_index → short 指令）
⑥ 写出 JSON
```

**三条一致性保证**：

1. **归一化与训练同源。** `AtomActionDataset.__getitem__` 调用 `data/utils.py::normalize()`，后者从 `{dataset_root}/dataset_metadata.json::traj_stats` 读数据集级统计量（6 个字段，n = 2,047,276 帧）。导出以同一 `dataset_root` 调用同一函数，无需任何额外处理。
2. **不旁路数据管线。** 样本索引通过一个 `IndexedDataset` 薄包装携带（在调用训练用 collate 之前剥离），`AtomActionDataset` 与 `AtomActionDataset_collate_fn` 均不改动。
3. **eval 模式必须开启。** `channel_encoder` / `transformer_encoder` / 细节支路 `cross_attention` 均含 dropout；DDP 关闭，单卡确定性前向。

#### 4.2.3 产物 schema

`data/atomaction_codebook_index.json`

```jsonc
{
  "meta": {
    "checkpoint": "...", "checkpoint_sha256": "...", "checkpoint_epoch": 250, "checkpoint_stage": 1,
    "dataset_root": "...", "train_actions": [...17...], "instruction_style": "short",
    "global_codebook_size": 36, "detail_codebook_size": 192, "num_detail": 9,
  },
  "records": [
    {
      "phase_id": "grasp/close_jar/variation0/phase_000",
      "action": "grasp", "task": "close_jar", "variation": "variation0", "phase": "phase_000",
      "trajectory_length": 18,
      "k_global": 12, "k_detail": [3, 88, 41, 7, 155, 20, 91, 3, 62],
      "start_frame": 48, "end_frame": 66, "keyframe_index": 65,
      "img_front": "front_rgb/48.png", "img_wrist": "wrist_rgb/48.png",
      "instruction": "Grasp the jar lid"
    }
  ]
}
```

> **含 `split` 字段。** 由 `data/build_adapter_split.py` 原地写回（§4.8.2），不生成额外 manifest。
> ⚠️ 重跑导出会覆盖 `split`；划分完全由 `(seed, val_ratio, 记录内容)` 决定，重跑导出后再执行一次划分脚本即可恢复逐条相同的划分。
> **含 `source_phase_index` 字段。** 指令必须按它回查，不能用目录名 `phase_XXX`（后者是样本序号）——实测 `push/close_drawer/variation1/phase_001` 用目录名会查到语义相反的指令且不报错。
> **不含码向量。** 512 维向量由 `k_global` / `k_detail` 查冻结码本即得，存进 JSON 只会让文件膨胀两个数量级。

#### 4.2.4 运行环境

`low_dim_obs.pkl` 存的是 `rlbench.backend.observation.Observation` 实例，unpickle 会 `import rlbench` → `pyrep` → `libcoppeliaSim.so.1`。运行前必须设置（训练脚本有同样依赖）：

```bash
export COPPELIASIM_ROOT=/home/weizeming/weizeming/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
```


#### 4.2.5 导出后必跑的断言（`data/import_codebook_index.py` 内已实现，✅ 全部通过）

| # | 断言 | 实测 |
|---|---|---|
| A1 | 记录数 == 56,496（17 类白名单，排除 `pose-adjust`） | ✅ |
| A2 | `phase_id` 唯一 | ✅ 56,496 |
| A3 | `k_global ∈ [0,36)`、`k_detail` 长 9 且 ∈ [0,192) | ✅ |
| A4 | 指令覆盖率 100% | ✅ 0 缺失 |
| A5 | 双相机起始帧文件存在 | ✅ 0 缺失 |
| A6 | 图像帧号 == `start_frame` | ✅ 0 不一致 |
| A7 | 唯一 `(task, variation, source_phase_index)` 组合数 == **572** | ✅（与 `phase_descriptions` 条目双射：孤儿 0、缺失 0） |
| A8 | **批内困惑度对账** | ✅ `ppl_g` **17.78 == 训练末态 17.78**；`ppl_d` 49.33 vs 51.21 |

> ⚠️ **困惑度对账必须用批内口径。** ckpt 记录的 17.78 / 51.21 是 `batch_size=64`（4 卡 DDP，每卡 64）的**批内**困惑度按 epoch 平均；数据集级的值是 23.66 / 167.06，二者不是同一个量。脚本按 64 随机分组再平均来比对（随机打散是必需的：按导出顺序切组会把同 task/variation 的样本挤在一起，人为压低批内多样性）。

**导出实测补充**：全局码覆盖 36/36、细节码 192/192；样本内细节槽位唯一索引数 **1.00 / 9**（见 §7 R1）。


### 4.3 输入构造

#### 4.3.1 图像管线（固定，训练与部署同一条）

```
图像（任意来源：AtomAction 256×256 / RLBench 128×128 / 评测环境任意分辨率）
   │
   ├─ 双线性抗锯齿降采样 ──────► 128 × 128        ← 固定中间分辨率
   │
   ├─ resize ─────────────────► 224 × 224        ← DINOv2 ViT-B/14 输入（224 = 16 × 14）
   │
   ├─ ImageNet mean/std 归一化
   │
   ▼
DINOv2 ViT-B/14 ❄️（官方权重，全新独立实例，无 LoRA）
   └─► x_norm_patchtokens [16, 16, 768]
        └─ AvgPool2d(2) ─► [8, 8, 768] ─ flatten ─► 64 token ─ Linear(768→512) ─► [64, 512]
```

- **为什么固定 128 中间分辨率**：Adapter 训练源图是 AtomAction 的 256×256，而 PerAct 的观测是 128×128。把管线固定在 128 使两侧输入分布**完全一致**
- **为什么 2×2 池化**：Adapter 需要的是粗粒度空间语义（哪个物体、大致方位、夹爪开合），不是 patch 级细节；把每相机 256 token 降到 64，总序列 **215**，模型真正轻量。配置项 `patch_pool: 2`（设 1 则不池化，序列长 599）。
- 两相机 token 各加**相机类型嵌入** $\in\mathbb R^{2\times512}$ 与 **8×8 可学习 2D 位置嵌入**。

#### 4.3.2 指令管线

```
short 指令（去尾句号）── CLIP ViT-B/16 text tower ❄️ ─► token embeddings [77, 512]
   └─ 保留 77 位 + attention_mask ─ Linear(512→512) ─► [77, 512]
```

**格式规范**（Adapter 训练与 planner 输出共用）：

| 项 | 规则 | 依据 |
|---|---|---|
| 来源 | `variation_metadata.json::phase_descriptions` 中 `style == "short"` 的文本 | 实测 572 条（白名单内），**100% 符合"大写开头 + 单句 + 句号结尾"**，词数 3–12（中位数 4） |
| 归一化 | **去掉句尾句号**，其余不动 | 句号会成为独立 BPE token，是与 RLBench 原生指令格式的唯一实质差异 |
| 大小写 | **不处理** | CLIP 分词器在 `helpers/clip/core/simple_tokenizer.py:123` 强制 `.lower()`，Adapter 的 CLIP ViT-B/16 与 PerAct 的 CLIP RN50 走同一套分词逻辑，大小写在进入模型前已被抹平 |
| 长度 | **不截断**，保留 CLIP 全部 77 个位置 | 有效 token 由 tokenizer 的 `attention_mask` 标记（实测 6–8 个），padding 由注意力可见性矩阵屏蔽 |

归一化后的 short 指令（`"grasp the jar lid"`）与 RLBench 原生指令（`"close the red jar"` / `"open bottom drawer"`）在句式、长度、格式上完全同分布，因此**同一条字符串可同时喂给 Adapter 与 baseline**，不需要 planner 输出两套。



#### 4.3.3 输入序列拼装

Adapter 的输入**只有图像与指令两种模态**（决策 7），不引入低维状态。序列固定为四段共 **215** token：

| 段 | 索引 | 长度 | 内容 |
|---|---|---|---|
| **I** 图像 | `0:128` | $N_{cam}P_{cam}=128$ | 每相机 64 token + 相机类型嵌入 + 8×8 可学习 2D 位置嵌入 |
| **T** 文本 | `128:205` | $T_{txt}=77$ | CLIP token 特征经 `Linear(512→512)`；**不加额外位置嵌入**（CLIP 输出已含） |
| **G** 全局 query | `205` | 1 | 可学习参数 |
| **D** 细节 query | `206:215` | $N_{detail}=9$ | 可学习参数，第 $n$ 行对应码本第 $n$ 个细节槽位 |

段间可见性由 §4.4.1.1 的矩阵约束。因主干不含跨段位置编码，**拼接顺序在数学上不影响结果**（注意力对 token 顺序置换等变），此处的顺序只是索引约定。



### 4.4 模型结构

#### 4.4.1 主干

| 组件 | 规格 |
|---|---|
| 层数 $L_{ada}$ | **4** |
| 隐藏维 $d_{ada}$ | **512** |
| 头数 $H_{ada}$ | **8**（每头 64） |
| FFN 维 | **2048** |
| 归一化 | Pre-LN（LayerNorm） |
| 激活 | GELU |
| Dropout | 0.1（训练）/ 0（推理） |
| 注意力 | **可见性矩阵**（§4.4.1.1），非全可见 |
| 位置编码 | 视觉 8×8 可学习 2D + 相机类型嵌入；**文本不额外加位置嵌入**（CLIP 输出已含）；查询 token 无位置编码 |
| 模态类型嵌入 | **未采用** —— 与三段各自投影层的 bias 数学冗余（图像/文本各有独立投影，query 本身是自由参数） |

#### 4.4.1.1 注意力可见性矩阵

序列布局固定为 `[I | T | G | D]`：图像 128（front 64 + wrist 64）、文本 77、全局 query 1、细节 query 9，共 **215**。

| 行(query)＼列(key) | I `0:128` | T `128:205` | G `205` | D `206:215` |
|---|:---:|:---:|:---:|:---:|
| **I** | ✓ | ✓ | ✗ | ✗ |
| **T** | ✓ | ✓ | ✗ | ✗ |
| **G** | ✓ | ✓ | ✓ 自身 | ✗ |
| **D** | ✓ | ✓ | ✓ | ✓ |

三条性质：① 观测段看不到任何 query，图像/文本表征不被码预测任务污染；② G 看不到 D，全局码只由观测决定；③ D 可见 G，细节码以全局码为条件（与 §4.4.2 细节头拼接 $h_g$ 同向，一个隐式一个显式）。

与 padding 的合成：`attention_mask[b,q,k] = block_mask[q,k] AND key_valid[b,k]`，其中 `key_valid` 的图像段恒 True、文本段取自 CLIP 的 `attention_mask`、query 段恒 True。因每行至少可见 128 个恒有效的图像 key，**不存在整行被屏蔽**，softmax 不会出 NaN。

实现：`AdapterAttention` / `AdapterEncoderLayer`（`model/module/encoder.py`，专为 Adapter 新增，未改动 `MultiHeadAttention` 与 `VisualSelfAttentionLayer`）。


#### 4.4.2 输出头

取 10 个查询 token 在最后一层的隐状态 $h_g,\{h_{d,n}\}\in\mathbb R^{512}$：

**全局头**
$$\text{logits}_g=\text{Linear}(512\to36)\bigl(\text{GELU}(\text{Linear}(512\to512)(h_g))\bigr)$$

**细节头（9 个槽位共享参数，显式接收全局上下文）**
$$\text{logits}_{d,n}=\text{Linear}(512\to192)\Bigl(\text{GELU}\bigl(\text{Linear}(1024\to512)([\,h_{d,n};\,h_g\,])\bigr)\Bigr)$$

拼接 $h_g$ 即实现"细节码以全局码为条件"的层级依赖。


#### 4.4.3 参数量

| 部分 | 参数量 |
|---|---|
| Transformer 主干（4 层 × ~3.15 M） | 12.61 M |
| 视觉投影 768→512 | 0.39 M |
| 文本投影 512→512 | 0.26 M |
| 位置 / 类型 / 相机嵌入 + 查询 token | 0.05 M |
| 全局头（512→512→36） | 0.28 M |
| 细节头（共享，1024→512→192） | 0.62 M |
| **可训小计** | **14,209,252（实测）** |
| DINOv2 ViT-B/14 ❄️ | ≈ 86.6 M |
| CLIP ViT-B/16 text tower ❄️ | ≈ 63.4 M |

**推理成本**：每次子任务切换 = 2 次 DINOv2 前向 + 1 次 CLIP 文本前向 + 215-token 的 4 层 Transformer，单次 **15–25 ms**（单卡）。每 episode 约 5–15 次，可忽略。

### 4.5 训练目标

只有分类损失（决策 8，不引入码空间辅助损失）：

$$L=\text{CE}(\text{logits}_g,\;k_g)+\lambda_d\cdot\frac{1}{N_{detail}}\sum_{n=1}^{9}\text{CE}(\text{logits}_{d,n},\;k_{d,n}),\qquad \lambda_d=1.0,\ \epsilon=0.05$$

⚠️ 细节项的 9 个槽位在标签上恒等（§7 R1），因此该项实质只有 1 个自由度。

### 4.6 代码落点（✅ 全部实现）

| 环节 | 文件 |
|---|---|
| 硬量化与标签导出 | `model/module/nsvq.py`（`quantize_hard` / 两个 `encode_indices`）、`model/module/atomaction_nsvq.py`（`encode_codebook_indices`）、`data/import_codebook_index.py` |
| 模型 | `model/module/encoder.py`（`AdapterImageTokenizer` / `AdapterTextTokenizer` / `AdapterAttention` / `AdapterEncoderLayer`）、`model/adapter.py`（`Adapter`） |
| 数据 | `data/build_adapter_split.py`、`data/adapter_dataset.py` |
| 训练 | `scripts/train_adapter.py`、`config/train_adapter.yaml`、`run/train_adapter.sh`、`utils/loss_func.py`、`utils/evaluate_func.py` |

`AdapterDataset` 返回**原始 uint8 图像**（`[3,256,256]`），全部预处理由 `AdapterImageTokenizer` 在 forward 内完成，训练与部署共用同一条管线。`adapter_collate_fn` 的输出直接对齐 `Adapter.forward(camera_images, instructions)` 的签名。

### 4.7 类平衡与数据增广

| 项 | 决定 | 理由 |
|---|---|---|
| 类平衡 | **不做**（不重加权、不重采样） | Adapter 是冻结编码器的摊还，部署时的先验分布就是训练分布；重加权会牺牲高频码精度，而高频码最常被用到。实测 `k_global` 不平衡比 **27:1**（最多 13.1%、最少 0.48%），在 36 类里不算极端 |
| 图像增广 | **不做** | 先拿到干净 baseline；不缓存特征，随时可加 |

代价是准确率被高频类主导，因此评估必须同时报 micro 与 macro（指标体系见 §4.9.2）。

### 4.8 数据划分

#### 4.8.1 划分策略

| 项 | 设计 |
|---|---|
| 单位 | `(task, variation)` 下的 `phase_XXX` 编号，**所有 action 目录共用同一份选择** |
| 依据 | 同一 `(task, variation)` 下 `phase_XXX` 在各 action 目录中指向同一个 demo episode（136 组中 **124 组**各 action 目录的 phase 集合完全相同），因此这样切不会把同一 demo 的兄弟 phase 分到两侧 |
| 比例 | **val 5%**，每组取 `max(1, round(N×0.05))` |
| 分层 | 按 `(task, variation)` 逐组抽，保证 **531 个 `action/task/variation` 目录全部含 val 样本** |
| 可复现 | 每组种子由 `sha256(seed｜task｜variation)` 派生，与遍历顺序无关；同 seed 重跑逐条相同 |
| 不做任务级留出 | Adapter 与码本一样是预训练产物，在全部 69 个任务上学习（不按 Seen/UnSeen 划分） |
| 不设 test 集 | 真正的 test 是 Stage 3 的 RLBench 成功率 |

```bash
python data/build_adapter_split.py --val-ratio 0.05 --seed 0
```

#### 4.8.2 产物

划分**原地写回** `data/atomaction_codebook_index.json`（每条记录一个 `split` 字段 + `meta` 记 `split_strategy` / `val_ratio` / `split_seed` / `split_counts`），不生成额外 manifest。

#### 4.8.3 实测结果（seed=0, val_ratio=0.05）

| 项 | 实测 |
|---|---|
| train / val | **53,703 / 2,793**（val 占 4.94%） |
| 每目录 val 比例 | min 1.0% / 中位 5.0% / max 6.0%，**无 val 的目录 0 个**（min 落在 `change_channel/variation1` 那类 phase 编号不对齐的组） |
| episode 完整性 | 15,272 个 episode，**跨 split 0 个** |
| 覆盖 | train 覆盖 69 任务 / 17 原子 / 36 全局码；val 覆盖 69 任务 / 17 原子 / 36 全局码 |
| **指令-only 基线（全局码）** | train 自身 56.7%｜**train 众数 → val 58.0%** |
| **指令-only 基线（细节码槽0）** | train 自身 27.6%｜train 众数 → val 26.8% |

> ⚠️ **val 的基线必须用 train 拟合的众数评估。** val 每个语义组只有约 4.9 个样本，用 val 自拟合的众数会被小样本偏差抬高（66.5% vs 诚实值 58.0%）。
>
> **58.0% 是 Adapter 的对照线**：若 val top-1 贴着 58%，说明模型退化成"背指令"、没用上图像。

### 4.9 训练与结果（✅ 已完成，2026-08-29）

#### 4.9.1 配置与吞吐

| 项 | 值 | 依据 |
|---|---|---|
| 硬件 | **单卡** RTX 5880 Ada | 2.1 min/epoch，DDP 复杂度与收益不成比例 |
| batch / num_workers | **64 / 32** | batch 64 与 128 耗时相同（都约 2 min/ep），故按效果选 64：839 步/ep × 100 ep = **83,900 次更新**，是 128 的两倍 |
| epochs | 100（实际 20–30 即足够，见 §4.9.3） | — |
| lr / 调度 | 3e-4，warmup 3 ep + cosine → 5% | 与 `train.yaml::stage0_main` 同量级 |
| 优化器 | AdamW，β=(0.9, 0.99)，wd 1e-3，grad_clip 1.0 | norm / bias / embedding / query_tokens 不加 weight decay |
| 精度 / 显存 | bf16 autocast / **3.4 GiB** | — |
| 总耗时 | **3 h 23 min** | — |

**吞吐瓶颈是 PNG 解码，不是 GPU**（GPU-only 124 ms/step）。实测 batch=64 端到端：

| num_workers | 16 | **32** | 48 |
|---|---|---|---|
| ms/step | 318 | **147** | 133 |
| min/epoch | 4.5 | **2.1** | 1.9 |

机器 208 核 / 251 GB 内存，10.3 GiB 的 front+wrist PNG 可完全驻留 page cache。取 32 是因为已达 GPU 上限的 84%，48 只快 10% 却多占 16 个进程（共享服务器）。

#### 4.9.2 指标体系

围绕"给定指令和图像能否正确预测码索引"分三层：

| 层 | 指标 | 角色 |
|---|---|---|
| **1 单索引** | `k_global` micro top-1 | **best checkpoint 判据**（细节码只有 1 个自由度，不适合当判据） |
| | top-5 / macro top-1 / 按类频分层（head·mid·tail 各 12 类） | 诊断：macro 暴露长尾塌陷，分层回答"是否全靠高频类" |
| **2 联合** | **联合 top-1** = `k_g` 正确 **且** 9 个 `k_d` 全对 | 最贴近部署：一次子任务注入的整组码向量是否完全正确 |
| | 细节码 9 槽平均 top-1、槽间预测多样性 | 诊断（⚠️ 只有 1 个自由度） |
| **3 码向量** | $\lVert z^{pred}-z^{true}\rVert / \bar d$（$\bar d$ = 平均码间距） | Stage 3 消费的是**码向量**不是索引；0 = 全对，1 ≈ 随机 |
| 分组 | 分原子（17）/ 分任务（69）top-1 | 定位问题来源 |

> **第 3 层为何必要**：码本实测几何为「全局码字范数 8.13 / 两两距离均值 9.32 / 最近邻均值 5.72」，即便"只错到最近邻"，误差也已是平均码间距的 **61%** —— 索引错误没有便宜的错法，但错得远近对 VLA 的影响天差地别，而这在 top-1 里是同一个"错"。

`micro` 按样本平均、`macro` 按类别平均。主指标取 micro，因为 Adapter 是摊还，**部署时的样本分布就是训练分布**，micro 直接等于"平均多久对一次"。

#### 4.9.3 结果

**最佳 checkpoint：epoch 94**

| 指标 | best (ep94) | 指令-only 基线 | Δ |
|---|---|---|---|
| **val 全局码 micro top-1** | **89.69%** | 58.0% | **+31.7** |
| val top-5 | 97.82% | — | — |
| val macro top-1 | 86.92% | — | — |
| head / mid / tail | 0.912 / 0.866 / **0.846** | — | 长尾未塌 |
| **联合 top-1** | **68.39%** | 23.0% | **+45.4** |
| 细节码 9 槽平均 top-1 | 72.60% | 26.8% | +45.8 |
| **码向量归一化误差**（全局 / 细节） | **0.072 / 0.170** | 1.0 ≈ 随机 | — |

**收敛轨迹**

| epoch | 1 | 10 | 20 | 30 | 50 | 80 | **94** | 100 |
|---|---|---|---|---|---|---|---|---|
| train top-1 | 0.679 | 0.920 | 0.963 | 0.986 | 0.998 | 1.000 | 1.000 | 1.000 |
| val top-1 | 0.8188 | 0.8833 | 0.8876 | 0.8915 | 0.8915 | 0.8955 | **0.8969** | 0.8955 |
| val macro | 0.777 | 0.852 | 0.859 | 0.869 | 0.868 | 0.872 | 0.869 | 0.870 |

**epoch 10 即达 88.3%，之后 90 个 epoch 仅再涨 1.4 个点**；train top-1 自 epoch 80 起锁死 1.000（完全记住训练集）但 val 未下滑，只是进入平台。**结论：20–30 epoch 足够，100 是过量的**，复现时可省约 2.5 小时。

**分原子 top-1**（best epoch）

| 层级 | 原子 |
|---|---|
| 1.000 | flip-close、flip-open、**hang**、pull、push、revolve-in、revolve-out、slide、wipe |
| 0.92–0.97 | place 0.961、lift 0.960、press 0.945、grasp 0.929、**insert 0.929** |
| < 0.91 | transfer 0.903、rotate 0.868、**approach 0.743** |

§7 R2/R3 担心的稀有原子（`hang` 全库仅 100 条、`insert` 283 条）**在 val 上表现最好**；反而样本量最大的 `approach`（11,998 条）最差 —— 符合直觉："朝某物移动"的起始帧最难预示后续轨迹落到哪个码。

#### 4.9.4 产物

| 文件 | 大小 | 用途 |
|---|---|---|
| `checkpoints/vqap_adapter/best.pth` | **56.9 MB** | **部署产物**，插件 `vqap_adapter.pth` 的来源。只含 14,209,252 个可训参数 + meta（两个冻结塔的身份、标签 JSON 的 sha256、split seed/ratio） |
| `checkpoints/vqap_adapter/latest.pth` | 170.7 MB | 续训用，另含 optimizer(108 MB) / scheduler / RNG 状态 |
| `log/vqap_adapter_*.log` | — | 每 epoch 的三层指标 + 分原子明细 |

加载 `best.pth` 时必须断言 `unexpected_keys` 为空、`missing_keys`（371 个）全部属于两个冻结塔 —— 冻结塔在 `Adapter.__init__` 时从 torch.hub / HuggingFace 缓存加载，不入盘。

```bash
python scripts/train_adapter.py --overfit-batch 64 --max-steps 300   # 过拟合自检
bash run/train_adapter.sh                                            # 正式训练
```


## 五、码向量注入：对外接口契约

> 本节只给**与 baseline 无关的接口**与**PerAct 上的注入点/注入方式**。完整的 Stage 3 训练方案（冻结边界细目、code-dropout、replay schema、数据生成）见 `VLA_Design.md`（待重写）。

### 5.1 插件 API

```python
class AtomActionCodebook(nn.Module):                       # 冻结
    global_codebook: Tensor      # [36, 512]
    detail_codebook: Tensor      # [192, 512]
    def lookup(self, k_global, k_detail) -> Tuple[Tensor, Tensor]   # → z_g [512], Z_d [9,512]

class CodeInjector(nn.Module):                             # 可训，每个 baseline 实现一次
    def forward(self, feat, z_g, Z_d, code_mask) -> feat_prime
```

**适配一个新 baseline 只需三步**：① 定位其「冻结主干之后、动作解码之前的最后一个共享张量」；② 在该处插一个 `CodeInjector`；③ 提供子任务边界的触发钩子。Adapter、码本、Planner 均无需改动。

### 5.2 PerAct 的注入点

PerAct 前向的关键张量（`voxel_size=100`, `im_channels=64`, `lang_fusion='seq'`, `patch 5/5`, `latents 2048×512`, `depth 6`）：

```
voxel_grid [B,10,100³] ─.detach()─► input_preprocess ─► d0 [B,64,100³]
   ├─► feats += ss0(d0) ⊕ maxp(d0)                                     (256)
   ▼
patchify ─► [B,64,20³] ─ +proprio ─► [B,128,20³] ─ flatten+lang ─► [B,8077,128]
   ▼
Perceiver（cross_attn + 6× self-attn）─► x [B,2048,512]
   ▼
decoder_cross_attn ─► ★ latents [B,128,20,20,20]    ← 注入点 D
   ├─► feats += ss1(latents) ⊕ maxp(latents)                           (512)
   ▼
up0 ─► u0 [B,64,100³] ─ final(cat[d0,u0]) ─► u [B,64,100³]
   ├─► trans_decoder ─► q_trans [B,1,100³]                    平移热图
   ├─► feats += ss_final(u) ⊕ maxp(u)                                  (256)
   └─► dense0(1024→256) ─► dense1(256→64) ─► rot_grip_collision_ff ─► 220
                                                    72×3 旋转 + 2 夹爪 + 2 碰撞
```

**选定注入点 D（`latents`，[B, 128, 20, 20, 20]，1.02 M 元素/样本）**。逐条对比：

| 候选 | 张量 | 元素/样本 | 梯度可达可训参数 | 同时影响平移与旋转 | 结论 |
|---|---|---|---|---|---|
| A | `voxel_grid` [B,10,100³] | 10.0 M | ❌ 已 `.detach()`，且需解冻 `input_preprocess` | ✓ | 否决 |
| B | Perceiver 输入 [B,8077,128] | 1.03 M | ❌ 需解冻主干 | ✓ | 否决 |
| C | 追加 code token 到 [B,8078,128] | ~0 | ❌ 需解冻主干；新增 token 改变 softmax 归一化，**无法零初始化到精确等价**；`pos_encoding` 长度固定 8077 | ✓ | 否决 |
| **D** | **`latents` [B,128,20³]** | **1.02 M** | ✅ 下游 `up0/final/trans_decoder/dense*` 全可训 | ✅ 两支路都过 | **✅ 选定** |
| E | `u0` [B,64,100³] | 64.0 M | ✅ | ✗ 只到平移 + `ss_final` | 否决（贵 63×，覆盖更窄） |
| F | `u` [B,64,100³] | 64.0 M | ✅ | 部分 | 否决（同上） |
| G/H/I | `feats` / `dense0` / `dense1` | ≤ 1024 | ✅ | ✗ 只到旋转/夹爪/碰撞 | 单独不够 |

D 是**冻结主干之后的第一个张量**，也是**唯一同时通向两个动作分支**的张量：平移经 `up0 → final → trans_decoder`，旋转/夹爪/碰撞经 `ss1/maxp → feats → dense0`。显存代价比 `u` 便宜 **63 倍**。

> **已知的通路强弱差异**：`SpatialSoftmax3D` 对每通道在空间维做 softmax，而通道级 FiLM 的 $\beta_c$ 在空间上是常数，**在 softmax 中被完全抵消**，$\gamma_c$ 只起逐通道温度作用。因此码信息进入旋转分支的路径是：`global_maxp(latents)`（128 维，完整）+ `ss1(latents)`（384 维，仅 $\gamma$ 的温度效应）+ `ss_final(u)`/`global_maxp(u)`（256 维，经 `up0/final` 的卷积与 LReLU 非线性后完整）。**单点 D 可用**，但若实验中出现"平移明显改善、旋转/夹爪几乎不动"，第一嫌疑就在这里，届时可补一个 `dense0` 输出上的 FiLM（0.13 M 参数，随时可加）。

### 5.3 注入方式：全局码 FiLM + 细节码 cross-attention

`latents` 展平为 **8000 个体素 token，维度 128**。

**(a) 全局码 → FiLM（作用于通道）**

$$c_g=\text{SiLU}\bigl(W_2\,\text{SiLU}(W_1\,\text{LN}(z_g))\bigr)\in\mathbb R^{256},\qquad [\gamma,\beta]=W_{\text{film}}\,c_g\in\mathbb R^{2\times128}$$
$$h=\text{latents}\odot(1+\gamma)[:,:,\text{None},\text{None},\text{None}]+\beta[\dots]$$

**(b) 细节码 → cross-attention（作用于空间）**

$$S=Z_d+\text{slot\_embed}\in\mathbb R^{9\times512}$$
$$Q=W_q\,\text{LN}(h_{\text{tok}})\in\mathbb R^{8000\times128},\quad K=W_kS,\ V=W_vS\in\mathbb R^{9\times128}$$
$$O=W_o\bigl(\text{softmax}(QK^\top/\sqrt{d_h})\,V\bigr),\qquad H=4\ \text{头},\ d_h=32$$
$$\text{latents}'=h+g\odot O,\qquad g\in\mathbb R^{128}\ \text{可学习门}$$

**cross-attention 的设计意图**：把 9 个细节码池化成 1 个 512 维向量再 FiLM，等于把 9 个码平均掉。改成 cross-attention 后，每个体素 token 可以独立选择关注哪几个细节码 —— 夹爪附近的体素关注"接近角度"码、目标物体附近的体素关注"接触方式"码，使 9 个细节码获得空间分工。`slot_embed` 用于区分 9 个槽位的语义（细节槽位本身无序）。

> ⚠️ **该设计意图当前无法实现（§7 R1）**：实测 9 个细节码在样本内**恒等**，$Z_d$ 的 9 行完全相同，$K,V$ 只差一个与输入无关的 `slot_embed`，这条支路退化成"一个码向量 + 一组常量"。Stage 3 实施前需先决定：① 照旧实现（等价于单向量条件）；② 简化为单个细节码；③ 回头修 Stage 0/1 的细节支路。


**成本**

| 项 | 参数量 | 显存/样本 | 计算/样本 |
|---|---|---|---|
| FiLM 支路（$W_1,W_2,W_{\text{film}}$） | 0.26 M | ~0 | ~0 |
| cross-attn 支路（$W_q,W_k,W_v,W_o$, slot_embed, $g$） | 0.17 M | 注意力矩阵 [4,8000,9] = 1.15 MB；输出 O = 4 MB | Q 投影 131 MFLOPs |
| **合计** | **≈ 0.43 M** | **≈ 5 MB** | 可忽略 |

**三条实现红线**

1. **零初始化等价性**：$W_{\text{film}}$ 的 weight 与 bias 全部 zeros-init → $\gamma=\beta=0$；门 $g$ zeros-init → $g\odot O=0$。训练开始时前向与原版 PerAct **逐位等价**。
2. **⚠️ 门控残差的零初始化陷阱**：**不能把 $g$ 与 $W_o$ 同时零初始化**。若 $W_o=0$ 则 $O\equiv 0$，而 $\partial L/\partial g\propto O$，$g$ 的梯度恒为零，**这条支路会永久失活**。正确做法：**$W_o$ 用 small-normal 初始化，只把 $g$ 置零**。FiLM 那条没有这个问题（$\partial L/\partial W_{\text{film}}\propto\partial L/\partial h\odot\text{latents}\neq 0$），照常零初始化。
3. **`code_mask` 硬关断**：`use_codebook=false` 时强制 $\gamma=\beta=0$ 且 $g\odot O=0$ —— 是**精确等价**而非近似退化。必须有单元测试断言：`code_mask=0` 时的输出与冻结的原版 PerAct **bit-exact**。

细节码消融只需关掉 (b) 支路，不影响 (a)。

### 5.4 冻结 / 可训边界

| ❄️ 冻结 | ✅ 可训 |
|---|---|
| `input_preprocess`, `patchify`, `proprio_preprocess`, `pos_encoding`, `latents`, `cross_attend_blocks`, `layers`(6 层 latent self-attn), `decoder_cross_attn`, CLIP RN50 | `up0`, `final`, `trans_decoder`, `dense0`, `dense1`, `rot_grip_collision_ff`（合计 ≈ 2.05 M） |
| 码本、Adapter、DINOv2、CLIP ViT-B/16 | 注入层 FiLM + cross-attn（≈ 0.43 M） |
| — | **`lang_preprocess`**（`Linear(512→128)`，65 K）—— 整任务指令→子任务指令是语言输入分布的实质改变，此层是语言进入冻结主干的唯一门户，必须可适应 |

**损失完全沿用 PerAct 原生损失**（平移体素 CE + 3 轴旋转 CE + 夹爪 CE + 碰撞 CE），不新增任何损失项。SE3 数据增广照常开启：码是旋转不变的语义条件，与增广无冲突。

### 5.5 Stage 3 其余细目（TBD）

code-dropout 调度、replay buffer 的 `subtask_id` / 码缓存 schema、训练数据生成、评测时序 —— 见 `VLA_Design.md`（待重写）。

---

## 六、Planner 契约（TBD）

Planner 采用 VLM API，不训练，靠 prompt 约束。其职责边界、输出 schema、状态机、调用时机、以及 §4.3.2 提到的**进度/序数字段**要求，待后续轮次讨论后写入 `VLA_Design.md`。

---

## 七、已知风险与实测结论

> 本节记录 Stage 2 实施过程中**实测发现**的问题。每条给出证据、当前处置、以及**是否已兑现**。

| # | 风险 | 状态 |
|---|---|---|
| R1 | 细节码 9 槽位恒等 | ⚠️ **未解决**（已接受） |
| R2 | 4 个部署任务不在训练集 | ⚠️ **未解决**，且因未做任务级留出而**无法在 val 上度量** |
| R3 | 码标签高度任务特异 | ✅ 未兑现（val +31.7 点），但跨任务结论未被推翻 |
| R4 | checkpoint 膨胀 | ✅ 已处理 |
| R5 | fp32 vs bf16 导出精度 | ✅ 已处理 |
| R6 | `split` 被导出脚本覆盖 | ✅ 可重建 |
| R7 | 指令须按 `source_phase_index` 回查 | ✅ 已处理 |
| R8 | top-5 随过拟合下降 | ⚠️ **未解决**，影响 Stage 3 的 top-k 方案 |
| R9 | 续训存在 0.1% 偏差 | ⚠️ **未隔离根因** |
| R10 | $\lambda_d$ 的 A/B 从未执行 | ⚠️ **未探索** |

### R1 细节码 9 个槽位在样本内恒等（严重，已接受）

**现象**：全库 56,496 条记录，样本内 9 个细节槽位的平均唯一索引数 **1.00 / 9**（600 样本抽查中 598 个是 9 槽全同）。

**根因**：细节支路的 cross-attention 退化成了**均匀平均池化**。

| 证据 | 数值 |
|---|---|
| 注意力熵 / 均匀分布上限 | **96.4% / 100.0% / 100.0%**（三个样本） |
| 9 个 query 的 cross-attn 输出两两余弦 | **1.0000**（槽间 std 0.0003–0.001） |
| 9 个 `learnable_queries` 两两余弦 | 0.0398（**query 本身没塌，仍近似正交**） |
| `learnable_queries` 范数 | 0.315–0.338（初始化值 `0.02×√512 ≈ 0.45`，训练中反而略缩） |

query 有区分度但范数太小，经 `q_proj` 后打分幅度不足以让 softmax 分化，9 个 query 拿到同一个 uniform 平均。细节码于是退化成"另一个投影 + 另一张码本"作用在**与全局码同一个均值池化特征**上 —— 二者归一化互信息 **0.818**。

**为什么训练指标没报警**：`ppl_d` 在 `[B×9]` 展平后统计，9 个重复索引只把每样本计数放大 9 倍、不改变分布形状，所以 51.21 看着"健康"（阈值 48）。**这个指标结构上发现不了槽位塌陷。**

**影响**：① Stage 2 的 9 路细节分类 = 1 路重复 9 次；② Stage 3 §5.3 的细节码 cross-attention（"让 9 个细节码获得空间分工"）失去意义 —— $Z_d$ 的 9 行完全相同，$K,V$ 只差与输入无关的 `slot_embed`；③ 细节码消融的结论会失去意义。

**处置**：**接受现状**（方案 A），照常导出 9 个重复标签、Adapter 照常预测 9 路，先跑通到 Stage 3 拿到端到端结果再决定是否回头修 Stage 0/1。若要修：放大 `learnable_queries` 初始化尺度 / 给 query 加 LayerNorm 或缩放 / 加槽间多样性损失，代价是重训 250 epoch。

**训练后实测（2026-08-29）**：模型的槽间预测多样性 **1.06 / 9**，与标签的 1.00 一致 —— 模型忠实学到了标签的塌陷，符合预期。细节码 9 槽平均 top-1 = 72.60%（基线 26.8%）。

**排查提示**：报告 9 槽准确率时必须注明只有 1 个自由度。槽间预测多样性已作为常规指标写入 `utils/evaluate_func.py`。

### R2 4 个部署任务不在 Adapter 训练集中（更正 `Exp_Design.md` 声明 B）

`Exp_Design.md` 声明 B 原称"UnSeen 6 个任务中，仅 `stack_cups` 完全不在 AtomAction_Dataset 中"。**实测是 4 个不在**（已扫描含 `pose-adjust` 在内的全部 18 个动作目录）：

| 部署任务 | 在 AtomAction（69 任务）中？ |
|---|---|
| Seen12 的 9 个：close_jar, light_bulb_in, open_drawer, place_shape_in_shape_sorter, push_buttons, put_groceries_in_cupboard, reach_and_drag, stack_blocks, sweep_to_dustpan | ✅ |
| **Seen12 的 3 个：`place_cups`, `slide_block_to_target`, `stack_wine`** | ❌ |
| UnSeen6 的 5 个：insert_onto_square_peg, meat_off_grill, put_item_in_drawer, put_money_in_safe, turn_tap | ✅ |
| **UnSeen6 的 1 个：`stack_cups`** | ❌ |

**叠加风险**：`place_cups`（独占原子 `hang`）与 `stack_wine`（独占原子 `insert`）恰是"独占原子任务"，而 `hang` 全库仅 **100 条且只来自 1 个任务**（`hang_frame_on_hanger`）、`insert` 仅 **283 条**（来自 4 个任务，不含 stack_wine）。**最稀缺的原子，其部署载体任务恰好没见过。**

**处置**：记录风险，**不为此调整训练数据**，先跑出端到端 VLA 结果再决定。Stage 3 的缓解手段：对这些子任务由 planner 置 `use_codebook=false` 关闭码注入（planner 契约本就有此开关）。

**训练后实测**：担心的两个稀有原子在 val 上表现**最好**（`hang` 1.000、`insert` 0.929，§4.9.3）。但这**不能**解除风险 —— val 里全是训练见过的任务，而 R2 说的是**任务缺失**这一条件；因为没做任务级留出（决策 20），该条件在 val 上**根本不存在**。⚠️ **风险仍然悬着，只能等 Stage 3 的端到端结果**。

### R3 码标签高度任务特异，文本侧无可迁移信息

| 预测方式 | `k_global` 命中率 |
|---|---|
| 仅靠 action（17 类众数） | 25.0% |
| **跨任务迁移（用其它任务同 action 的众数预测本任务）** | **21.3%** |
| (action, task) 众数 | 54.4% |
| 指令组 (task, variation, source_phase_index) 众数 | 56.8% |

分动作的跨任务迁移更差：`insert 0%`、`slide 0%`、`push 0%`、`revolve-in 0%`、`wipe 4%`、`rotate 10%`。

**推论**：(action, task) 的 54.4% 已逼近完整指令组的 56.8%，说明**文本侧信号几乎全部来自"是哪个任务"，而非"是哪个原子"**——与组合泛化相反。因此把文本输入换成 `action-object` 形式**无法**缓解 R2：它改变的是输入编码，创造不出标签里不存在的信息。

（附：CLIP 处理单词与连字符本身没有问题 —— `place-cups` 被切成 `['place</w>','-</w>','cups</w>']`，与 `place cups` 的余弦达 **0.948**，`grasp jar lid` 与 `Grasp the jar lid` 达 0.929，action-object 短语同动作内 0.808 / 跨动作 0.622。技术可行，只是无益于 R2。）

**处置**：保持 short 指令原文。真正的杠杆在图像侧（冻结 DINOv2 的跨任务泛化能力）与 Stage 3 的 `use_codebook` 开关。

**训练后实测**：由此推出的「Adapter 可能训不过 58.0% 基线」**未兑现** —— val top-1 89.69%，高出 31.7 个点，说明图像侧确实提供了指令之外的大量信息。但表中"跨任务迁移仅 21.3%"的结论**未被推翻**：val 中没有任何新任务，该数字度量的条件未被检验（同 R2）。

### R4 checkpoint 膨胀（✅ 已处理）

`Adapter.state_dict()` 含两个冻结塔的 149.7 M 参数，**每存一次 625.6 MB**，其中 571 MB 是可无损重建的官方权重。按 `requires_grad` 过滤后 **54.2 MB（11.5×）**。

保存端：
```python
trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
trainable_state = {k: v for k, v in model.state_dict().items() if k in trainable_names}
```
加载端必须校验 `missing_keys` 全部属于两个冻结塔、`unexpected_keys` 为空（实测 missing 371 个且全属冻结塔，可训权重逐位一致）。

**实际落地**：`best.pth` **56.9 MB**（部署产物，不含优化器状态）、`latest.pth` **170.7 MB**（含 optimizer 108 MB + scheduler + RNG，供续训）。用 `best.pth` 续训会明确报错而非静默失败。

代价：checkpoint 不再自包含，依赖 `~/.cache/torch/hub` 与 `~/.cache/huggingface`。meta 已记录两个塔的身份标识（`dinov2_vitb14` / `openai/clip-vit-base-patch16`）与标签 JSON 的 sha256。

### R5 导出精度：fp32 vs bf16

训练用 bf16 autocast，标签导出用 **fp32**（硬量化取 argmin，fp32 不会在近邻边界翻码）。512 样本实测差异：`k_global` **0.00%**、`k_detail` **0.98%**。

### R6 `split` 字段会被导出脚本覆盖

重跑 `data/import_codebook_index.py` 会重写 index JSON 并清掉 `split`。划分完全由 `(seed, val_ratio, 记录内容)` 决定，重跑导出后再执行一次 `build_adapter_split.py` 即可恢复**逐条相同**的划分。

### R7 指令必须按 `source_phase_index` 回查

`phase_XXX` 目录名是 `output_phase_index`（样本序号），`variation_metadata.json::phase_descriptions` 用的是 `source_phase_stats.phase_index`（语义 phase 序号）。用目录名查会**静默查错**：实测 `push/close_drawer/variation1/phase_001` 目录索引 1 查到「Start pushing the middle drawer inward.」，而正确的（source 索引 2）是「Finish pushing the middle drawer closed.」——语义相反且不报错。

### R8 top-5 随过拟合下降（⚠️ 未解决，影响 Stage 3）

| epoch | 10 | 30 | 50 | 94 |
|---|---|---|---|---|
| val top-1 | 0.8833 | 0.8915 | 0.8915 | **0.8969** |
| val top-5 | **0.9946** | 0.9807 | 0.9753 | 0.9782 |

top-1 在涨而 **top-5 在跌**，是过拟合的典型特征：模型把概率质量压到单一候选上，次优候选的质量变差。

**影响**：若 Stage 3 采用 top-k 候选注入（而非只注入 argmax），应选 **epoch 10–20 的 checkpoint**（top-5 ≈ 0.99）而非 `best.pth`。当前只保存了 `best.pth` 与 `latest.pth`，**中间 epoch 的权重已丢失**，届时需重训（20 epoch 约 40 min）。

### R9 续训存在 0.1% 偏差，根因未隔离（⚠️ 未解决）

| 运行 | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| A 连续 3 epoch | 4.7255 / 0.6790 | 3.2059 / 0.8071 | **2.7843 / 0.8395** |
| B 2 epoch + 续训 1 epoch | 4.7255 / 0.6790 | 3.2059 / 0.8071 | **2.7871 / 0.8390** |

跨进程同种子的 epoch 1–2 **逐位一致**（数据管线与初始化确定性成立），但续训后的 epoch 3 loss 差 0.1%。状态恢复已逐项核对无误（epoch / global_step / best / optimizer / scheduler / RNG / 权重），怀疑是 `cudnn.benchmark=True` 的算法自动调优 + bf16 归约顺序，**未证实**。

**影响**：续训是崩溃恢复的保险，单 epoch 0.1% 的损失差不改变结论。要严格复现可关掉 `cudnn.benchmark` 并开 `torch.use_deterministic_algorithms(True)`，代价是训练变慢。

### R10 $\lambda_d$ 的 A/B 从未执行（⚠️ 未探索）

初始时细节损失 ≈ $\ln 192 = 5.26$，全局损失 ≈ $\ln 36 = 3.58$ —— **一个已知退化（9 个标签恒等、基线仅 26.8%）的任务，在共享主干上的梯度权重反而更大**。这是否压低了全局码准确率，**仍是未知数**。

主跑取 $\lambda_d=1.0$（忠实设计文档）且结果达标（89.69%），故未执行对照。`config/train_adapter.yaml::loss.lambda_detail` 是配置项，一次 $\lambda_d=0.1$ 的对照实验约 3.5 h（按 §4.9.3 的结论只需 20–30 epoch，约 1 h）。
