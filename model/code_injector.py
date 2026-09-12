"""码向量注入模块：把 VQAP 的双码本条件注入 PerAct 的 latents。

作用点是 PerAct 的 `latents` [B, 128, 20, 20, 20] —— 冻结主干之后的第一个张量，
也是**唯一同时通向平移与旋转两个动作分支**的张量：
    latents ─┬─ ss1/global_maxp → feats → dense0 → 旋转/夹爪/碰撞
             └─ up0 → final → trans_decoder → 平移

结构（Adapter_Design §5.3）
    (a) 全局码 z_g  → FiLM，作用于**通道**维
    (b) 细节码 Z_d  → cross-attention，作用于**空间**维（8000 个体素 token）

三条实现红线
    1. `w_film` 的 weight 与 bias 全部 zeros-init → 初始 γ=β=0，训练起点与原版逐位等价。
    2. **`w_o` 用 small-normal，只把 `gate` 置零。** 若 w_o 也置零则 O≡0，而 ∂L/∂gate ∝ O，
       gate 的梯度恒为零，这条支路会**永久失活**。FiLM 那条没有此问题
       （∂L/∂w_film ∝ ∂L/∂h ⊙ latents ≠ 0），照常零初始化。
    3. `code_mask` 必须以**乘法**作用在 γ/β 与门控项上，不能用 if 分支——
       batch 内 mask 混合时 if 无法向量化，且容易漏掉某一支。
       乘 0.0 / 乘 1.0 / 加 0.0 在浮点下都是精确运算，因此 mask=0 时输出与输入**逐位相同**。

⚠️ 已知通路强弱差异（Adapter_Design §5.2）：`SpatialSoftmax3D` 对每通道在空间维做 softmax，
   而通道级 FiLM 的 β 在空间上是常数、**在 softmax 中被完全抵消**，γ 只起逐通道温度作用。
   码进入旋转分支的完整路径主要是 `global_maxp(latents)`（128 维，完整）与经 up0/final
   卷积后的 `ss_final(u)`。若实验出现「平移明显改善、旋转几乎不动」，第一嫌疑就在这里，
   届时可补一个 dense0 输出上的 FiLM（0.13 M，随时可加）。

⚠️ 细节码支路的已知退化（Adapter_Design §7 R1）：实测 9 个细节码在样本内**恒等**，
   Z_d 的 9 行完全相同，K/V 只差一个与输入无关的 slot_embed，这条支路实际退化为
   「一个码向量 + 一组常量」。保留结构以便后续修复 Stage 0/1 后直接受益。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CodeInjector(nn.Module):
    #: 相位嵌入表的行数。实测 replay 里最大段下标是 19（stack_blocks 最长 20 段）。
    MAX_PHASE = 24

    def __init__(self, dim: int = 128, code_dim: int = 512, n_slots: int = 9,
                 hidden: int = 256, heads: int = 4, use_detail: bool = True,
                 use_phase: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.heads = heads
        self.use_detail = use_detail
        self.use_phase = use_phase

        # ---- (c) 相位嵌入：当前是计划里的第几段 ----
        # 为什么加这一路（实测依据）：
        #   段数 vs 成功率 r = −0.41(B0) / −0.46(B1)；
        #   短程(≤3段) 3 个任务 B0 均值 64.0%，长程(≥5段) 13 个任务只有 32.9%，差 31.1 pp。
        # PerAct 在一个关键帧只看到当前视觉 + 一句**恒定**的整任务指令，
        # 长程任务里同一视觉状态会出现在不同阶段（stack_blocks 摞第 2 块和第 3 块
        # 画面几乎一样），模型无从判断自己走到哪一步 —— 这正是 planner 知道、
        # 而模型拿不到的信息。
        # 它**任务无关**（只是「第几段」），所以能迁移到 UnSeen；
        # 参数量 24×128 = 3,072，可忽略。
        if use_phase:
            self.phase_embed = nn.Embedding(self.MAX_PHASE, dim)
            nn.init.normal_(self.phase_embed.weight, std=0.01)

        # ---- (a) 全局码 → FiLM ----
        self.ln_g = nn.LayerNorm(code_dim)
        self.mlp_g = nn.Sequential(
            nn.Linear(code_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.w_film = nn.Linear(hidden, 2 * dim)

        # ---- (b) 细节码 → cross-attention ----
        if use_detail:
            assert dim % heads == 0, f"dim {dim} 必须能被 heads {heads} 整除"
            self.slot_embed = nn.Parameter(torch.randn(n_slots, code_dim) * 0.02)
            self.ln_h = nn.LayerNorm(dim)
            self.w_q = nn.Linear(dim, dim, bias=False)
            self.w_k = nn.Linear(code_dim, dim, bias=False)
            self.w_v = nn.Linear(code_dim, dim, bias=False)
            self.w_o = nn.Linear(dim, dim, bias=False)
            self.gate = nn.Parameter(torch.zeros(dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 红线 1：FiLM 输出层全零 → 初始 γ=β=0
        nn.init.zeros_(self.w_film.weight)
        nn.init.zeros_(self.w_film.bias)
        for m in self.mlp_g:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        if self.use_detail:
            for w in (self.w_q, self.w_k, self.w_v):
                nn.init.xavier_uniform_(w.weight)
            # 红线 2：w_o 必须非零，否则 gate 永远收不到梯度
            nn.init.normal_(self.w_o.weight, std=0.02)
            nn.init.zeros_(self.gate)

    # ------------------------------------------------------------------
    def _cross_attn(self, h: torch.Tensor, Z_d: torch.Tensor) -> torch.Tensor:
        """h [B,C,X,Y,Z] × Z_d [B,S,code_dim] → O [B,C,X,Y,Z]"""
        b, c = h.shape[0], h.shape[1]
        spatial = h.shape[2:]
        tok = h.reshape(b, c, -1).transpose(1, 2)             # [B,N,C]
        s = Z_d + self.slot_embed.unsqueeze(0)                # [B,S,code_dim]

        q = self.w_q(self.ln_h(tok))                          # [B,N,C]
        k = self.w_k(s)                                       # [B,S,C]
        v = self.w_v(s)
        hd = c // self.heads
        q = q.view(b, -1, self.heads, hd).transpose(1, 2)     # [B,H,N,hd]
        k = k.view(b, -1, self.heads, hd).transpose(1, 2)     # [B,H,S,hd]
        v = v.view(b, -1, self.heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)           # [B,H,N,hd]
        o = o.transpose(1, 2).reshape(b, -1, c)               # [B,N,C]
        o = self.w_o(o)
        return o.transpose(1, 2).reshape(b, c, *spatial)

    def forward(self, latents: torch.Tensor, z_g: torch.Tensor,
                Z_d: torch.Tensor | None = None,
                code_mask: torch.Tensor | None = None,
                phase: torch.Tensor | None = None) -> torch.Tensor:
        """
        latents   : [B, C, X, Y, Z]   PerAct 的 latents，C=128
        phase     : 仅为与 v2 统一调用签名，**v1 不使用** —— v1 行为逐位不变
        z_g       : [B, code_dim]     全局码向量
        Z_d       : [B, S, code_dim]  细节码向量（use_detail=False 时可为 None）
        code_mask : [B] 或 [B,1]      0/1；0 表示该样本关闭码注入，输出与 latents 逐位相同
        """
        b, c = latents.shape[0], latents.shape[1]
        if code_mask is None:
            m = latents.new_ones(b, 1)
        else:
            m = code_mask.reshape(b, 1).to(latents.dtype)

        # ---- (a) FiLM ----
        n_spatial = latents.dim() - 2                         # 3（X,Y,Z）
        bc_shape = (b, c) + (1,) * n_spatial                  # [B,C,1,1,1] 供 γ/β 广播
        gate_shape = (1, c) + (1,) * n_spatial                # [1,C,1,1,1] gate 是逐通道参数
        m_shape = (b, 1) + (1,) * n_spatial                   # [B,1,1,1,1] mask 是逐样本标量

        gamma, beta = self.w_film(self.mlp_g(self.ln_g(z_g))).chunk(2, dim=-1)
        gamma = gamma * m                                     # mask=0 → 精确为 0
        beta = beta * m
        h = latents * (1.0 + gamma).view(bc_shape) + beta.view(bc_shape)

        # ---- (b) cross-attention ----
        if not self.use_detail or Z_d is None:
            return h
        o = self._cross_attn(h, Z_d)
        return h + self.gate.view(gate_shape) * o * m.view(m_shape)

    # ------------------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CodebookLookup(nn.Module):
    """冻结的码本查表：(k_global, k_detail[9]) → (z_g [B,512], Z_d [B,9,512])。

    码本来自 checkpoints/vqap_pretrain/stage1/codebook.pth，全程冻结、不参与训练。
    """

    def __init__(self, global_codebook: torch.Tensor, detail_codebook: torch.Tensor) -> None:
        super().__init__()
        assert global_codebook.dim() == 2 and detail_codebook.dim() == 2
        self.register_buffer("global_codebook", global_codebook.float(), persistent=False)
        self.register_buffer("detail_codebook", detail_codebook.float(), persistent=False)

    @property
    def n_global(self) -> int:
        return int(self.global_codebook.shape[0])

    @property
    def n_detail(self) -> int:
        return int(self.detail_codebook.shape[0])

    @torch.no_grad()
    def forward(self, k_global: torch.Tensor, k_detail: torch.Tensor):
        kg = k_global.long().reshape(-1)
        kd = k_detail.long()
        if not (0 <= int(kg.min()) and int(kg.max()) < self.n_global):
            raise IndexError(f"k_global 越界: [{int(kg.min())}, {int(kg.max())}] "
                             f"不在 [0, {self.n_global})")
        if not (0 <= int(kd.min()) and int(kd.max()) < self.n_detail):
            raise IndexError(f"k_detail 越界: [{int(kd.min())}, {int(kd.max())}] "
                             f"不在 [0, {self.n_detail})")
        return self.global_codebook[kg], self.detail_codebook[kd]


def load_codebook(path: str, device: str | torch.device = "cpu") -> CodebookLookup:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    g = next(v for v in ck["global_codebook"].values() if v.ndim == 2)
    d = next(v for v in ck["detail_codebook"].values() if v.ndim == 2)
    return CodebookLookup(g, d).to(device).eval()


# ==================================================================== v2

class CodeInjectorV2(nn.Module):
    """B4 的注入层 —— 去掉门控与 FiLM，改成两条**纯残差相加**的支路。

        h   = latents + M(z_g)          全局码：MLP 后直接逐通道相加，空间上广播
        out = h + cross_attn(h, Z_d)    细节码：cross-attn 残差

    # 为什么这么改（v1 的实测证据）

    v1 用 `gate ⊙ o` 做门控、`w_film` 做 FiLM，两者都**零初始化**。实测走完
    100000 步：

        gate 范数        0.0036 → 0.0061（1.7×）
        注入改变 latents  0.0056% → 0.0095%
        B1@40000 = B3@40000 = 35.33%    码本贡献为零

    根因是 LAMB 的更新式 `‖Δp‖ ≡ lr·‖p‖` —— 步长正比于参数自身范数，
    零初始化张量只有第一步是自由的（`‖p‖=0` 时 trust_ratio=1），之后被锁进
    每步至多长 lr 的倍增。100000 步允许 e¹⁰=22026 倍，实际只长了 1.7 倍，
    说明驱动力也只有天花板的 5%（打开门带来的损失下降太小，自我锁死）。

    **v2 直接删掉门控。** 实测 `o` 的 rms 是 latents 的 0.177 ——
    去掉 gate 之后注入幅度从 0.0064% 变成 17.7%，**2760 倍**，
    不需要改优化器、不需要参数分组，那个坑从源头消失。

    # 初始化

        w_g  normal(std=0.01)   全局码初始注入约 16%
        w_o  normal(std=0.02)   细节码初始注入约 17.7%（与 v1 相同，未改）

    两条都是**正常尺度**（不是零），所以 `lr·‖p‖` 从一开始就是正常步长。
    不加 warmup（保留现有训练框架），代价是训练第一步就有约三成扰动 ——
    这是刻意的取舍：v1 的教训是「注入太弱」，宁可偏大也不要再被锁死。

    # 关于 code_mask

    `code_mask=0` **只在 action == pose-adjust 时发生，实测占 112/10323 = 1.08%**。
    所以「mask=0 时与原版逐位等价」不是一条有实际保护作用的安全网
    （98.9% 的样本走 mask=1 的路径）。保留乘法结构是为了**单测锚点**：
    mask=0 时输出必须与输入 bit-exact，这条断言能抓住乘/加写反之类的实现 bug。

    # 细节码支路的已知退化（保留，如实记录）

    实测 9 个槽位**完全相同**的段占 82.8%，此时 cross-attn 的 K/V 只差一个
    与输入无关的 slot_embed，退化为「1 个码向量 + 9 个常量」。
    剩下 17.2% 的样本槽位确实不同，那部分 cross-attn 是有意义的。
    报告细节码相关指标时须注明这一点（`Adapter_Design §7 R1`）。
    """

    #: 全局码投影的初始化尺度。决定初始注入幅度（0.01 → 约 16%）。
    W_G_STD = 0.01
    #: 细节码输出投影的初始化尺度。与 v1 相同，实测注入约 17.7%。
    W_O_STD = 0.02

    #: 相位嵌入表的行数。实测 replay 里最大段下标是 19（stack_blocks 最长 20 段）。
    MAX_PHASE = 24

    def __init__(self, dim: int = 128, code_dim: int = 512, n_slots: int = 9,
                 hidden: int = 256, heads: int = 4, use_detail: bool = True,
                 use_phase: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.heads = heads
        self.use_detail = use_detail
        self.use_phase = use_phase

        # ---- (c) 相位嵌入：当前是计划里的第几段 ----
        # 为什么加这一路（实测依据）：
        #   段数 vs 成功率 r = −0.41(B0) / −0.46(B1)；
        #   短程(≤3段) 3 个任务 B0 均值 64.0%，长程(≥5段) 13 个任务只有 32.9%，差 31.1 pp。
        # PerAct 在一个关键帧只看到当前视觉 + 一句**恒定**的整任务指令，
        # 长程任务里同一视觉状态会出现在不同阶段（stack_blocks 摞第 2 块和第 3 块
        # 画面几乎一样），模型无从判断自己走到哪一步 —— 这正是 planner 知道、
        # 而模型拿不到的信息。
        # 它**任务无关**（只是「第几段」），所以能迁移到 UnSeen；
        # 参数量 24×128 = 3,072，可忽略。
        if use_phase:
            self.phase_embed = nn.Embedding(self.MAX_PHASE, dim)
            nn.init.normal_(self.phase_embed.weight, std=0.01)

        # ---- (a) 全局码 → 逐通道残差 ----
        self.ln_g = nn.LayerNorm(code_dim)
        self.mlp_g = nn.Sequential(
            nn.Linear(code_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.w_g = nn.Linear(hidden, dim)          # v1 这里是 w_film(hidden, 2*dim)

        # ---- (b) 细节码 → cross-attention 残差（结构同 v1，只是没有 gate）----
        if use_detail:
            assert dim % heads == 0, f"dim {dim} 必须能被 heads {heads} 整除"
            self.slot_embed = nn.Parameter(torch.randn(n_slots, code_dim) * 0.02)
            self.ln_h = nn.LayerNorm(dim)
            self.w_q = nn.Linear(dim, dim, bias=False)
            self.w_k = nn.Linear(code_dim, dim, bias=False)
            self.w_v = nn.Linear(code_dim, dim, bias=False)
            self.w_o = nn.Linear(dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.mlp_g:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # 🔴 w_g 用小随机而非零初始化 —— 零初始化正是 v1 被 LAMB 锁死的原因
        nn.init.normal_(self.w_g.weight, std=self.W_G_STD)
        nn.init.zeros_(self.w_g.bias)
        if self.use_detail:
            for w in (self.w_q, self.w_k, self.w_v):
                nn.init.xavier_uniform_(w.weight)
            nn.init.normal_(self.w_o.weight, std=self.W_O_STD)

    # ------------------------------------------------------------------
    def _cross_attn(self, h: torch.Tensor, Z_d: torch.Tensor) -> torch.Tensor:
        """h [B,C,X,Y,Z] × Z_d [B,S,code_dim] → O [B,C,X,Y,Z]（与 v1 相同）"""
        b, c = h.shape[0], h.shape[1]
        spatial = h.shape[2:]
        tok = h.reshape(b, c, -1).transpose(1, 2)             # [B,N,C]
        s = Z_d + self.slot_embed.unsqueeze(0)                # [B,S,code_dim]

        q = self.w_q(self.ln_h(tok))
        k = self.w_k(s)
        v = self.w_v(s)
        hd = c // self.heads
        q = q.view(b, -1, self.heads, hd).transpose(1, 2)
        k = k.view(b, -1, self.heads, hd).transpose(1, 2)
        v = v.view(b, -1, self.heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, -1, c)
        o = self.w_o(o)
        return o.transpose(1, 2).reshape(b, c, *spatial)

    def forward(self, latents: torch.Tensor, z_g: torch.Tensor,
                Z_d: torch.Tensor | None = None,
                code_mask: torch.Tensor | None = None,
                phase: torch.Tensor | None = None) -> torch.Tensor:
        b, c = latents.shape[0], latents.shape[1]
        n_spatial = latents.dim() - 2
        bc_shape = (b, c) + (1,) * n_spatial
        m_shape = (b, 1) + (1,) * n_spatial
        m = (latents.new_ones(b, 1) if code_mask is None
             else code_mask.reshape(b, 1).to(latents.dtype))

        # (a) 全局码：逐通道残差，空间上广播
        g = self.w_g(self.mlp_g(self.ln_g(z_g)))              # [B,C]
        # (c) 相位：当前是计划里的第几段。与码走同一条残差，同样受 code_mask 约束
        #     —— mask=0 时整条注入必须与原版逐位相同，这是单测的锚点。
        if self.use_phase and phase is not None:
            idx = phase.reshape(-1).long().clamp_(0, self.MAX_PHASE - 1)
            g = g + self.phase_embed(idx)                     # [B,C]
        h = latents + g.view(bc_shape) * m.view(m_shape)

        # (b) 细节码：cross-attn 残差，**没有门控**
        if not self.use_detail or Z_d is None:
            return h
        o = self._cross_attn(h, Z_d)
        return h + o * m.view(m_shape)

    # ------------------------------------------------------------------
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
