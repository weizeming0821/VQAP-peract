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
    def __init__(self, dim: int = 128, code_dim: int = 512, n_slots: int = 9,
                 hidden: int = 256, heads: int = 4, use_detail: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.heads = heads
        self.use_detail = use_detail

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
                code_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        latents   : [B, C, X, Y, Z]   PerAct 的 latents，C=128
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
