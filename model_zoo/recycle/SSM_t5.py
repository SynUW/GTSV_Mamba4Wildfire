"""
SSM_t5.py

================================================================
说明 / Note (中文)
================================================================
本文件来自 forecasting_pin/model_zoo/SSM_stable_v2.py，为目前 json F
缓存上单变体 F1 最高的版本（test F1=0.7539，胜过原 SSM_stable 的
约 0.7418；与 MambaHSI 0.7765 仍差约 0.023）。

相对于原始 SSM_stable，本文件累积了 v1 + v2 两轮改动：

【v1：Norm / 激活手术（稳定性改造）】
  1. RMSNorm 替换原先的 DyT (Dynamic Tanh)，去除可能的训练不稳定来源。
  2. 所有 BatchNorm2d 改为 GroupNorm（通过 _safe_group_norm 自适应分组，
     当 num_channels 不能被 requested_groups 整除时回退到能整除的最大约数），
     消除 BN 在 H5 缓存小 batch 上的统计漂移。
  3. GELU 激活全部替换为 SiLU，对齐 MambaHSI 的激活栈。

【v2：MambaHSI 风格 stem（跨波段信息保留）】
  原 SSM_stable 的 stem 使用 groups=55 的 3x3 + 1x1 分组卷积，等价于在
  入口处把 550 个输入通道切成 55 组、组间互不交流，造成跨波段信息在第一层
  就被截断。v2 将 stem 替换为：

      Conv2d(num_bands=550, hidden_dim=550, kernel_size=1)
        -> _safe_group_norm(hidden_dim, 55)
        -> SiLU()

  即一个 1x1 全混合卷积 + GroupNorm + SiLU，让全部 550 通道从第一层就
  自由交互。后续 conv（spatial 块之后的 refine）也对齐为
  1x1 + GroupNorm(55) + SiLU。

【未改动部分（与 SSM_stable_v1 一致）】
  - SparseDeformableMambaBlock（spatial：top-k 与中心 token 余弦相似度，
    sparsity_ratio=0.3）。
  - SparseDeformableChannelMambaBlock（temporal/spectral：多头注意力
    打分 + top-k，sparsity_ratio=0.3）。
  - 三路 mamba 顺序：temporal(2 层) -> spectral(2 层, 残差) -> spatial(2 层, 残差)。
  - DropPath=0.3、Dropout=0.5、d_state=16、d_conv=4、expand=2。
  - DOY embedding 接口保留但当前 forward 未使用（与 v2 行为对齐）。
  - Model.forward 输入仍为 (B, T=10, C=55, H=13, W=13)，permute+reshape 后
    送入 MambaModel；输出 shape 为 (B, pred_len)。

【已经验证的假设（可作为后续 t5 衍生工作的起点）】
  - DyT/BN/GELU 的混合不如 RMSNorm/GroupNorm/SiLU 稳定（v1 已涨点）。
  - groups=55 stem 在入口处切断跨波段信息是显著瓶颈（v2 在 v1 基础上再涨点）。

【尚未在本文件中尝试（留给 t6+ 系列）】
  - token-grouped 无损 spectral（SpeMamba 风格）。
  - 稠密时间维 mamba（不在 10 步上做 sparse）。
  - 双向 sparse spatial mamba（forward + reverse 融合）。
  - 可学习的 deformable offsets（per-query (dx,dy) MLP + grid_sample）。
  - 去掉或调小 DropPath 与 Dropout（测试是否过度正则）。
  - 2D sin-cos 位置编码注入 spatial 块。
================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


def _safe_group_norm(num_channels: int, requested_groups: int) -> nn.GroupNorm:
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    g = max(1, min(int(requested_groups), num_channels))
    if num_channels % g != 0:
        for c in range(g, 0, -1):
            if num_channels % c == 0:
                g = c
                break
    return nn.GroupNorm(g, num_channels)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))


def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.mamba = Mamba(d_model=self.expanded_dim, d_state=d_state, d_conv=d_conv, expand=2)

    def forward(self, x):
        B, L, C = x.shape
        residual = x
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)
        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx + 1, :]
        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)
        center_norm = F.normalize(center, p=2, dim=-1)
        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)
        sim = torch.softmax(sim, dim=-1)
        k = max(1, int(L * self.sparsity_ratio))
        _, topk_idx = torch.topk(sim, k=k, dim=-1)
        x_sparse = batched_index_select(x_proj, 1, topk_idx)
        x_processed = self.mamba(x_sparse)
        x_processed = self.proj_out(x_processed)
        output = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)
        return self.drop_path(output) + residual


class SparseDeformableChannelMambaBlock(nn.Module):
    def __init__(self, dim, num_heads=13, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_ratio = sparsity_ratio

        self.norm = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        B, L, C = x.shape
        residual = x
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn_scores_full = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        importance = attn_scores_full.mean(dim=-1).mean(dim=1)
        k_sparse = max(1, int(L * self.sparsity_ratio))
        _, topk_idx = torch.topk(importance, k_sparse, dim=-1)
        x_sparse = batched_index_select(x_norm, 1, topk_idx)
        x_mamba = self.mamba(x_sparse)
        out = torch.zeros_like(x)
        out.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_mamba)
        return self.drop_path(out) + residual


class MambaModel(nn.Module):
    def __init__(self, num_classes=1, patch_size=13, num_bands=550, hidden_dim=550):
        super().__init__()
        self.patch_size = patch_size

        # MambaHSI-style stem: simple 1x1 conv + GroupNorm + SiLU. Full cross-channel mixing.
        self.stem = nn.Sequential(
            nn.Conv2d(num_bands, hidden_dim, kernel_size=1, stride=1, padding=0),
            _safe_group_norm(hidden_dim, 55),
            nn.SiLU(),
        )

        self.temporal_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(dim=13 * 13, drop_rate=0.3) for _ in range(2)]
        )
        self.spectral_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(dim=13 * 13, drop_rate=0.3) for _ in range(2)]
        )
        self.spatial_mamba = nn.Sequential(
            *[SparseDeformableMambaBlock(dim=hidden_dim, drop_rate=0.3) for _ in range(2)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            _safe_group_norm(hidden_dim, 55),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape

        x = x.reshape(B, C, H * W).reshape(B, 55, 10, H * W).reshape(B * 55, 10, H * W)
        x = self.temporal_mamba(x)
        x = x.reshape(B, 55, 10, H, W).reshape(B, 550, H, W)

        x = x.reshape(B, C, H * W).reshape(B, 55, 10, H * W).permute(0, 2, 1, 3).reshape(B * 10, 55, H * W)
        x = self.spectral_mamba(x) + x
        x = x.reshape(B, 10, 55, H, W).permute(0, 2, 1, 3, 4).reshape(B, 550, H, W)

        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.spatial_mamba(x) + x
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)

        features = x.reshape(B, C, H, W)
        x = self.head(features)
        return x, features


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        patch_size = getattr(configs, 'patch_size', 13)
        self.enc_in = getattr(configs, 'enc_in', 55)
        pred_len = getattr(configs, 'pred_len', 1)
        d_model = 550
        self.pred_len = pred_len
        self.model = MambaModel(num_classes=pred_len, patch_size=patch_size, num_bands=550, hidden_dim=d_model)
        self.doy_embedding = nn.Embedding(366 + 1, d_model, padding_idx=0)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            _safe_group_norm(d_model, 55),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, pred_len),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        B, T, C, H, W = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1, 3, 4).reshape(B, C * T, H, W)
        hidden_features, _ = self.model(x_enc)
        out = self.fc_out(hidden_features)
        return out
