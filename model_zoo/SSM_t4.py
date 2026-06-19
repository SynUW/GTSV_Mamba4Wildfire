"""
基于learnable UMAP2的改动，主要是
删除了umap，
删除了burn history laplacian
删除了x_t，只保留了hidden_features
"""

import torch
import torch.nn as nn
from torch.nn import AvgPool2d, MaxPool2d
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from contextlib import contextmanager

import numpy as np
import joblib

from mamba_ssm import Mamba

import math
from torch.autograd import Variable

@contextmanager
def disable_tqdm():
    """临时禁用 tqdm 进度条的上下文管理器"""
    import tqdm
    old_disable = getattr(tqdm.tqdm, 'disable', None)
    tqdm.tqdm.disable = True
    try:
        yield
    finally:
        if old_disable is not None:
            tqdm.tqdm.disable = old_disable
        else:
            delattr(tqdm.tqdm, 'disable')


# =============================================================================
# 图/矩阵可视化工具（统一外部函数，供 UMAPAnchorLaplacian、construct_batch_laplacian 等调用）
# =============================================================================

def visualize_adjacency_heatmap(A, save_path=None, title=None, annotate=True, figsize=(12, 10)):
    """
    邻接矩阵热力图，可选在格子内标注数值。

    Args:
        A: [N, N] 邻接/相似度矩阵，torch.Tensor 或 np.ndarray
        save_path: 保存路径，None 则不保存
        title: 图标题
        annotate: 是否在每个格子内写数值（矩阵过大时建议 False）
        figsize: 图尺寸
    """
    if hasattr(A, 'detach'):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(A_np, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="A[i,j]")
    if title:
        ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    if annotate and A_np.shape[0] <= 32:
        n = A_np.shape[0]
        vmin, vmax = float(A_np.min()), float(A_np.max())
        thresh = (vmin + vmax) / 2.0
        for i in range(n):
            for j in range(n):
                val = float(A_np[i, j])
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_graph_network(A, save_path=None, title=None, max_nodes=64, figsize=(10, 10)):
    """
    从邻接矩阵绘制网络图（节点+连边）。节点数过大时只画热力图，不画网络图。

    Args:
        A: [N, N] 邻接矩阵，torch.Tensor 或 np.ndarray
        save_path: 保存路径，None 则不保存
        title: 图标题
        max_nodes: 超过此节点数则不画网络图（仅适合小图）
        figsize: 图尺寸
    """
    if hasattr(A, 'detach'):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)
    n = A_np.shape[0]
    if n > max_nodes:
        return
    try:
        import networkx as nx
    except ImportError:
        return
    G = nx.from_numpy_array(A_np)
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="lightblue", ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_channel_time_window(x_enc, channel_indices=(1, 2, 3, 4), window_len=5, batch_idx=0, save_path=None):
    """
    从 x_enc (B, T, C, H, W) 中取多个通道，在 T 维上随机取长度为 window_len 的时间窗口并可视化。
    每个通道一行，一行内为同一通道的多个时间切片。
    """
    B, T, C, H, W = x_enc.shape
    channel_indices = [c for c in channel_indices if c < C]
    if not channel_indices:
        channel_indices = [min(1, C - 1)]
    if T < window_len:
        window_len = T
    t_max = T - window_len
    t_start = torch.randint(0, t_max + 1, (1,)).item() if t_max >= 0 else 0

    n_rows = len(channel_indices)
    n_cols = window_len
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    im = None
    for row, ch_idx in enumerate(channel_indices):
        frames = x_enc[batch_idx, t_start : t_start + window_len, ch_idx, :, :].detach().cpu().numpy()
        for col in range(n_cols):
            ax = axes[row, col]
            im = ax.imshow(frames[col], cmap='viridis', aspect='equal')
            if row == 0:
                ax.set_title(f't={t_start + col}')
            if col == 0:
                ax.set_ylabel(f'Ch {ch_idx}', fontsize=10)
            ax.axis('off')
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.5)
    plt.suptitle(f'Channels {channel_indices} (rows), time window [{t_start}, {t_start + window_len})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return t_start


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from contextlib import contextmanager


@contextmanager
def disable_tqdm():
    import tqdm
    old_disable = getattr(tqdm.tqdm, 'disable', None)
    tqdm.tqdm.disable = True
    try:
        yield
    finally:
        if old_disable is not None:
            tqdm.tqdm.disable = old_disable
        else:
            delattr(tqdm.tqdm, 'disable')


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


   
# ==================== MODEL ARCHITECTURE ====================
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

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


class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = DyT(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.A = nn.Parameter(torch.zeros(d_state, d_state))

        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv-1,
            groups=self.expanded_dim,
            bias=False
        )

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n-1):
            A[i, i+1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        #L = H * W
        residual = x

        # Flatten spatial dimensions
        x_flat = x #x.reshape(B, L, C)

        # Normalize and project
        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Token selection
        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx+1, :]


        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)        # [B, L, D]
        center_norm = F.normalize(center, p=2, dim=-1)        # [B, 1, D]

        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)

        #im = torch.matmul(x_proj, center.transpose(-1, -2)).squeeze(-1)  # [B, L]
        sim = torch.softmax(sim, dim=-1)  # Normalized probabilities

        k = max(1, int(L * self.sparsity_ratio))

        _, topk_idx = torch.topk(sim, k=k, dim=-1)

        x_sparse = batched_index_select(x_proj, 1, topk_idx)  # [B, k, expanded_dim]

        # Conv processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        # SSM processing
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Combine with residual
        #x_processed = x_processed + batched_index_select(residual.reshape(B, L, C), 1, topk_idx)

        # Scatter back to original positions (dtype must match for scatter_; autocast can make x_processed fp16)
        output = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed.to(dtype=output.dtype))

        #return output.reshape(B, H, W, C) + x
        return output + residual

class SimplifiedMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        # SSM parameters
        self.A = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.B = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.proj_in(x)

        # Conv branch
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :x.shape[1]]
        x_conv = x_conv.transpose(1, 2)

        # SSM branch
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.sigmoid(self.A.unsqueeze(0)) * h + Bx
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x = torch.stack(outputs, dim=1)
        x = self.proj_out(x)
        return x  # + residual


class MambaBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mamba = SimplifiedMambaBlock(dim=dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.mamba(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        B, N, C = x.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        feature = self.to_out(out).permute(0, 2, 1)

        return feature

class SparseDeformableChannelMambaBlock(nn.Module):
    """
    Sparse self-attention block:
    - Input/output shape is consistent with SparseDeformableMambaBlock: x: [B, L, C]
    - Use standard multi-head self-attention, but only keep the top-k keys (by attention scores) for each query, implementing sparse attention
    - No longer rely on "center point similarity" to select tokens, but directly use the attention distribution of self-attention for sparsity
    """
    def __init__(self, dim, num_heads=13, sparsity_ratio=0.3, drop_rate=0.3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sparsity_ratio = sparsity_ratio

        self.norm = DyT(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        """
        x: [B, L, C]
        """
        B, L, C = x.shape
        residual = x

        x_norm = self.norm(x)  # [B, L, C]

        qkv = self.qkv(x_norm)  # [B, L, 3C]
        q, k, _ = qkv.chunk(3, dim=-1)

        # [B, L, C] -> [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # attn_scores_full: [B, num_heads, L, L]
        attn_scores_full = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # importance: [B, L]
        importance = attn_scores_full.mean(dim=-1).mean(dim=1)

        k_sparse = max(1, int(L * self.sparsity_ratio))
        # topk_idx: [B, k_sparse]
        _, topk_idx = torch.topk(importance, k_sparse, dim=-1)

        x_sparse = batched_index_select(x_norm, 1, topk_idx)

        x_mamba = self.mamba(x_sparse)

        out = torch.zeros_like(x)
        out.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_mamba.to(dtype=out.dtype))

        out = self.drop_path(out) + residual
        return out

class SpatialSpectralMambaBlock(nn.Module):
    def __init__(self, dim: int, patch_size: int, num_heads = 8):
        super().__init__()
        head_dim = dim // num_heads
        self.spatial_mamba = SparseDeformableMambaBlock(dim=dim, drop_rate=0.5)
        self.spectral_mamba = SparseDeformableChannelMambaBlock(dim=13*13, drop_rate=0.5)
        self.temporal_mamba = SparseDeformableChannelMambaBlock(dim=13*13, drop_rate=0.5)
        self.norm = nn.BatchNorm2d(dim)
        self.max_pool = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=dim,
                                    heads=num_heads, dim_head=head_dim, dropout=0.)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        
        x_temporal = x.reshape(B, C, H * W).reshape(B, 38, 10, H*W).reshape(B*38, 10, H*W)
        x_temporal = self.temporal_mamba(x_temporal)
        x_temporal = x_temporal.reshape(B, 38, 10, H, W).reshape(B, 380, H, W)
        
        
        x_channel = x_temporal.reshape(B, C, H * W).reshape(B, 38, 10, H*W).permute(0, 2, 1, 3).reshape(B*10, 38, H*W)
        x_spectral = self.spectral_mamba(x_channel)
        x_spectral = x_spectral.reshape(B, 10, 38, H, W).permute(0, 2, 1, 3, 4).reshape(B, 380, H, W)
        
        x_spatial = x_temporal.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_spatial = self.spatial_mamba(x_spatial)
        x_spatial = x_spatial.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        fusion = x_spatial + x_spectral + x_temporal
        
        x = self.conv(fusion) + x

        return x


def construct_batch_laplacian(burn_history, k=16*16, visualize=True, visualize_path_prefix="ssm_batch_laplacian"):
    """
    使用广义 Jaccard (Tanimoto) 系数构建稀疏 Laplacian。
    适用于非二值的稀疏连续变量 (如火灾强度、概率)。

    Args:
        burn_history: [B, T] 连续数值矩阵 (要求非负，如 ReLU 后的特征或概率)
        k:            int, Top-K 近邻数
        visualize:    是否保存邻接矩阵 A 的热力图与网络图（使用统一外部可视化函数）
        visualize_path_prefix: 保存路径前缀，热力图与网络图分别加 _heatmap.png / _network.png

    Returns:
        L: [B, B] 归一化拉普拉斯矩阵
    """
    B = burn_history.size(0)
    device = burn_history.device
    
    # 1. 确保数据非负 (Tanimoto 要求非负向量)
    # 如果已经是非负的(如概率)，这一步没有副作用
    bh = F.relu(burn_history.float())

    # ==========================================
    # 2. 计算广义 Jaccard (Tanimoto) 相似度
    # ==========================================
    # 公式: J(A, B) = (A . B) / (|A|^2 + |B|^2 - A . B)
    
    # 分子: Dot Product (点积)
    # [B, T] @ [T, B] -> [B, B]
    # intersection[i, j] = sum(bh[i] * bh[j])
    dot_prod = torch.mm(bh, bh.t())
    
    # 分母准备: Squared Norm (模的平方)
    # |A|^2 = sum(a_i^2)
    # [B, 1]
    norm_sq = torch.sum(bh ** 2, dim=1, keepdim=True)
    
    # 分母: Union (广义并集)
    # 利用广播: |A|^2 + |B|^2 - (A . B)
    union = norm_sq + norm_sq.t() - dot_prod
    
    # 计算相似度
    # 添加 epsilon 防止除零 (处理完全无火的历史，即 norm_sq=0 的情况)
    J = dot_prod / (union + 1e-6)
    

    J_no_self = J.clone()
    J_no_self.fill_diagonal_(float('-inf'))

    vals, inds = torch.topk(J_no_self, k=min(k, B-1), dim=1)
    A = torch.zeros_like(J)
    A.scatter_(1, inds, vals.clamp_min(0))
    A = (A + A.t()) / 2.0


    # 可选：图可视化（热力图 + 网络图，统一使用外部函数）
    # visualize_adjacency_heatmap(
    #     A, save_path=f"{visualize_path_prefix}_heatmap.png",
    #     title="Batch Laplacian (Jaccard) Adjacency A", annotate=(B <= 32)
    # )
        # visualize_graph_network(
        #     A, save_path=f"{visualize_path_prefix}_network.png",
        #     title="Batch Laplacian (Jaccard) Graph", max_nodes=64
        # )
    # pdb.set_trace()
    # 添加自环
    I = torch.eye(B, device=device)
    A_hat = A + I

    # 度矩阵归一化
    D_hat_diag = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.pow(D_hat_diag, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    D_mat = torch.diag(D_inv_sqrt)

    # L = D^-0.5 * A_hat * D^-0.5
    L = torch.mm(torch.mm(D_mat, A_hat), D_mat)

    return L


class MambaModel(nn.Module):
    def __init__(self, num_classes=1, patch_size=13, num_bands=380, hidden_dim=128):
        super().__init__()
        self.patch_size = patch_size

        # Stem layer for initial feature extraction
        
        self.stem = nn.Sequential(
            nn.Conv2d(num_bands, num_bands*2, kernel_size=3, stride=1, padding=1, groups=38),
            nn.BatchNorm2d(num_bands*2),
            nn.GELU(),
            nn.Conv2d(num_bands*2, hidden_dim, kernel_size=1, stride=1, padding=0, groups=38),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.temporal_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(dim=13*13, drop_rate=0.3) for _ in range(2)]
        )
        
        self.spectral_mamba = nn.Sequential(
            *[SparseDeformableChannelMambaBlock(dim=13*13, drop_rate=0.3) for _ in range(2)]
        )
        
        self.spatial_mamba = nn.Sequential(
            *[SparseDeformableMambaBlock(dim=hidden_dim, drop_rate=0.3) for _ in range(2)]
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.Linear(hidden_dim, num_classes)
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        
    def forward(self, x):
        # Input x: [B, C, H, W]
        # Single year format: C = 138 = 6 bands × 23 time steps
        # Use stem layer to map input channels to hidden_dim        
        x = self.stem(x)  # [B, hidden_dim, H, W] = 
        
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).reshape(B, 38, 10, H*W).reshape(B*38, 10, H*W)
        x = self.temporal_mamba(x)
        x = x.reshape(B, 38, 10, H, W).reshape(B, 380, H, W)
        
        x = x.reshape(B, C, H * W).reshape(B, 38, 10, H*W).permute(0, 2, 1, 3).reshape(B*10, 38, H*W)
        x = self.spectral_mamba(x) + x
        x = x.reshape(B, 10, 38, H, W).permute(0, 2, 1, 3, 4).reshape(B, 380, H, W)
        
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = self.spatial_mamba(x) + x
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        
        features = x.reshape(B, C, H, W)
        
        # features = self.spectral_blocks(x)

        x = self.head(features)
        return x, features


class MiniGCNLayer(nn.Module):
    """
    Single GCN layer, supports batch matrix multiplication.
    Formula: H' = ReLU(BN( L @ (H @ W) + b ))
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x, laplacian):
        # x: [B, in_features]
        # laplacian: [B, B] (Mini-batch Laplacian)
        
        # 1. Linear transform (Feature projection): H @ W
        
        support = torch.mm(x, self.weight) 
        
        # 2. Graph Convolution (Message passing): L @ support
        # Here L is (B, B), support is (B, out_features)
        out = torch.mm(laplacian, support)
        
        # 3. Bias and Norm
        out = out + self.bias
        out = self.bn(out)
        return out

class MiniGCNStream(nn.Module):
    """
    GCN Stream according to miniGCN paper logic.
    Input: Spectral Signatures (Pixels) + Mini-batch Laplacian
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.3):
        super().__init__()
        
        # Layer 1
        self.gc1 = MiniGCNLayer(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Layer 2
        self.gc2 = MiniGCNLayer(hidden_features, out_features)
        
    def forward(self, x, laplacian):
        # x: [B, num_bands]
        # laplacian: [B, B]
        x = self.gc1(x, laplacian)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.gc2(x, laplacian)
        # In the paper, the second layer output is usually directly the feature, no need to add ReLU/Softmax (before fusion)
        return x


class LULCOneHot(nn.Module):
    """
    LULC 最后一通道改为 one-hot，与 UMAP 预处理一致。
    类别 ID 1..n_lulc_classes → one-hot 下标 0..n_lulc_classes-1；nodata/非法为全 0。
    输入: x [B, T, C]，最后一列为 LULC (1..17)
    返回: y [B, T, (C-1) + n_lulc_classes]
    """
    def __init__(self, n_lulc_classes: int = 17, nodata_ids=(0, 255, -1), lulc_col: int = -1):
        super().__init__()
        self.n_lulc_classes = n_lulc_classes
        self.nodata_ids = set(int(x) for x in nodata_ids)
        self.lulc_col = int(lulc_col)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, C]，最后一列为 LULC 类别"""
        if x.ndim != 3:
            raise ValueError("输入张量必须是 [B, T, C]")
        x_cont = x[..., :self.lulc_col] if self.lulc_col != -1 else x[..., :-1]
        lulc_raw = x[..., self.lulc_col]
        if lulc_raw.dtype.is_floating_point:
            lulc_raw = torch.round(lulc_raw)
        lulc = lulc_raw.long().clamp(min=-1, max=255)
        valid = (lulc >= 1) & (lulc <= self.n_lulc_classes)
        for bad in self.nodata_ids:
            valid = valid & (lulc != bad)
        # one-hot 下标 = 类别 - 1
        idx = torch.where(valid, lulc - 1, 0)
        onehot = F.one_hot(idx, num_classes=self.n_lulc_classes).float()  # [B, T, n_lulc_classes]
        return torch.cat([x_cont.float(), onehot], dim=-1)

class PositionalEncoder(nn.Module):

    """ Positional Encoding """

    def __init__(self, d_model: int = 256, n_days: int = 366, device: str = 'cuda'):
        super(PositionalEncoder, self).__init__()

        """
        Parameters
        ----------
        d_model : int (default 256)
            number of dimensions for encoding
        n_days : int (default 366)
            number of days
        device : str (default cuda)
            device GPU or CPU
        """

        self.d_model = d_model
        self.n_days = n_days
        self.device = device

        pe = torch.zeros(n_days, d_model).to(device)

        # precompute the encoding
        for pos in range(n_days):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10 ** ((2 * (i+1)) / d_model)))

        # store in buffer for fast access
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input day of the year x_t [N]
        """
        # 使用输入 x 的设备，避免 model.to(cuda:1) 后仍 .to('cuda') 导致 cuda:0/cuda:1 混用
        x = Variable(self.pe[x[:, -1] - 1, :], requires_grad=False).to(x.device)
        return x

class Model(nn.Module):
    """Wrapper for train_all_h5: accepts configs and (x_enc, x_mark_enc, x_dec, x_mark_dec) interface."""

    def __init__(self, configs):
        super().__init__()
        patch_size = getattr(configs, 'patch_size', 13)
        self.enc_in = getattr(configs, 'enc_in', 38)
        seq_len = getattr(configs, 'seq_len', 365)  # 参数从外部传入
        pred_len = getattr(configs, 'pred_len', 1)
        d_model = getattr(configs, 'd_model', 128)
        n_lulc_classes = getattr(configs, 'n_lulc_classes', 17)
        lulc_emb_dim = getattr(configs, 'lulc_emb_dim', 4)
        
        # LULC one-hot：最后一通道为 LULC (1..17)，替换为 17 维 one-hot，与 UMAP 预处理一致
        self.lulc_onehot = LULCOneHot(
            n_lulc_classes=n_lulc_classes,
            nodata_ids=(0, 255, -1),
            lulc_col=-1,
        )
        # LULC embedding：将类别 ID 映射为 4 维嵌入（用于主模型），保留索引语义
        # 0 用作 padding / NoData，1..n_lulc_classes 对应有效 LULC 类别
        self.lulc_embedding = nn.Embedding(n_lulc_classes + 1, lulc_emb_dim, padding_idx=0)
        # one-hot 后通道数：37 + 17 = 54（用于图拉普拉斯）
        self.enc_in_after_onehot = (self.enc_in - 1) + n_lulc_classes
        # embedding 后通道数：37 + 4 = 41（用于主模型）
        self.enc_in_after_lulc = (self.enc_in - 1) + lulc_emb_dim
        num_bands = self.enc_in_after_lulc * seq_len

        self.pred_len = pred_len
        self.model = MambaModel(num_classes=pred_len, patch_size=patch_size, num_bands=380, hidden_dim=380)
        self.gcn_stream = MiniGCNStream(in_features=d_model, hidden_features=d_model//2, out_features=d_model, dropout=0.5)
        self.gcn_stream_manifold = MiniGCNStream(in_features=d_model, hidden_features=d_model//2, out_features=d_model, dropout=0.5)
        # 时间信息 embedding：doy 1..366，用于预测时刻的季节性
        time_emb_dim = getattr(configs, 'time_emb_dim', 32)
        self.doy_embedding = nn.Embedding(366 + 1, d_model//10, padding_idx=0)  # 0=padding, 1..366=doy

        # self.fc_out = nn.Linear(d_model * 3, pred_len)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, pred_len)
        )
        
        
        self.attn_net = nn.Sequential(
            nn.Linear(d_model, 16), 
            nn.GELU(), 
            nn.Linear(16, 1))
        
        # 不硬编码 device，由 model.to(device) 统一迁移，避免 cuda:0/cuda:1 混用
        self.PositionalEncoding = PositionalEncoder(d_model, 366, 'cpu')
        # PE_Weights should be used carefully i.e., with nonlinear activation/dropout otherwise they have no effect
        self.PE_Weights = torch.nn.Parameter(torch.zeros(d_model))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        # x_enc: (B, T, C, H, W)，最后一通道为 LULC (1..17)
        B, T, C, H, W = x_enc.shape
        B, T, N_mark = x_mark_enc.shape  
        
        if burn_history is not None:
            # burn_history 可能是 (B, 1, H, W, Tb) 或 (B, H, W, Tb)
            bh_shape = burn_history.shape
            if len(bh_shape) == 5:
                if bh_shape[1] == 1:
                    # (B, 1, H, W, Tb) -> (B, H, W, Tb)
                    burn_history = burn_history.squeeze(1)
                else:
                    raise ValueError(f"Unexpected burn_history shape: {bh_shape}, expected (B, 1, H, W, Tb) or (B, H, W, Tb)")
            elif len(bh_shape) != 4:
                raise ValueError(f"Unexpected burn_history shape: {bh_shape}, expected (B, 1, H, W, Tb) or (B, H, W, Tb)")

            # (B, H, W, Tb) -> (B, Tb)
            burn_history = burn_history.sum(dim=(1, 2))  # sum over H, W
            Tb = burn_history.shape[1]
            ws = 60
            if Tb < ws:
                # 如果 Tb < 30，直接 sum 成一个值
                burn_history = burn_history.sum(dim=1, keepdim=True)  # (B, 1)
            else:
                n, r = Tb // ws, Tb % ws
                if n > 0:
                    # 前 n*ws 个时间步：reshape 成 (B, n, ws) 然后 sum
                    full_part = burn_history[:, :n * ws].reshape(B, n, ws).sum(dim=2)  # (B, n)
                    parts = [full_part]
                else:
                    parts = []
                if r > 0:
                    # 剩余部分：sum -> (B, 1)
                    remainder_part = burn_history[:, n * ws:].sum(dim=1, keepdim=True)  # (B, 1)
                    parts.append(remainder_part)
                burn_history = torch.cat(parts, dim=1) if parts else burn_history.sum(dim=1, keepdim=True)

        x_enc = x_enc.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)

        hidden_features, spatial_features = self.model(x_enc)  # (B, d_model), (B, d_model, H, W)

        out = self.fc_out(hidden_features)
        return out


if __name__ == "__main__":
    model = MambaModel(num_classes=13, patch_size=13, num_bands=138, hidden_dim=64)
    x = torch.randn(128, 138, 13, 13)
    print(model(x).shape)

    # 可视化：x_enc (B, T, C, H, W) 取第 2、3、4、5 通道，每通道一行；T 上随机长度 5 的时间窗口
    x_enc = torch.randn(2, 365, 38, 13, 13)