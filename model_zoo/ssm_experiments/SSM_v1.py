"""
SSM_v0.py
1. backbone
    1.1 stem convolution with 38 groups with 38 variables, for each group, there is a time series data with 10 time steps.
    1.2 mamba blocks with separate and sequential temporal, spectral, and spatial processing.
2. graph layer
    2.1 laplacian matrix construction based on cossine similarity 
    2.2 graph layer extraction based on laplacian matrix and mamba hidden features
3. head
    3.1 linear output layer
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

import pdb

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

class UMAPAnchorLaplacian(nn.Module):
    def __init__(self, umap_model_path, library_path, sigma=1.0, device='cuda'):
        """
        Args:
            umap_model_path: 预训练好的 UMAP 模型 (.joblib) 路径
            library_path: 光谱库 (.npz) 路径，需包含 'library_embeddings'
            sigma: RBF 核宽度 (可以设为 learnable parameter)
            device: 运行设备
        """
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True)
        self.device = device

        # 1. 加载 UMAP 模型 (CPU)
        print(f"[Graph] Loading UMAP model from {umap_model_path}...")
        loaded_obj = joblib.load(umap_model_path)
        self.umap_model = loaded_obj.umap_model if hasattr(loaded_obj, 'umap_model') else loaded_obj
        
        # 2. 加载光谱库 (Anchors)
        print(f"[Graph] Loading Anchor Library from {library_path}...")
        lib_data = np.load(library_path)
        
        # 支持两种格式：
        # 1. 旧格式：library_embeddings (K, T, c)
        # 2. 新格式：pos_library_embeddings 和 neg_library_embeddings (分别对应正负样本库)
        if 'library_embeddings' in lib_data:
            # 旧格式：直接使用 library_embeddings
            anchors_np = lib_data['library_embeddings']  # (K, T, c)
        elif 'pos_library_embeddings' in lib_data and 'neg_library_embeddings' in lib_data:
            # 新格式：合并正负样本库
            pos_emb = lib_data['pos_library_embeddings']  # (K_pos, T, c)
            neg_emb = lib_data['neg_library_embeddings']  # (K_neg, T, c)
            anchors_np = np.concatenate([pos_emb, neg_emb], axis=0)  # (K_pos + K_neg, T, c)
        else:
            available_keys = list(lib_data.keys())
            raise KeyError(
                f"Expected 'library_embeddings' or ('pos_library_embeddings', 'neg_library_embeddings') "
                f"in {library_path}, but found keys: {available_keys}"
            )
        
        # 假设 library_embeddings 形状为 [K, T, c]，需要展平为 [K, Tc] 以匹配输入
        K, T_lib, c = anchors_np.shape
        anchors_flat = anchors_np.reshape(K, -1)
        
        # 保存库的时序长度和通道数，用于后续处理
        self.T_lib = T_lib  # 库的时序长度（例如 10）
        self.c = c  # UMAP 输出维度（例如 5）
        self.C = None  # 原始通道数，将在第一次 forward 时推断
        
        # 将 Anchors 注册为 Buffer (固定参数，不更新梯度，但随模型移动)
        self.register_buffer('anchors', torch.from_numpy(anchors_flat).float())
        print(f"[Graph] Ready. Anchors shape: {self.anchors.shape} (K={K}, T={T_lib}, c={c})")

    def forward(self, x):
        """
        Args:
            x: [B, C*T] 输入特征 (Flattened Time Series)
               其中 C 是通道数（例如 38），T 是时序长度（例如 365）
        Returns:
            L: [B, B] Normalized Laplacian Matrix
        """
        B, CT = x.shape
        
        # ==========================================
        # Step 1: 推断通道数和时序长度
        # ==========================================
        # UMAP 模型是在 (N, C) 上训练的，其中 C 是通道数
        # 我们需要从 (B, C*T) 中提取最后 T_lib 天
        if self.C is None:
            # 尝试推断：假设 T 是 365（常见值），C = CT / 365
            # 或者从 anchors 的形状推断：anchors 是 (K, T_lib*c)
            # 但这里我们需要知道 C，所以先尝试常见值
            possible_T = [365, 366, 10, 20, 30]  # 常见的时序长度
            for T_guess in possible_T:
                if CT % T_guess == 0:
                    self.C = CT // T_guess
                    break
            if self.C is None:
                # 如果无法推断，假设 C=38（从代码中看到）
                self.C = 38
        
        T_input = CT // self.C
        
        # ==========================================
        # Step 2: 提取最后 T_lib 天并 reshape
        # ==========================================
        # Reshape: (B, C*T) -> (B, T, C)
        x_reshaped = x.view(B, T_input, self.C)
        
        # 提取最后 T_lib 天: (B, T, C) -> (B, T_lib, C)
        if T_input >= self.T_lib:
            x_last = x_reshaped[:, -self.T_lib:, :]  # (B, T_lib, C)
        else:
            # 如果输入时序长度小于库的时序长度，进行填充
            padding = torch.zeros(B, self.T_lib - T_input, self.C, device=x.device)
            x_last = torch.cat([padding, x_reshaped], dim=1)  # (B, T_lib, C)
        
        # 展平为 (B*T_lib, C) 以便输入 UMAP
        x_flat = x_last.reshape(-1, self.C)  # (B*T_lib, C)
        
        # ==========================================
        # Step 3: Manifold Embedding (UMAP)
        # ==========================================
        # 注意：UMAP transform 是 CPU 操作且不可导
        # 梯度会在这一步截断，无法传回 x 之前的层
        with torch.no_grad():
            x_cpu = x_flat.detach().cpu().numpy()
            # 临时禁用 UMAP 的进度条输出
            old_verbose = getattr(self.umap_model, 'verbose', None)
            old_tqdm_kwds = getattr(self.umap_model, 'tqdm_kwds', None)
            self.umap_model.verbose = False
            self.umap_model.tqdm_kwds = None
            # 投影到低维空间: (B*T_lib, C) -> (B*T_lib, c)
            with disable_tqdm():
                z_np = self.umap_model.transform(x_cpu)
            # 恢复原始设置
            if old_verbose is not None:
                self.umap_model.verbose = old_verbose
            if old_tqdm_kwds is not None:
                self.umap_model.tqdm_kwds = old_tqdm_kwds
            z_flat = torch.from_numpy(z_np).float().to(self.device)
        
        # Reshape 回时序结构: (B*T_lib, c) -> (B, T_lib, c)
        z = z_flat.view(B, self.T_lib, self.c)
        
        # 展平为 (B, T_lib*c) 以匹配 anchors 的形状 (K, T_lib*c)
        z = z.reshape(B, -1)  # (B, T_lib*c)
            
        # ==========================================
        # Step 4: 计算与 Anchor 的相似度矩阵 S
        # ==========================================
        # z: [B, dim], anchors: [K, dim]
        # dist: [B, K]
        dist = torch.cdist(z, self.anchors, p=2)
        
        # S_ik = exp(-dist^2 / sigma^2)
        # 物理意义：样本 i 与第 k 个 Anchor 的相似概率
        S = torch.exp(-(dist ** 2) / (self.sigma ** 2 + 1e-6))
        
        # ==========================================
        # Step 5: 构建邻接矩阵 A (Low-Rank Reconstruction)
        # ==========================================
        # A = S * S^T
        # 维度: [B, K] @ [K, B] -> [B, B]
        # 物理意义：如果样本 i 和 j 都像 Anchor k，则它们建立连接
        A = torch.mm(S, S.t())
        
        # ==========================================
        # Step 6: 计算归一化拉普拉斯矩阵 L
        # ==========================================
        # 添加自环
        I = torch.eye(B, device=x.device)
        A_hat = A + I
        
        # 计算度矩阵 D
        D_hat_diag = torch.sum(A_hat, dim=1) # [B]
        
        # 计算 D^-0.5
        D_inv_sqrt = torch.pow(D_hat_diag, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag(D_inv_sqrt)
        
        # L = D^-0.5 * A_hat * D^-0.5
        L = torch.mm(torch.mm(D_mat, A_hat), D_mat)
        
        return L

class UMAPSpectralGraphLayer(nn.Module):
    def __init__(self, 
                 builder_path, 
                 library_path, 
                 seq_len, 
                 n_channels, 
                 sigma=1.0, 
                 device='cuda'):
        """
        自动加载 UMAP 模型和光谱库，并构建空间拉普拉斯矩阵的层。

        参数:
        - builder_path: .joblib 文件路径 (包含训练好的 SpectralLibraryBuilder)
        - library_path: .npz 文件路径 (包含 library_embeddings)
        - seq_len (T): 时间序列长度
        - n_channels (C): 原始波段数
        - sigma: RBF 核的宽度参数
        - device: 运行设备的字符串 ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=True) # 可学习的 sigma
        self.device = device
        
        # print(f"[GraphLayer] Loading UMAP model from {builder_path} ...")
        # 1. 加载 UMAP 模型 (CPU 对象)
        # 注意: joblib 加载的对象通常包含 umap_model 属性
        loaded_obj = joblib.load(builder_path)
        if hasattr(loaded_obj, 'umap_model'):
            self.umap_model = loaded_obj.umap_model
        else:
            self.umap_model = loaded_obj # 假设直接保存了 umap 对象
            
        # 2. 加载光谱库 Anchors
        # print(f"[GraphLayer] Loading Spectral Library from {library_path} ...")
        lib_data = np.load(library_path)
        
        # 支持两种格式：
        # 1. 旧格式：library_embeddings (K, T, c)
        # 2. 新格式：pos_library_embeddings 和 neg_library_embeddings (分别对应正负样本库)
        if 'library_embeddings' in lib_data:
            # 旧格式：直接使用 library_embeddings
            anchors_np = lib_data['library_embeddings']  # (K, T, c)
        elif 'pos_library_embeddings' in lib_data and 'neg_library_embeddings' in lib_data:
            # 新格式：合并正负样本库
            pos_emb = lib_data['pos_library_embeddings']  # (K_pos, T, c)
            neg_emb = lib_data['neg_library_embeddings']  # (K_neg, T, c)
            anchors_np = np.concatenate([pos_emb, neg_emb], axis=0)  # (K_pos + K_neg, T, c)
        else:
            available_keys = list(lib_data.keys())
            raise KeyError(
                f"Expected 'library_embeddings' or ('pos_library_embeddings', 'neg_library_embeddings') "
                f"in {library_path}, but found keys: {available_keys}"
            )
        
        # 假设 library_embeddings 形状为 (K, T, c)
        # 我们需要将其展平为 (K, T*c) 以计算时序相似度
        self.K, _, self.n_components = anchors_np.shape
        
        anchors_flat = anchors_np.reshape(self.K, -1)  # (K, T*c)
        
        # 将 Anchors 注册为 Buffer (不参与梯度更新，但随模型保存/移动)
        self.register_buffer('anchors', torch.from_numpy(anchors_flat).float())
        
        print(f"[GraphLayer] Ready. Anchors shape: {self.anchors.shape} (K={self.K}, Dim={self.seq_len}*{self.n_components})")

    def forward(self, x):
        """
        Args:
            x: 输入特征张量 [B, C_total, H, W]
               假设 C_total = seq_len * n_channels
               这是 Minibatch 内的空间数据
        
        Returns:
            D: 度矩阵 [B, HW, HW]
            L: 归一化拉普拉斯矩阵 [B, HW, HW]
        """
        B, C_total, H, W = x.shape
        N_pixels = B * H * W
        
        # ==========================================
        # Step 1: 维度重塑与准备
        # ==========================================
        # 目标: 将 x 转换为 (N_pixels * T, C) 以便输入 UMAP
        # 假设 x 的通道排列是 (T, C) 混合的，我们需要拆开
        
        # View: [B, T, C, H, W] (假设输入是按时间堆叠的，如果按通道堆叠请调整为 B, C, T, H, W)
        # Permute -> [B, H, W, T, C] -> Flatten -> [B*H*W*T, C]
        x_reshaped = x.view(B, self.seq_len, self.n_channels, H, W)
        x_pixels_flat = x_reshaped.permute(0, 3, 4, 1, 2).reshape(-1, self.n_channels)
        
        # ==========================================
        # Step 2: UMAP 投影 (CPU 操作)
        # ========================================== 
        # 警告: UMAP transform 不可导且运行在 CPU 上
        # 在训练循环中这可能会成为瓶颈
        
        # 1. Move to CPU Numpy
        x_np = x_pixels_flat.detach().cpu().numpy()
        
        # 2. UMAP Transform (N*T, C) -> (N*T, c)
        # 这一步使用了你预训练好的投影参数
        # 临时禁用 UMAP 的进度条输出
        old_verbose = getattr(self.umap_model, 'verbose', None)
        old_tqdm_kwds = getattr(self.umap_model, 'tqdm_kwds', None)
        self.umap_model.verbose = False
        self.umap_model.tqdm_kwds = None
        with disable_tqdm():
            x_emb_np = self.umap_model.transform(x_np)
        # 恢复原始设置
        if old_verbose is not None:
            self.umap_model.verbose = old_verbose
        if old_tqdm_kwds is not None:
            self.umap_model.tqdm_kwds = old_tqdm_kwds
        
        # 3. Move back to GPU Tensor
        x_emb = torch.from_numpy(x_emb_np).float().to(x.device)
        
        # ==========================================
        # Step 3: 时序特征重组
        # ==========================================
        # (N_pixels * T, c) -> (N_pixels, T * c)
        # 每个空间位置现在由一条低维轨迹表示
        x_emb_spatial = x_emb.view(B, H, W, self.seq_len, self.n_components)
        
        # Flatten spatial dims for graph construction: [B, HW, T*c]
        features = x_emb_spatial.reshape(B, H*W, -1) 
        
        # ==========================================
        # Step 4: 基于 Anchor 的构图 (Scheme 3)
        # ==========================================
        # features: [B, HW, feat_dim]
        # anchors:  [K, feat_dim]
        
        # 1. 计算相似度 S (Similarity to Anchors)
        # dist: [B, HW, K]
        dist_to_anchors = torch.cdist(features, self.anchors, p=2)
        
        # Gaussian Kernel
        S = torch.exp(-(dist_to_anchors ** 2) / (self.sigma ** 2 + 1e-6))
        
        # 2. 构建邻接矩阵 A = S * S^T
        # [B, HW, K] @ [B, K, HW] -> [B, HW, HW]
        A = torch.bmm(S, S.transpose(1, 2))
        
        # 3. 添加自环
        I = torch.eye(H*W, device=x.device).unsqueeze(0)
        A_hat = A + I
        
        # 4. 计算度矩阵 D
        D_hat_diag = torch.sum(A_hat, dim=2) # [B, HW]
        
        # 5. 计算归一化拉普拉斯矩阵 L
        # L = D^-0.5 * A_hat * D^-0.5
        D_inv_sqrt = torch.pow(D_hat_diag, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag_embed(D_inv_sqrt)
        
        L = torch.bmm(torch.bmm(D_mat, A_hat), D_mat)
        
        # 为了后续 GCN 计算方便，通常也返回 D_hat_diag 或 D_matrix 的逆
        # 这里按要求返回 D (原始度矩阵形式，非逆) 和 L
        D_raw = torch.diag_embed(D_hat_diag)
        
        return D_raw, L

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

        # Scatter back to original positions
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

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
        out.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_mamba)

        out = self.drop_path(out) + residual
        return out

class SpatialSpectralMambaBlock(nn.Module):
    def __init__(self, dim: int, patch_size: int, num_heads = 8):
        super().__init__()
        head_dim = dim // num_heads
        self.spatial_mamba = SparseDeformableMambaBlock(dim=dim, drop_rate=0.3)
        self.spectral_mamba = SparseDeformableChannelMambaBlock(dim=13*13, drop_rate=0.3)
        self.temporal_mamba = SparseDeformableChannelMambaBlock(dim=13*13, drop_rate=0.3)
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


def construct_batch_laplacian_jaccard_lag(burn_history, k=16*16, buffer_size=1):
    """
    基于 VUS 论文思想构建的鲁棒火灾拉普拉斯矩阵。
    核心改进：引入 Buffer Region (通过 MaxPool) 以容忍火灾传播的时间滞后 (Lag)。

    Args:
        burn_history: [B, T] 原始火灾历史张量。
                      可以是二值 (0/1) 或连续值 (FRP/概率)。
        k:            int, Top-K 近邻数，控制图的稀疏度。
        buffer_size:  int, 缓冲窗口大小 (论文中的 \ell)。
                      建议为奇数 (如 3, 5)。
                      值越大，对“时间滞后”的容忍度越高，能连接跨度更大的火灾传播；
                      值越小，对时间对齐要求越严格。

    Returns:
        L: [B, B] 归一化拉普拉斯矩阵 (Normalized Laplacian)
    """
    B, T = burn_history.shape
    device = burn_history.device
    
    # 1. 数据准备与预处理
    # 确保数据为浮点型且非负
    bh = burn_history.float()
    if bh.min() < 0:
        bh = F.relu(bh)

    # ============================================================
    # Step 1: 引入 Buffer Region (Lag Tolerance)
    # ============================================================
    # 论文启发：异常检测评估中引入 Buffer 来解决 lag 问题。
    # 这里我们用 1D Max Pooling 来模拟这个 Buffer。
    # 物理意义：只要在窗口 [t-w, t+w] 内着火，就认为 t 时刻“有火灾风险”。
    # 这使得 t 时刻着火的节点可以与 t+1 或 t+2 时刻着火的节点建立连接。
    
    if buffer_size > 1:
        # 输入需要是 [N, C, L] -> [B, 1, T]
        bh_input = bh.unsqueeze(1)
        
        # Padding 保证时间长度不变 (Same Padding)
        padding = buffer_size // 2
        
        # Max Pooling: 相当于形态学操作中的 "Dilation" (膨胀)
        # 如果数据是概率值，也可以用 AvgPool，但 MaxPool 对稀疏事件捕捉更敏锐
        bh_buffered = F.max_pool1d(
            bh_input, 
            kernel_size=buffer_size, 
            stride=1, 
            padding=padding
        ).squeeze(1) # [B, T]
    else:
        bh_buffered = bh

    # ============================================================
    # Step 2: 计算广义 Jaccard (Tanimoto) 相似度
    # ============================================================
    # 针对稀疏数据的最佳度量，忽略 0-0 背景。
    # J(A, B) = (A . B) / (|A|^2 + |B|^2 - A . B)
    
    # 分子: Intersection (Buffered)
    # [B, T] @ [T, B] -> [B, B]
    # 经过 Buffer 后，原本时间错位的火灾现在会有交集
    intersection = torch.mm(bh_buffered, bh_buffered.t())
    
    # 分母: Union
    # |A|^2
    norm_sq = torch.sum(bh_buffered ** 2, dim=1, keepdim=True) # [B, 1]
    
    # Union = |A|^2 + |B|^2 - Intersection
    union = norm_sq + norm_sq.t() - intersection
    
    # 计算相似度矩阵 J
    # epsilon 防止除零 (处理完全无火的样本)
    J = intersection / (union + 1e-6)

    # ============================================================
    # Step 3: 图稀疏化 (Top-K Sparsification)
    # ============================================================
    # 只保留最相似的 k 个邻居，去除噪音连接
    
    # 确保 k 不超过 batch size
    k = min(k, B)
    
    # topk_vals: [B, k], topk_inds: [B, k]
    vals, inds = torch.topk(J, k=k, dim=1)
    
    # 构建稀疏邻接矩阵 A
    A = torch.zeros_like(J)
    A.scatter_(1, inds, vals)

    # ============================================================
    # Step 4: 拉普拉斯矩阵构建
    # ============================================================
    
    # 1. 对称化 (Symmetrization) -> 无向图
    A = (A + A.t()) / 2.0
    
    # 2. 添加自环 (Self-loops) -> A_hat
    I = torch.eye(B, device=device)
    A_hat = A + I
    
    # 3. 计算度矩阵 D_hat
    D_hat_diag = torch.sum(A_hat, dim=1)
    
    # 4. 归一化: L = D^-0.5 * A_hat * D^-0.5
    D_inv_sqrt = torch.pow(D_hat_diag, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0. # 处理孤立点
    D_mat = torch.diag(D_inv_sqrt)
    
    L = torch.mm(torch.mm(D_mat, A_hat), D_mat)
    
    return L

def construct_batch_laplacian(burn_history, k=16*16):
    """
    使用广义 Jaccard (Tanimoto) 系数构建稀疏 Laplacian。
    适用于非二值的稀疏连续变量 (如火灾强度、概率)。

    Args:
        burn_history: [B, T] 连续数值矩阵 (要求非负，如 ReLU 后的特征或概率)
        k:            int, Top-K 近邻数

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
    
    # ==========================================
    # 3. 图稀疏化 (Top-K Sparsification)
    # ==========================================
    # 只保留最相似的 k 个邻居
    k = min(k, B)
    
    vals, inds = torch.topk(J, k=k, dim=1)
    
    # 构建稀疏邻接矩阵
    A = torch.zeros_like(J)
    A.scatter_(1, inds, vals)
    
    # ==========================================
    # 4. 对称化与拉普拉斯构建
    # ==========================================
    # 对称化
    A = (A + A.t()) / 2.0
    
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

def construct_batch_laplacian_cosine(features, k=16*16, epsilon=1e-6):
    """
    基于余弦相似度构建 k-NN 归一化拉普拉斯矩阵。
    适用于高维光谱数据，关注波形形状而非数值大小。

    Args:
        features: [B, C] 输入特征张量。
                  建议先经过 Stem 层提取特征，或者直接使用 Flatten 后的高维输入。
        k: int, 每个节点保留的最近邻居数量 (k-NN)。
           控制图的稀疏度。k 越大图越稠密，信息传播越广但噪声也越大。
        epsilon: float, 防止除零的微小值。

    Returns:
        L: [B, B] Normalized Laplacian Matrix (D^-0.5 * A_hat * D^-0.5)
    """
    # 1. 归一化特征 (L2 Norm)
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # 先将特征归一化到单位球面上，之后的点积就是余弦相似度
    features_norm = F.normalize(features, p=2, dim=1)
    
    # 2. 计算相似度矩阵 S (全连接)
    # [B, C] @ [C, B] -> [B, B]
    # S_ij 的范围是 [-1, 1]
    S = torch.mm(features_norm, features_norm.t())
    
    # 3. 处理负相关 (可选，但推荐)
    # 在构建图神经网络时，通常只考虑正相关作为“连接”。
    # 负值意味着方向相反，作为邻居可能引入歧义。
    S = F.relu(S)  # 将 [-1, 0) 截断为 0
    
    # 4. 构建稀疏邻接矩阵 A (Top-K Sparsification)
    # 这一步至关重要：
    # 如果不进行 Top-K 过滤，A 是一个全连接稠密矩阵，
    # 会导致 GCN 在一层之后就发生严重的 Over-smoothing (所有节点特征趋同)。
    
    B = features.size(0)
    # 确保 k 不超过 Batch Size
    k = min(k, B)
    
    # 选取每行最大的 k 个值 (即最相似的 k 个邻居)
    # topk_vals: [B, k], topk_inds: [B, k]
    topk_vals, topk_inds = torch.topk(S, k=k, dim=1)
    
    # 创建稀疏的邻接矩阵 A
    A = torch.zeros_like(S)
    # scatter_ 的作用：将 topk_vals 填充到 A 中 topk_inds 指定的位置
    A.scatter_(1, topk_inds, topk_vals)
    
    # 5. 对称化 (Symmetrization)
    # k-NN 图是有向的 (A 是 B 的邻居，但 B 不一定是 A 的邻居)。
    # 拉普拉斯矩阵通常要求无向图 (对称矩阵)。
    # 策略：如果 i 连接 j 或者 j 连接 i，则保留连接 (OR 逻辑)
    # 或者取平均值: A_sym = (A + A^T) / 2
    A = (A + A.t()) / 2.0
    
    # 6. 添加自环 (Self-loops)
    # A_hat = A + I
    # 保证节点自身的信息在卷积过程中被保留
    I = torch.eye(B, device=features.device)
    A_hat = A + I
    
    # 7. 计算度矩阵 D_hat
    # D_ii = sum_j(A_hat_ij)
    D_hat_diag = torch.sum(A_hat, dim=1)
    
    # 8. 计算归一化拉普拉斯矩阵
    # L = D^-0.5 * A_hat * D^-0.5
    
    # 计算 D^-0.5
    D_inv_sqrt = torch.pow(D_hat_diag, -0.5)
    # 处理孤立点或度极小的情况 (虽加了自环一般不会除零，但为了稳健)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    
    # 构建对角矩阵
    D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
    
    # 矩阵乘法 L = D^-0.5 @ A_hat @ D^-0.5
    L = torch.mm(torch.mm(D_mat_inv_sqrt, A_hat), D_mat_inv_sqrt)
    
    return L

def construct_batch_laplacian_rbf(features, sigma=1.0):
    """
    在 Batch 内部动态构建拉普拉斯矩阵 (Based on RBF Kernel)
    Args:
        features: [B, C] tensor, 这里的 features 通常是中心像素的光谱
        sigma: RBF 核的宽度参数
    Returns:
        laplacian: [B, B] Normalized Laplacian Matrix
    """
    # 1. 计算欧氏距离矩阵 [B, B]
    # dist[i, j] = ||x_i - x_j||
    dist = torch.cdist(features, features, p=2)
    
    # 2. 构建邻接矩阵 A (RBF Kernel)
    # A_ij = exp(-dist^2 / sigma^2)
    A = torch.exp(-(dist ** 2) / (sigma ** 2))
    
    # 3. 添加自环 (Self-loops) 以保持自身特征
    B = features.size(0)
    I = torch.eye(B, device=features.device)
    A_hat = A + I
    
    # 4. 计算度矩阵 D_hat
    D_hat_diag = torch.sum(A_hat, dim=1)
    D_hat_inv_sqrt = torch.pow(D_hat_diag, -0.5)
    D_hat_inv_sqrt[torch.isinf(D_hat_inv_sqrt)] = 0. # 处理除零
    D_hat_inv_sqrt_mat = torch.diag(D_hat_inv_sqrt)
    
    # 5. 计算归一化拉普拉斯矩阵: L = D^-0.5 * A_hat * D^-0.5
    L = torch.mm(torch.mm(D_hat_inv_sqrt_mat, A_hat), D_hat_inv_sqrt_mat)
    
    return L

class InternalSplitStem(nn.Module):
    def __init__(self, 
                 total_bands_per_step=41,  # 你的C=41
                 seq_len=10,               # 你的T=10 (或其他值)
                 hidden_dim=128,
                 static_indices=None):
        super().__init__()
        
        # 1. 定义索引
        if static_indices is None:
            # 默认为你提供的列表
            self.static_indices = [0, 13, 14, 15, 16, 17, 18, 19, 20, 37, 38, 39, 40]
        else:
            self.static_indices = static_indices
            
        # 自动计算动态变量索引 (0-40中排除静态的)
        all_indices = torch.arange(total_bands_per_step)
        # 使用mask来筛选动态索引
        is_static = torch.zeros(total_bands_per_step, dtype=torch.bool)
        is_static[self.static_indices] = True
        self.dynamic_indices = all_indices[~is_static].tolist()
        
        # 将索引注册为buffer，以免在设备移动时出错，但它们不是模型参数
        self.register_buffer('static_idx_tensor', torch.tensor(self.static_indices, dtype=torch.long))
        self.register_buffer('dynamic_idx_tensor', torch.tensor(self.dynamic_indices, dtype=torch.long))
        
        self.seq_len = seq_len
        self.total_bands = total_bands_per_step
        
        num_static = len(self.static_indices)      # 13
        num_dynamic = len(self.dynamic_indices)    # 28
        
        # ==========================================
        # 2. 定义处理流 (Streams)
        # ==========================================
        
        # Stream A: 动态流 (处理 T * 28 个通道)
        # 策略: 先用 1x1 卷积压缩巨大的时间维度，再用 3x3 提取空间特征
        self.dynamic_branch = nn.Sequential(
            nn.Conv2d(num_dynamic * seq_len, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Stream B: 静态流 (处理 13 个通道)
        # 策略: 只需要处理一次 (T=0)，直接用 3x3 卷积提取语义/空间特征
        self.static_branch = nn.Sequential(
            nn.Conv2d(num_static, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # ==========================================
        # 3. 融合层
        # ==========================================
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

    def forward(self, x):
        """
        Args:
            x: [B, Total_Channels, H, W] 
               其中 Total_Channels = seq_len * 41 = 10 * 41 = 410
               实际数据排列是 (C*T) 顺序：(C0_T0, C0_T1, ..., C0_T9, C1_T0, C1_T1, ..., C40_T9)
        """
        B, C_total, H, W = x.shape
        
        # 1. 恢复时间维度视图 [B, T, 41, H, W]
        # 输入是 (B, C*T, H, W) = (B, 41*10, H, W)，需要先 reshape 为 (B, C, T, H, W) 再 permute
        x_reshaped = x.view(B, self.total_bands, self.seq_len, H, W)  # (B, 41, 10, H, W)
        x_reshaped = x_reshaped.permute(0, 2, 1, 3, 4)  # (B, 10, 41, H, W)
        
        # 2. 内部拆分 (Slicing)
        
        # 提取动态部分: 取所有时间步 [B, T, 28, H, W]
        x_dyn = torch.index_select(x_reshaped, 2, self.dynamic_idx_tensor)
        # 展平回 [B, T*28, H, W]
        x_dyn = x_dyn.reshape(B, -1, H, W)
        
        # 提取静态部分: 只取第 1 个时间步 (T=0) [B, 13, H, W]
        # 既然输入里重复了10次，我们只算一次，省去90%的静态卷积计算量
        x_stat = torch.index_select(x_reshaped[:, 0, ...], 1, self.static_idx_tensor)
        
        # 3. 分别通过卷积层
        feat_dyn = self.dynamic_branch(x_dyn)   # [B, hidden, H, W]
        feat_stat = self.static_branch(x_stat)  # [B, hidden, H, W]
        
        # 4. 融合
        out = torch.cat([feat_dyn, feat_stat], dim=1)
        out = self.fusion(out)
        
        return out

class MambaModel(nn.Module):
    def __init__(self, num_classes=1, patch_size=13, num_bands=13870, hidden_dim=128):
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
        
        # self.stem = InternalSplitStem(
        #     total_bands_per_step=41, # 41
        #     seq_len=10,                     # 10
        #     hidden_dim=hidden_dim,
        #     static_indices=[0, 13, 14, 15, 16, 17, 18, 19, 20, 37, 38, 39, 40]
        # )
        
        self.spectral_blocks = nn.Sequential(
            *[SpatialSpectralMambaBlock(dim=hidden_dim, patch_size=patch_size) for _ in range(2)]
        )
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Input x: [B, C, H, W]
        # Single year format: C = 138 = 6 bands × 23 time steps
        # Use stem layer to map input channels to hidden_dim        
        x = self.stem(x)  # [B, hidden_dim, H, W]
        
        features = self.spectral_blocks(x)

        x = self.head(features)
        return x


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
        self.model = MambaModel(num_classes=pred_len, patch_size=patch_size, num_bands=380, hidden_dim=d_model)
        self.gcn_stream = MiniGCNStream(in_features=d_model, hidden_features=d_model * 2, out_features=d_model, dropout=0.3)
        self.gcn_stream_manifold = MiniGCNStream(in_features=d_model, hidden_features=d_model * 2, out_features=d_model, dropout=0.3)
        # 时间信息 embedding：doy 1..366，用于预测时刻的季节性
        time_emb_dim = getattr(configs, 'time_emb_dim', 32)
        self.doy_embedding = nn.Embedding(366 + 1, d_model, padding_idx=0)  # 0=padding, 1..366=doy
        self.fc_out = nn.Linear(d_model * 3, pred_len)
        
        # self.graph_layer = UMAPSpectralGraphLayer(
        #     builder_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10.joblib",
        #     library_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10_libraries_by_label.npz",
        #     seq_len=seq_len,
        #     n_channels=38,
        # )
        
        self.graph_layer = UMAPAnchorLaplacian(
            umap_model_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10.joblib",
            library_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10_libraries_by_label.npz",
            sigma=1.0,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        # x_enc: (B, T, C, H, W)，最后一通道为 LULC (1..17)
        B, T, C, H, W = x_enc.shape
        
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
            ws = 30
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
        # pdb.set_trace()
        x_enc = x_enc.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)
        _, T_mark, _ = x_mark_enc.shape
        # 解码器时间信息：最后一维为 doy_norm [0,1]，转成 1..366 做 embedding
        doy_norm = x_mark_enc[:, :, -1]  # (B, T_dec)
        doy_idx = (doy_norm * 365.0 + 1).long().clamp(1, 366)  # (B, T_dec)，取预测时刻（最后一时间步）
        doy_emb = self.doy_embedding(doy_idx[:, -1])  # (B, time_emb_dim)
        # 若 LULC 在 [0,1] 归一化，缩放到 1..17
        # x_enc_copy = x_enc.clone()
        # x_enc_copy = x_enc_copy.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)
        
        # # if x_enc[:, :, -1].max() < 1.0:
        # #     x_enc = x_enc.clone()
        # x_enc[:, :, -1] = x_enc[:, :, -1] * 17.0
        
        # # 按空间点展平
        # x_flat = x_enc.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)  # (B*H*W, T, C)
        # # LULC 做 17 类 one-hot（用于图拉普拉斯）
        # x_onehot = self.lulc_onehot(x_flat)  # (B*H*W, T, 54) = (B*H*W, T, 37+17)
        
        # # 图拉普拉斯用 54 维（one-hot 后）
        # x_onehot_reshaped = x_onehot.view(B, H, W, T, self.enc_in_after_onehot).permute(0, 3, 4, 1, 2)  # (B, T, 54, H, W)
        # x_onehot_for_graph = x_onehot_reshaped.permute(0, 2, 1, 3, 4).reshape(B, self.enc_in_after_onehot * T, H, W)  # (B, 54*T, H, W)
        # x_center_onehot = x_onehot_for_graph[:, :, H//2, W//2]  # (B, 54*T)
        laplacian_manifold = self.graph_layer(x_enc[:, :, H//2, W//2])  # (B, B)
        
        # # 主模型用 41 维（embedding 后）：将类别 ID 映射为 4 维
        # x_cont = x_flat[:, :, :-1]  # (B*H*W, T, 37) 连续特征
        # lulc_raw = x_flat[:, :, -1]  # (B*H*W, T) 原始 LULC
        # if lulc_raw.dtype.is_floating_point:
        #     lulc_raw = torch.round(lulc_raw)
        # lulc_idx = lulc_raw.long().clamp(min=0, max=self.lulc_embedding.num_embeddings - 1)
        # # 将 nodata / 非法值映射到 0（padding_idx）
        # for bad in (0, 255, -1):
        #     lulc_idx[lulc_idx == bad] = 0
        # lulc_emb = self.lulc_embedding(lulc_idx)  # (B*H*W, T, 4)
        # x_emb = torch.cat([x_cont.float(), lulc_emb], dim=-1)  # (B*H*W, T, 41)
        
        # # 还原为空间形状并送入主模型
        # x_emb_reshaped = x_emb.view(B, H, W, T, self.enc_in_after_lulc).permute(0, 3, 4, 1, 2)  # (B, T, 41, H, W)
        # x = x_emb_reshaped.permute(0, 2, 1, 3, 4).reshape(B, self.enc_in_after_lulc * T, H, W)  # (B, 41*T, H, W)
        hidden_features = self.model(x_enc)  # (B, d_model)
        laplacian = construct_batch_laplacian(burn_history)
        # pdb.set_trace()
        out = self.gcn_stream(hidden_features, laplacian)  # (B, d_model)
        out2 = self.gcn_stream_manifold(hidden_features, laplacian_manifold)  # (B, d_model)
        out = self.fc_out(torch.cat([out, out2, hidden_features], dim=1))
        # out = self.fc_out(out)
        return out


def visualize_channel_time_window(x_enc, channel_indices=(1, 2, 3, 4), window_len=5, batch_idx=0, save_path=None):
    """
    从 x_enc (B, T, C, H, W) 中取多个通道（默认第 2、3、4、5 个），在 T 维上随机取长度为 window_len 的时间窗口并可视化。
    每个通道一行，一行内为同一通道的 5 个时间切片。
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


if __name__ == "__main__":
    model = MambaModel(num_classes=13, patch_size=13, num_bands=138, hidden_dim=64)
    x = torch.randn(128, 138, 13, 13)
    print(model(x).shape)

    # 可视化：x_enc (B, T, C, H, W) 取第 2、3、4、5 通道，每通道一行；T 上随机长度 5 的时间窗口
    x_enc = torch.randn(2, 365, 38, 13, 13)
