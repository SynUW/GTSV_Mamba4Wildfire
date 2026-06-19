import torch
import torch.nn as nn
from torch.nn import AvgPool2d, MaxPool2d
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt
from STGMamba import KFGN_Mamba as MambaBlock
from STGMamba import ModelArgs

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
    

class SpatialSpectralMambaBlock(nn.Module):
    def __init__(self, dim: int, patch_size: int, num_heads = 8):
        super().__init__()
        head_dim = dim // num_heads
        kfgn_mamba_args = ModelArgs(
            K=3,
            A=torch.ones(169, 169),
            feature_size=169,
            d_model=169,  # hidden_dim is fea_size
            n_layer=4,
            features=169
        )
        self.spatial_mamba = MambaBlock(kfgn_mamba_args)
        self.spectral_mamba = SparseDeformableMambaBlock(dim=13*13, drop_rate=0.3)
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
        B, C, H, W = x.shape  # C = hidden_dim == 128
        x = self.norm(x)
        x_spectral = x.reshape(B, C, H * W)
        x_spectral = self.spectral_mamba(x_spectral)
        
        x_spatial = x.reshape(B, C, -1)  # b, c, h*w
        x_spatial = self.spatial_mamba(x_spatial)
        x_spatial = x_spatial.reshape(B, C, H, W)
        
        fusion = x_spatial + x_spectral.reshape(B, C, H, W)
        
        x = self.conv(fusion) + x

        return x

def construct_batch_laplacian(features, sigma=1.0):
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

class MambaModel(nn.Module):
    def __init__(self, num_classes=1, patch_size=13, num_bands=13870, hidden_dim=128):
        super().__init__()
        self.patch_size = patch_size

        # Stem layer for initial feature extraction
        
        self.stem = nn.Sequential(
            nn.Conv2d(num_bands, num_bands//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_bands//2),
            nn.GELU(),
            nn.Conv2d(num_bands//2, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
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

class Model(nn.Module):
    """Wrapper for train_all_h5: accepts configs and (x_enc, x_mark_enc, x_dec, x_mark_dec) interface."""

    def __init__(self, configs):
        super().__init__()
        patch_size = getattr(configs, 'patch_size', 13)
        self.enc_in = getattr(configs, 'enc_in', 38)
        seq_len = getattr(configs, 'seq_len', 365)
        pred_len = getattr(configs, 'pred_len', 1)
        d_model = getattr(configs, 'd_model', 128)
        num_bands = self.enc_in * seq_len  # (B, C, H, W, T) -> stem expects (B, C*T, H, W)
        self.pred_len = pred_len
        self.model = MambaModel(num_classes=pred_len, patch_size=patch_size, num_bands=num_bands, hidden_dim=d_model)
        
        self.gcn_stream = MiniGCNStream(in_features=d_model, hidden_features=d_model, out_features=d_model, dropout=0.3)
        # 将 GCN 输出 (B, d_model) 投影到 (B, pred_len)，否则 test 时 test_probs 会是 (N, d_model)，flatten 后与 (N,) 的 targets 维度不匹配
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc: (B, seq_len, C, H, W) from adapter
        # visualize_channel_time_window(x_enc, channel_indices=(-1), window_len=10, batch_idx=0, save_path='ssm_ch2_time5.png')
        B, T, C, H, W = x_enc.shape
        x = x_enc.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)  # (B, C, H, W, T) -> MambaModel expects (B, C, H, W, T)
        hidden_features = self.model(x)  # (B, d_model)，MambaModel 的 head 未做 num_classes 线性层

        laplacian = construct_batch_laplacian(hidden_features)
        out = self.gcn_stream(hidden_features, laplacian)  # (B, d_model)
        out = self.fc_out(out)  # (B, pred_len)，与 test 期望的 [B, L] 一致
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
    model = MambaModel(num_classes=13, patch_size=13, num_bands=138, hidden_dim=128)
    x = torch.randn(10, 138, 13, 13)
    print("Output shape:", model(x).shape)

    # 参数量与计算复杂度 (与 __main__ 中构造的 model/x 一致)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,}) ≈ {total_params/1e6:.2f}M")

    try:
        from thop import profile
        with torch.no_grad():
            flops, _ = profile(model, inputs=(x,), verbose=False)
        print(f"FLOPs (thop): {flops:,} ≈ {flops/1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"FLOPs: 未安装 thop 或计算失败 (pip install thop), {e}")

    # 可视化：x_enc (B, T, C, H, W) 取第 2、3、4、5 通道，每通道一行；T 上随机长度 5 的时间窗口
    # x_enc = torch.randn(2, 365, 38, 13, 13)
