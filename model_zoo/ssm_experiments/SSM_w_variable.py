"""
将UMAP之后的x进行投影，然后计算adjacency matrix，这个投影是可学习的。
UMAP2相较于UMAP1，主要区别在于：
UMAP2使用的是可学习的相似度计算头，而UMAP1使用的是固定的相似度计算头。
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

class UMAPAnchorLaplacian(nn.Module):
    """
    基于 UMAP + Anchor 库的多头可学习自适应图拉普拉斯矩阵构建模块。
 
    设计思路：
        1. UMAP 投影（不可导）
           将输入样本投影到低维语义空间，提供地表类型先验。
 
        2. 多头可学习相似度（可导，参考 DASGCN MAAM）
           每个头使用独立的 sample_proj / anchor_proj 投影样本和 anchor，
           在投影空间内计算余弦相似度 Si [B, K]。
           最终相似度 S = sum(wi * Si)，wi 为可学习头权重（类比多头注意力）。
           不同头捕捉不同视角的 anchor 相关模式，head_weights 本身可解释。
 
        3. Anchor 中介建图（保留核心语义中介思想）
           A_umap = S_sparse @ S_sparse^T
           两个样本通过共享高权重 anchor 建立连接。
 
        4. 自适应调制（可导，参考 AAMGCRN）
           W = sigmoid(normalize(adapt_proj(hidden_features)) @
                       normalize(adapt_proj(hidden_features))^T)
           A = W ∘ A_umap（Hadamard 乘积）
           任务驱动的细化，梯度路径最短直接连通 Mamba 主干。
 
        5. Anchor 利用率正则（参考图聚类文献）
           同时约束 Affinity（稀疏性）和 Balance（均匀性），
           比单纯负熵正则更完整。
 
    梯度路径：
        路径一（任务驱动）：
            pred_loss → GCN → L → A → W → adapt_proj → hidden_features → Mamba
        路径二（语义驱动）：
            pred_loss → GCN → L → A → A_umap → S_sparse → S →
            sample_proj[i] / anchor_proj[i] → head_weights
 
    参数：
        umap_model_path  : 预训练 UMAP 模型的 .joblib 文件路径。
        library_path     : 光谱库 .npz 文件路径。
                           旧格式：'library_embeddings' (K, T, c)
                           新格式：'pos_library_embeddings' +
                                   'neg_library_embeddings'，自动合并。
        n_input_channels : 原始输入波段数，默认 38。
        hidden_dim       : 每个头的 sample_proj / anchor_proj 输出维度，默认 32。
        d_model          : hidden_features 的维度（主干网络输出），默认 380。
        adapt_dim        : adapt_proj 的中间维度，默认 32。
        num_heads        : 相似度计算的头数（类比多头注意力），默认 4。
        temperature      : Softmax 稀疏化温度，越小越趋近硬 Top-K，默认 1.0。
        lambda_anchor    : anchor 利用率正则的损失权重，默认 0.01。
        c1               : Affinity 正则权重（稀疏性），默认 3.0。
        c2               : Balance 正则权重（均匀性），默认 1.0。
        c3               : L2 正则权重，默认 1e-6。
        device           : 运行设备，默认 'cuda'。
 
    输入：
        x               : Tensor [B, C*T]    Flatten 后的时序光谱数据。
        hidden_features : Tensor [B, d_model] 主干网络输出的隐特征。
        temperature     : float (可选)        覆盖初始化时的温度。
 
    输出：
        L          : Tensor [B, B]   归一化拉普拉斯矩阵，直接送入 GCN。
        graph_loss : Tensor (标量)   anchor 利用率正则损失。
        S          : Tensor [B, K]   融合后的 anchor 相似度（供调试）。
    """
 
    def __init__(
        self,
        umap_model_path: str,
        library_path: str,
        n_input_channels: int = 38,
        hidden_dim: int = 32,
        d_model: int = 380,
        adapt_dim: int = 32,
        num_heads: int = 4,
        temperature: float = 1.0,
        lambda_anchor: float = 0.01,
        c1: float = 3.0,
        c2: float = 1.0,
        c3: float = 1e-6,
        device: str = 'cuda',
    ):
        super().__init__()
 
        self.C = n_input_channels
        self.num_heads = num_heads
        self.temperature = temperature
        self.lambda_anchor = lambda_anchor
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.device = device
 
        # ==============================================================
        # 1. 加载 UMAP 模型
        # ==============================================================
        print(f"[UMAPAnchorLaplacian] Loading UMAP model from {umap_model_path} ...")
        loaded_obj = joblib.load(umap_model_path)
        self.umap_model = (
            loaded_obj.umap_model
            if hasattr(loaded_obj, 'umap_model')
            else loaded_obj
        )
 
        # ==============================================================
        # 2. 加载 Anchor 库
        # ==============================================================
        print(f"[UMAPAnchorLaplacian] Loading anchor library from {library_path} ...")
        lib_data = np.load(library_path)
 
        if 'library_embeddings' in lib_data:
            anchors_np = lib_data['library_embeddings']              # (K, T, c)
        elif ('pos_library_embeddings' in lib_data
              and 'neg_library_embeddings' in lib_data):
            pos = lib_data['pos_library_embeddings']                 # (K_pos, T, c)
            neg = lib_data['neg_library_embeddings']                 # (K_neg, T, c)
            anchors_np = np.concatenate([pos, neg], axis=0)         # (K, T, c)
        else:
            raise KeyError(
                f"Expected 'library_embeddings' or "
                f"('pos_library_embeddings', 'neg_library_embeddings') "
                f"in {library_path}, got: {list(lib_data.keys())}"
            )
 
        self.K, self.T_lib, self.n_components = anchors_np.shape
        anchors_flat = anchors_np.reshape(self.K, -1)                # (K, T_lib*c)
        umap_feat_dim = self.T_lib * self.n_components
 
        self.register_buffer(
            'anchors_raw',
            torch.from_numpy(anchors_flat).float()                   # (K, T_lib*c)
        )
 
        print(
            f"[UMAPAnchorLaplacian] Ready. "
            f"K={self.K}, T_lib={self.T_lib}, c={self.n_components}, "
            f"umap_feat_dim={umap_feat_dim}, hidden_dim={hidden_dim}, "
            f"num_heads={num_heads}, d_model={d_model}"
        )
 
        # ==============================================================
        # 3. 多头可学习相似度投影头（参考 DASGCN MAAM）
        #
        #    每个头有独立的 sample_proj 和 anchor_proj，
        #    允许不同头捕捉不同视角的 anchor 相关模式。
        #    注意 sample_proj 和 anchor_proj 是两个独立网络（非对称设计），
        #    允许学习非对称相关性，最终通过 A_umap 对称化。
        # ==============================================================
        self.sample_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(umap_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_heads)
        ])
 
        self.anchor_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(umap_feat_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_heads)
        ])
 
        # 头权重：可学习，经 softmax 归一化后加权融合各头的相似度矩阵
        # 物理意义：模型自动学习哪种 anchor 相关视角对预测更重要
        self.head_weights = nn.Parameter(
            torch.ones(num_heads) / num_heads
        )
 
        # ==============================================================
        # 4. 自适应调制网络（参考 AAMGCRN）
        #
        #    对称双线性：
        #      h   = adapt_proj(hidden_features)   [B, adapt_dim]
        #      h_n = L2_normalize(h)               [B, adapt_dim]
        #      W   = sigmoid(h_n @ h_n^T)          [B, B]
        #    天然对称，值域 [0, 1]，梯度路径直连 hidden_features。
        # ==============================================================
        self.adapt_proj = nn.Sequential(
            nn.Linear(d_model, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, adapt_dim),
        )
 
    # ------------------------------------------------------------------
    # 私有工具方法
    # ------------------------------------------------------------------
 
    def _umap_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入 x [B, C*T] 通过预训练 UMAP 投影到低维嵌入 [B, T_lib*c]。
        取时序最后 T_lib 步参与投影（与训练时一致）。
        此操作在 CPU 上执行，不可导，梯度在此截断。
        """
        B, CT = x.shape
        T_input = CT // self.C
        x_reshaped = x.view(B, T_input, self.C)
 
        if T_input >= self.T_lib:
            x_last = x_reshaped[:, -self.T_lib:, :]                 # [B, T_lib, C]
        else:
            pad = torch.zeros(
                B, self.T_lib - T_input, self.C, device=x.device
            )
            x_last = torch.cat([pad, x_reshaped], dim=1)            # [B, T_lib, C]
 
        x_flat = x_last.reshape(-1, self.C)                         # [B*T_lib, C]
 
        with torch.no_grad():
            x_cpu = x_flat.detach().cpu().numpy()
            old_verbose = getattr(self.umap_model, 'verbose', None)
            old_tqdm    = getattr(self.umap_model, 'tqdm_kwds', None)
            self.umap_model.verbose   = False
            self.umap_model.tqdm_kwds = None
            with disable_tqdm():
                z_np = self.umap_model.transform(x_cpu)             # [B*T_lib, c]
            if old_verbose is not None:
                self.umap_model.verbose   = old_verbose
            if old_tqdm is not None:
                self.umap_model.tqdm_kwds = old_tqdm
 
        z_flat = torch.from_numpy(z_np).float().to(x.device)        # [B*T_lib, c]
        z = z_flat.view(B, self.T_lib * self.n_components)          # [B, T_lib*c]
        return z
 
    def _compute_multihead_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """
        多头可学习相似度计算（参考 DASGCN MAAM 的多通道设计）。
 
        每个头独立计算样本与 anchor 的余弦相似度 Si [B, K]，
        最终用可学习权重 head_weights 加权融合为 S [B, K]。
 
        参数：
            z : [B, T_lib*c]  样本 UMAP 嵌入（梯度截断后的输出）
        返回：
            S : [B, K]        多头融合后的相似度，负值截断为 0
        """
        weights = F.softmax(self.head_weights, dim=0)                # [num_heads]
        S = torch.zeros(z.shape[0], self.K, device=z.device)
 
        for i in range(self.num_heads):
            # 样本投影（可导）：[B, hidden_dim]
            z_q = F.normalize(self.sample_projs[i](z), p=2, dim=-1)
 
            # anchor 投影（可导）：[K, hidden_dim]
            z_k = F.normalize(
                self.anchor_projs[i](self.anchors_raw), p=2, dim=-1
            )
 
            # 余弦相似度，截断负值：[B, K]
            Si = F.relu(torch.mm(z_q, z_k.T))
 
            # 加权累加
            S = S + weights[i] * Si
 
        return S                                                      # [B, K]
 
    def _softmax_sparsify(
        self, S: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Softmax 软稀疏化，全程可导，替代硬 Top-K 截断。
 
        temperature 调度建议：
            训练初期 1.0 → 探索，所有 anchor 均匀贡献
            训练后期 0.1 → 收敛，权重集中于最相似 anchor
 
        参数：
            S           : [B, K]
            temperature : float
        返回：
            S_soft : [B, K]  行和为 1 的软权重
        """
        return F.softmax(S / (temperature + 1e-6), dim=-1)
 
    def _compute_adapt_weights(
        self, hidden_features: torch.Tensor
    ) -> torch.Tensor:
        """
        用主干网络隐特征计算 [B, B] 自适应调制矩阵（参考 AAMGCRN）。
 
        对称双线性内积：
            h   = adapt_proj(hidden_features)   [B, adapt_dim]
            h_n = L2_normalize(h)               [B, adapt_dim]
            W   = sigmoid(h_n @ h_n^T)          [B, B]
 
        天然对称，值域 [0, 1]，梯度路径最短。
 
        参数：
            hidden_features : [B, d_model]
        返回：
            W : [B, B]
        """
        h = self.adapt_proj(hidden_features)                         # [B, adapt_dim]
        h_norm = F.normalize(h, p=2, dim=-1)                        # [B, adapt_dim]
        W = torch.sigmoid(torch.mm(h_norm, h_norm.T))               # [B, B]
        return W
 
    @staticmethod
    def _normalize_laplacian(A: torch.Tensor) -> torch.Tensor:
        """
        对邻接矩阵 A [B, B] 计算对称归一化拉普拉斯矩阵。
        L = D^{-1/2} (A + I) D^{-1/2}
        """
        B = A.shape[0]
        I = torch.eye(B, device=A.device)
        A_hat = A + I
        D = A_hat.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat = torch.diag(D_inv_sqrt)
        L = torch.mm(torch.mm(D_mat, A_hat), D_mat)
        return L
 
    def _anchor_usage_loss(self, S: torch.Tensor) -> torch.Tensor:
        """
        Anchor 利用率正则损失（参考图聚类文献的 Affinity + Balance 设计）。
 
        N = S^T @ S  [K, K]，类比论文中的 N = B^T B。
 
        三项约束：
            Affinity  （最小化）：N 的非对角元素均值 / 对角元素均值
                       → 每个样本只属于一个 anchor（稀疏性）
            Balance   （最大化）：V = diag(N)^T @ diag(N) 的非对角 / 对角
                       → 每个 anchor 均匀被激活（均匀性）
            L2 正则              → 防止相似度权重爆炸
 
        注意：使用稀疏化前的原始 S，保证所有 anchor 参数均有梯度。
 
        参数：
            S : [B, K]  多头融合后、稀疏化前的相似度矩阵
        返回：
            loss : 标量
        """
        # N = S^T @ S : [K, K]
        N = torch.mm(S.T, S)
 
        diag_N    = torch.diag(N)                                    # [K]
        off_diag_N = N - torch.diag(diag_N)                         # [K, K]
 
        # Affinity：最小化（稀疏性）
        affinity = off_diag_N.sum() / (
            (self.K - 1) * diag_N.sum() + 1e-6
        )
 
        # Balance：最大化（均匀性）
        # V = v^T v，v = diag(N)
        V = diag_N.unsqueeze(1) * diag_N.unsqueeze(0)               # [K, K]
        diag_V    = torch.diag(torch.diag(V))
        off_diag_V = V - diag_V
 
        balance = off_diag_V.sum() / (
            (self.K - 1) * torch.diag(V).sum() + 1e-6
        )
 
        # L2 正则
        l2_reg = (S ** 2).sum() / (S.numel() + 1e-6)
 
        loss = (
            self.c1 * affinity
            + self.c2 * (1.0 - balance)
            + self.c3 * l2_reg
        )
 
        return self.lambda_anchor * loss
 
    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
 
    def forward(
        self,
        x: torch.Tensor,
        hidden_features: torch.Tensor,
        temperature: float = None,
    ):
        """
        参数：
            x               : Tensor [B, C*T]    Flatten 后的时序光谱数据。
            hidden_features : Tensor [B, d_model] 主干网络输出的隐特征。
            temperature     : float (可选)        覆盖初始化时的温度，
                                                  用于训练退火调度。
 
        返回：
            L          : Tensor [B, B]   归一化拉普拉斯矩阵。
            graph_loss : Tensor (标量)   anchor 利用率正则损失。
            S          : Tensor [B, K]   多头融合相似度（供调试/可视化）。
        """
        temp = temperature if temperature is not None else self.temperature
 
        # ----------------------------------------------------------
        # Step 1: UMAP 投影（不可导，梯度在此截断）
        # ----------------------------------------------------------
        z = self._umap_transform(x)                                  # [B, T_lib*c]
 
        # ----------------------------------------------------------
        # Step 2: 多头可学习相似度（可导）
        #   每个头捕捉不同 anchor 相关视角，head_weights 可学习融合
        # ----------------------------------------------------------
        S = self._compute_multihead_similarity(z)                    # [B, K]
 
        # ----------------------------------------------------------
        # Step 3: Softmax 软稀疏化（可导，替代硬 Top-K）
        # ----------------------------------------------------------
        S_sparse = self._softmax_sparsify(S, temp)                   # [B, K]
 
        # ----------------------------------------------------------
        # Step 4: Anchor 中介建图（UMAP 语义骨架）
        #   A_umap[i,j] 大 ⟺ 样本 i、j 共享高权重 anchor
        #   对称化保证无向图
        # ----------------------------------------------------------
        A_umap = torch.mm(S_sparse, S_sparse.T)                     # [B, B]
        A_umap = 0.5 * (A_umap + A_umap.T)                         # 显式对称化
 
        # ----------------------------------------------------------
        # Step 5: 自适应调制（任务驱动，梯度路径最短）
        #   W 由 hidden_features 驱动
        #   A = W ∘ A_umap（Hadamard）
        # ----------------------------------------------------------
        W = self._compute_adapt_weights(hidden_features)             # [B, B]
        A = W * A_umap                                               # [B, B]
 
        # ----------------------------------------------------------
        # Step 6: 归一化拉普拉斯
        # ----------------------------------------------------------
        L = self._normalize_laplacian(A)                             # [B, B]
 
        # ----------------------------------------------------------
        # Step 7: Anchor 利用率正则损失
        #   使用稀疏化前的 S，保证所有 anchor 参数均有梯度
        # ----------------------------------------------------------
        graph_loss = self._anchor_usage_loss(S)
 
        return L, graph_loss, S

   
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


def construct_batch_laplacian_jaccard_weighted(
    burn_history,
    k=40, # Top-K 近邻数
    buffer_size=3, # 事件相似性中的时间缓冲窗口，用于容忍火灾滞后。前后各 buffer_size//2 个时间步。
    alpha_event=0.7,
    fire_threshold=0.0, # 将 burn_history > fire_threshold 视为“活跃/着火”窗口。
    recent_window=3, # 背景统计里“最近窗口火活动”的长度。
    sigma_bg=None, # 背景状态 RBF 相似度的 sigma。
    mutual_knn=False, # 是否使用 mutual kNN（双向近邻才保留边）。
    self_loop_weight=1.0, # 自环权重。
    return_adjacency=False, # 若为 True，返回 (L, A, S_total, S_event, S_bg)
    eps=1e-6, # 数值稳定项。
    visualize=False, # 是否保存邻接矩阵 A 的热力图
    visualize_path_prefix="ssm_batch_laplacian_event", # 热力图保存路径前缀
):
    """
    基于“事件相似性 + 背景状态相似性”构建 batch 内图的归一化传播矩阵。

    参数:
        burn_history: [B, T]
            每个样本的历史野火序列（非负，允许连续值，例如烧毁面积/火强度/概率）。
        k: int
            每个节点保留的近邻数（不含自己）。
            建议:
              - B=16 时先试 3~5
              - B=32 时先试 5~8
        buffer_size: int
            事件相似性中的时间缓冲窗口，用于容忍火灾滞后。
            建议 3 或 5。
        alpha_event: float in [0, 1]
            事件相似性的权重。背景状态权重为 (1 - alpha_event)。
        fire_threshold: float
            将 burn_history > fire_threshold 视为“活跃/着火”窗口。
        recent_window: int
            背景统计里“最近窗口火活动”的长度。
        sigma_bg: float or None
            背景状态 RBF 相似度的 sigma。
            None 时自动用 batch 内 pairwise distance 的中位数估计。
        mutual_knn: bool
            是否使用 mutual kNN（双向近邻才保留边）。
        self_loop_weight: float
            自环权重。
        return_adjacency: bool
            若为 True，返回 (L, A, S_total, S_event, S_bg)
            否则只返回 L
        eps: float
            数值稳定项。

    返回:
        L: [B, B]
            归一化传播矩阵：D^{-1/2} (A + I*self_loop_weight) D^{-1/2}
            （与当前代码风格一致，适合直接送入 GCN）
    """

    # ============================================================
    # 0. 输入检查与预处理
    # ============================================================
    if burn_history.ndim != 2:
        raise ValueError(f"burn_history must be [B, T], got {tuple(burn_history.shape)}")

    bh = F.relu(burn_history.float())   # 保证非负
    device = bh.device
    B, T = bh.shape

    if B == 1:
        # 单节点图的退化情况
        L = torch.ones(1, 1, device=device)
        if return_adjacency:
            A = torch.zeros(1, 1, device=device)
            S = torch.ones(1, 1, device=device)
            return L, A, S, S, S
        return L

    # ============================================================
    # 1. 事件相似性 S_event: buffered Tanimoto / Jaccard
    # ============================================================
    # 用 max_pool1d 做时间缓冲，容忍传播滞后
    if buffer_size > 1:
        pad = buffer_size // 2
        bh_buf = F.max_pool1d(
            bh.unsqueeze(1), kernel_size=buffer_size, stride=1, padding=pad
        ).squeeze(1)
    else:
        bh_buf = bh

    # 广义 Tanimoto / Jaccard
    dot_event = torch.mm(bh_buf, bh_buf.t())                      # [B, B]
    norm_sq_event = torch.sum(bh_buf ** 2, dim=1, keepdim=True)  # [B, 1]
    union_event = norm_sq_event + norm_sq_event.t() - dot_event
    S_event = dot_event / (union_event + eps)

    # ============================================================
    # 2. 背景状态相似性 S_bg
    #    目的：把“共同不着火 / 共同低风险 / 共同稳定”也编码进去
    # ============================================================
    active = (bh > fire_threshold).float()        # [B, T]
    quiet = 1.0 - active                          # [B, T]

    quiet_ratio = quiet.mean(dim=1, keepdim=True)                 # 长期无火比例
    active_ratio = active.mean(dim=1, keepdim=True)               # 活跃比例
    mean_fire = bh.mean(dim=1, keepdim=True)                      # 平均火活动
    std_fire = bh.std(dim=1, keepdim=True, unbiased=False)        # 波动
    recent_len = min(max(1, recent_window), T)
    recent_fire = bh[:, -recent_len:].mean(dim=1, keepdim=True)   # 最近火活动

    # 背景状态描述向量: [B, 5]
    bg_feat = torch.cat(
        [quiet_ratio, active_ratio, mean_fire, std_fire, recent_fire], dim=1
    )

    # 标准化，避免不同量纲不平衡
    bg_feat = (bg_feat - bg_feat.mean(dim=0, keepdim=True)) / (bg_feat.std(dim=0, keepdim=True) + eps)

    # 用 RBF 做背景状态相似性
    dist_bg = torch.cdist(bg_feat, bg_feat, p=2)   # [B, B]

    if sigma_bg is None:
        # 用上三角非对角元素的中位数做自适应 sigma
        tri_mask = torch.triu(torch.ones_like(dist_bg, dtype=torch.bool), diagonal=1)
        valid_dists = dist_bg[tri_mask]
        if valid_dists.numel() == 0:
            sigma_bg = 1.0
        else:
            sigma_bg = torch.median(valid_dists).item()
            sigma_bg = max(sigma_bg, 1e-3)

    S_bg = torch.exp(-(dist_bg ** 2) / (2 * (sigma_bg ** 2) + eps))

    # ============================================================
    # 3. 融合总相似度
    # ============================================================
    alpha_event = float(alpha_event)
    alpha_event = max(0.0, min(1.0, alpha_event))
    S_total = alpha_event * S_event + (1.0 - alpha_event) * S_bg

    # 防止数值误差
    S_total = torch.clamp(S_total, min=0.0, max=1.0)

    # ============================================================
    # 4. kNN 图构建（先排除 self，再补自环）
    # ============================================================
    k_eff = min(k, B - 1)

    S_for_knn = S_total.clone()
    S_for_knn.fill_diagonal_(float("-inf"))

    vals, inds = torch.topk(S_for_knn, k=k_eff, dim=1)

    A = torch.zeros_like(S_total)
    A.scatter_(1, inds, vals)

    if mutual_knn:
        # 只保留双向都认为是近邻的边
        mutual_mask = (A > 0) & (A.t() > 0)
        A = ((A + A.t()) / 2.0) * mutual_mask.float()
    else:
        # 普通对称化
        A = (A + A.t()) / 2.0

    # mask = ~torch.eye(A.size(0), dtype=torch.bool, device=A.device)
    # thr = torch.quantile(A[mask], 0.85)
    # A[A < thr] = 0.0
    
    # A 的热力图可视化（仅 heatmap）
    # visualize_adjacency_heatmap(
    #     A, save_path=f"{visualize_path_prefix}_heatmap.png",
    #     title="Batch Laplacian (Event+Background) Adjacency A", annotate=(B <= 32),
    # )

    # ============================================================
    # 5. 加自环 + 归一化
    #    输出保持和你现有代码一致: D^{-1/2}(A_hat)D^{-1/2}
    # ============================================================
    I = torch.eye(B, device=device)
    A_hat = A + self_loop_weight * I

    D_hat_diag = torch.sum(A_hat, dim=1)
    D_inv_sqrt = torch.pow(D_hat_diag + eps, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_mat = torch.diag(D_inv_sqrt)

    L = torch.mm(torch.mm(D_mat, A_hat), D_mat)

    if return_adjacency:
        return L, A, S_total, S_event, S_bg
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
    
    # 可视化
    # visualize_adjacency_heatmap(
    #     A, save_path=f"ssm_batch_laplacian_cosine_heatmap.png",
    #     title="Batch Laplacian (Cosine Similarity) Adjacency A", annotate=(B <= 32),
    # )
    
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

def construct_batch_laplacian_euclidean_rbf(features, sigma=15.0):
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
    # print(torch.median(dist))
    
    # 2. 构建邻接矩阵 A (RBF Kernel)
    # A_ij = exp(-dist^2 / sigma^2)
    A = torch.exp(-(dist ** 2) / (sigma ** 2))
    
    # 可视化
    # visualize_adjacency_heatmap(
    #     A, save_path=f"ssm_batch_laplacian_rbf_heatmap.png",
    #     title="Batch Laplacian (RBF) Adjacency A", annotate=(features.size(0) <= 32),
    # )
    
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
        
        # self.stem = InternalSplitStem(
        #     total_bands_per_step=41, # 41
        #     seq_len=10,                     # 10
        #     hidden_dim=hidden_dim,
        #     static_indices=[0, 13, 14, 15, 16, 17, 18, 19, 20, 37, 38, 39, 40]
        # )
        
        self.spectral_blocks = nn.Sequential(
            *[SpatialSpectralMambaBlock(dim=hidden_dim, patch_size=patch_size) for _ in range(2)]
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
        x = self.stem(x)  # [B, hidden_dim, H, W]
        
        B, C, H, W = x.shape
        # x = x.reshape(B, C, H * W).reshape(B, 38, 10, H*W).reshape(B*38, 10, H*W)
        # x = self.temporal_mamba(x)
        # x = x.reshape(B, 38, 10, H, W).reshape(B, 380, H, W)
        
        x = x.reshape(B, C, H * W).reshape(B, 38, 10, H*W).permute(0, 2, 1, 3).reshape(B*10, 38, H*W)
        x = self.spectral_mamba(x) + x
        x = x.reshape(B, 10, 38, H, W).permute(0, 2, 1, 3, 4).reshape(B, 380, H, W)
        
        # x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # x = self.spatial_mamba(x) + x
        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
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
            nn.Linear(d_model*3, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(d_model, pred_len)
        )
        
        # self.graph_layer = UMAPSpectralGraphLayer(
        #     builder_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10.joblib",
        #     library_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10_libraries_by_label.npz",
        #     seq_len=seq_len,
        #     n_channels=38,
        # )
        
        self.graph_layer = UMAPAnchorLaplacian(
            umap_model_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10.joblib",
            library_path="./spectral_lib/spectral_lib_checkpoints/umap_dps_seq10_libraries_by_label.npz",
            n_input_channels=38,
            hidden_dim=32,
            d_model=380,        # 与主干网络输出一致
            adapt_dim=32,
            temperature=1.0,
            lambda_anchor=0.01,
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
        
        # year_norm, month_sin, month_cos, day_sin, day_cos, weekday_sin, weekday_cos, doy_norm
        
        doy = (x_mark_enc[:, :, -1]*365.0+1.0).round().clamp(1, 366).long()
        x_t = self.PositionalEncoding(doy)  # (B, T, d_model)
        x_t = x_t * self.PE_Weights
        
        
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
        # pdb.set_trace()
        x_enc = x_enc.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)
        # 编码器时间 mark 的步数应等于输入 x_enc 的 T；由 adapter.create_time_marks(adapter.seq_len) 决定，

        
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
        
        # laplacian_manifold, graph_loss, sparse_map = self.graph_layer(x_enc[:, :, H//2, W//2])  # (B, B)
        
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
        hidden_features, spatial_features = self.model(x_enc)  # (B, d_model), (B, d_model, H, W)
        hidden_features = hidden_features + x_t
        laplacian = construct_batch_laplacian(burn_history)
        laplacian_manifold, graph_loss, sparse_map = self.graph_layer(x_enc[:, :, H//2, W//2], hidden_features, temperature=1)  # (B, B)

        out = self.gcn_stream(hidden_features, laplacian)  # (B, d_model)
        
        out2 = self.gcn_stream_manifold(hidden_features, laplacian_manifold)  # (B, d_model)
        
        # stacked_feats = torch.stack([hidden_features, out, out2], dim=1)
        # # 为了防止特征维度混淆，可以用一个小型网络给每个分支打分
        # scores = self.attn_net(stacked_feats) # [B, 3, 1]
        # # Softmax 归一化权重
        # weights = torch.softmax(scores, dim=1)
        # fused_features = torch.sum(weights * stacked_feats, dim=1) # [B, hidden_dim]
        # out = self.fc_out(hidden_features)
        
        # out = self.fc_out(0.25*out + 0.75* hidden_features)
        out = self.fc_out(torch.cat([out, out2, hidden_features], dim=1))
        # out = self.fc_out(out)
        return out


if __name__ == "__main__":
    model = Model(configs=None).to('cuda')
    x = torch.randn(128, 380, 13, 13).to('cuda')
    print(model(x, x, x, x))
