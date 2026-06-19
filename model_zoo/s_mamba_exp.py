"""
Is Mamba Effective for Time Series Forecasting?
"""
import torch
import torch.nn as nn
import os
import sys

# Add project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from time_series_models.layers.Mamba_EncDec import Encoder, EncoderLayer
from time_series_models.layers.Embed import DataEmbedding_inverted
import datetime
import numpy as np

from mamba_ssm import Mamba
import torch.nn.functional as F

class Configs:
    def __init__(self, seq_len=10, pred_len=7, d_model=256, d_state=256, d_ff=2048, 
                 e_layers=5, dropout=0.1, activation='relu', output_attention=False,
                 use_norm=False, embed='timeF', freq='d'):
        # Model basic parameters
        self.seq_len = seq_len  # Input sequence length
        self.pred_len = pred_len  # Prediction length
        self.d_model = d_model  # Model dimension
        self.d_state = d_state  # SSM state expansion factor
        self.d_ff = d_ff   # Feed-forward network dimension
        
        # Model structure parameters
        self.e_layers = e_layers  # Number of encoder layers
        self.dropout = dropout  # Dropout rate
        self.activation = activation  # Activation function
        
        # Other parameters
        self.output_attention = output_attention  # Whether to output attention weights
        self.use_norm = use_norm  # Whether to use normalization
        self.embed = embed  # Embedding type
        self.freq = freq  # Frequency

class MixedEmbedding(nn.Module):
    """
    混合连续 + 类别特征的嵌入模块（面向 LULC 类别）。
    
    典型输入:
        x: [B, T, C], 其中最后一列是 LULC 类别 (原始标签值，如 1..17)
           其余 C-1 列为连续特征。
           
    返回:
        y: [B, T, (C-1) + lulc_emb_dim]
    
    关键特性:
    - 只处理有效的 LULC 类别（1..17），假设输入数据中没有无效值。
    """
    def __init__(
        self,
        valid_lulc_ids=None,           # e.g. list(range(1, 18)) → {1..17}
        lulc_emb_dim: int = 2,         # LULC 类别的嵌入维度
        lulc_col: int = -1,            # LULC 位于输入最后一列
        round_float_ids: bool = True   # 如果输入是浮点型，先四舍五入
    ):
        super().__init__()
        # 有效类别集合（默认 1..17）
        if valid_lulc_ids is None:
            valid_lulc_ids = list(range(1, 18))
        valid_lulc_ids = [int(v) for v in valid_lulc_ids]
        assert len(valid_lulc_ids) >= 1, "valid_lulc_ids 不能为空"

        self.valid_lulc_ids = sorted(valid_lulc_ids)
        self.lulc_col = int(lulc_col)
        self.round_float_ids = bool(round_float_ids)

        # --- 构建"原始ID → 嵌入索引"的映射 ---
        # LULC 实际值（1-17）映射到 embedding 索引（0-16）
        # 例如：LULC=1 → embedding_index=0, LULC=2 → embedding_index=1, ..., LULC=17 → embedding_index=16
        id2idx = {}
        for i, raw_id in enumerate(self.valid_lulc_ids):
            id2idx[raw_id] = i  # raw_id (1-17) -> embedding index (0-16)
        self.id2idx = id2idx  # python 字典，forward 里逐值替换（类别数量很少，循环成本低）

        num_embeddings = len(self.valid_lulc_ids)  # 17个embedding，索引0-16
        self.lulc_emb = nn.Embedding(num_embeddings, lulc_emb_dim)

    def _sanitize_and_map_ids(self, lulc_ids: torch.Tensor) -> torch.Tensor:
        """
        将输入的原始 LULC 值映射到连续的 embedding 索引空间。
        - 浮点数先四舍五入
        - 假设所有值都在 valid_lulc_ids 范围内
        """
        if lulc_ids.dtype.is_floating_point and self.round_float_ids:
            lulc_ids = torch.round(lulc_ids)
        lulc_ids = lulc_ids.long()

        # 初始化映射结果
        mapped = torch.zeros_like(lulc_ids)

        # 合法值按映射表替换
        for raw_id, emb_idx in self.id2idx.items():
            mapped[lulc_ids == raw_id] = emb_idx

        # 检查是否还有未映射的值（应该都在 valid_lulc_ids 中）
        # 检查所有值是否都在映射表中
        all_mapped = True
        unique_vals = lulc_ids.unique()
        for val in unique_vals:
            val_int = int(val.item())
            if val_int not in self.id2idx:
                all_mapped = False
                break
        
        if not all_mapped:
            unknown_vals = [int(v.item()) for v in unique_vals if int(v.item()) not in self.id2idx]
            raise ValueError(
                f"发现未映射的 LULC 值 {unknown_vals}；"
                f"请将这些值加入 valid_lulc_ids。当前有效值: {self.valid_lulc_ids}"
            )

        return mapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C]，最后一列为 LULC 类别
        """
        if x.ndim != 3:
            raise ValueError("输入张量必须是 [B, T, C]")

        # 分离连续特征与 LULC
        x_cont = x[..., :self.lulc_col] if self.lulc_col != -1 else x[..., :-1]
        lulc_raw = x[..., self.lulc_col]

        # 将原始 LULC 值映射到 embedding 索引
        lulc_mapped = self._sanitize_and_map_ids(lulc_raw)

        # 嵌入并拼接回连续特征
        e_lulc = self.lulc_emb(lulc_mapped)  # [B, T, lulc_emb_dim]
        return torch.cat([x_cont.float(), e_lulc], dim=-1)

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding - first parameter should be sequence length
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout, time_feat_dim=8)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.projector = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.BatchNorm1d(configs.d_model),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model, configs.pred_len),
            )
        self.lulc_embedding = MixedEmbedding(valid_lulc_ids=list(range(1, 18)), lulc_emb_dim=4, lulc_col=-1, round_float_ids=True)


    def forecast(self, x_enc, x_mark_enc):

        B, T, N, H, W = x_enc.shape
        x_enc = x_enc[:, :, :, H//2, W//2].squeeze() # B L N
        
        # x_enc = x_enc.clone()
        # if torch.min(x_enc[:, :, -1]) < 1:
        #     x_enc = x_enc.clone()
        #     x_enc[:, :, -1] = x_enc[:, :, -1] * 17.0
        # x_enc = self.lulc_embedding(x_enc)
        
        # x_mark_enc = None
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        fire_token = enc_out[:, 0, :]                           # [B, d_model]，Fire 是变量 0
        dec_out = self.projector(fire_token)                    # [B, pred_len]

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, burn_history=None):
        # x_enc: [B, L, D] where L=10 represents data from the previous 10 days
        # target_date: [B] list of date strings in yyyymmdd format
        
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out # [:, -self.pred_len:, :][:, :, 0]  # [B, L, N]
    
    
if __name__ == '__main__':
    configs = Configs(
    seq_len=10,
    pred_len=7,
    d_model=39,
    d_state=16,
    d_ff=256,
    e_layers=2,
    dropout=0.1,
)
    # Create model and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    
    # Test data
    batch_size = 32
    x_enc = torch.randn(batch_size, configs.seq_len, configs.d_model).to(device)  # [32, 10, 39]
    target_date = ['20010829'] * batch_size  # Example date
    
    # Forward propagation test
    output = model(x_enc, target_date)
    print(f"Input shape: {x_enc.shape}")
    print(f"Output shape: {output.shape}")
    # print(f"Model structure:\n{model}")