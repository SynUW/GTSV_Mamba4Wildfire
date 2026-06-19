import os
import io
import typing
import warnings
from urllib import request
from http import client

import numpy as np
import scipy as sp
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import PIL
    import PIL.Image
except ImportError:  # pragma: no cover
    PIL = None

# ==========================================
# CUSTOM LAYERS
# ==========================================

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.position_embedding = nn.Embedding(
            num_embeddings=num_patches, embedding_dim=projection_dim
        )

    def forward(self, patch):
        # 'patch' is used just to determine the device
        positions = torch.arange(
            0, self.num_patches, dtype=torch.long, device=patch.device
        )
        return self.position_embedding(positions)


class ClassToken(nn.Module):
    """Append a class token to an input layer."""
    def __init__(self, hidden_size):
        super(ClassToken, self).__init__()
        self.hidden_size = hidden_size
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        cls_broadcasted = self.cls.expand(batch_size, -1, -1)
        return torch.cat([cls_broadcasted, inputs], dim=1)


class AddPositionEmbs(nn.Module):
    """Adds learned positional embeddings to the inputs."""
    def __init__(self, seq_len, hidden_size):
        super(AddPositionEmbs, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, hidden_size) * 0.06)

    def forward(self, inputs):
        return inputs + self.pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, is_masked):
        super(MultiHeadSelfAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.num_heads = num_heads
        self.is_masked = is_masked
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads

        self.query_dense = nn.Linear(hidden_size, hidden_size)
        self.key_dense = nn.Linear(hidden_size, hidden_size)
        self.value_dense = nn.Linear(hidden_size, hidden_size)
        self.combine_heads = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs):
        batch_size, seq_length, _ = inputs.shape

        # Linear projections and reshape to (batch, num_heads, seq_len, proj_dim)
        q = self.query_dense(inputs).view(batch_size, seq_length, self.num_heads, self.projection_dim).transpose(1, 2)
        k = self.key_dense(inputs).view(batch_size, seq_length, self.num_heads, self.projection_dim).transpose(1, 2)
        v = self.value_dense(inputs).view(batch_size, seq_length, self.num_heads, self.projection_dim).transpose(1, 2)

        # Attention scores
        score = torch.matmul(q, k.transpose(-2, -1)) / (self.projection_dim ** 0.5)

        if self.is_masked:
            # Create a lower triangular mask
            mask = torch.tril(torch.ones((seq_length, seq_length), device=inputs.device)).view(1, 1, seq_length, seq_length)
            score = score.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(score, dim=-1)
        attention = torch.matmul(weights, v)

        # Reshape back to (batch, seq_len, hidden_size)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.combine_heads(attention)
        
        return output, weights


class TransformerBlock(nn.Module):
    """Implements a Transformer block."""
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout, is_masked):
        super(TransformerBlock, self).__init__()
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.att = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            is_masked=is_masked
        )
        self.dropout_layer = nn.Dropout(dropout)
        
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlpblock = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(approximate='none'),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        x = self.layernorm1(inputs)
        x_att, weights = self.att(x)
        x = self.dropout_layer(x_att) + inputs
        
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights


# ==========================================
# MODELS & CONFIGS
# ==========================================

CONFIG_Ti = {
    "dropout": 0.1,
    "mlp_dim": 768,
    "num_heads": 3,
    "num_layers": 12,
    "hidden_size": 192,
}

CONFIG_S = {
    "dropout": 0.1,
    "mlp_dim": 1664,
    "num_heads": 6,
    "num_layers": 12,
    "hidden_size": 384,
}

CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 768,
}

# 预设名称 -> 配置，供 Model 通过 configs.vit_size 选用
VIT_PRESETS = {
    "tiny": CONFIG_Ti,
    "small": CONFIG_S,
    "base": CONFIG_B,
    "large": CONFIG_L,
}


class ViTModel(nn.Module):
    """Build a ViT model in PyTorch."""
    def __init__(
        self,
        input_shape: tuple,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        classes: int,
        dropout: float = 0.1,
        include_top: bool = True,
        representation_size: int = None,
        return_sequence: bool = True,
        is_masked: bool = True
    ):
        super(ViTModel, self).__init__()
        self.return_sequence = return_sequence
        self.include_top = include_top
        
        # input_shape is expected to be (seq_len, feature_dim) like (10, 45)
        seq_len, feature_dim = input_shape
        
        self.proj = nn.Linear(feature_dim, hidden_size)
        self.patch_encoder = PatchEncoder(seq_len, hidden_size)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                is_masked=is_masked
            ) for _ in range(num_layers)
        ])
        
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.representation_size = representation_size
        if representation_size is not None:
            self.pre_logits = nn.Linear(hidden_size, representation_size)
            
        if self.include_top:
            head_in_features = representation_size if representation_size is not None else hidden_size
            # The original Keras code allowed an arbitrary activation, but typically 
            # PyTorch models output raw logits and use CrossEntropyLoss. 
            # I will output raw logits.
            self.head = nn.Linear(head_in_features, classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        proj = self.proj(x)
        pos_emb = self.patch_encoder(x)
        y = proj + pos_emb
        
        # Store weights if needed, though original build_model discarded them
        for block in self.blocks:
            y, _ = block(y)
            
        y = self.encoder_norm(y)
        
        if self.representation_size is not None:
            y = torch.tanh(self.pre_logits(y))
            
        if not self.return_sequence:
            # Equivalent to tf.keras.layers.Flatten()(y)
            y = y.view(y.size(0), -1)
            
        if self.include_top:
            y = self.head(y)
            
        return y


def vit_base(
    input_shape=(10, 45),
    classes=2,
    include_top=True,
    weights="imagenet21k+imagenet2012"
):
    model = ViTModel(
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
        **CONFIG_B
    )
    return model

def vit_tiny(
    input_shape=(10, 45),
    classes=2,
    include_top=True,
    weights="imagenet21k+imagenet2012"
):
    model = ViTModel(
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
        **CONFIG_Ti
    )
    return model

def vit_small(
    input_shape=(10, 45),
    classes=2,
    include_top=True,
    weights="imagenet21k+imagenet2012",
):
    model = ViTModel(
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
        **CONFIG_S
    )
    return model

def vit_tiny_custom(
    input_shape=(10, 45),
    classes=2,
    include_top=True,
    weights="imagenet21k+imagenet2012",
    num_heads=3,
    mlp_dim=768,
    num_layers=12,
    hidden_size=192,
    return_sequence=True,
    is_masked=True
):
    model = ViTModel(
        input_shape=input_shape,
        classes=classes,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=0.1,
        return_sequence=return_sequence,
        is_masked=is_masked
    )
    return model


# ==========================================
# 与 train_all_h5 兼容的 Model 包装（参考 TSMixer）
# ==========================================

class Model(nn.Module):
    """
    时序预测用 T4Fire/ViT 包装：接受 configs，forward 签名与 TSMixer 等一致。
    输入 x_enc (B, T, C, H, W)，取中心像素 (B, T, C) 送入 ViT，再投影到 (B, pred_len, c_out)。

    通过 configs.vit_size 指定 backbone 规模：'tiny' | 'small' | 'base' | 'large'（默认 'base'）。
    若 vit_size 不在上述预设中，则用 configs 的 d_model、e_layers 等做自定义规模。
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = getattr(configs, 'seq_len', 10)
        self.pred_len = getattr(configs, 'pred_len', 1)
        enc_in = getattr(configs, 'enc_in', 38)
        c_out = getattr(configs, 'c_out', 38)
        vit_size = getattr(configs, 'vit_size', 'base').lower().strip()

        self.enc_in = enc_in
        self.c_out = c_out
        self.pred_len = self.pred_len

        if vit_size in VIT_PRESETS:
            cfg = VIT_PRESETS[vit_size].copy()
            dropout = cfg.pop("dropout", 0.1)
            self.backbone = ViTModel(
                input_shape=(self.seq_len, enc_in),
                classes=2,
                include_top=False,
                representation_size=None,
                return_sequence=True,
                is_masked=True,
                dropout=dropout,
                **cfg,
            )
            hidden_size = cfg["hidden_size"]
        else:
            d_model = getattr(configs, 'd_model', 256)
            e_layers = getattr(configs, 'e_layers', 2)
            dropout = getattr(configs, 'dropout', 0.1)
            num_heads = max(1, d_model // 64)
            mlp_dim = max(d_model, d_model * 4)
            self.backbone = ViTModel(
                input_shape=(self.seq_len, enc_in),
                num_layers=e_layers,
                hidden_size=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                classes=2,
                include_top=False,
                representation_size=None,
                return_sequence=True,
                is_masked=True,
                dropout=dropout,
            )
            hidden_size = d_model

        self.channel_proj = nn.Linear(hidden_size, c_out)
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        # x_enc: (B, T, C, H, W) -> 取中心像素 (B, T, C)
        B, T, C, H, W = x_enc.shape
        x = x_enc[:, :, :, H // 2, W // 2]  # (B, T, C)
        if T != self.seq_len:
            x = x[:, -self.seq_len:, :]
        # ViT: (B, seq_len, enc_in) -> (B, seq_len, d_model)
        h = self.backbone(x)
        h = self.channel_proj(h)  # B, seq_len, c_out
        out = self.time_proj(h.transpose(1, 2))  # B, c_out, pred_len
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, 0, :]
        raise ValueError('Only forecast tasks implemented.')