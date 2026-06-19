"""
Time-axis Transformer forecasting model — mirrors LSTM.py / gru.py architecture;
the only difference is nn.LSTM/nn.GRU is replaced by a Transformer encoder
stack. The encoder already provides global self-attention, so the extra
MultiheadAttention head used in LSTM/GRU is omitted (would be redundant
attention-on-attention).

NOTE: The original iTransformer design (variables-as-tokens, time collapsed
via Linear(seq_len -> d_model)) was empirically poor on this sparse-event
task. Channel 0 in this dataset is identically zero, so picking
`enc_out[:, 0, :]` as the prediction token was sampling a dead variable.
This rewrite uses time steps as tokens.
"""
import math
import os
import sys

import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from time_series_models.layers.Transformer_EncDec import Encoder, EncoderLayer
from time_series_models.layers.SelfAttention_Family import FullAttention, AttentionLayer


class Configs:
    def __init__(self, config=None):
        if config is not None:
            self.seq_len = getattr(config, 'seq_len', 365)
            self.pred_len = getattr(config, 'pred_len', 7)
            self.enc_in = getattr(config, 'enc_in', 40)
            self.d_model = getattr(config, 'd_model', 256)
            self.n_heads = getattr(config, 'n_heads', 8)
            self.e_layers = getattr(config, 'e_layers', 2)
            self.d_ff = getattr(config, 'd_ff', 512)
            self.factor = getattr(config, 'factor', 1)
            self.dropout = getattr(config, 'dropout', 0.1)
            self.activation = getattr(config, 'activation', 'gelu')
            self.output_attention = getattr(config, 'output_attention', False)
            self.use_norm = getattr(config, 'use_norm', True)
        else:
            self.seq_len = 365
            self.pred_len = 7
            self.enc_in = 40
            self.d_model = 256
            self.n_heads = 8
            self.e_layers = 2
            self.d_ff = 512
            self.factor = 1
            self.dropout = 0.1
            self.activation = 'gelu'
            self.output_attention = False
            self.use_norm = True


class _SinusoidalPositional(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class Model(nn.Module):
    """Transformer encoder over the temporal axis."""

    def __init__(self, configs):
        super(Model, self).__init__()

        if hasattr(configs, 'model_name'):
            configs = Configs(configs)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Per-timestep feature projection: C -> d_model
        self.input_proj = nn.Linear(configs.enc_in, configs.d_model)
        self.position_embedding = _SinusoidalPositional(configs.d_model)
        self.embed_dropout = nn.Dropout(configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        # Project final temporal context to pred_len
        self.projector = nn.Linear(configs.d_model, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc):
        # x_enc: [B, T, C, H, W] -> center pixel [B, T, C]
        B, T, C, H, W = x_enc.shape
        x = x_enc[:, :, :, H // 2, W // 2]
        x = torch.where(x == -9999, torch.full_like(x, 0.5), x)

        x = self.input_proj(x) + self.position_embedding(x)   # [B, T, d_model]
        x = self.embed_dropout(x)

        enc_out, _ = self.encoder(x, attn_mask=None)          # [B, T, d_model]

        last = enc_out[:, -1, :]                              # last-step context
        return self.projector(last)                           # [B, pred_len]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        return self.forecast(x_enc, x_mark_enc)


if __name__ == '__main__':
    configs = Configs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)

    B, T, C, H, W = 4, configs.seq_len, configs.enc_in, 5, 5
    x_enc = torch.randn(B, T, C, H, W).to(device)
    x_mark_enc = torch.randn(B, T, 8).to(device)
    out = model(x_enc, x_mark_enc, None, None)
    print(f"Output shape: {out.shape}  (expected [{B}, {configs.pred_len}])")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
