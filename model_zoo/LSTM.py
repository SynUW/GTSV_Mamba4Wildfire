"""
LSTM forecasting model — operates on the TIME axis (fixed).

Bug fix vs. previous version: the previous code piped variables through
`DataEmbedding_inverted` which collapses the time axis with a Linear(seq_len -> d_model)
and produces tokens shaped [B, N, D]. The LSTM then rolled over N (variables),
not over time, so temporal dynamics were never modeled. This rewrite runs the
LSTM directly over the time axis on the center-pixel time series.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class Configs:
    def __init__(self, config=None):
        if config is not None:
            self.seq_len = getattr(config, 'seq_len', 365)
            self.pred_len = getattr(config, 'pred_len', 7)
            self.enc_in = getattr(config, 'enc_in', 40)
            self.d_model = getattr(config, 'd_model', 256)
            self.hidden_size = getattr(config, 'hidden_size', 128)
            self.num_layers = getattr(config, 'num_layers', 2)
            self.dropout = getattr(config, 'dropout', 0.1)
            self.bidirectional = getattr(config, 'bidirectional', True)
            self.use_norm = getattr(config, 'use_norm', True)
        else:
            self.seq_len = 365
            self.pred_len = 7
            self.enc_in = 40
            self.d_model = 256
            self.hidden_size = 128
            self.num_layers = 2
            self.dropout = 0.1
            self.bidirectional = True
            self.use_norm = True


class Model(nn.Module):
    """LSTM over the temporal axis with optional self-attention head."""

    def __init__(self, configs):
        super(Model, self).__init__()

        if hasattr(configs, 'model_name'):
            configs = Configs(configs)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Per-timestep feature projection: C -> d_model
        self.input_proj = nn.Linear(configs.enc_in, configs.d_model)

        self.lstm = nn.LSTM(
            input_size=configs.d_model,
            hidden_size=configs.hidden_size,
            num_layers=configs.num_layers,
            dropout=configs.dropout if configs.num_layers > 1 else 0.0,
            bidirectional=configs.bidirectional,
            batch_first=True,
        )

        lstm_out_dim = configs.hidden_size * (2 if configs.bidirectional else 1)

        self.attn = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=8,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(configs.dropout)

        # Project final temporal context to pred_len
        self.projector = nn.Linear(lstm_out_dim, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc):
        # x_enc: [B, T, C, H, W] -> center pixel [B, T, C]
        B, T, C, H, W = x_enc.shape
        x = x_enc[:, :, :, H // 2, W // 2]
        x = torch.where(x == -9999, torch.full_like(x, 0.5), x)

        x = self.input_proj(x)                       # [B, T, d_model]
        lstm_out, _ = self.lstm(x)                   # [B, T, lstm_out_dim]

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        h = self.attn_norm(lstm_out + attn_out)
        h = self.dropout(h)

        last = h[:, -1, :]                           # last-step context
        return self.projector(last)                  # [B, pred_len]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, burn_history=None):
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
