"""
Bidirectional classic ConvLSTM forecasting model.

Stacked ConvLSTM cells run forward and backward over the time axis (mirroring
nn.LSTM(bidirectional=True)). The two directions are concatenated along the
channel dim. We read the last temporal step, pool spatially, and project
linearly to pred_len. No upstream feature projection, no post-backbone
multi-head attention — classic ConvLSTM with bidirectional wrapping only.

Aligned with LSTM/GRU/iTransformer: configs-driven dims, output dim = pred_len,
-9999 sentinel handling, full-patch spatial input preserved (cropping to
center pixel would degenerate ConvLSTM into a plain LSTM).

ConvLSTM-specific config keys are used (`convlstm_hidden_channels`,
`convlstm_num_layers`, `convlstm_kernel_size`) so the shared training
Config's `num_layers`/`hidden_size` cannot silently shrink this model.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=pad)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=pad)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=pad)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=pad)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=pad)

    def forward(self, x, h_prev, c_prev):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
        c = f * c_prev + i * torch.tanh(self.Wxc(x) + self.Whc(h_prev))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))
        h = o * torch.tanh(c)
        return h, c


class _UniConvLSTM(nn.Module):
    """Stack of ConvLSTM cells, returns the last layer's per-step output sequence."""
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        cells = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.cell_list = nn.ModuleList(cells)

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, T, hidden, H, W]
        B, T, _, H, W = x.shape
        cur = x
        for layer_idx, cell in enumerate(self.cell_list):
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            c = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
            outs = []
            for t in range(T):
                h, c = cell(cur[:, t], h, c)
                outs.append(h)
            cur = torch.stack(outs, dim=1)
            if layer_idx < self.num_layers - 1:
                cur = self.dropout(cur.reshape(B * T, self.hidden_channels, H, W)).reshape(
                    B, T, self.hidden_channels, H, W
                )
        return cur


class BiConvLSTM(nn.Module):
    """Bidirectional ConvLSTM analogue of nn.LSTM(bidirectional=True)."""
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers,
                 dropout=0.0, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.fwd = _UniConvLSTM(input_channels, hidden_channels, kernel_size,
                                 num_layers, dropout=dropout)
        if bidirectional:
            self.bwd = _UniConvLSTM(input_channels, hidden_channels, kernel_size,
                                     num_layers, dropout=dropout)

    def forward(self, x):
        # x: [B, T, C, H, W] -> [B, T, (2*)hidden, H, W]
        fwd = self.fwd(x)
        if not self.bidirectional:
            return fwd
        bwd = self.bwd(torch.flip(x, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])
        return torch.cat([fwd, bwd], dim=2)


class Configs:
    def __init__(self, config=None):
        if config is not None:
            self.seq_len = getattr(config, 'seq_len', 365)
            self.pred_len = getattr(config, 'pred_len', 7)
            self.enc_in = getattr(config, 'enc_in', 40)
            self.hidden_channels = getattr(config, 'convlstm_hidden_channels', 128)
            self.kernel_size = getattr(config, 'convlstm_kernel_size', 3)
            self.num_layers = getattr(config, 'convlstm_num_layers', 3)
            self.dropout = getattr(config, 'dropout', 0.1)
            self.bidirectional = getattr(config, 'bidirectional', True)
            self.use_norm = getattr(config, 'use_norm', True)
        else:
            self.seq_len = 365
            self.pred_len = 7
            self.enc_in = 40
            self.hidden_channels = 128
            self.kernel_size = 3
            self.num_layers = 3
            self.dropout = 0.1
            self.bidirectional = True
            self.use_norm = True


class Model(nn.Module):
    """Bidirectional classic ConvLSTM: BiConvLSTM -> last step -> spatial pool -> Linear."""

    def __init__(self, configs):
        super().__init__()

        if hasattr(configs, 'model_name'):
            configs = Configs(configs)

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.convlstm = BiConvLSTM(
            input_channels=configs.enc_in,
            hidden_channels=configs.hidden_channels,
            kernel_size=configs.kernel_size,
            num_layers=configs.num_layers,
            dropout=configs.dropout if configs.num_layers > 1 else 0.0,
            bidirectional=configs.bidirectional,
        )

        out_dim = configs.hidden_channels * (2 if configs.bidirectional else 1)
        self.projector = nn.Linear(out_dim, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc):
        # x_enc: [B, T, C, H, W] — full patch preserved.
        x = torch.where(x_enc == -9999, torch.full_like(x_enc, 0.5), x_enc)

        seq = self.convlstm(x)                            # [B, T, (2*)hidden, H, W]
        last = seq[:, -1]                                 # [B, (2*)hidden, H, W]
        last = F.adaptive_avg_pool2d(last, (1, 1)).flatten(1)  # [B, (2*)hidden]
        return self.projector(last)                       # [B, pred_len]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                mask=None, burn_history=None):
        return self.forecast(x_enc, x_mark_enc)


if __name__ == '__main__':
    configs = Configs()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)

    B, T, C, H, W = 2, 60, configs.enc_in, 5, 5
    x_enc = torch.randn(B, T, C, H, W).to(device)
    out = model(x_enc)
    print(f"Output shape: {out.shape}  (expected [{B}, {configs.pred_len}])")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
