"""
ST-Mamba (Spatio-Temporal Mamba) adapted for wildfire forecasting.

Reference: https://github.com/xian1234/ST-Mamba
Decouples static spatial context (last timestep U-Net skips) and dynamic temporal
changes (Mamba over bottleneck sequence).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sm_mamba requires mamba_ssm. Install in mamba_env or: pip install mamba-ssm"
    ) from e


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TemporalFusionModule(nn.Module):
    """Mamba-based temporal fusion over U-Net bottleneck vectors."""

    def __init__(self, in_features=512, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.temporal_model = Mamba(
            d_model=in_features,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.out_features = in_features

    def forward(self, x_seq):
        fused_seq = self.temporal_model(x_seq)
        return fused_seq[:, -1, :]


class STMamba(nn.Module):
    """
    ST-Mamba core: input (B, T, C, H, W) -> dense map (B, n_classes, H, W).
    """

    def __init__(
        self,
        n_channels=55,
        n_classes=1,
        bottleneck_h=1,
        bottleneck_w=1,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.bottleneck_h = int(bottleneck_h)
        self.bottleneck_w = int(bottleneck_w)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.temporal_fusion = TemporalFusionModule(
            in_features=512, d_state=d_state, d_conv=d_conv, expand=expand
        )
        dec_flat = 512 * self.bottleneck_h * self.bottleneck_w
        self.decoder_projection = nn.Linear(self.temporal_fusion.out_features, dec_flat)
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        bottleneck_vectors = []
        skips_last_t = {}
        for ti in range(t):
            xt = x[:, ti, :, :, :]
            s1 = self.inc(xt)
            s2 = self.down1(s1)
            s3 = self.down2(s2)
            s4 = self.down3(s3)
            v = self.pool(s4).view(b, -1)
            bottleneck_vectors.append(v)
            if ti == t - 1:
                skips_last_t["s1"] = s1
                skips_last_t["s2"] = s2
                skips_last_t["s3"] = s3

        bottleneck_sequence = torch.stack(bottleneck_vectors, dim=1)
        v_fused = self.temporal_fusion(bottleneck_sequence)
        d_start = self.decoder_projection(v_fused)
        d = d_start.view(b, 512, self.bottleneck_h, self.bottleneck_w)
        d = self.up1(d, skips_last_t["s3"])
        d = self.up2(d, skips_last_t["s2"])
        d = self.up3(d, skips_last_t["s1"])
        return self.outc(d)


def _bottleneck_spatial(img_size: int, n_down: int = 3) -> int:
    size = int(img_size)
    for _ in range(n_down):
        size = max(1, size // 2)
    return size


class Model(nn.Module):
    """
    train_all_h5 兼容包装：
      输入 x_enc: (B, seq_len, enc_in, H, W)
      输出: (B, pred_len) — 中心像素的 FIRMS logit
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(getattr(configs, "seq_len", 10))
        self.pred_len = int(getattr(configs, "pred_len", 1))
        self.enc_in = int(getattr(configs, "enc_in", 55))
        img_h = int(getattr(configs, "img_size_h", 13))
        img_w = int(getattr(configs, "img_size_w", img_h))
        self.center_row = img_h // 2
        self.center_col = img_w // 2

        bh = _bottleneck_spatial(img_h)
        bw = _bottleneck_spatial(img_w)
        self.backbone = STMamba(
            n_channels=self.enc_in,
            n_classes=1,
            bottleneck_h=bh,
            bottleneck_w=bw,
            d_state=int(getattr(configs, "d_state", 16)),
            d_conv=int(getattr(configs, "d_conv", 4)),
            expand=int(getattr(configs, "expand", 2)),
        )
        if self.pred_len > 1:
            self.time_proj = nn.Linear(1, self.pred_len)
        else:
            self.time_proj = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, burn_history=None):
        if x_enc.dim() != 5:
            raise ValueError(f"sm_mamba expects x_enc (B,T,C,H,W), got {tuple(x_enc.shape)}")

        b, t, c, h, w = x_enc.shape
        if t > self.seq_len:
            x_enc = x_enc[:, -self.seq_len :, :, :, :]

        logits_map = self.backbone(x_enc)  # (B, 1, H, W)
        center = logits_map[:, 0, self.center_row, self.center_col]  # (B,)
        out = center.unsqueeze(-1)  # (B, 1)
        if self.time_proj is not None:
            out = self.time_proj(out)
        return out


if __name__ == "__main__":
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SimpleNamespace(seq_len=10, pred_len=1, enc_in=55, img_size_h=13, img_size_w=13)
    model = Model(cfg).to(device)
    x = torch.randn(2, 10, 55, 13, 13, device=device)
    y = model(x, None, None, None)
    print("sm_mamba:", x.shape, "->", y.shape)
