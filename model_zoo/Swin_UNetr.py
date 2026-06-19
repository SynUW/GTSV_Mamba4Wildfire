import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# window partition (2D)
# ===============================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B,
               H // window_size, window_size,
               W // window_size, window_size,
               C)
    windows = x.permute(0,1,3,2,4,5).contiguous()
    windows = windows.view(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H*W/window_size/window_size))
    x = windows.view(B,
                     H // window_size,
                     W // window_size,
                     window_size,
                     window_size,
                     -1)
    x = x.permute(0,1,3,2,4,5).contiguous()
    x = x.view(B,H,W,-1)
    return x

# ===============================
# window partition 3D (T, H, W)
# ===============================

def window_partition_3d(x, window_size):
    """x: (B, T, H, W, C) -> windows (-1, ws^3, C)"""
    B, T, H, W, C = x.shape
    ws = window_size
    x = x.view(B, T // ws, ws, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()  # B, nT, nH, nW, ws, ws, ws, C
    windows = x.view(-1, ws * ws * ws, C)
    return windows

def window_reverse_3d(windows, window_size, T, H, W):
    ws = window_size
    nT, nH, nW = T // ws, H // ws, W // ws
    B = windows.shape[0] // (nT * nH * nW)
    x = windows.view(B, nT, nH, nW, ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()  # B, nT, ws, nH, ws, nW, ws, C
    x = x.view(B, T, H, W, -1)
    return x

# ===============================
# window attention
# ===============================

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B,N,C = x.shape

        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads)
        qkv = qkv.permute(2,0,3,1,4)

        q,k,v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)

        return x

# ===============================
# Swin block (no downsample)
# ===============================

class SwinBlock(nn.Module):

    def __init__(self, dim, num_heads, window_size=3, shift=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

        self.window_size = window_size
        self.shift = shift

    def forward(self, x):

        B,C,H,W = x.shape

        shortcut = x

        x = x.permute(0,2,3,1)   # B,H,W,C
        x = self.norm1(x)

        if self.shift:
            x = torch.roll(x, shifts=(-self.window_size//2,
                                      -self.window_size//2),
                           dims=(1,2))

        windows = window_partition(x, self.window_size)

        windows = self.attn(windows)

        x = window_reverse(windows,
                           self.window_size,
                           H,
                           W)

        if self.shift:
            x = torch.roll(x,
                           shifts=(self.window_size//2,
                                   self.window_size//2),
                           dims=(1,2))

        x = x + shortcut.permute(0,2,3,1)

        x = x + self.mlp(self.norm2(x))

        x = x.permute(0,3,1,2)

        return x

# ===============================
# Swin block 3D (T, H, W)
# ===============================

class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=2, shift=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size
        self.shift = shift

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        ws = self.window_size
        shortcut = x
        x = x.permute(0, 2, 3, 4, 1)  # B, T, H, W, C
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2, -ws // 2), dims=(1, 2, 3))
        windows = window_partition_3d(x, ws)
        windows = self.attn(windows)
        x = window_reverse_3d(windows, ws, T, H, W)
        if self.shift:
            x = torch.roll(x, shifts=(ws // 2, ws // 2, ws // 2), dims=(1, 2, 3))
        x = x + shortcut.permute(0, 2, 3, 4, 1)
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 4, 1, 2, 3)  # B, C, T, H, W
        return x

# ===============================
# Mini Swin 3D (时空联合建模)
# ===============================

class Model(nn.Module):
    """3D Swin：Conv3d stem + 3D window attention，输入 (B, T, C, H, W)，输出 (B, num_classes)。"""

    def __init__(self,
                 configs,
                 in_ch=38,
                 num_classes=1,
                 embed_dim=64,
                 depth=4,
                 window_size=2):
        super().__init__()
        self.window_size = window_size
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[
            SwinBlock3D(dim=embed_dim, num_heads=4, window_size=window_size, shift=(i % 2 == 1))
            for i in range(depth)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, x_mark_enc=None, x_mark_dec=None, mask=None, burn_history=None):
        # 输入 (B, T, C, H, W) -> (B, C, T, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # B, C, T, H, W
            x = self.stem(x)
            ws = self.window_size
            pad_t = (ws - x.size(2) % ws) % ws
            pad_h = (ws - x.size(3) % ws) % ws
            pad_w = (ws - x.size(4) % ws) % ws
            if pad_t or pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
            x = self.blocks(x)
            x = self.head(x)
            return x
        # 4D (B, C, H, W) 也支持：视为 T=1
        if x.dim() == 4:
            x = x.unsqueeze(2)  # B, C, 1, H, W
        x = self.stem(x)
        ws = self.window_size
        pad_t = (ws - x.size(2) % ws) % ws
        pad_h = (ws - x.size(3) % ws) % ws
        pad_w = (ws - x.size(4) % ws) % ws
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        x = self.blocks(x)
        x = self.head(x)
        return x


# ===============================
# test
# ===============================

if __name__ == "__main__":
    B, T, C, H, W = 16, 10, 38, 13, 13
    model = Model(in_ch=C, num_classes=1)  # in_ch 与输入通道一致
    x = torch.randn(B, T, C, H, W)
    y = model(x)
    print(y.shape)  # (B, num_classes) e.g. (16, 2)
