"""
adapted from:
https://github.com/Intelligent-Computing-Lab-Panda/STAtten
"""
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

from module import *


class Model(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[6, 8, 6],
        sr_ratios=[8, 4, 2],
        T=4,
        chunk_size=2,
        pooling_stat="1111",
        spike_mode="lif",
        attn_mode="direct_xor",
        dvs_mode=False,
        TET=False,
        attention_mode="STAtten",
        pretrained=False,
        pretrained_cfg=None,
    ):
        # 兼容 train_all_h5 的 Model(config) 单参数调用：从 config 解析为整数，避免 config 被当 img_size_h 传入下游导致 Config // int 报错
        if hasattr(img_size_h, "enc_in") or hasattr(img_size_h, "seq_len"):
            config = img_size_h
            img_size_h = int(getattr(config, "img_size_h", 13))
            img_size_w = int(getattr(config, "img_size_w", 13))
            in_channels = int(getattr(config, "enc_in", 38))
            num_classes = 1
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.T = T
        self.TET = TET
        self.dvs = dvs_mode
        self.attention_mode = attention_mode

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        patch_embed = MS_SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        blocks = nn.ModuleList(
            [
                MS_Block_Conv(
                    dim=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    sr_ratio=sr_ratios,
                    attn_mode=attn_mode,
                    spike_mode=spike_mode,
                    dvs=dvs_mode,
                    layer=j,
                    attention_mode=self.attention_mode,
                    chunk_size=chunk_size
                )
                for j in range(sum(depths))
            ]
        )

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", blocks)

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:
            self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        elif spike_mode == "plif":
            self.head_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="torch"
            )
        self.head = (
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, hook=None):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, _, hook = patch_embed(x, hook=hook)
        for blk in block:
            x, _, hook = blk(x, hook=hook)

        x = x.flatten(3).mean(3)
        return x, hook

    def forward(self, x, hook=None, x_mark_enc=None, x_mark_dec=None, mask=None, burn_history=None):
        hook = {}
        input_B, input_T = x.shape[0], x.shape[1]  # 用于最后 mean 时区分时间维与 batch 维
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, hook = self.forward_features(x, hook=hook)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x.detach()

        x = self.head(x)
        if not self.TET:
            # 对时间维做 mean，保证输出 (B, 1)。下游可能是 (T,B,1) 或 (B,T,1)
            if x.size(0) == input_T:
                x = x.mean(0)
            else:
                x = x.mean(1)
        return x


@register_model
def sdt(**kwargs):
    model = Model(
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model

if __name__ == "__main__":
    B, T, C, H, W = 16, 10, 38, 13, 13
    x = torch.randn(B, T, C, H, W)
    statten = Model(in_channels=C, num_classes=1)
    y = statten(x)  # forward 只返回 logits
    print(y.shape)