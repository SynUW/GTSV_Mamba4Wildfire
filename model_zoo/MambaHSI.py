import math
import torch
from torch import nn
from mamba_ssm import Mamba


class SpeMamba(nn.Module):
    def __init__(self,channels, token_num=8, use_residual=False, group_num=38):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self,x):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_flat = self.mamba(x_flat)
        x_recon = x_flat.view(B, H, W, C_pad)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(x_recon)
        # 裁剪到原始输入通道数，确保与SpaMamba输出维度匹配
        x_proj = x_proj[:, :x.shape[1], :, :]
        if self.use_residual:
            return x + x_proj
        else:
            return x_proj


class SpaMamba(nn.Module):
    def __init__(self,channels,use_residual=False,group_num=4,use_proj=True):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
                           d_model=channels,  # Model dimension d_model
                           d_state=16,  # SSM state expansion factor
                           d_conv=4,  # Local convolution width
                           expand=2,  # Block expansion factor
                           )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.GroupNorm(group_num, channels),
                nn.SiLU()
            )

    def forward(self,x):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B,H,W,C = x_re.shape
        x_flat = x_re.view(1,-1, C)
        x_flat = self.mamba(x_flat)

        x_recon = x_flat.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon


class BothMamba(nn.Module):
    def __init__(self,channels,token_num,use_residual,group_num=4,use_att=True):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual
        if self.use_att:
            self.weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)

        self.spa_mamba = SpaMamba(channels,use_residual=use_residual,group_num=group_num)
        self.spe_mamba = SpeMamba(channels,token_num=token_num,use_residual=use_residual,group_num=group_num)

    def forward(self,x):
        spa_x = self.spa_mamba(x)
        spe_x = self.spe_mamba(x)
        if self.use_att:
            weights = self.softmax(self.weights)
            fusion_x = spa_x * weights[0] + spe_x * weights[1]
        else:
            fusion_x = spa_x + spe_x
        if self.use_residual:
            return fusion_x + x
        else:
            return fusion_x


class Model(nn.Module):
    def __init__(self,configs=None, in_channels=380, hidden_dim=380, num_classes=1, use_residual=False, mamba_type='both', token_num=4, group_num=38, use_att=True):
        super(Model, self).__init__()

        self.mamba_type = mamba_type

        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=hidden_dim,kernel_size=1,stride=1,padding=0),
                                             nn.GroupNorm(group_num,hidden_dim),
                                             nn.SiLU())
        if mamba_type == 'spa':
            self.mamba = nn.Sequential(SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        SpaMamba(hidden_dim,use_residual=use_residual,group_num=group_num),
                                        )
        elif mamba_type == 'spe':
            self.mamba = nn.Sequential(SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num),
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                        SpeMamba(hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num)
                                        )

        elif mamba_type=='both':
            self.mamba = nn.Sequential(BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

                                       BothMamba(channels=hidden_dim,token_num=token_num,use_residual=use_residual,group_num=group_num,use_att=use_att),
                                       )


        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=3, stride=2, padding=0),
                                      nn.BatchNorm2d(128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128,out_channels=num_classes,kernel_size=1,stride=1,padding=0),
                                      )

    def forward(self,x, x_mark_enc=None, x_mark_dec=None, mask=None, burn_history=None):
        B, C, T, H, W = x.shape
        x = x.reshape(B, C*T, H, W)
        x = self.patch_embedding(x)
        x = self.mamba(x)

        logits = self.cls_head(x)  # [B, num_classes, H, W]
        # 对空间维做显式汇聚；勿用 squeeze(logits)：当 B=1 且 num_classes=1 且 H=W=1 时会变成 0 维标量，
        # 再 unsqueeze(1) 会触发 IndexError。
        logits = logits.mean(dim=(-2, -1))  # [B, num_classes]

        return logits



if __name__=='__main__':
    batch, length, dim = 2, 13, 138
    x = torch.randn(batch, dim, length, length).to("cuda")
    model = Model(
        in_channels=23*6,
        hidden_dim=416,
        group_num=1,
        num_classes=11

    ).to("cuda")
    y = model(x)
    print(y.shape)
    # assert y.shape == x.shape


if __name__ == '__main__':
    import time
    try:
        from thop import profile, clever_format
        has_thop = True
    except ImportError:
        has_thop = False
        print("警告: 未安装thop库，无法计算FLOPs，请运行: pip install thop")
    
    print("=== MambaHSI 模型性能测试 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型参数 (参考 our_model.py 中的配置)
    batch_size = 1024
    num_bands = 138
    patch_size = 13
    num_classes = 11
    hidden_dim = 138
    
    # 创建模型
    model = Model(
        in_channels=num_bands,
        hidden_dim=hidden_dim,
        group_num=2,  # 138和140都能被2整除
        num_classes=num_classes
    ).to(device)
    
    # 创建输入数据
    x = torch.randn(batch_size, num_bands, patch_size, patch_size).to(device)
    print(f"输入数据形状: {x.shape}")
    
    # 计算模型参数量
    print("\n=== 模型复杂度分析 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 计算FLOPs (如果可用)
    if has_thop:
        try:
            flops, params = profile(model, inputs=(x,), verbose=False)
            flops, params = clever_format([flops, params], "%.3f")
            print(f"参数量 (thop): {params}")
            print(f"FLOPs: {flops}")
            
            # 计算GFLOPs
            flops_num = profile(model, inputs=(x,), verbose=False)[0]
            gflops = flops_num / 1e9
            print(f"GFLOPs: {gflops:.3f}")
        except Exception as e:
            print(f"FLOPs计算失败: {e}")
    
    # 测试推理速度
    print("\n=== 推理性能测试 ===")
    model.eval()
    num_runs = 10
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # 正式测试
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            output = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()
    
    inference_time = (end_time - start_time) / num_runs
    throughput = batch_size / inference_time
    
    print(f"推理时间: {inference_time:.4f} 秒/批次")
    print(f"吞吐量: {throughput:.2f} 样本/秒")
    print(f"输出形状: {output.shape}")
    
    # 内存使用情况
    if device.type == 'cuda':
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"\n=== GPU内存使用 ===")
        print(f"已分配: {memory_allocated:.2f} GB")
        print(f"已保留: {memory_reserved:.2f} GB")
    
    print("\n=== 测试完成 ===")