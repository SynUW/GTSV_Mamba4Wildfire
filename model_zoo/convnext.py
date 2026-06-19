import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """针对 2D 特征的 LayerNorm"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        # 输入格式: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # 恢复原始维度
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt 基础块 (无下采样)"""

    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        hidden_dim = dim * expansion_ratio
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return residual + x

class Model(nn.Module):
    def __init__(
            self,
            configs=None,
            in_chans=380,
            depths=[2, 2],  # 仅保留两个阶段（避免下采样后尺寸过小）
            dims=[128, 128],  # 对应通道数
            num_classes=1,  # 根据你的任务调整
            patch_size=1  # 初始卷积的步长（设为1避免尺寸变化）
    ):
        super().__init__()

        # 调整初始卷积（无下采样）
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=patch_size, padding=1),
            LayerNorm2d(dims[0])
        )

        # 仅构建不降分辨率的阶段
        self.stages = nn.ModuleList()
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # 分类头（全局平均池化适应小尺寸）
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(dims[-1]),
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x, x_mark_enc=None, x_mark_dec=None, mask=None, burn_history=None):
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)  # (B, C, H, W, T) -> MambaModel expects (B, C, H, W, T)
        x = self.stem(x)  # [B, 138, 13, 13] -> [B, 96, 13, 13]
        for stage in self.stages:
            x = stage(x)  # 保持尺寸不变
        x = self.head(x)  # [B, num_classes]
        return x


def test():
    model = Model()
    dummy = torch.randn(2, 138, 13, 13)
    print("Input shape:", dummy.shape)
    out = model(dummy)
    print("Output shape:", out.shape)  # 应为 [2, 11]


if __name__ == '__main__':
    import time
    try:
        from thop import profile, clever_format
        has_thop = True
    except ImportError:
        has_thop = False
        print("警告: 未安装thop库，无法计算FLOPs，请运行: pip install thop")
    
    print("=== ConvNext 模型性能测试 ===")
    
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
        num_classes=num_classes,
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