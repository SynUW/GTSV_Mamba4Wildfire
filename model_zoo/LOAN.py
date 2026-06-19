import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.manual_seed(0)
import math
from torch.autograd import Variable

np.set_printoptions(suppress=True, precision=2)


class Conv3D_Block(nn.Module):
    """ 3D Conv Block used for the dynamic branch """

    def __init__(self, in_channels: int = 18, out_channels: int = 32, kernel_size: int = (3, 3, 3),
                 stride: int = (1, 1, 1), padding: int = (1, 1, 1), bias: bool = True, norm: bool = True):

        super(Conv3D_Block, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 18)
            number of input channels
        out_channels : int (default 32)
            number of output channels
        kernel_size : int (default 3)
            kernel size of output channels 
        stride : int (default 1)
            stride of convolution filters
        padding : int (default 1)
            padding for input image
        bias : bool (default True)
            option to use bias
        norm : bool (default True)
            option to do normalization after convolution     
        """

        if norm:
            self.Block = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

        else:
            self.Block = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                # nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor):

        """ input tensor x [N, K, D, W, H] """

        return self.Block(x)


class Conv2D_Block(nn.Module):
    """ 2D Conv Block used for the dynamic branch """

    def __init__(self, in_channels: int = 18, out_channels: int = 32, kernel_size: int = (3, 3),
                 stride: int = (1, 1), padding: int = (1, 1), bias: bool = True, norm: bool = True):

        super(Conv2D_Block, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 18)
            number of input channels
        out_channels : int (default 32)
            number of output channels
        kernel_size : int (default 3)
            kernel size of output channels 
        stride : int (default 1)
            stride of convolution filters
        padding : int (default 1)
            padding for input image
        bias : bool (default True)
            option to use bias
        norm : bool (default True)
            option to do normalization after convolution     
        """

        if norm:
            self.Block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.Block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                # nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor):

        """ input tensor x [N, K, W, H] """

        return self.Block(x)


class LOAN(nn.Module):

    """ Location-aware Adaptive Normalization layer """

    def __init__(self, in_channels: int, cond_channels: int, free_norm: str = 'batch',
                 kernel_size: int = 3, norm: bool = True):
        super(LOAN, self).__init__()

        """
        Parameters
        ----------
        in_channels : int
            number of input channels
        cond_channels : int
            number of channels for conditional map
        free_norm : int (default batch)
            type of normalization to be used for the modulated map
        kernel_size : int (default 3)
            kernel size of output channels 
        norm : bool (default True)
            option to do normalization of the modulated map
        """

        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.kernel_size = kernel_size
        self.norm = norm
        self.k_channels = cond_channels

        if norm:
            if free_norm == 'batch':
                self.free_norm = nn.BatchNorm3d(self.in_channels, affine=False)
            else:
                raise ValueError('%s is not a recognized free_norm type in SPADE' % free_norm)

        # self.mlp = nn.Sequential(
        #    nn.Conv2d(in_channels=self.cond_channels, out_channels=self.k_channels, kernel_size=self.kernel_size,
        #             padding=self.kernel_size//2, padding_mode='replicate'),
        #    nn.ReLU(inplace=True)
        #    )

        # projection layers
        self.mlp_gamma = nn.Conv2d(self.k_channels, self.in_channels, kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.mlp_beta = nn.Conv2d(self.k_channels, self.in_channels, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2)

        # initialize projection layers
        # self.mlp.apply(self.init_weights)
        self.mlp_beta.apply(self.init_weights)
        self.mlp_gamma.apply(self.init_weights)

        # normalization for the conditional map
        self.free_norm_cond = torch.nn.BatchNorm2d(cond_channels, affine=False)

    def init_weights(self, m):
        # classname = m.__class__.__name__
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    def generate_one_hot(self, labels: torch.Tensor):

        """
        Convert the semantic map into one-hot encoded
        This method can be used for the CORINE land cover data_m
        """

        con_map = torch.nn.functional.one_hot(labels, num_classes=10)
        con_map = torch.permute(con_map, (0, 3, 2, 1))
        return con_map.float()

    def forward(self, x: torch.Tensor, con_map: torch.Tensor):

        """
        input tensor x [N, K, D, W, H]
        conditional map tensor con_map [N, K, W, H]
        """

        # parameter-free normalized map
        if self.norm:
            normalized = self.free_norm(x)
        else:
            normalized = x

        # used for data_m
        # con_map = self.generate_one_hot(con_map)
        # con_map = con_map.float()

        # produce scaling and bias conditioned on semantic map
        # con_map = F.interpolate(con_map, size=x.size()[-2:], mode='nearest')

        # normalize the conditional map
        actv = self.free_norm_cond(con_map)
        actv = nn.functional.relu(actv)

        # actv = self.mlp(con_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias after duplication along the D time dimension
        out = normalized * (1 + gamma[:, :, None, :, :]) + beta[:, :, None, :, :]

        return out

class Model(nn.Module):

    """
    A PyTorch implementation of:
    Location-aware Adaptive Normalization: A Deep Learning Approach For Wildfire Danger Forecasting
    https://arxiv.org/abs/2212.08208 - https://doi.org/10.1109/TGRS.2023.3285401

    CNN, the model is hard codded
    """

    # 默认静态变量通道索引（与论文/数据约定一致）
    DEFAULT_STATIC_CHANNELS = [13, 14, 15, 16, 17, 18, 19, 20, 37]

    def __init__(self, Config, static_channels: list = DEFAULT_STATIC_CHANNELS, total_channels: int = 38, num_frames_d: int = 10,
                 n_classes: int = 1, drop_out: float = 0.5, pe: bool = True, device: str = 'cuda'):

        super(Model, self).__init__()

        """
        Parameters
        ----------
        static_channels : list of int (default [13,14,...,20,37])
            静态变量对应的通道索引；输入 (B,T,C,H,W) 中这些通道在时间维上复制填充，取 t=0 作为静态分支输入
        total_channels : int (default 38)
            输入总通道数 C
        num_frames_d : int (default 10)
            时间步数 T
        n_classes : int (default 2)
            number of classes
        drop_out : float (default 0.5)
            dropout ratio
        pe : bool (default True)
            option to use positional encoding
        device : str (default cuda)
            device GPU or CPU
        """
        # if static_channels is None:
        #     static_channels = list(Model.DEFAULT_STATIC_CHANNELS)
        self.static_channels = sorted(static_channels)
        self.total_channels = total_channels
        self.num_frames_d = num_frames_d
        self.n_static = len(self.static_channels)
        self.n_dynamic = total_channels - self.n_static
        assert self.n_dynamic > 0, "total_channels must be greater than len(static_channels)"

        self.drop_out = drop_out
        self.n_classes = n_classes
        self.pe = pe
        self.device = device

        cond_channels = 32
        self.Block1 = Conv3D_Block(self.n_dynamic, 32, (3, 3, 3), (1, 1, 1), (0, 0, 0), True, False)
        self.loan1 = LOAN(in_channels=32, cond_channels=cond_channels, free_norm='batch', kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        cond_channels = 64
        self.Block2 = Conv3D_Block(32, 64, (3, 3, 3), (1, 1, 1), (0, 0, 0), True, False)
        self.loan2 = LOAN(in_channels=64, cond_channels=cond_channels, free_norm='batch', kernel_size=3, norm=False)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.Block3 = Conv3D_Block(64, 128, (2, 2, 2), (1, 2, 2), (0, 0, 0), True, False)
        self.GAP = nn.AdaptiveAvgPool3d((2, 1, 1))

        self.Block4 = Conv2D_Block(self.n_static, 32, (3, 3), (1, 1), (0, 0), True, False)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.Block5 = Conv2D_Block(32, 64, (3, 3), (1, 1), (0, 0), True, False)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.Block6 = Conv2D_Block(64, 128, (2, 2), (2, 2), (0, 0), True, False)
        self.GAP6 = nn.AdaptiveAvgPool2d(1)

        self.lastlayer1 = nn.Conv1d(128 * 2 + 128, 256, 1)
        self.lastlayer2 = nn.Conv1d(256, 128, 1)
        self.lastlayer3 = nn.Conv1d(128, 32, 1)
        self.lastlayer4 = nn.Conv1d(32, n_classes, 1)

        self.drop = nn.Dropout(drop_out)

        self.activation = nn.LogSoftmax(dim=1)

        if pe:
            self.PositionalEncoding = PositionalEncoder(256, 366, device).to(device)
            # PE_Weights should be used carefully i.e., with nonlinear activation/dropout otherwise they have no effect
            self.PE_Weights = torch.nn.Parameter(torch.zeros(256))

    def forward(self, x: torch.Tensor, x_t: torch.Tensor, x_mark_enc: torch.Tensor=None, x_mark_dec: torch.Tensor=None):

        """
        Input: x [N, T, C, H, W] (BTCHW)。x_t 可为 [N] 的 day-of-year，或 [N, T, 8] 的 x_mark_enc（从第 8 维 doy_norm 反算 doy）。
        """
        B, T, C, H, W = x.shape
        # day-of-year：若 x_t 为 (B,) 则直接用；若为 (B, T, 8) 则从最后一维 doy_norm 反算
        if x_t.dim() == 1:
            doy = x_t.long()  # [N]
        else:
            doy_norm = x_t[:, -1, 7]  # 最后一时间步的第 8 维 (doy_norm)
            doy = (doy_norm * 365.0 + 1).long().clamp(1, 366)
        x_t = doy

        assert C == self.total_channels, f"Expected {self.total_channels} channels, got {C}"
        assert T == self.num_frames_d, f"Expected {self.num_frames_d} time steps, got {T}"

        # 静态：取 t=0 的指定通道（静态在时间维上复制，取一帧即可）
        x_s = x[:, 0, self.static_channels, :, :]   # [N, n_static, H, W]
        # 动态：其余通道，保持 (N, T, n_dynamic, H, W) -> Conv3d 需要 (N, C, D, H, W)
        dynamic_inds = [i for i in range(C) if i not in self.static_channels]
        x_d = x[:, :, dynamic_inds, :, :]           # [N, T, n_dynamic, H, W]
        x_d = x_d.permute(0, 2, 1, 3, 4)            # [N, n_dynamic, T, H, W]

        x_d = self.Block1(x_d)
        x_s = self.Block4(x_s)

        x_d = self.loan1(x_d, x_s)

        x_s = self.bn4(x_s)
        x_s = F.relu(x_s, inplace=True)
        x_d = F.relu(x_d, inplace=True)
        x_d = self.pool1(x_d)
        x_s = self.pool4(x_s)

        x_d = self.Block2(x_d)
        x_s = self.Block5(x_s)

        x_d = self.loan2(x_d, x_s)

        x_d = F.relu(x_d, inplace=True)
        x_s = F.relu(x_s, inplace=False)
        x_d = self.pool2(x_d)
        x_s = self.pool5(x_s)

        # 小空间输入(如 13×13)经 pool 后可能变为 (D,1,1)，Block3 kernel (2,2,2) 会报错，先做自适应池化保证每维≥2
        d, h, w = x_d.shape[2], x_d.shape[3], x_d.shape[4]
        if d < 2 or h < 2 or w < 2:
            x_d = F.adaptive_avg_pool3d(x_d, (max(2, d), max(2, h), max(2, w)))
        x_d = self.Block3(x_d)
        # 静态分支：小空间(如 13×13)经 pool 后可能变为 (1,1)，Block6 kernel (2,2) 会报错，先做自适应池化保证 H,W≥2
        h_s, w_s = x_s.shape[2], x_s.shape[3]
        if h_s < 2 or w_s < 2:
            x_s = F.adaptive_avg_pool2d(x_s, (max(2, h_s), max(2, w_s)))
        x_s = self.Block6(x_s)
        x_d = F.relu(x_d, inplace=True)
        x_s = F.relu(x_s, inplace=True)

        x_d = self.GAP(x_d)
        x_d = x_d.view(-1, 128 * 2, 1)
        x_s = self.GAP6(x_s)
        x_s = x_s.view(-1, 128, 1)
        if self.pe:
            x_t = self.PositionalEncoding(x_t)
            x_t = x_t * (1 + self.PE_Weights)
            x_d = x_d + x_t.unsqueeze(-1)

        #x_d = self.drop(x_d)

        x = torch.cat((x_d, x_s), dim=1)

        x = self.lastlayer1(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.lastlayer2(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.lastlayer3(x)
        x = F.relu(x)

        x = self.lastlayer4(x)

        x = torch.squeeze(x, -1)
        # x = self.activation(x)

        return x


class PositionalEncoder(nn.Module):

    """ Positional Encoding """

    def __init__(self, d_model: int = 256, n_days: int = 366, device: str = 'cuda'):
        super(PositionalEncoder, self).__init__()

        """
        Parameters
        ----------
        d_model : int (default 256)
            number of dimensions for encoding
        n_days : int (default 366)
            number of days
        device : str (default cuda)
            device GPU or CPU
        """

        self.d_model = d_model
        self.n_days = n_days
        self.device = device

        pe = torch.zeros(n_days, d_model).to(device)

        # precompute the encoding
        for pos in range(n_days):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10 ** ((2 * (i+1)) / d_model)))

        # store in buffer for fast access
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        input day of the year x_t [N]
        """
        x = Variable(self.pe[x - 1, :], requires_grad=False).to(self.device)
        # x = Variable(self.pe, requires_grad=False).cuda()

        return x


if __name__ == '__main__':

    device = 'cuda'
    
    class Configs:
        def __init__(self):
            self.static_channels = [13, 14, 15, 16, 17, 18, 19, 20, 37]
            self.total_channels = 38
            self.num_frames_d = 10
            self.n_classes = 2
            self.drop_out = 0.5
            self.pe = True
            self.device = device

    # 输入 BTCHW，包含所有变量；静态通道 [13,14,15,16,17,18,19,20,37] 在时间维上复制填充
    B, T, C, H, W = 16, 10, 38, 13, 13
    x = torch.randn(B, T, C, H, W).to(device)
    test_t = torch.randint(1, 366, (B,)).long().to(device)

    model = Model(
        Config=Configs(),
        static_channels=[13, 14, 15, 16, 17, 18, 19, 20, 37],
        total_channels=C,
        num_frames_d=T,
        device=device,
    ).to(device)

    test = model(x, test_t)

    print(test.shape)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(model))