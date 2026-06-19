"""
adapted from:
https://github.com/SELGroup/MultiSPANS
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model.traffic_flow_prediction.layers.pe_layers import *
from libcity.model.traffic_flow_prediction.layers.STTransformer_layers import *
from libcity.model.traffic_flow_prediction.layers.patch_layers import *
from libcity.model.traffic_flow_prediction.layers.mask_layers import *


def construct_spatial_adj_cosine(features, k=None, epsilon=1e-6):
    """
    基于余弦相似度在空间节点上构建归一化前的邻接矩阵 A_hat = A + I。
    用于 H*W 个空间位置作为节点、特征为 [N, D] 时的图构建（参考 SSM.construct_batch_laplacian_cosine）。

    Args:
        features: [N, D] 节点特征（如 H*W 个空间点，每点 D 维）。
        k: 每节点保留的 k-NN 数量；默认 min(32, N)。
        epsilon: 未使用，保留接口兼容。

    Returns:
        A_hat: [N, N] 对称邻接 + 自环，tensor。
    """
    N, D = features.shape
    if k is None:
        k = min(32, N)
    k = min(k, N)
    features_norm = F.normalize(features, p=2, dim=1)
    S = torch.mm(features_norm, features_norm.t())
    S = F.relu(S)
    topk_vals, topk_inds = torch.topk(S, k=k, dim=1)
    A = torch.zeros_like(S, device=features.device, dtype=features.dtype)
    A.scatter_(1, topk_inds, topk_vals)
    A = (A + A.t()) / 2.0
    I = torch.eye(N, device=features.device, dtype=A.dtype)
    A_hat = A + I
    return A_hat


class Model(AbstractTrafficStateModel):
    def __init__(self, config=None, data_feature=None, in_channels=None, out_channels=None,
                 input_H=13, input_W=13, embed_dim=64, num_layers=3, num_heads=8,
                 input_window=12, output_window=1, **kwargs):
        # Fire 模式：由 in_channels / out_channels 调用，邻接在 forward 里用 H*W 空间相似度计算
        if in_channels is not None and out_channels is not None:
            nn.Module.__init__(self)
            self._init_fire(
                in_channels=in_channels, out_channels=out_channels,
                input_H=input_H, input_W=input_W, embed_dim=embed_dim,
                num_layers=num_layers, num_heads=num_heads,
                input_window=input_window, output_window=output_window, **kwargs
            )
            return
        # 标准训练只传 config、未传 data_feature 时，从 config 取参数走 fire 模式
        if data_feature is None and config is not None:
            nn.Module.__init__(self)
            in_ch = getattr(config, 'enc_in', 38)
            # 本流程预测单通道(FIRMS)，与 test 的 target 形状 (B,) 一致，固定输出 1
            out_ch = 1
            seq_len = getattr(config, 'seq_len', 10)
            pred_len = getattr(config, 'pred_len', 1)
            dev = getattr(config, 'device', torch.device('cpu'))
            self._init_fire(
                in_channels=in_ch, out_channels=out_ch,
                input_H=input_H, input_W=input_W, embed_dim=embed_dim,
                num_layers=num_layers, num_heads=num_heads,
                input_window=seq_len, output_window=pred_len,
                device=dev, **kwargs
            )
            return
        super().__init__(config, data_feature)
        self._init_libcity(config, data_feature)

    def _init_fire(self, in_channels, out_channels, input_H=13, input_W=13, embed_dim=64,
                   num_layers=3, num_heads=8, input_window=12, output_window=1, **kwargs):
        self._scaler = None
        self.adj_mx = None  # 在 forward 中由 H*W 空间相似度计算
        self.feature_dim = in_channels
        self.output_dim = out_channels
        self.num_nodes = input_H * input_W
        self._input_H, self._input_W = input_H, input_W  # fire 模式输出 reshape 用
        self.load_external = False
        self._logger = getLogger()
        self.device = kwargs.get('device', torch.device('cpu'))
        self.embed_dim = embed_dim
        self.skip_conv_flag = True
        self.residual_conv_flag = True
        self.skip_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_window = input_window
        self.output_window = output_window
        self.gconv_hop_num = kwargs.get('gconv_hop_num', 3)
        self.gconv_alpha = kwargs.get('gconv_alpha', 0.0)
        self.conv_kernels = kwargs.get('conv_kernels', [2, 3, 6, 12])
        self.conv_stride = kwargs.get('conv_stride', 1)
        self.conv_if_gc = False  # 不用图卷积编码，避免依赖预定义 adj
        self.norm_type = kwargs.get('norm_type', 'BatchNorm')
        self.att_scale = kwargs.get('att_scale', None)
        self.att_dropout = kwargs.get('att_dropout', 0.1)
        self.ffn_dropout = kwargs.get('ffn_dropout', 0.1)
        self.Spe_type = kwargs.get('Satt_pe_type', 'laplacian')
        self.Spe_learnable = kwargs.get('Spe_learnable', False)
        self.Tpe_type = kwargs.get('Tatt_pe_type', 'sincos')
        self.Tpe_learnable = kwargs.get('Tpe_learnable', False)
        self.Smask_flag = False  # 动态图时不使用预建 mask
        self.block_forward_mode = kwargs.get('block_forward_mode', 0)
        self.sstore_attn = kwargs.get('sstore_attn', False)
        self.activition_fn = nn.ReLU
        self.patchencoder = patching_conv(
            in_channel=self.feature_dim, embed_dim=self.embed_dim,
            in_seq_len=self.input_window, kernel_sizes=self.conv_kernels, stride=self.conv_stride
        )
        self.hid_seq_len = self.patchencoder.out_seq_len
        self.externalPEencoder = External_Encoding(d_model=self.embed_dim, device=self.device)
        self.nodePEencoder = S_Positional_Encoding(
            pe_type=self.Spe_type, learn_pe=self.Spe_learnable, node_num=self.num_nodes,
            d_model=self.embed_dim, device=self.device
        )
        self.tseqPEencoder = Positional_Encoding(
            pe_type=self.Tpe_type, learn_pe=self.Tpe_learnable, q_len=self.hid_seq_len,
            d_model=self.embed_dim, device=self.device
        )
        self.STencoders = nn.ModuleList([
            STBlock(
                seq_len=self.hid_seq_len, node_num=self.num_nodes, embed_dim=self.embed_dim, num_heads=self.num_heads,
                forward_mode=self.block_forward_mode, norm=self.norm_type, scale=self.att_scale,
                global_nodePE=self.nodePEencoder, global_tseqPE=self.tseqPEencoder, smask_flag=self.Smask_flag, sbias_flag=False,
                tmask_flag=False, tbias_flag=False, key_missing_mask_flag=False,
                attention_dropout=self.att_dropout, proj_dropout=self.ffn_dropout, activation_fn=self.activition_fn,
                pre_norm=False, sstore_attn=self.sstore_attn
            ) for _ in range(self.num_layers)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1)
            for _ in range(self.num_layers + 1)
        ])
        self.residual_convs = nn.ModuleList([
            nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1)
            for _ in range(self.num_layers)
        ])
        self.lineardecoder = depatching_conv(
            embed_dim=self.skip_dim, unpatch_channel=self.skip_dim // 2, out_channel=self.output_dim,
            hid_seq_len=self.hid_seq_len, out_seq_len=self.output_window
        )
        self.droput_layer = nn.Dropout(p=self.ffn_dropout)

    def _init_libcity(self, config, data_feature):
        self._scaler = self.data_feature.get('scaler')
        self.adj_mx = data_feature.get('adj_mx')
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        outfeat_dim = config.get('outfeat_dim',None)
        self.output_dim = outfeat_dim if outfeat_dim is not None else self.data_feature.get('output_dim', 1)
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        # self.ext_dim = self.data_feature.get("ext_dim", 0)
        # self.num_batches = self.data_feature.get('num_batches', 1)
        self.load_external = config.get('load_external', False)
        if self.load_external:
            self.feature_dim -= 8
        self._logger = getLogger()
        
        self.device = config.get('device', torch.device('cpu'))
        self.embed_dim = config.get('embed_dim', 64)
        self.skip_conv_flag = config.get('skip_conv_flag', True)
        self.residual_conv_flag = config.get('residual_conv_flag', True)
        self.skip_dim = config.get('skip_dim', self.embed_dim)
        self.num_layers = config.get('num_layers', 3)
        self.num_heads = config.get('num_heads', 8)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)

        self.gconv_hop_num = config.get('gconv_hop_num',3)
        self.gconv_alpha = config.get('gconv_alpha',0)

        self.conv_kernels = config.get('conv_kernels',[2,3,6,12])
        self.conv_stride = config.get('conv_stride',1)
        self.conv_if_gc = config.get('conv_if_gc',False)

        self.norm_type = config.get('norm_type','BatchNorm')
        
        self.att_scale = config.get('att_scale',None)
        self.att_dropout = config.get('att_dropout',0.1)
        self.ffn_dropout = config.get('ffn_dropout',0.1)
        self.Spe_type = config.get('Satt_pe_type','laplacian')
        self.Spe_learnable = config.get('Spe_learnable',False)
        self.Tpe_type = config.get('Tatt_pe_type','sincos')
        self.Tpe_learnable = config.get('Tpe_learnable',False)
        self.Smask_flag = config.get('Smask_flag',True)
        self.block_forward_mode = config.get('block_forward_mode',0)
        self.sstore_attn = config.get('sstore_attn',False)
        # static parameters 
        self.activition_fn = nn.ReLU

        if self.skip_conv_flag is False:
            self.skip_dim = self.embed_dim
        """
            3/28: 需要skip connection
                  需要depatch解码器/ST解码器
                  加入multi-mask机制
        """

        self.patchencoder = patching_STconv(
            in_channel=self.feature_dim , embed_dim=self.embed_dim, 
            in_seq_len=self.input_window, 
            gdep = self.gconv_hop_num, alpha = self.gconv_alpha,
            kernel_sizes=self.conv_kernels,stride=self.conv_stride,device=self.device
            )  if self.conv_if_gc else patching_conv(
            in_channel=self.feature_dim , embed_dim=self.embed_dim, 
            in_seq_len=self.input_window, kernel_sizes=self.conv_kernels,stride=self.conv_stride
            )
        self.hid_seq_len = self.patchencoder.out_seq_len
        if self.Smask_flag:
            self.infomask = Infomap_Multi_Mask_Generator(self.num_nodes,self.adj_mx)
            self.graphmask = Graph_Mask_Generator(self.num_nodes,self.adj_mx)
        self.externalPEencoder = External_Encoding(d_model=self.embed_dim, device=self.device)
        self.nodePEencoder = S_Positional_Encoding(
            pe_type=self.Spe_type, learn_pe=self.Spe_learnable, node_num=self.num_nodes, 
            d_model=self.embed_dim,device = self.device)
        self.tseqPEencoder = Positional_Encoding(
            pe_type=self.Tpe_type, learn_pe=self.Tpe_learnable, q_len=self.hid_seq_len, 
            d_model=self.embed_dim,device = self.device)
        self.STencoders = nn.ModuleList(
            [STBlock(
                seq_len=self.hid_seq_len,node_num=self.num_nodes,embed_dim=self.embed_dim,num_heads=self.num_heads,
                forward_mode=self.block_forward_mode,norm=self.norm_type,scale=self.att_scale,
                global_nodePE=self.nodePEencoder,global_tseqPE=self.tseqPEencoder,smask_flag=self.Smask_flag,sbias_flag=False,
                tmask_flag=False,tbias_flag=False,key_missing_mask_flag=False,
                attention_dropout=self.att_dropout,proj_dropout=self.ffn_dropout,activation_fn=self.activition_fn,
                pre_norm=False,sstore_attn=self.sstore_attn
            ) for _ in range(self.num_layers)]
        )
        
        if self.skip_conv_flag:
            self.skip_convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
                ) for _ in range(self.num_layers+1)
            ])

        if self.residual_conv_flag:
            self.residual_convs = nn.ModuleList([
                nn.Conv2d(
                    in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=1,
                ) for _ in range(self.num_layers)
            ])

        self.lineardecoder = depatching_conv(embed_dim=self.skip_dim, unpatch_channel=self.skip_dim//2, out_channel=self.output_dim, 
                                            hid_seq_len = self.hid_seq_len, out_seq_len=self.output_window)

        # self.lineardecoder = nn.Sequential( 
        #     # in [b,n,patch_seq_len,embed_dim] 
        #     # out [b,n,out_seq_len,b,n,out_dim]
        #     nn.Linear(self.skip_dim,self.output_dim),
        #     Permution(0,1,3,2),
        #     nn.Linear(self.hid_seq_len,self.output_window),
        #     Permution(0,1,3,2)
        # )

        self.droput_layer = nn.Dropout(p=self.ffn_dropout)

    def forward(self, batch, x_mark_enc=None, x_mark_dec=None, mask=None, burn_history=None):
        # Fire 模式：batch 可为 (B,T,C,H,W)，邻接由 H*W 空间相似度动态计算
        if self.adj_mx is None:
            if isinstance(batch, dict):
                x = batch['X'].permute(0, 2, 1, 3).contiguous()  # btnc -> bntc
            else:
                # x: (B, T, C, H, W) -> (B, N, T, C), N=H*W
                x = batch
                B, T, C, H, W = x.shape
                x = x.permute(0, 2, 3, 4, 1).reshape(B, H * W, T, C).contiguous()
            # 用空间节点特征 [N, T*C] 计算相似度邻接 A_hat
            B, N, T_in, C_in = x.shape
            feat = x[0].reshape(N, -1)  # 第一个样本的 (N, T*C)
            A_hat = construct_spatial_adj_cosine(feat, k=min(32, N))
            dense_adj_mx = A_hat.detach().cpu().numpy()
        else:
            dense_adj_mx = self.adj_mx
            if isinstance(batch, dict):
                x = batch['X'].permute(0, 2, 1, 3).contiguous()
            else:
                x = batch
                B, T, C, H, W = x.shape
                x = x.permute(0, 2, 3, 4, 1).reshape(B, H * W, T, C).contiguous()

        if self.Smask_flag:
            multimask = get_static_multihead_mask(self.num_heads,[self.infomask,self.graphmask],device=self.device)
        else:
            multimask = None
        npe = self.nodePEencoder(dense_adj_mx).reshape(1,-1,1,self.embed_dim).contiguous()
        tpe = self.tseqPEencoder().reshape(1,1,-1,self.embed_dim).contiguous()
        # 保证与输入同设备（LaplacianPE 用 numpy 计算后 .to(self.device)，self.device 可能为 cpu）
        npe = npe.to(x.device)
        tpe = tpe.to(x.device)
        if self.load_external:
            x, epe = self.externalPEencoder(x)
            npe, tpe = npe+epe, tpe+epe
        if self.conv_if_gc:
            x = self.patchencoder(x,dense_adj_mx)
        else: x = self.patchencoder(x) # [b,n,patch_seq_len,embed_dim]

        skip = self.skip_convs[-1](x.permute(0,3,2,1)) if self.skip_conv_flag else x
        if self.sstore_attn:
            for i,block in enumerate(self.STencoders):
                h,attention_score, attention_weight = block(x,dense_adj_mx, npe, tpe, sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
                skip = skip+self.skip_convs[i](h.permute(0,3,2,1)) if self.skip_conv_flag else skip+h
                x = self.residual_convs[i](x.permute(0,3,2,1)).permute(0,3,2,1)+h if self.residual_conv_flag else x+h
                if self.training is not True:
                    import time
                    t = time.localtime()
                    torch.save({'attention_score':attention_score, 'attention_weight':attention_weight},"./attn_save/{}_att.pt".format(time.strftime("%d_%H_%M_%S",t)))
        
        else:
            for i,block in enumerate(self.STencoders):
                h = block(x,dense_adj_mx, npe, tpe, sattn_mask=multimask)  # [b,n,patch_seq_len,embed_dim]
                skip = skip+self.skip_convs[i](h.permute(0,3,2,1)) if self.skip_conv_flag else skip+h
                x = self.residual_convs[i](x.permute(0,3,2,1)).permute(0,3,2,1)+h if self.residual_conv_flag else x+h
        skip = skip.permute(0,3,2,1) if self.skip_conv_flag else skip
        # out = torch.sum(torch.stack(skips))
        skip = self.droput_layer(skip)
        out = self.lineardecoder(skip).permute(0,2,1,3).contiguous()  # (B, T_out, N, C_out)
        # Fire 模式：去掉空间维，对 N 做全局平均池化 -> (B, T_out, C_out)
        if self.adj_mx is None and hasattr(self, '_input_H'):
            out = out.mean(dim=2)  # (B, T_out, C_out)
        # 损失函数要求 predictions 为 (B, C)，保证 2D
        out = out.reshape(out.size(0), -1)
        return out
       
    # def calculate_loss(self, batch):
    #     y_true = batch['y']
    #     y_predicted = self.predict(batch)
    #     y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
    #     y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
    #     return loss.masked_mae_torch(y_predicted, y_true)

    def predict(self, batch):
        return self.forward(batch)

if __name__ == "__main__":
    B, T, C, H, W = 16, 10, 38, 13, 13
    model = Model(
        in_channels=C, out_channels=1,
        input_window=T, output_window=1,
        input_H=H, input_W=W,
    )
    x = torch.randn(B, T, C, H, W)
    y = model(x)
    print(y.shape)