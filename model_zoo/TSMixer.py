import torch.nn as nn
import torch

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.pred_len = configs.pred_len
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        
        self.projection = nn.Sequential(
              nn.Linear(configs.seq_len, configs.d_model),
              nn.BatchNorm1d(configs.d_model),
              nn.GELU(),
              nn.Dropout(configs.dropout),
              nn.Linear(configs.d_model, configs.pred_len),
          )
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        # print(torch.unique(x_enc[:, :, 0, :, :]))
        B, T, C, H, W = x_enc.shape
        # Center patch → [B, T, C]. Do not use squeeze(): batch B==1 would drop the
        # batch dim and break ResBlock (transpose(1,2) needs rank-3).
        x_enc = x_enc[:, :, :, H // 2, W // 2]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        # x_enc = x_enc.permute(0, 2, 1)[:, 0, :]
        # 取 channel 0（即 Fire 经过 ResBlock 跨变量混合后的"feature 0"），10 天历史 → [B, 10]
        fire_feat = x_enc[:, :, 0]
        enc_out = self.projection(fire_feat)
        # print(torch.unique(enc_out))
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, burn_history=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out# [:, -self.pred_len:, :][:, :, 0]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')