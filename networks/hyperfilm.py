import torch
import torch.nn as nn
from networks.blocks import *
import numpy as np

class HyperFiLM(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            cnn_layers,
            rnn_layers,
            batch_size,
            condition,
            film_dim,
        ):
        super().__init__()
        self.condition = condition
        if cnn_layers > 0:
            if condition=='hyper':
                con_ch = 3*in_ch
                self.cond = HyperConv(
                    input_size=15,
                    in_ch=con_ch,
                    out_ch=1,
                    kernel_size=1,
                )
            elif condition=='film':
                con_ch = 3 + 3*in_ch
                _cond = []
                self.cond_layers = 3
                for n in range(self.cond_layers):
                    _cond.append(FilmResBlk(
                        in_ch=con_ch,
                        out_ch=1 if n==self.cond_layers-1 else con_ch,
                        kernel_size=1,
                        #activation=torch.sin,
                        activation=torch.tanh,
                        z_dim=12,
                    ))
                self.cond = nn.ModuleList(_cond)
            elif condition=='none':
                self.cond = nn.Conv1d(15+3*in_ch,1,1,stride=1,padding=0)

        _cnn = []
        self.cnn_layers = cnn_layers
        hidden_ch = 4*in_ch
        for i in range(cnn_layers):
            _cnn.append(FilmResBlk(
                in_ch=in_ch if i==0 else hidden_ch,
                out_ch=1 if rnn_layers==0 and i==cnn_layers-1 else hidden_ch,
                kernel_size=3,
                z_dim=129,
                activation=nn.Tanh(),
                permute=True if film_dim=='freq' else False,
            ))
        self.cnn = nn.ModuleList(_cnn)
        self.rnn_layers = rnn_layers
        if self.rnn_layers > 0:
            self.rnn = nn.GRU(
                input_size=hidden_ch,
                hidden_size=out_ch,
                num_layers=rnn_layers,
                bidirectional=True,
                batch_first=True,
            )
            self.out = nn.Conv1d(2,1,1,stride=1,padding=0,bias=False)
        self.linear = nn.Conv1d(in_ch, out_ch, 1, stride=1, padding=0, bias=False)

    def forward(self, x, p_s, p_t, a):
        lens = x.shape[-1]
        if self.cnn_layers > 0:
            if self.condition=='hyper':
                p_s[:,:,0::3] = p_s[:,:,0::3] - p_t[:,:,0::3]
                p_s[:,:,1::3] = p_s[:,:,1::3] - p_t[:,:,1::3]
                p_s[:,:,2::3] = p_s[:,:,2::3] - p_t[:,:,2::3]
                z = self.sin_enc(p_s, lens)
                print(p_t.shape)
                print(a.shape)
                a = self.sin_enc(torch.cat((p_t, a),dim=-1), lens)
                print("...",a.shape)
                z = self.cond(z, a)
            elif self.condition=='film':
                z = self.sin_enc(torch.cat((p_s,p_t),dim=-1), lens)
                for n in range(self.cond_layers):
                    z = self.cond[n](z, a)
            elif self.condition=='none':
                z = self.sin_enc(torch.cat((p_s,p_t,a),dim=-1), lens)
                z = self.cond(z)

        o = self.linear(x)
        for n in range(self.cnn_layers):
            x = self.cnn[n](x, z)
        if self.rnn_layers > 0:
            x, _ = self.rnn(x.permute(0,2,1))
            x = self.out(x.permute(0,2,1))
        if self.rnn_layers==0 and self.cnn_layers==0:
            x = 0
        return o + x, x

    def sin_enc(self, encs, lens=64, eps=1e-5):
        encs = encs.permute(0,2,1)    # (B, C, 1)
        encs = encs.repeat(1,1,lens)
        w = 2**((torch.ones_like(encs).cumsum(-1)-1) / lens)
        encs[:,:,0::2] = torch.sin(w[:,:,0::2]*np.pi*encs[:,:,0::2])
        encs[:,:,1::2] = torch.cos(w[:,:,1::2]*np.pi*encs[:,:,1::2])
        return encs


