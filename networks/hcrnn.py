import torch
import torch.nn as nn
from networks.blocks import *
import numpy as np

class HyperRNN(nn.Module):
    def __init__(self, 
                 n_layers, batch_size, input_size, hidden_size,
                 in_ch, out_ch, hidden_ch, z_ch,
                 activation=None, bidirectional=True,
        ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.activation = activation
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1
        _cell_1 = []
        for n in range(n_layers):
            _cell_1.append(RNNCell(
                input_size, hidden_size,
                in_ch, out_ch, hidden_ch, z_ch,
                layer_num=n, reverse=False,
            ))
        self.cell_1 = nn.ModuleList(_cell_1)
        _h_1 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
        self.h_1 = nn.Parameter(_h_1)
        if self.bidirectional:
            _cell_2 = []
            for n in range(n_layers):
                _cell_2.append(RNNCell(
                    input_size, hidden_size,
                    in_ch, out_ch, hidden_ch, z_ch,
                    layer_num=n, reverse=True,
                ))
            self.cell_2 = nn.ModuleList(_cell_2)
            _h_2 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
            self.h_2 = nn.Parameter(_h_2)
        self.who_model = WeightModel(
            in_ch=z_ch, out_ch=D*hidden_ch,
            in_dim=input_size, out_dim=out_ch, hidden_size=hidden_size,
        )
        #self.bo_model = BiasModel(
        #    in_dim=input_size, out_dim=out_ch, hidden_size=hidden_size,
        #)

    def forward(self, x, z):
        B, C, F = x.shape

        # compute weights and bias
        # z: (B, in_ch, lens)
        W_ho = self.who_model(z)
        #b_o = self.bo_model(z)

        # recurrent computation
        # x: (B, 1, in_ch)
        # h: (B, 1, hidden_ch)
        # o: (B, out_ch, F)
        # Wx (B, 1, hidden_ch) =   xs (B, 1, in_ch)
        #                      * W_ih (B, in_ch, hidden_ch)
        # Wh (B, 1, hidden_ch) =    h (B, 1, hidden_ch)
        #                      * W_hh (B, hidden_ch, hidden_ch)
        # Wo (B, 1, out_ch) =    h (B, 1, D*hidden_ch)
        #                      * W_ho (B, D*hidden_ch, out_ch)
        h_l2r = self.layer_fwd(self.cell_1, self.h_1, x, z)
        if self.bidirectional:
            h_r2l = self.layer_fwd(self.cell_2, self.h_2, x, z)
            h_l2r = torch.cat((h_l2r, h_r2l), dim=1)
        o = []
        # h: (B, hidden_ch, F) --> F * (B, 1, in_ch)
        ho = h_l2r.permute(0,2,1).split(1,dim=1)
        for h in ho:
            #o += [torch.bmm(h, W_ho) + b_o]
            o += [torch.bmm(h, W_ho).permute(0,2,1)]
        o = torch.cat(o, dim=-1)

        if self.activation is not None:
            o = self.activation(o)
        return o, h

    def layer_fwd(self, cell, hs, input, z):
        for n in range(self.n_layers):
            cell[n].set_weight(z)
            h_out = cell[n](input, hs[n])
            #input = h_out
            input = torch.tanh(h_out)
        return h_out    # (B, hidden_ch, F)

class HyperLSTM(nn.Module):
    def __init__(self, 
                 n_layers, batch_size, input_size, hidden_size,
                 in_ch, out_ch, hidden_ch, z_ch,
                 activation=None, bidirectional=True,
        ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.activation = activation
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1
        _cell_1 = []
        for n in range(n_layers):
            _cell_1.append(LSTMCell(
                input_size, hidden_size,
                in_ch, out_ch, hidden_ch, z_ch,
                layer_num=n, reverse=False,
            ))
        self.cell_1 = nn.ModuleList(_cell_1)
        _h_1 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
        _c_1 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
        self.h_1 = nn.Parameter(_h_1)
        self.c_1 = nn.Parameter(_c_1)
        if self.bidirectional:
            _cell_2 = []
            for n in range(n_layers):
                _cell_2.append(LSTMCell(
                    input_size, hidden_size,
                    in_ch, out_ch, hidden_ch, z_ch,
                    layer_num=n, reverse=True,
                ))
            self.cell_2 = nn.ModuleList(_cell_2)
            _h_2 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
            _c_2 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
            self.h_2 = nn.Parameter(_h_2)
            self.c_2 = nn.Parameter(_c_2)
        self.who_model = WeightModel(
            in_ch=z_ch, out_ch=D*hidden_ch,
            in_dim=input_size, out_dim=out_ch, hidden_size=hidden_size,
        )
        #self.bo_model = BiasModel(
        #    in_dim=input_size, out_dim=out_ch, hidden_size=hidden_size,
        #)

    def forward(self, x, z):
        B, C, F = x.shape

        W_ho = self.who_model(z)
        #b_o = self.bo_model(z)

        h_l2r = self.layer_fwd(self.cell_1, self.h_1, self.c_1, x, z)
        if self.bidirectional:
            h_r2l = self.layer_fwd(self.cell_2, self.h_2, self.c_2, x, z)
            h_l2r = torch.cat((h_l2r, h_r2l), dim=1)
        o = []
        ho = h_l2r.permute(0,2,1).split(1,dim=1)
        for h in ho:
            #o += [torch.bmm(h, W_ho) + b_o]
            o += [torch.bmm(h, W_ho).permute(0,2,1)]
        o = torch.cat(o, dim=-1)

        if self.activation is not None:
            o = self.activation(o)
        return o, h

    def layer_fwd(self, cell, hs, cs, input, z):
        for n in range(self.n_layers):
            cell[n].set_weight(z)
            h_out, c_out = cell[n](input, hs[n], cs[n])
            #input = h_out
            input = torch.tanh(h_out)
        return h_out    # (B, hidden_ch, F)

class HyperGRU(nn.Module):
    def __init__(self, 
                 n_layers, batch_size, input_size, hidden_size,
                 in_ch, out_ch, hidden_ch, z_ch,
                 activation=None, bidirectional=True,
        ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.activation = activation
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1
        _cell_1 = []
        for n in range(n_layers):
            _cell_1.append(GRUCell(
                input_size, hidden_size,
                in_ch, out_ch, hidden_ch, z_ch,
                layer_num=n, reverse=False,
            ))
        self.cell_1 = nn.ModuleList(_cell_1)
        _h_1 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
        self.h_1 = nn.Parameter(_h_1)
        if self.bidirectional:
            _cell_2 = []
            for n in range(n_layers):
                _cell_2.append(GRUCell(
                    input_size, hidden_size,
                    in_ch, out_ch, hidden_ch, z_ch,
                    layer_num=n, reverse=True,
                ))
            self.cell_2 = nn.ModuleList(_cell_2)
            _h_2 = torch.zeros(n_layers,batch_size,1,self.hidden_ch)
            self.h_2 = nn.Parameter(_h_2)
        self.who_model = WeightModel(
            in_ch=z_ch, out_ch=D*hidden_ch,
            in_dim=input_size, out_dim=out_ch, hidden_size=hidden_size,
        )
        #self.bo_model = BiasModel(
        #    in_dim=input_size, out_dim=out_ch, hidden_size=hidden_size,
        #)

    def forward(self, x, z):
        B, C, F = x.shape
        x_seq = x.permute(0,2,1).split(1,dim=1)

        W_ho = self.who_model(z)
        #b_o = self.bo_model(z)

        h_l2r = self.layer_fwd(self.cell_1, self.h_1, x, z)
        if self.bidirectional:
            h_r2l = self.layer_fwd(self.cell_2, self.h_2, x, z)
            h_l2r = torch.cat((h_l2r, h_r2l), dim=1)

        o = []
        ho = h_l2r.permute(0,2,1).split(1,dim=1)
        for h in ho:
            #o += [torch.bmm(h, W_ho) + b_o]
            o += [torch.bmm(h, W_ho).permute(0,2,1)]
        o = torch.cat(o, dim=-1)

        if self.activation is not None:
            o = self.activation(o)
        return o, h

    def layer_fwd(self, cell, hs, input, z):
        for n in range(self.n_layers):
            cell[n].set_weight(z)
            h_out = cell[n](input, hs[n])
            #input = h_out
            input = torch.tanh(h_out)
        return h_out    # (B, hidden_ch, F)


class HyperCRNN(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            cnn_layers,
            rnn_layers,
            batch_size,
        ):
        super().__init__()
        con_ch = 3*in_ch
        self.cond = HyperConv(
            input_size=15,
            in_ch=con_ch,
            out_ch=1,
            kernel_size=1,
        )
        _cnn = []
        self.cnn_layers = cnn_layers
        hidden_ch = 4*in_ch
        for i in range(cnn_layers):
            _cnn.append(FilmResBlk(
                in_ch=in_ch if i==0 else hidden_ch,
                out_ch=hidden_ch,
                kernel_size=3,
                z_dim=129,
                activation=nn.Tanh(),
                permute=True,
            ))
        self.cnn = nn.ModuleList(_cnn)
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
        p_s[:,:,0::3] = p_s[:,:,0::3] - p_t[:,:,0::3]
        p_s[:,:,1::3] = p_s[:,:,1::3] - p_t[:,:,1::3]
        p_s[:,:,2::3] = p_s[:,:,2::3] - p_t[:,:,2::3]
        z = self.pos_enc(p_s, lens)
        a = self.pos_enc(torch.cat((p_t, a),dim=-1), lens)
        z = self.cond(z, a)
        lin = self.linear(x)
        for n in range(self.cnn_layers):
            x = self.cnn[n](x, z)
        y, _ = self.rnn(x.permute(0,2,1))
        y = self.out(y.permute(0,2,1))
        return lin + y, y

    def pos_enc(self, pos, lens=64, eps=1e-5):
        pos = pos.permute(0,2,1)    # (B, C, 1)
        pos = pos.repeat(1,1,lens)
        w = 2**((torch.ones_like(pos).cumsum(-1)-1) / lens)
        pos[:,:,0::2] = torch.sin(w[:,:,0::2]*np.pi*pos[:,:,0::2])
        pos[:,:,1::2] = torch.cos(w[:,:,1::2]*np.pi*pos[:,:,1::2])
        return pos


