import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 padding=None,
                 bias=True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, input):
        x = F.pad(input.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super(CausalConv1d, self).forward(x)


class HyperConv(nn.Module):
    def __init__(self, input_size, in_ch, out_ch, kernel_size, activation=None, w_hidden_size=32):
        '''
        HyperConv implements a temporal convolution that has different convolution weights for each time step.
        :param input_size: (int) dimension of the weight generating input variable
        :param in_ch: (int) number of input channels of the temporal convolution
        :param out_ch: (int) number of output channels of the temporal convolution
        :param kernel_size: (int) kernel size of the temporal convolution
        '''
        super().__init__()
        weight_regressor_hidden_size = w_hidden_size
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        if activation is None:
            activation = nn.ReLU()
        self.weight_model = nn.Sequential(
            nn.Conv1d(input_size, weight_regressor_hidden_size, kernel_size=1),
            activation,
            nn.Conv1d(weight_regressor_hidden_size, in_ch * out_ch * kernel_size, kernel_size=1)
        )
        self.bias_model = nn.Sequential(
            nn.Conv1d(input_size, weight_regressor_hidden_size, kernel_size=1),
            activation,
            nn.Conv1d(weight_regressor_hidden_size, out_ch, kernel_size=1)
        )
        # initialize weights such that regressed weights are distributed in a suitable way for sine activations
        self.weight_model[0].weight.data.zero_()
        self.weight_model[0].bias.data.zero_()

    def forward(self, x, z):
        '''
        :param x: the input signal as a B x in_ch x T tensor
        :param z: the weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: a B x out_ch x T tensor as the result of the hyper-convolution
        '''
        B = x.shape[0]
        assert x.shape[-1] % z.shape[-1] == 0

        # linearize input by appending receptive field in channels
        start, end = 0, x.shape[-1]
        x = torch.cat([x[:, :, start:end] for i in range(self.kernel_size)], dim=1)

        # rearrange input to blocks for matrix multiplication
        x = x.permute(0,2,1).contiguous().view(x.shape[0] * z.shape[-1], x.shape[-1]//z.shape[-1], x.shape[1])

        # compute weights and bias
        weight = self.weight_model(z).view(B, self.in_ch * self.kernel_size, self.out_ch, z.shape[-1])
        weight = weight.permute(0,3,1,2).contiguous().view(B * z.shape[-1], self.in_ch * self.kernel_size, self.out_ch)
        bias = self.bias_model(z).view(B, self.out_ch, z.shape[-1])
        bias = bias.permute(0,2,1).contiguous().view(B * z.shape[-1], self.out_ch)

        # compute result of dynamic convolution
        y = torch.bmm(x, weight)
        y = y + bias[:, None, :]
        y = y.view(B, -1, self.out_ch).permute(0, 2, 1).contiguous()

        return y


class HyperConvCausal(nn.Module):
    def __init__(self, input_size, in_ch, out_ch, kernel_size, dilation=1):
        '''
        HyperConvCausal implements a temporal convolution that has different convolution weights for each time step.
        :param input_size: (int) dimension of the weight generating input variable
        :param in_ch: (int) number of input channels of the temporal convolution
        :param out_ch: (int) number of output channels of the temporal convolution
        :param kernel_size: (int) kernel size of the temporal convolution
        :param dilation: (int) dilation of the temporal convolution
        '''
        super().__init__()
        weight_regressor_hidden_size = 32
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight_model = nn.Sequential(
            nn.Conv1d(input_size, weight_regressor_hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(weight_regressor_hidden_size, in_ch * out_ch * kernel_size, kernel_size=1)
        )
        self.bias_model = nn.Sequential(
            nn.Conv1d(input_size, weight_regressor_hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(weight_regressor_hidden_size, out_ch, kernel_size=1)
        )
        # initialize weights such that regressed weights are distributed in a suitable way for sine activations
        self.weight_model[0].weight.data.zero_()
        self.weight_model[0].bias.data.zero_()
        #self.weight_model[-1].bias.data.uniform_(-np.sqrt(6.0/(self.in_ch*self.kernel_size)),
        #                                         np.sqrt(6.0/(self.in_ch*self.kernel_size)))

    def forward(self, x, z):
        '''
        :param x: the input signal as a B x in_ch x T tensor
        :param z: the weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: a B x out_ch x T tensor as the result of the hyper-convolution
        '''
        B = x.shape[0]
        assert x.shape[-1] % z.shape[-1] == 0

        # causal padding
        padding = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, [padding, 0])

        # linearize input by appending receptive field in channels
        start, end = padding, x.shape[-1]
        x = torch.cat([x[:, :, start-i*self.dilation:end-i*self.dilation] for i in range(self.kernel_size)], dim=1)

        # rearrange input to blocks for matrix multiplication
        x = x.permute(0,2,1).contiguous().view(x.shape[0] * z.shape[-1], x.shape[-1]//z.shape[-1], x.shape[1])

        # compute weights and bias
        weight = self.weight_model(z).view(B, self.in_ch * self.kernel_size, self.out_ch, z.shape[-1])
        weight = weight.permute(0,3,1,2).contiguous().view(B * z.shape[-1], self.in_ch * self.kernel_size, self.out_ch)
        bias = self.bias_model(z).view(B, self.out_ch, z.shape[-1])
        bias = bias.permute(0,2,1).contiguous().view(B * z.shape[-1], self.out_ch)

        # compute result of dynamic convolution
        y = torch.bmm(x, weight)
        y = y + bias[:, None, :]
        y = y.view(B, -1, self.out_ch).permute(0, 2, 1).contiguous()

        return y

#class HyperConvBlk(nn.Module):
#    def __init__(
#            self,
#            in_ch, out_ch, z_dim,
#            kernel_size, dilation=1,
#        ):
#        super().__init__()
#
#        self.kernel_size = kernel_size
#        self.dilation = dilation
#        self.in_ch = in_ch
#        self.out_ch = out_ch
#        self.hyp_conv = HyperConv(z_dim, in_ch, 4*in_ch, kernel_size, dilation)
#        self.out_conv = nn.Sequential(
#            nn.BatchNorm1d(4*in_ch),
#            nn.Conv1d(4*in_ch, 8*in_ch, 3, stride=1, padding=1),
#            nn.Tanh(),
#            nn.BatchNorm1d(8*in_ch),
#            nn.Conv1d(8*in_ch, out_ch, 1, stride=1, padding=0),
#            #nn.Tanh(),
#        )
#        #for i in [1,4,7]:
#        #    self.out_conv[i].weight.data.normal_(mean=0, std=0.02)
#
#
#    def forward(self, x, z):
#        assert x.shape[-1] % z.shape[-1] == 0
#        x = self.hyp_conv(x, z)
#        x = self.out_conv(x)
#        return x
#
#    def receptive_field(self):
#        return (self.kernel_size - 1) * self.dilation + 1
#
class HyperConvBlk(nn.Module):
    def __init__(
            self,
            in_ch, out_ch, z_dim,
            kernel_size, activation,
            do_hyper_conv=True,
        ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dhc = do_hyper_conv
        if do_hyper_conv:
            self.conv = HyperConv(z_dim, in_ch, out_ch, kernel_size, w_hidden_size=64, activation=nn.Tanh())
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(in_ch)
        self.ac = activation
        if not in_ch == out_ch:
            self.equalize_channels = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, z):
        if z is not None:
            assert x.shape[-1] % z.shape[-1] == 0
        y = self.bn(x)
        if self.dhc:
            y = self.conv(y, z)
        else:
            y = self.conv(y)
        if self.ac is not None:
            y = self.ac(y)
        if not self.in_ch == self.out_ch:
            x = self.equalize_channels(x)
        return x + y

class HyperConvRes(nn.Module):
    def __init__(self, in_ch, out_ch, z_dim, kernel_size, dilation=1):
        '''
        :param in_ch: (int) input channels
        :param out_ch: (int) output channels
        :param z_dim: (int) dimension of the weight-generating input
        :param kernel_size: (int) size of the filter
        :param dilation: (int) dilation
        '''
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = HyperConv(z_dim, in_ch, out_ch, kernel_size, dilation)
        self.residual = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.residual.weight.data.uniform_(-np.sqrt(6.0/out_ch), np.sqrt(6.0/out_ch))
        self.skip = nn.Conv1d(out_ch, out_ch, kernel_size=1)
        self.skip.weight.data.uniform_(-np.sqrt(6.0/out_ch), np.sqrt(6.0/out_ch))
        if not in_ch == out_ch:
            self.equalize_channels = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            self.equalize_channels.weight.data.uniform_(-np.sqrt(6.0 / in_ch), np.sqrt(6.0 / in_ch))

    def forward(self, x, z):
        '''
        :param x: input signal as a B x in_ch x T tensor
        :param z: weight-generating input as a B x z_dim x K tensor (K s.t. T is a multiple of K)
        :return: output: B x out_ch x T tensor as layer output
                 skip: B x out_ch x T tensor as skip connection output
        '''
        assert x.shape[-1] % z.shape[-1] == 0
        y = self.conv(x, z)
        y = torch.sin(y)
        # residual and skip
        residual = self.residual(y)
        if not self.in_ch == self.out_ch:
            x = self.equalize_channels(x)
        skip = self.skip(y)
        return (residual + x) / 2, skip

    def receptive_field(self):
        return (self.kernel_size - 1) * self.dilation + 1

class TransConvBlk(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TransConvBlk, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = HyperConv(z_dim, in_ch, out_ch, kernel_size, dilation)
        self.residual = nn.ConvTranspose1d(out_ch, out_ch, kernel_size=1)
        #self.residual.weight.data.uniform_(-np.sqrt(6.0/out_ch), np.sqrt(6.0/out_ch))
        self.skip = nn.ConvTranspose1d(out_ch, out_ch, kernel_size=1)
        #self.skip.weight.data.uniform_(-np.sqrt(6.0/out_ch), np.sqrt(6.0/out_ch))
        if not in_ch == out_ch:
            self.equalize_channels = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=1)
            #self.equalize_channels.weight.data.uniform_(-np.sqrt(6.0 / in_ch), np.sqrt(6.0 / in_ch))


    def forward(self, x, y):
        skp = self.skip(y)
        res = self.residual(y)
        y = torch.sin(res)
        if not self.in_ch == self.out_ch:
            x = self.equalize_channels(x)
        y = self.conv(x, z)
        return x


class WeightModel(nn.Module):
    def __init__(self, in_ch, out_ch, in_dim, out_dim, hidden_size):
        super().__init__()
        self.c_model = nn.Sequential(
            nn.Linear(in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, out_ch),
        )
        self.s_model = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, out_dim),
        )
        #mu, std = 0, 0.04
        mu, std = 0, 0.02
        for p in self.c_model.parameters():
            nn.init.normal_(p.data, mean=mu, std=std)
        for p in self.s_model.parameters():
            nn.init.normal_(p.data, mean=mu, std=std)

    def forward(self, x):
        return self.c_model(self.s_model(x).permute(0,2,1)).permute(0,2,1)

class BiasModel(nn.Module):
    def __init__(self, in_ch, out_ch, in_dim, out_dim, hidden_size):
        super().__init__()
        self.c_model = nn.Sequential(
            nn.Linear(in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, 2*in_ch),
            nn.ReLU6(),
            nn.Linear(2*in_ch, out_ch),
        )
        self.s_model = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU6(),
            nn.Linear(hidden_size, out_dim),
        )
        mu, std = 0, 0.01
        for p in self.c_model.parameters():
            nn.init.normal_(p.data, mean=mu, std=std)
        for p in self.s_model.parameters():
            nn.init.normal_(p.data, mean=mu, std=std)

    def forward(self, x):
        return self.c_model(self.s_model(x).permute(0,2,1)).permute(0,2,1)

class RNNCell(nn.Module):
    def __init__(
            self,
            input_size, hidden_size,
            in_ch, out_ch, hidden_ch, z_ch,
            layer_num=0, reverse=False,
        ):
        super().__init__()
        self.reverse = reverse
        self.wih_model = WeightModel(
            in_ch=z_ch, out_ch=in_ch if layer_num==0 else hidden_ch,
            in_dim=input_size, out_dim=hidden_ch, hidden_size=hidden_size,
        )
        self.whh_model = WeightModel(
            in_ch=z_ch, out_ch=hidden_ch,
            in_dim=input_size, out_dim=hidden_ch, hidden_size=hidden_size,
        )
        self.bh_model = BiasModel(
            in_ch=z_ch, out_ch=1,
            in_dim=input_size, out_dim=hidden_ch, hidden_size=hidden_size,
        )

    def forward(self, x, h):
        # h: (B, 1, hidden_ch)
        # x: (B, in_ch, F) --> F * (B, 1, in_ch)
        x_s = x.permute(0,2,1).split(1,dim=1)
        if self.reverse:
            x_s = reversed(x_s)
        h_s = []
        for x in x_s:
            Wx = torch.bmm(x, self.W_ih)    # (B, 1, hidden_ch)
            Wh = torch.bmm(h, self.W_hh)    # (B, 1, hidden_ch)
            h = torch.tanh(Wx + Wh + self.b_h)
            h_s += [h]
        # h: F * (B, 1, hidden_ch) --> (B, hidden_ch, F)
        h = torch.cat(h_s, dim=1).permute(0,2,1)
        if self.reverse:
            h = h.flip(dims=[-1])
        return h

    def set_weight(self, z):
        self.W_ih = self.wih_model(z)
        self.W_hh = self.whh_model(z)
        self.b_h = self.bh_model(z)

class LSTMCell(nn.Module):
    def __init__(
            self,
            input_size, hidden_size,
            in_ch, out_ch, hidden_ch, z_ch,
            layer_num=0, reverse=False,
        ):
        super().__init__()
        self.reverse = reverse
        self.wih_model = WeightModel(
            in_ch=z_ch, out_ch=in_ch if layer_num==0 else hidden_ch,
            in_dim=input_size, out_dim=4*hidden_ch, hidden_size=hidden_size,
        )
        self.whh_model = WeightModel(
            in_ch=z_ch, out_ch=hidden_ch,
            in_dim=input_size, out_dim=4*hidden_ch, hidden_size=hidden_size,
        )
        self.bh_model = BiasModel(
            in_ch=z_ch, out_ch=1,
            in_dim=input_size, out_dim=4*hidden_ch, hidden_size=hidden_size,
        )

    def forward(self, x, h, c):
        # x: (B, in_ch, F) --> F * (B, 1, in_ch)
        x_s = x.permute(0,2,1).split(1,dim=1)
        if self.reverse:
            x_s = reversed(x_s)
        h_s = []
        for x in x_s:
            Wx = torch.bmm(x, self.W_ih)
            Wh = torch.bmm(h, self.W_hh)
            gates = torch.tanh(Wx + Wh + self.b_h)
            i_gate, f_gate, cell_gate, o_gate = gates.chunk(4,-1)
            i_t = torch.sigmoid(i_gate)
            f_t = torch.sigmoid(f_gate)
            g_t = torch.tanh(cell_gate)
            o_t = torch.sigmoid(o_gate)
            c = c * f_t + i_t * g_t
            h = o_t * torch.tanh(c)
            h_s += [h]
        # h: F * (B, 1, hidden_ch) --> (B, hidden_ch, F)
        h = torch.cat(h_s, dim=1).permute(0,2,1)
        if self.reverse:
            h = h.flip(dims=[-1])
        return h, c

    def set_weight(self, z):
        self.W_ih = self.wih_model(z)
        self.W_hh = self.whh_model(z)
        self.b_h = self.bh_model(z)


class GRUCell(nn.Module):
    def __init__(
            self,
            input_size, hidden_size,
            in_ch, out_ch, hidden_ch, z_ch,
            layer_num=0, reverse=False,
        ):
        super().__init__()
        self.reverse = reverse
        self.wih_model = WeightModel(
            in_ch=z_ch, out_ch=in_ch if layer_num==0 else hidden_ch,
            in_dim=input_size, out_dim=3*hidden_ch, hidden_size=hidden_size,
        )
        self.whh_model = WeightModel(
            in_ch=z_ch, out_ch=hidden_ch,
            in_dim=input_size, out_dim=3*hidden_ch, hidden_size=hidden_size,
        )
        self.bih_model = BiasModel(
            in_ch=z_ch, out_ch=1,
            in_dim=input_size, out_dim=3*hidden_ch, hidden_size=hidden_size,
        )
        self.bhh_model = BiasModel(
            in_ch=z_ch, out_ch=1,
            in_dim=input_size, out_dim=3*hidden_ch, hidden_size=hidden_size,
        )

    def forward(self, x, h):
        # x: (B, in_ch, F) --> F * (B, 1, in_ch)
        x_s = x.permute(0,2,1).split(1,dim=1)
        if self.reverse:
            x_s = reversed(x_s)
        h_s = []
        for x in x_s:
            xt = torch.tanh(torch.bmm(x, self.W_ih) + self.b_ih)
            ht = torch.tanh(torch.bmm(h, self.W_hh) + self.b_hh)
            x_reset, x_upd, x_new = xt.chunk(3,-1)
            h_reset, h_upd, h_new = ht.chunk(3,-1)
            reset_gate = torch.sigmoid(x_reset + h_reset)
            update_gate = torch.sigmoid(x_upd + h_upd)
            new_gate = torch.tanh(x_new + (reset_gate * h_new))
            h = update_gate * h + (1 - update_gate) * new_gate
            h_s += [h]
        # h: F * (B, 1, hidden_ch) --> (B, hidden_ch, F)
        h = torch.cat(h_s, dim=1).permute(0,2,1)
        if self.reverse:
            h = h.flip(dims=[-1])
        return h

    def set_weight(self, z):
        self.W_ih = self.wih_model(z)
        self.W_hh = self.whh_model(z)
        self.b_ih = self.bih_model(z)
        self.b_hh = self.bhh_model(z)


class AdaIN(nn.Module):
    def __init__(self, dim, num_features,permute=False):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(dim, num_features*2)
        self.permute=permute

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(-1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        if self.permute:
            return ((1 + gamma) * self.norm(x.permute(0,2,1)) + beta).permute(0,2,1)
        else:
            return (1 + gamma) * self.norm(x) + beta


class FiLM(nn.Module):
    def __init__(self, dim, num_features,permute=False):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(dim, num_features*2)
        self.permute=permute

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(-1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        if self.permute:
            return ((1 + gamma) * x.permute(0,2,1) + beta).permute(0,2,1)
        else:
            return (1 + gamma) * x + beta

class FilmResBlk(nn.Module):
    def __init__(
            self,
            in_ch, out_ch, kernel_size,
            z_dim, activation,
            permute=False,
        ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        padding = kernel_size//2
        self.pre = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.Tanh(),
        )
        if permute:
            self.film = FiLM(z_dim,129,True)
        else:
            self.film = FiLM(z_dim,out_ch)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
        )
        self.ac = activation

    def forward(self, x, z):
        x = self.pre(x)
        y = self.conv(x)
        y = self.film(y,z)
        if self.ac is not None:
            y = self.ac(y)
        return x + y


