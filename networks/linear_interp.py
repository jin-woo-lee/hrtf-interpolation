import torch
import numpy as np

class Interpolator(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
        ):
        super().__init__()


    def forward(self, x, p):
        lens = 64 if self.dom_size==256 else 129
        if self.pos_encode:
            p = self.pos_enc(p, lens)
        if self.do_hyper_conv:
            out = self.cnn(x, p)
        else:
            out = self.cnn(x)
        out = torch.sin(out)
        out = self.bn(out)
        if self.do_hyper_rnn:
            out, _ = self.rnn(out, p)
        else:
            out = out.permute(0,2,1)
            out, _ = self.rnn(out)
            out = out.permute(0,2,1)
            out = torch.tanh(out)
        if self.op is not None:
            out = self.op(out, x)
        if self.post is not None:
            out = self.post(out)
        return out

    def pos_enc(self, pos, lens=64, eps=1e-5):
        pos = pos.permute(0,2,1)    # (B, C, 1)
        pos = pos.repeat(1,1,lens)
        w = 2**((torch.ones_like(pos).cumsum(-1)-1) / lens)
        pos[:,:,0::2] = torch.sin(w[:,:,0::2]*np.pi*pos[:,:,0::2])
        pos[:,:,1::2] = torch.cos(w[:,:,1::2]*np.pi*pos[:,:,1::2])
        return pos


