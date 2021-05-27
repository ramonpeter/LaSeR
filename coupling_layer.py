import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor=None,
                 clamp=2.,
                 act_norm=1.,
                 act_norm_type='SOFTPLUS',
                 permute_soft=False,
                 dropout=0.0,
                 num_layers=2,
                 internal_size=None):

        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels - channels // 2
        self.split_len2 = channels // 2
        self.splits = [self.split_len1, self.split_len2]

        self.in_channels = channels
        self.clamp = clamp

        if not all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]):
            raise(ValueError(F"Dimensions of input and one or more conditions don't agree: {dims_c} vs {dims_in}."))

        if act_norm_type == 'SIGMOID':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: 10 * torch.sigmoid(a - 2.))
        elif act_norm_type == 'SOFTPLUS':
            act_norm = 10. * act_norm
            self.softplus = nn.Softplus(beta=0.5)
            self.actnorm_activation = (lambda a: 0.1 * self.softplus(a))
        elif act_norm_type == 'EXP':
            act_norm = np.log(act_norm)
            self.actnorm_activation = (lambda a: torch.exp(a))
        else:
            raise ValueError('Please, SIGMOID, SOFTPLUS or EXP, as actnorm type')

        assert act_norm > 0., "please, this is not allowed. don't do it. take it... and go."
        self.act_norm = nn.Parameter(torch.ones(1, self.in_channels) * float(act_norm))
        self.act_offset = nn.Parameter(torch.zeros(1, self.in_channels))
        self.act_norm_trigger = True

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels,channels))
            for i,j in enumerate(np.random.permutation(channels)):
                w[i,j] = 1.
        w_inv = w.T

        self.w = nn.Parameter(torch.FloatTensor(w).view(channels, channels),
                              requires_grad=False)
        self.w_inv = nn.Parameter(torch.FloatTensor(w_inv).view(channels, channels),
                              requires_grad=False)

        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s = subnet_constructor(num_layers, self.split_len1 + condition_length, 2 * self.split_len2, internal_size = internal_size, dropout = dropout)
        self.last_jac = None

    def log_e(self, s):
        s = self.clamp * torch.tanh(0.1 * s)
        return s

    def permute(self, x, rev=False):
        scale = self.actnorm_activation( self.act_norm)
        if rev:
            return (torch.matmul(x, self.w_inv) - self.act_offset) / scale
        else:
            return torch.matmul((x * scale + self.act_offset), self.w)

    def affine(self, x, a, rev=False):
        ch = x.shape[1]
        sub_jac = self.log_e(a[:,:ch])
        if not rev:
            return (x * torch.exp(sub_jac) + 0.1 * a[:,ch:],
                    torch.sum(sub_jac, dim=(1)))
        else:
            return ((x - 0.1 * a[:,ch:]) * torch.exp(-sub_jac),
                    -torch.sum(sub_jac, dim=(1)))

    def forward(self, x, c=[], rev=False):
        if rev:
            x = [self.permute(x[0], rev=True)]

        x1, x2 = torch.split(x[0], self.splits, dim=1)
        if not rev:
            a1 = self.s(torch.cat([x1, *c], 1) if self.conditional else x1)
            x2, j2 = self.affine(x2, a1)
        else: # names of x and y are swapped!
            a1 = self.s(torch.cat([x1, *c], 1) if self.conditional else x1)
            x2, j2 = self.affine(x2, a1, rev=True)

        self.last_jac = j2
        x_out = torch.cat((x1, x2), 1)
        n_pixels = 1
        self.last_jac += ((-1)**rev * n_pixels) * (torch.log(self.actnorm_activation(self.act_norm) + 1e-12).sum())
        if not rev:
            x_out = self.permute(x_out, rev=False)
        return [x_out]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims
