# Rational Quadratic Spline code taken from https://github.com/bayesiains/nsf

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-6


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
            inputs[..., None] >= bin_locations,
            dim=-1
        ) - 1


def unconstrained_rational_quadratic_spline(inputs,
                                            unnormalized_widths,
                                            unnormalized_heights,
                                            unnormalized_derivatives,
                                            inverse=False,
                                            tails='linear',
                                            tail_bound=3.,
                                            min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == 'linear':
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError('{} tails are not implemented.'.format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    return outputs, logabsdet

def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=-1., right=1., bottom=-1., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    assert torch.min(inputs) >= left and torch.max(inputs) <= right, f'Inputs outside spline range: min {torch.min(inputs)}, max {torch.max(inputs)}'

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + unnormalized_derivatives

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet



class NeuralSplineCouplingBlock(nn.Module):
    '''bla

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.'''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, n_bins=10):
        super().__init__()

        self.channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.split_len1
        self.n_bins = n_bins

        self.widths = nn.Parameter(0.5 * torch.randn([1, self.split_len1, n_bins]) + 3)
        self.heights = nn.Parameter(0.5 * torch.randn([1, self.split_len1, n_bins]) + 3)
        self.slopes = nn.Parameter(0.25 * torch.randn([1, self.split_len1, n_bins-1]) + 1)
        self.coupling_spline_plot = None
        self.internal_spline_plot = None

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s = subnet_constructor(self.split_len1 + condition_length, self.split_len2 * (3 * self.n_bins - 1))

    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)

        if not rev:
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            s = self.s(x1_c).view([-1, self.split_len2, (3 * self.n_bins - 1)])
            widths, heights, slopes = torch.split(s, [self.n_bins, self.n_bins, self.n_bins-1], dim=-1)
            y2, jac2 = unconstrained_rational_quadratic_spline(x2, widths, heights, slopes, inverse=False)
            # y1, jac1 = unconstrained_rational_quadratic_spline(x1,
            #                                                    nn.functional.softplus(self.widths).repeat(x1.shape[0], 1, 1),
            #                                                    nn.functional.softplus(self.heights).repeat(x1.shape[0], 1, 1),
            #                                                    nn.functional.softplus(self.slopes).repeat(x1.shape[0], 1, 1),
            #                                                    inverse=False)
            y1 = x1
            jac1 = 0
            self.last_jac = jac1 + jac2 + (self.widths**2).sum() - (self.heights**2).sum() + (1000*(self.slopes - 1)**2).sum()
            return [torch.cat((y1, y2), 1)]
        else:
            y1, y2 = x1, x2
            x1, jac1 = unconstrained_rational_quadratic_spline(y1,
                                                               nn.functional.softplus(self.widths).repeat(y1.shape[0], 1, 1),
                                                               nn.functional.softplus(self.heights).repeat(y1.shape[0], 1, 1),
                                                               nn.functional.softplus(self.slopes).repeat(y1.shape[0], 1, 1),
                                                               inverse=True)
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            s = self.s(x1_c).view([-1, self.split_len2, (3 * self.n_bins - 1)])
            widths, heights, slopes = torch.split(s, [self.n_bins, self.n_bins, self.n_bins-1], dim=-1)
            x2, jac2 = unconstrained_rational_quadratic_spline(y2, widths, heights, slopes, inverse=True)
            self.last_jac = jac1 + jac2
            return [torch.cat((x1, x2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims

    def plot_coupling_spline(self, x, c=[]):
        with torch.no_grad():
            x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
            x1_c = torch.cat([x1, *c], 1) if self.conditional else x1
            s = self.s(x1_c).view([-1, self.split_len2, (3 * self.n_bins - 1)])
            widths, heights, slopes = torch.split(s, [self.n_bins, self.n_bins, self.n_bins-1], dim=-1)
            inputs = torch.Tensor(np.linspace(-3.5, 3.5, 5000)).to(x1_c.device)
            outputs, jac = unconstrained_rational_quadratic_spline(inputs,
                                                                   widths[0,0].expand(inputs.shape[0], 1, self.n_bins).squeeze(),
                                                                   heights[0,0].expand(inputs.shape[0], 1, self.n_bins).squeeze(),
                                                                   slopes[0,0].expand(inputs.shape[0], 1, self.n_bins-1).squeeze(),
                                                                   inverse=False)
            if not self.coupling_spline_plot:
                self.coupling_spline_plot, = plt.plot(inputs.data.cpu().numpy(), outputs.data.cpu().numpy())
            else:
                self.coupling_spline_plot.set_ydata(outputs.data.cpu().numpy())

    def plot_internal_spline(self):
        with torch.no_grad():
            inputs = torch.Tensor(np.linspace(-3.5, 3.5, 5000)).to(self.widths.device)
            outputs, jac = unconstrained_rational_quadratic_spline(inputs,
                                                                   nn.functional.softplus(self.widths[0,0]).expand(inputs.shape[0], 1, self.n_bins).squeeze(),
                                                                   nn.functional.softplus(self.heights[0,0]).expand(inputs.shape[0], 1, self.n_bins).squeeze(),
                                                                   nn.functional.softplus(self.slopes[0,0]).expand(inputs.shape[0], 1, self.n_bins-1).squeeze(),
                                                                   inverse=False)
            print(self.widths.grad[0,0].cpu().numpy())
            print(self.heights.grad[0,0].cpu().numpy())
            print(self.slopes.grad[0,0].cpu().numpy())
            if not self.internal_spline_plot:
                self.internal_spline_plot, = plt.plot(inputs.data.cpu().numpy(), outputs.data.cpu().numpy())
            else:
                self.internal_spline_plot.set_ydata(outputs.data.cpu().numpy())

if __name__ == '__main__':
    from time import time

    # n_points = 10000
    # n_bins = 10
    # inputs = torch.Tensor(np.linspace(-3.5, 3.5, n_points))
    # unnormalized_widths = torch.Tensor(np.tile(np.random.rand(n_bins) + 0.2, [n_points,1]))
    # unnormalized_heights = torch.Tensor(np.tile(np.random.rand(n_bins) + 0.2, [n_points,1]))
    # unnormalized_derivatives = torch.Tensor(np.tile([np.sin(a)/np.cos(a) for a in np.random.rand(n_bins-1) * np.pi/2], [n_points,1]))
    # t0 = time()
    # outputs, logabsdet = unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False)
    # t1 = time()
    # inverse, logabsdetinv = unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=True)
    # t2 = time()
    # print(t1 - t0, t2 - t1)
    # print(logabsdet.shape)
    # plt.plot(inputs.data.numpy(), outputs.data.numpy())
    # plt.plot(inputs.data.numpy(), inverse.data.numpy())
    # # plt.plot(outputs.data.numpy(), inputs.data.numpy(), ':')
    # plt.gca().set_aspect('equal')
    # plt.show()

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, 512),
                             # nn.ReLU(),
                             # nn.Linear(512, 512),
                             nn.ReLU(),
                             nn.Linear(512, c_out),
                             nn.Softplus())
    nscb = NeuralSplineCouplingBlock(dims_in=[[128]], subnet_constructor=subnet_fc)

    for param in nscb.parameters():
        print(type(param.data), param.size(), param.requires_grad)

    x = torch.randn([10,128])
    x.requires_grad = True
    y = nscb([x])
    x_ = nscb(y, rev=True)[0]
    print((x - x_).abs().mean().item())

    # nscb.plot_coupling_spline([x])
    # nscb.plot_internal_spline()
    # plt.gca().set_aspect('equal')
    # plt.show()
