import torch
from torch import nn


def bernstein_composition(p_coeff, q_coeff, B):
    '''
    Compute coefficients of the Bernstein polynomials h(x) = q(p(x)) for many p at once.
    See "Composing Bézier simplexes", A.D. DeRose, 1988.
    Expects p_coeff of shape [batchsize, degree+1] and q_coeff of shape [degree+1].
    '''
    m, n = p_coeff.shape[1] - 1, q_coeff.shape[0] - 1
    h = torch.zeros((p_coeff.shape[0], n+1, n+1, m*n+1), device=p_coeff.device) # indexed as h[batch,s,i,j]
    h[:,0,:,0] = q_coeff

    for s in range(1, n + 1):
        for j in range(m*s + 1):
            i = n - s + 1
            kmin = max(0, j - m)
            kmax = min(j, m*s - m) + 1

            h[:, s, :i, j] = torch.sum(B[None, None, m*s-m, kmin:kmax] * B[None, None, m, j-kmax+1:j-kmin+1].flip(-1) *
                                       ((1 - p_coeff[:, None, j-kmax+1:j-kmin+1].flip(-1)) * h[:, s-1, :i, kmin:kmax] +
                                        p_coeff[:, None, j-kmax+1:j-kmin+1].flip(-1) * h[:, s-1, 1:i+1, kmin:kmax]),
                                       dim=-1) / B[m*s, j]

    return h[:,n,0,:]


def evaluate_bernstein_polynomial(x, coefficients):
    '''
    Evaluate Bernstein polynomials defined by given coefficients at locations x, using De Casteljau's algorithm.
    Expects x of shape [batchsize] and coefficients of shape [batchsize, degree+1].
    '''
    # t = time()
    x = x[:,None]
    y = coefficients
    for i in range(coefficients.shape[1] - 1):
        y = (1-x) * y[:,:-1] + x * y[:,1:]
    # print(f'de casteljau: {time()-t:.8f}')
    return y[:,0]


def bernstein_polynomial_derivative(x, coefficients):
    '''
    Derivatives of the Bernstein polynomials defined by given coefficients at locations x.
    Expects x of shape [batchsize] and coefficients of shape [batchsize, degree+1].
    '''
    return (coefficients.shape[1] - 1) * evaluate_bernstein_polynomial(x, coefficients[:,1:] - coefficients[:,:-1])


def bernstein_inverse_coefficients(coefficients, M, B, maxinvdegree=12, tolerance=0.01):
    '''
    Approximate coefficients of the Bernstein polynomials that are inverses of the ones defined by the given coefficients.
    See "Convergent inversion approximations for polynomials in Bernstein form", Rida T. Farouki, 1999.
    Expects coefficients of shape [batchsize, degree+1].
    '''

    # Step 0: initial Bernstein coefficient for degree 0 approximation
    # t = time()
    c = 1 - (1 / coefficients.shape[1]) * torch.sum(coefficients, dim=1)[:,None]
    j = 0
    error = tolerance + 1
    while(error > tolerance):
        j += 1
        # Step 1: compute Bernstein coefficients alpha for composed Legendre polynomials of degree j-1 and j+1
        alpha_low = bernstein_composition(coefficients, M[j-1,:j], B)
        alpha_high = bernstein_composition(coefficients, M[j+1,:j+2], B)
        # Step 2: compute coefficients for j-th Legendre polynomial
        m = coefficients.shape[1] - 1
        l_j = 0.5 * ((1/(m*j - m + 1)) * torch.sum(alpha_low[:,:m*j-m+1], dim=1) -
                     (1/(m*j + m + 1)) * torch.sum(alpha_high[:,:m*j+m+1], dim=1))
        # Step 3: elevate degree of previous Bernstein coefficients to j
        kj = torch.arange(1, j, dtype=torch.float32, device=coefficients.device) / j
        c = torch.cat([c[:,:1], (1 - kj) * c[:,1:j] + kj * c[:,:j-1], c[:,-1:]], dim=1)
        # Step 4: add contributions l_j*M_jk from j-th Legendre polynomial to elevated Bernstein coefficients
        c += l_j[:,None] * M[j:j+1, :j+1]
        # Step 5: compute approximation error, end loop if good enough, else increase j
        # TODO
        error = 1 if (j < maxinvdegree) else 0

    # print(f'inversion: {time()-t:.8f}')
    return c


def evaluate_bernstein_spline(x, coeffs, heights, widths, M=None, B=None, rev=False):
    '''
    Evaluate splines that are defined by monotone Bernstein polynomials
    centered at zero, scaled by given widths and heights and extended according
    to the slope at the polynomials' end points.
    '''
    widths = widths + 0.1 * widths.sign()
    heights = heights + 0.1 * heights.sign()

    if not rev:
        x = x / widths + 0.5
    else:
        x = x / heights + 0.5

    l_indices = x < 0
    xl = x[l_indices]
    r_indices = x > 1
    xr = x[r_indices]
    m_indices = ~(l_indices | r_indices)
    y = torch.zeros_like(x)
    derivatives = torch.zeros_like(x)

    if not rev:
        l_derivatives = bernstein_polynomial_derivative(torch.zeros_like(xl), coeffs[l_indices])
        derivatives[l_indices] = l_derivatives
        derivatives[m_indices] = bernstein_polynomial_derivative(x[m_indices], coeffs[m_indices])
        r_derivatives = bernstein_polynomial_derivative(torch.ones_like(xr), coeffs[r_indices])
        derivatives[r_indices] = r_derivatives

        y[l_indices] = xl * l_derivatives
        y[m_indices] = evaluate_bernstein_polynomial(x[m_indices], coeffs[m_indices])
        y[r_indices] = 1 + (xr - 1) * r_derivatives
        y = (y - 0.5) * heights
        derivatives = derivatives * heights/widths

    else:
        inv_coeffs = bernstein_inverse_coefficients(coeffs, M, B)
        ym = evaluate_bernstein_polynomial(x[m_indices], inv_coeffs[m_indices])
        y[m_indices] = ym

        l_derivatives = 1 / bernstein_polynomial_derivative(torch.zeros_like(xl), coeffs[l_indices])
        derivatives[l_indices] = l_derivatives
        derivatives[m_indices] = 1 / bernstein_polynomial_derivative(ym, coeffs[m_indices])
        r_derivatives = 1 / bernstein_polynomial_derivative(torch.ones_like(xr), coeffs[r_indices])
        derivatives[r_indices] = r_derivatives

        y[l_indices] = xl * l_derivatives
        y[r_indices] = 1 + (xr - 1) * r_derivatives
        y = (y - 0.5) * widths
        derivatives = derivatives * widths/heights

    return y, derivatives


class BernsteinSplineCouplingBlock(nn.Module):
    '''
    Invertible coupling block where half of the variables are transformed with
    an invertible Bernstein polynomial spline parameterized based on the other
    half and optional conditioning input.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    '''

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None, degree=10):
        super().__init__()

        self.channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.split_len1
        self.degree = degree

        assert all([dims_c[i][1:] == dims_in[0][1:] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s = subnet_constructor(self.split_len1 + self.condition_length, self.split_len2 * (self.degree+2))

        # Pascal's triangle to look up binomial coefficients
        self.B = torch.zeros((2*degree**2, 2*degree**2))
        self.B[0,0] = 1
        self.B[1,0] = 1
        self.B[1,1] = 1
        for i in range(2, self.B.shape[0]):
            self.B[i,0] = 1
            self.B[i,1:i] = self.B[i-1,:i-1] + self.B[i-1,1:i]
            self.B[i,i] = 1
        self.B = torch.nn.Parameter(self.B, requires_grad=False)

        # Pascal's triangle with alternating signs, as needed for spline composition
        self.M = self.B * torch.tensor([[1., -1.], [-1., 1.]]).repeat(self.B.shape[0]//2, self.B.shape[1]//2)
        self.M = torch.nn.Parameter(self.M, requires_grad=False)


    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        if rev: y1, y2 = x1, x2

        # Concatenate conditional input
        x1_c = torch.cat([x1, *c], 1) if self.conditional else x1

        # Prepare spline parameters from subnetwork output
        s = self.s(x1_c).view([-1, (self.degree+2)])
        coeffs, widths, heights = torch.split(s, [self.degree, 1, 1], dim=-1)
        coeffs = torch.cat([torch.zeros((*coeffs.shape[:-1], 1), device=x1.device),
                            torch.cumsum(nn.functional.softplus(coeffs), dim=-1)], dim=-1)
        coeffs = coeffs / coeffs[:,-1,None]
        widths = nn.functional.softplus(widths.squeeze(-1))
        heights = heights.squeeze(-1)

        if not rev:
            # Apply coupling transform, compute log Jacobian determinant
            y2, derivatives = evaluate_bernstein_spline(x2.contiguous().view(-1), coeffs, heights, widths)
            self.last_jac = derivatives.view([-1, self.split_len2]).log().sum(dim=-1)
            return [torch.cat((x1, y2.view([-1, self.split_len2])), dim=1)]
        else:
            # Apply coupling transform, compute log Jacobian determinant
            x2, derivatives = evaluate_bernstein_spline(y2.contiguous().view(-1), coeffs, heights, widths, self.M, self.B, rev=True)
            self.last_jac = derivatives.view([-1, self.split_len2]).log().sum(dim=-1)
            return [torch.cat((y1, x2.view([-1, self.split_len2])), dim=1)]

    def jacobian(self, x, c=[], rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims

    def plot_spline(self, start=-1, end=1, n_points=5000, rev=False):
        from matplotlib import pyplot as plt
        plt.ion()

        with torch.no_grad():
            x = torch.linspace(start,end,n_points)[:,None].repeat(1,2)
            x = torch.cat([5*torch.randn((1,1)).repeat(n_points,2), x], dim=-1).to(self.B.device)
            c = torch.tensor([1.5, 0.0]).repeat(n_points,1).to(self.B.device)
            y = self.forward([x], c=[c], rev=rev)[0]

            plt.figure(2); plt.gcf().clear()
            plt.plot(x[:,-1].data.cpu().numpy(), y[:,-1].data.cpu().numpy())
            plt.axhline(0, c='k', lw=.5); plt.axvline(0, c='k', lw=.5)
            plt.plot([-1,1], [-1,1], ':', color='gray')
            plt.grid(); plt.figure(2).canvas.flush_events(); plt.pause(0.1)


if __name__ == '__main__':
    # Visually check correctness of Bernstein inversion
    import numpy as np
    from matplotlib import pyplot as plt

    torch.manual_seed(12)
    n_points = 5000
    x = torch.linspace(-1,1,n_points)[:,None].repeat(1,3)
    x = torch.cat([5*torch.randn((1,3)).repeat(n_points,1), x], dim=-1).cuda()
    c = 10*torch.randn((1,10)).repeat(n_points,1).cuda()

    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, 8),
                             nn.LeakyReLU(),
                             nn.Linear(8, c_out))
    BernsteinBlock = BernsteinSplineCouplingBlock(dims_in=[list(x.shape[1:])],
                                                  dims_c=[list(c.shape[1:])],
                                                  subnet_constructor=subnet_fc,
                                                  degree=10).cuda()

    y = BernsteinBlock([x], c=[c])[0]
    plt.plot(x[:,-1].data.cpu().numpy(), y[:,-1].data.cpu().numpy())

    # d = BernsteinBlock.jacobian([x], c=[c])
    # plt.plot(x[:,-1].data.cpu().numpy(), (torch.cumsum(d[:,-1].data, dim=0) * 2 / d.shape[0]).cpu().numpy() + y[0,-1].item())

    # inv = BernsteinBlock([x], c=[c], rev=True)[0]
    # plt.plot(x[:,-1].data.cpu().numpy(), inv[:,-1].data.cpu().numpy())

    # cycle = BernsteinBlock([y], c=[c], rev=True)[0]
    # plt.plot(x[:,-1].data.cpu().numpy(), cycle[:,-1].data.cpu().numpy(), color=(.7,.7,.7))

    plt.axhline(0, c='k', lw=.5); plt.axvline(0, c='k', lw=.5)
    plt.plot([-1,1], [-1,1], ':', color='gray')
    # plt.gca().set_aspect('equal'); plt.xlim((-0.3,1.3)); plt.ylim((-0.3,1.3))
    plt.grid(); plt.show()
