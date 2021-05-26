import torch
from torch import nn
import torch.nn.functional as F
import random, math, os


# Largely adapted from https://github.com/pbloem/former

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false (in place operation)
    """
    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval


def d(tensor=None):
    """ Returns a device string either for the best available device or argument """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'


class SelfAttentionCouplingBlock(nn.Module):
    """
    Compute self-attention features like in a standard Transformer network, but
    based on only half the embedding dimensions, and add it to the other half.
    """

    def __init__(self, dims_in, dims_c=[], n_heads=8, mask=False):
        super().__init__()

        self.channels = dims_in[0][-1]
        self.ndims = len(dims_in[0])
        self.split_idx = self.channels // 2
        self.n_heads = n_heads
        self.mask = mask

        assert all([dims_c[i][:1] == dims_in[0][:1] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][-1] for i in range(len(dims_c))])

        self.tokeys     = nn.Linear(self.split_idx + condition_length, self.split_idx * n_heads, bias=False)
        self.toqueries  = nn.Linear(self.split_idx + condition_length, self.split_idx * n_heads, bias=False)
        self.tovalues   = nn.Linear(self.split_idx + condition_length, self.split_idx * n_heads, bias=False)
        self.unifyheads = nn.Linear(self.split_idx * n_heads + condition_length, self.channels - self.split_idx)

    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], [self.split_idx, self.channels - self.split_idx], dim=-1)
        b, t, e = x1.size() # batch, token, embedding dimensions
        x1_c = torch.cat([x1, *c], dim=-1) if self.conditional else x1

        keys    =    self.tokeys(x1_c).view(b, t, self.n_heads, e)
        queries = self.toqueries(x1_c).view(b, t, self.n_heads, e)
        values  =  self.tovalues(x1_c).view(b, t, self.n_heads, e)

        # Compute scaled dot-product self-attention
        # Fold heads into the batch dimension
        keys    =    keys.transpose(1,2).contiguous().view(b*self.n_heads, t, e)
        queries = queries.transpose(1,2).contiguous().view(b*self.n_heads, t, e)
        values  =  values.transpose(1,2).contiguous().view(b*self.n_heads, t, e)
        # Instead of dividing the dot products by sqrt(e), scale the keys and values (more memory efficient)
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*self.n_heads, t, t)

        # Mask out the upper half of the dot matrix, excluding the diagonal
        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        # Row-wise self-attention probabilities
        dot = F.softmax(dot, dim=2)
        # Apply the self attention to the values
        out = torch.bmm(dot, values).view(b, self.n_heads, t, e)
        # Swap self.n_heads, t back, unify heads
        out = out.transpose(1,2).contiguous().view(b, t, self.n_heads * e)
        out_c = torch.cat([out, *c], dim=-1) if self.conditional else out
        unified = self.unifyheads(out_c)

        if not rev:
            return [torch.cat((x1, x2 + unified), dim=-1)]
        else:
            return [torch.cat((x1, x2 - unified), dim=-1)]
        # Missing layer norm and dropout from standard Transformer

    def jacobian(self, x, c=[], rev=False):
        # TODO
        pass

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use one input."
        return input_dims



class FeedForwardCouplingBlock(nn.Module):
    """
    Transforms half the embedding dimensions with a simple MLP and adds it to the other half.
    Parameters are shared between all tokens.

    subnet_constructor: function or class, with signature constructor(dims_in, dims_out).
                        The result should be a torch nn.Module, that takes dims_in input channels,
                        and dims_out output channels. See tutorial for examples.
    """

    def __init__(self, dims_in, dims_c=[], subnet_constructor=None):
        super().__init__()

        self.channels = dims_in[0][1]
        self.ndims = len(dims_in[0])
        self.split_idx = self.channels // 2

        assert all([dims_c[i][:1] == dims_in[0][:1] for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        condition_length = sum([dims_c[i][-1] for i in range(len(dims_c))])

        self.ff = subnet_constructor(self.split_idx + condition_length, self.channels - self.split_idx)

    def forward(self, x, c=[], rev=False):
        x1, x2 = torch.split(x[0], [self.split_idx, self.channels - self.split_idx], dim=-1)
        x1_c = torch.cat([x1, *c], dim=-1) if self.conditional else x1

        out = self.ff(x1_c)

        if not rev:
            return [torch.cat((x1, x2 + out), dim=-1)]
        else:
            return [torch.cat((x1, x2 - out), dim=-1)]
        # Missing layer norm and dropout from standard Transformer

    def jacobian(self, x, c=[], rev=False):
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use one input."
        return input_dims
