"""Embedding and Aggregation Layers"""

import tensorflow as tf
# import kernels
from lsrflow.tensorflow.embedding.kernels import Kernel

#########################################
# Kernel embedding
#########################################

class KernelEmbedding(tf.keras.layers.Layer):
    """A custom Layer to create an embedding
       taking batch information into account
       with a pre-defined kernel

    # Arguments:
         kernel: kernel to apply batchwise

    # Input shape
        2D tensor with shape:
        [batch_size, n_features]

    # Output shape
        3D tensor with shape:
        [batch_size, batch_size, n_features]
    """

    def __init__(
        self, sigma: float = 1.0, kernel: str = "batch_pairwise_diff", **kwargs
    ):

        self.sigma = sigma
        self.kernel = getattr(Kernel(self.sigma), kernel)

        super().__init__(**kwargs)

    def call(self, x):
        """Build the actual logic."""

        return self.kernel(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[0], input_shape[1])

    def get_config(self):
        config = {
            "sigma": self.sigma,
            "kernel": self.kernel,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


#########################################
# Aggregation Layer
#########################################


class Aggregation(tf.keras.layers.Layer):
    """A custom Layer to apply an aggregation function

    # Arguments

     # Input shape
        3D tensor with shape:
        [batch_size, batch_size, n_features]

    # Output shape
        2D tensor with shape:
        [batch_size, n_features]
    """

    def __init__(self, agg_std=1, agg_mean=0, agg_sum=0, agg_max=0, **kwargs):

        self.agg_std = agg_std
        self.agg_mean = agg_mean
        self.agg_sum = agg_sum
        self.agg_max = agg_max

        super().__init__(**kwargs)

    def call(self, x):
        """Build the actual logic."""

        final_out = 0

        # (batch_size_1, batch_size_2, n_units) -> (batch_size_1, dim_out)
        if self.agg_std:
            final_out += tf.math.reduce_std(x, 1)

        if self.agg_mean:
            final_out += tf.math.reduce_mean(x, 1)

        if self.agg_sum:
            final_out += tf.math.reduce_sum(x, 1)

        if self.agg_max:
            final_out += tf.math.reduce_max(x, 1)

        return final_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = {
            "agg_std": self.agg_std,
            "agg_mean": self.agg_mean,
            "agg_sum": self.agg_sum,
            "agg_max": self.agg_max,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
