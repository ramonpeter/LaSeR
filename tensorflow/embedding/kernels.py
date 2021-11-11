"""Kernel class"""

import tensorflow as tf

#########################################
# Kernel
#########################################

# pylint: disable=C0103
class Kernel:
    """Embedding kernel class.

    Defines which Observables will be plotted depending on the
    specified dataset.
    """

    def __init__(self, sigma):

        self.sigma = sigma

    @staticmethod
    def batch_pairwise_diff(x):
        """Computes the pairwise difference between features in a batch

        Args:
            x: a tensor of shape [batch_size, n_features]

        Returns:
            y: a distance matrix of dimensions [batch_size, batch_size, n_features]

        """
        # to expand the dimensions with none type dimension properly
        magic = tf.ones((tf.shape(x)[0]))

        xc = tf.tensordot(magic, x, axes=0)
        xb = tf.transpose(xc, (1, 0, 2))

        # Calculate the difference in all dimensions
        y = xb - xc

        return y

    def squared_batch_pairwise_diff(self, x):
        """Computes the squared pairwise difference between features in a batch

        Args:
            x: a tensor of shape [batch_size, n_features]

        Returns:
            y: a distance matrix of dimensions [batch_size, batch_size, n_features]

        """
        return tf.square(self.batch_pairwise_diff(x))

    def batch_exponential_kernel(self, x):
        """Computes the exponential kernel between features in a batch
        k(x) = exp(-(B-x)**2/sigma**2)

         Args:
             x: a tensor of shape [batch_size, n_features]

         Returns:
             y: a distance matrix of dimensions [batch_size, batch_size, n_features]

        """
        gamma = 1.0 / (2.0 * self.sigma ** 2)
        kernel = tf.exp(-gamma * self.squared_batch_pairwise_diff(x))

        return kernel
