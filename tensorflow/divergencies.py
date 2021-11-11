import tensorflow as tf

class Divergencies:
    """Loss class conatiner."""

    def __init__(self):
        self.divergences = [x for x in dir(self)
                            if ('__' not in x)]

    @staticmethod
    def chi2(true, test, logp, logq):
        """ Implement Neyman chi2 divergence.

        This function returns the Neyman chi2 divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the chi2 is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): not used in chi2
            logq (tf.tensor or array(nbatch) of floats): not used in chi2

        Returns:
            tf.tensor(float): computed Neyman chi2 divergence

        """
        return tf.reduce_mean((tf.stop_gradient(true) - test)**2
                              / test / tf.stop_gradient(test))

    # pylint: disable=invalid-name
    @staticmethod
    def kl(true, test, logp, logq):
        """ Implement Kullback-Leibler (KL) divergence.

        This function returns the Kullback-Leibler divergence for two given sets
        of probabilities, true and test. It uses importance sampling, i.e. the
        estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the KL is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed KL divergence

        """
        return tf.reduce_mean(input_tensor=tf.stop_gradient(true/test)
                              * (tf.stop_gradient(logp) - logq))

    # pylint: disable=invalid-name
    @staticmethod
    def js(true, test, logp, logq):
        """ Implement Jensen-Shannon divergence.

        This function returns the Jensen-Shannon divergence for two given
        sets of probabilities, true and test. It uses importance sampling,
        i.e. the estimator is divided by an additional factor of 'test'.

        tf.stop_gradient is used such that the correct gradient is returned
        when the Jenson-Shannon is used as loss function.

        Arguments:
            true (tf.tensor or array(nbatch) of floats): true probability of points.
            test (tf.tensor or array(nbatch) of floats): estimated probability of points
            logp (tf.tensor or array(nbatch) of floats): logarithm of the true probability
            logq (tf.tensor or array(nbatch) of floats): logarithm of the estimated probability

        Returns:
            tf.tensor(float): computed Jensen-Shannon divergence

        """
        logm = tf.math.log(0.5*(test+tf.stop_gradient(true)))
        return tf.reduce_mean(input_tensor=(
            tf.stop_gradient(0.5/test) * ((tf.stop_gradient(true)
                                           * (tf.stop_gradient(logp)-logm))
                                          + (test * (logq-logm)))))

    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError('The requested loss function {} '
                                'is not implemented. Allowed '
                                'options are {}.'.format(
                                    name, self.divergences))

