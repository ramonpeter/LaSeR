from typing import Union
import tensorflow as tf

class MALAMCMC:
    """
    Metropolis-adjusted Langevin algorithm (MALA)
    which samples from an energy-based model
    based on tf.keras.Model.
    """
    def __init__(
        self,
        generator: tf.keras.Model, 
        discriminator: tf.keras.Model,
        epsilon: float=1e-2
    ):

        self.generator = generator
        self.discriminator = discriminator
        self.epsilon = epsilon
    
    @tf.function
    def log_p(self, x):
        sq_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        return - sq_norm/2 + self.discriminator(self.generator(x))

    @tf.function
    def p_posterior(self, x):
        return tf.math.exp(self.log_p(x))

    @tf.function
    def grad_log_p(self, x):
        with tf.GradientTape(persistent=True) as gt:
            gt.watch(x)
            log_prob =  self.log_p(x)
        
        return gt.gradient(log_prob, x)

    @tf.function
    def q_transition(self, y, x):
        q_log = y - x - self.epsilon**2/2 * self.grad_log_p(x) 
        q_log_sq_norm = tf.reduce_sum(tf.square(q_log), axis =-1, keepdims=True)
        return tf.math.exp(-1/(2*self.epsilon**2) * q_log_sq_norm )
    
    @tf.function
    def langevin_step(self, z):
        z_proposal = z + self.epsilon**2/2 * self.grad_log_p(z) + self.epsilon * tf.random.normal(tf.shape(z))
        return z_proposal

    @tf.function
    def alpha(self, z, z_prop):
        r = self.p_posterior(z_prop) * self.q_transition(z,z_prop)/ (self.p_posterior(z) * self.q_transition(z_prop,z))
        return tf.minimum(1.0, r)

    @tf.function
    def metropolis_hastings(self, z, z_prop):
        alpha = self.alpha(z, z_prop)
        u = tf.random.uniform((1,1))
        accepted = u <= alpha
        z_accepted = z_prop if accepted else z
        return z_accepted, accepted

    def sample(self, latent_dim, n_samples):
        z = tf.random.normal((1,latent_dim))
        sample = []
        accepted = 0

        # Burn in
        for _ in range(1000):
            z_prop = self.langevin_step(z)
            z, _ = self.metropolis_hastings(z, z_prop)

        for _ in range(n_samples):
            z_prop = self.langevin_step(z)
            z, bool_accepted = self.metropolis_hastings(z, z_prop)
            if bool_accepted:
                accepted += 1
            sample.append(z)

        acc_rate = accepted/n_samples
        return tf.concat(sample, axis=0), acc_rate

class HamiltonMCMC:
    """
    Hamilton Markov-Chain Monte Carlo 
    which samples from an energy-based model
    based on tf.keras.Model.
    """
    def __init__(
        self,
        generator: tf.keras.Model, 
        discriminator: tf.keras.Model,
        latent_dim: int,
        M: tf.Tensor = None,
        L: int = 100,
        epsilon: float=1e-2
    ):

        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

        if M  == None:
            self.M = tf.linalg.diag(([1] * self.latent_dim))
        else:
            self.M = M

        self.L = L
        self.epsilon = epsilon
    
    @tf.function
    def U(self, q):
        sq_norm = tf.reduce_sum(tf.square(q), axis=-1, keepdims=True)
        return sq_norm/2 - self.discriminator(self.generator(q))

    @tf.function
    def K(self, p):            
        return tf.linalg.matmul(p ,tf.linalg.matmul(self.M,p, transpose_b=True))

    @tf.function
    def grad_U(self, q):
        with tf.GradientTape() as gt:
            gt.watch(q)
            U =  self.U(q)
        
        return gt.gradient(U, q)
    
    @tf.function
    def leapfrog_step(self, q_init):
        q = q_init
        p_init =  tf.random.normal(tf.shape(q))
        p = p_init

        # Make half a step for momentum at the beginning
        p -= self.epsilon * self.grad_U(q)/2

        # Alternate full steps for position and momentum
        for i in range(self.L):
            # Make a full step for the position
            q += self.epsilon * p
            # Make a full step for the momentum, except at end of trajectory
            if i != self.L-1:
                p -=  self.epsilon * self.grad_U(q)

        # Make a half step for momentum at the end.
        p -= self.epsilon * self.grad_U(q) / 2
        # Negate momentum at end of trajectory to make the proposal symmetric
        p *= -1

        # Evaluate potential and kinetic energies at start and end of trajectory
        U_init = self.U(q_init)
        K_init = tf.reduce_sum(tf.square(p_init), axis=-1, keepdims=True)/2
        U_proposed = self.U(q)
        K_proposed = tf.reduce_sum(tf.square(p), axis=-1, keepdims=True)/2
        
        # Accept or reject the state at end of trajectory, returning either
        # the position at the end of the trajectory or the initial position
        u = tf.random.uniform((1,1))
        if u < tf.math.exp(U_init - U_proposed + K_init - K_proposed):
            return q, True
        else:
            return q_init, False

    def sample(self, latent_dim, n_samples):
        q = tf.random.normal((1,latent_dim))
        sample = []
        accepted = 0

        # Burn in
        for _ in range(1000):
            q, _  = self.leapfrog_step(q)

        # actual sampling
        for _ in range(n_samples):
            q, bool_accepted = self.leapfrog_step(q)
            if bool_accepted:
                accepted += 1
            sample.append(q)

        acc_rate = accepted/n_samples
        return tf.concat(sample, axis=0), acc_rate