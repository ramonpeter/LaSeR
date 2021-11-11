###############################################################################
# Imports
###############################################################################

# import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import rand
from numpy.random import randn
from sklearn.model_selection import train_test_split
import time

# import tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from mcmc import MALAMCMC, HamiltonMCMC
#mcmc import MALAMCMC, HamiltonMCMC

# Import keras modules from tf.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Lambda, Dense, Input, Layer, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.utils import to_categorical

###############################################################################
# Build the models
###############################################################################

# define the generator model
def define_generator(latent_dim, n_outputs, spectral_norm=False):
    g_in = Input(shape=(latent_dim,))

    # two hidden layers
    if spectral_norm:
        x = tfa.layers.SpectralNormalization(
            Dense(25, activation="relu", kernel_initializer="he_uniform")
        )(g_in)
        x = tfa.layers.SpectralNormalization(
            Dense(25, activation="relu", kernel_initializer="he_uniform")
        )(x)
        g_out = tfa.layers.SpectralNormalization(Dense(n_outputs))(x)
    else:
        x = Dense(25, activation="relu", kernel_initializer="he_uniform")(g_in)
        x = Dense(25, activation="relu", kernel_initializer="he_uniform")(x)
        g_out = Dense(n_outputs)(x)

    model = Model(g_in, g_out, name="generator")
    return model


# define the iscriminator model
def define_discriminator(n_inputs, spectral_norm=True):
    d_in = Input(shape=(n_inputs,))

    # three hidden layers
    if spectral_norm:
        x = tfa.layers.SpectralNormalization(
            Dense(25, activation="relu", kernel_initializer="he_uniform")
        )(d_in)
        x = tfa.layers.SpectralNormalization(
            Dense(25, activation="relu", kernel_initializer="he_uniform")
        )(x)
        x = tfa.layers.SpectralNormalization(
            Dense(25, activation="relu", kernel_initializer="he_uniform")
        )(x)
        d_out = tfa.layers.SpectralNormalization(Dense(1))(x)
    else:
        x = Dense(25, activation="relu", kernel_initializer="he_uniform")(d_in)
        x = Dense(25, activation="relu", kernel_initializer="he_uniform")(x)
        x = Dense(25, activation="relu", kernel_initializer="he_uniform")(x)
        d_out = Dense(1)(x)

    model = Model(d_in, d_out, name="discriminator")
    return model


###############################################################################
# Get the real samples
###############################################################################


def generate_real_samples_twomodes(n):
    Z1 = tf.random.normal((int(n / 2), 1), -1, 0.4)
    Z2 = tf.random.normal((n - int(n / 2), 1), 1, 0.4)
    X = tf.concat([Z1, Z2], axis=0)
    x = tf.random.shuffle(X)
    # generate class labels
    return x


def load_data(path: str):
    """Load data"""
    data = pd.read_hdf(path)
    data = data.iloc[:, :].values
    return data


def sample_data(data, batch_size):
    """Get array of samples from loaded data"""
    index = np.arange(data.shape[0])
    if batch_size <= data.shape[0]:
        choice = np.random.choice(index, batch_size, replace=False)
    else:
        choice = np.random.choice(index, batch_size, replace=True)

    batch = data[choice]
    return tf.convert_to_tensor(batch, dtype=tf.float32)


###############################################################################
# Define the MCMC latent dim resampling
###############################################################################


def mcmc_latent_dim(generator, discriminator, latent_dim, n=1000, MALA=False):

    # =======================================
    # MALA MCMC
    # ---------------------------------------
    # = Langevin dynamics (q)
    #   + Metropolis-Hastings
    # =======================================

    if MALA:
        # optimal: epsilon = 0.4 gives optimal acceptance rate (0.574)
        mala = MALAMCMC(generator, discriminator, epsilon=0.48)
        z, rate = mala.sample(latent_dim, n)
        print(rate)

    # =======================================
    # Hamilton MCMC
    # --------------------------------------
    # = Hamilton dynamics (q,p)
    #   + Metropolis-Hastings
    # =======================================
    else:
        # (dim = 5): optimal: epsilon = 0.04, L=30 gives optimal acceptance rate (0.65)
        hamilton = HamiltonMCMC(
            generator, discriminator, latent_dim, L=30, epsilon=0.04
        )
        z, rate = hamilton.sample(latent_dim, n)
        print(rate)

    # =======================================
    # Trace plot
    # =======================================
    FONTSIZE = 16
    fig, axs = plt.subplots(1)
    x = np.arange(n)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    axs.scatter(x, z[:, 0].numpy(), c="black", s=1.5)
    axs.set_ylabel(r"First position coordinate", fontsize=FONTSIZE)
    # axs.set_yscale('log')
    for label in (
        [axs.yaxis.get_offset_text()] + axs.get_xticklabels() + axs.get_yticklabels()
    ):
        label.set_fontsize(FONTSIZE - 2)

    fig.savefig(f"trace_plot.pdf", format="pdf")
    plt.close()

    return z


###############################################################################
# Plotting
###############################################################################

# evaluate the discriminator and plot real and fake points
def plot_resampled(
    epoch,
    generator,
    discriminator,
    latent_dim,
    n=100,
    bins=50,
    save_weighted_latent=False,
):
    # prepare real samples
    gcolor = "#3b528b"
    dcolor = "#e41a1c"
    x_real = generate_real_samples_twomodes(10 * n).numpy()

    # Use Latex to compile
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

    # get fake samples
    noise = tf.random.normal((n, latent_dim))
    x_fake = generator(noise).numpy()
    weights = np.exp(discriminator(generator(noise)))

    if save_weighted_latent:
        print("Save weighted latent space")
        data = np.concatenate([noise, weights], axis=-1)
        s = pd.HDFStore("camel_latent.h5")
        s.append("data", pd.DataFrame(data))
        s.close()

    # Get the better samples latent space
    if True:
        print("Start sampling...")
        start_time = time.time()
        noise_mod = mcmc_latent_dim(generator, discriminator, latent_dim, n=n)
        end_time = time.time()
        print("--- Sample time: %s mins ---" % ((end_time - start_time) / 60))
        print("--- Sample time: %s secs ---" % ((end_time - start_time)))
        x_mod = generator(noise_mod).numpy()

    # get the refined points
    print("Load unweighted data...")
    unweighted_latent = load_data("5d_camel_latent_unweighted_new.h5")
    noise_uw = sample_data(unweighted_latent, n)
    x_re = generator(noise_uw).numpy()

    # scatter plot real and fake data points
    FONTSIZE = 16
    fig, axs = plt.subplots(1)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    y_r, x_r = np.histogram(x_real, bins, range=(-3, 3))
    axs.step(x_r[:50], y_r / 10, "black", label="Truth", linewidth=1.0, where="mid")
    axs.hist(
        x_fake, color=gcolor, alpha=0.5, bins=np.linspace(-3, 3, bins), label="Base"
    )
    axs.hist(
        x_fake,
        weights=weights,
        color="darkgreen",
        alpha=0.5,
        bins=np.linspace(-3, 3, bins),
        label="DCTR",
    )
    axs.hist(x_mod, color=dcolor, alpha=0.5, bins=np.linspace(-3, 3, bins), label="HMC")
    axs.hist(
        x_re, color="orange", alpha=0.5, bins=np.linspace(-3, 3, bins), label="LaSeR"
    )

    axs.set_ylabel(r"Events", fontsize=FONTSIZE)
    axs.set_xlabel(r"$x$", fontsize=FONTSIZE)
    # axs.set_yscale('log')
    for label in (
        [axs.yaxis.get_offset_text()] + axs.get_xticklabels() + axs.get_yticklabels()
    ):
        label.set_fontsize(FONTSIZE - 2)

    axs.legend(
        frameon=False,
        loc="upper right",
        prop={"size": (FONTSIZE - 2)},
    )
    fig.savefig("two_mode_{}".format(epoch) + ".pdf", format="pdf")
    plt.close()

    # weighted correlation
    rmin = -3
    rmax = 3
    with PdfPages(f"weighted_latent_dim.pdf") as pp:
        for i, j in enumerate(list(range(latent_dim))):
            for k in list(range(latent_dim))[i + 1 :]:
                fig, axs = plt.subplots(1, 2)
                axs[0].hist2d(
                    noise[:, j],
                    noise[:, k],
                    bins=[np.linspace(rmin, rmax, bins), np.linspace(rmin, rmax, bins)],
                )
                axs[0].set_xlabel(f"$z_{{{j}}}$")
                axs[0].set_ylabel(f"$z_{{{k}}}$")
                axs[1].hist2d(
                    noise[:, j],
                    noise[:, k],
                    bins=[np.linspace(rmin, rmax, bins), np.linspace(rmin, rmax, bins)],
                    weights=weights[:, 0],
                )
                axs[1].set_xlabel(f"$z_{{{j}}}$")
                axs[1].set_ylabel(f"$z_{{{k}}}$")
                fig.savefig(pp, format="pdf")
                plt.close()

    # one-correlation
    with PdfPages(f"refined_latent_dim.pdf") as pp:
        for i, j in enumerate(list(range(latent_dim))):
            for k in list(range(latent_dim))[i + 1 :]:
                fig, axs = plt.subplots(1, 2)
                axs[0].hist2d(
                    noise[:, j],
                    noise[:, k],
                    bins=[np.linspace(rmin, rmax, bins), np.linspace(rmin, rmax, bins)],
                )
                axs[0].set_xlabel(f"$z_{{{j}}}$")
                axs[0].set_ylabel(f"$z_{{{k}}}$")
                axs[1].hist2d(
                    noise_mod[:, j],
                    noise_mod[:, k],
                    bins=[np.linspace(rmin, rmax, bins), np.linspace(rmin, rmax, bins)],
                )
                axs[1].set_xlabel(f"$z_{{{j}}}$")
                axs[1].set_ylabel(f"$z_{{{k}}}$")
                fig.savefig(pp, format="pdf")
                plt.close()


###############################################################################
# Define the training functions
###############################################################################

# Define the optimizers
# use of beta_1=0.5, beta_2=0.9 imroves results
d_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, beta_2=0.9)
g_optimizer = tf.keras.optimizers.Adam(beta_1=0.5, beta_2=0.9)

# Instantiate the bce loss function with logits
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# ============
# Train both
# ============
@tf.function
def train_step(generator, discriminator, real_batch, batch_size, latent_dim):

    # Sample random points in the latent space
    random_noise = tf.random.normal(shape=(batch_size, latent_dim))

    # Decode them to fake images
    gen_batch = generator(random_noise)

    # Assemble labels discriminating real from fake images
    ones = tf.ones((batch_size, 1))
    zeros = tf.zeros((batch_size, 1))

    # Train the discriminator
    with tf.GradientTape() as tape:
        logit_real = discriminator(real_batch)
        logit_fake = discriminator(gen_batch)

        # Add gen and real loss
        d_loss = loss(ones, logit_real)
        d_loss += loss(zeros, logit_fake)

    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Sample random points in the latent space
    random_noise = tf.random.normal(shape=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    ones = tf.ones((batch_size, 1))

    # Train the generator
    with tf.GradientTape() as tape:
        logit_fake = discriminator(generator(random_noise))
        g_loss = loss(ones, logit_fake)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))

    return d_loss, g_loss


# ======================
# Train classifier only
# ======================
@tf.function
def train_d(generator, discriminator, real_batch, batch_size, latent_dim):

    # Sample random points in the latent space
    random_noise = tf.random.normal(shape=(batch_size, latent_dim))

    # Decode them to fake images
    gen_batch = generator(random_noise)

    # Assemble labels discriminating real from fake images
    ones = tf.ones((batch_size, 1))
    zeros = tf.zeros((batch_size, 1))

    # Train the discriminator
    with tf.GradientTape() as tape:
        logit_real = discriminator(real_batch)
        logit_fake = discriminator(gen_batch)

        # Add gen and real loss
        d_loss = loss(ones, logit_real)
        d_loss += loss(zeros, logit_fake)

    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    return d_loss


def train_twomodes(generator, discriminator, latent_dim, n_epochs=10000, n_batch=128):
    #
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real = generate_real_samples_twomodes(n_batch)

        # Train the discriminator & generator on one batch of real images.
        d_loss, g_loss = train_step(
            generator, discriminator, x_real, n_batch, latent_dim
        )
        if (i + 1) % 1000 == 0:
            # Print metrics
            print(
                "Epoch #{}: Generative Loss: {}, Discriminator Loss: {}".format(
                    i + 1, g_loss, d_loss
                )
            )
    #        # Logging.
    #        if (i+1) % n_eval == 0:
    #            plot_resampled(i+1, generator, discriminator, latent_dim, 10000, steps=100)
    plot_resampled("d0", generator, discriminator, latent_dim, n=10000)

    for i in range(n_epochs * 2):
        x_real = generate_real_samples_twomodes(2 * n_batch)
        d_loss = train_d(generator, discriminator, x_real, 2 * n_batch, latent_dim)
        if (i + 1) % 1000 == 0:
            # Print metrics
            print("Epoch #{}: Discriminator Loss: {}".format(i + 1, d_loss))

    #        # Logging.
    #        if (i+1) % n_eval == 0:
    #            plot_resampled(i+2, generator, discriminator, latent_dim, 10000, steps=100)

    plot_resampled("d1", generator, discriminator, latent_dim, n=10000)


###############################################################################
# Create all models and run it
###############################################################################

# size of the latent space
latent_dim = 5
# create the discriminator
discriminator_twomodes = define_discriminator(1, spectral_norm=False)
# create the generator
generator_twomodes = define_generator(latent_dim, 1)

TRAIN = False
# train model
if TRAIN:
    start_time = time.time()
    print("Start training...")
    train_twomodes(generator_twomodes, discriminator_twomodes, latent_dim, n_batch=256)
    print("--- Run time: %s mins ---" % ((time.time() - start_time) / 60))
    print("--- Run time: %s secs ---" % ((time.time() - start_time)))
    generator_twomodes.save_weights("weights_g.h5")
    discriminator_twomodes.save_weights("weights_d.h5")
else:
    print("Load weights and plot...")
    generator_twomodes.load_weights("weights_g.h5")
    discriminator_twomodes.load_weights("weights_d.h5")
    # Final_plot
    plot_resampled(
        "HMC_test",
        generator_twomodes,
        discriminator_twomodes,
        latent_dim,
        n=1000,
        bins=50,
    )
