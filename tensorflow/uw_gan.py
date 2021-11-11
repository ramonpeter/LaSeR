###############################################################################
# Imports
###############################################################################

# import standard libraries
import numpy as np
import os
import sys
import time

# import tensorflow
# I use tensorflow 2.X (2.1.0)
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend as K

# Import keras modules from tf.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input


TRAIN = False

###############################################################################
# Build the models
###############################################################################

## define the generator model
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
# Regularizer if not with spectral norm
###############################################################################

def bc_loss_weighted(label, logit, weight):
    """Calculates weighted BC loss
    """
    # Sum of the weights
    n = tf.reduce_sum(weight)
    cost =  tf.reduce_sum(weight * tf.nn.sigmoid_cross_entropy_with_logits(label, logit))/n
    return cost


###############################################################################
# Define the training function
###############################################################################

# Settings
max_iter = 20000
batch_size = 2048
latent_dim = 5
d_updates = 10
step_factor = 200



# Optimizer and scheduler

lr_schedule_g = tf.keras.optimizers.schedules.InverseTimeDecay(
                1e-3,
                decay_steps=step_factor,
                decay_rate=0.2,
                staircase=True)

lr_schedule_d = tf.keras.optimizers.schedules.InverseTimeDecay(
                1e-3,
                decay_steps=step_factor*d_updates,
                decay_rate=0.2,
                staircase=True)


# Instantiate one optimizer for the discriminator and another for the generator.
d_optimizer = tf.keras.optimizers.Adam(lr_schedule_d, beta_1=0.5, beta_2=0.9)
g_optimizer = tf.keras.optimizers.Adam(lr_schedule_g, beta_1=0.5, beta_2=0.9)


# define the networks
generator = define_generator(latent_dim, n_outputs=2)
#generator = define_inn(latent_dim, n_outputs=1)
discriminator = define_discriminator(n_inputs=2)
# discriminator = define_discriminator_embedding(n_inputs=1)

generator.summary()
discriminator.summary()

@tf.function
def train_step():
    
    # Assemble labels discriminating real from fake images
    ones = tf.ones((batch_size, 1))
    zeros = tf.zeros((batch_size, 1))
    gen_weight = tf.ones((batch_size, 1))

    # Train the discriminator
    for i in range(d_updates):         
        real_batch, real_weight = sample_circle_weighted(batch_size)
        # Sample random points in the latent space  
        random_noise = tf.random.normal(shape=(batch_size, latent_dim))
        # Decode them to fake images
        gen_batch = generator(random_noise)
        
        with tf.GradientTape() as tape:
            # logit_real = discriminator([real_batch, real_weight])
            # logit_fake = discriminator([gen_batch, gen_weight])
            logit_real = discriminator(real_batch)
            logit_fake = discriminator(gen_batch)
            
            # Add gen and real loss
            d_loss = bc_loss_weighted(ones, logit_real, real_weight)
            d_loss += bc_loss_weighted(zeros, logit_fake, gen_weight)
            
            
        grads = tape.gradient(d_loss, discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Sample random points in the latent space
    random_noise = tf.random.normal(shape=(batch_size, latent_dim))
    
    # Train the generator
    with tf.GradientTape() as tape:
        # logit_fake = discriminator([generator(random_noise), gen_weight])
        logit_fake = discriminator(generator(random_noise))
        g_loss = bc_loss_weighted(ones, logit_fake, gen_weight)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    
    return d_loss, g_loss

def train_online(iterations=2000, evals=1000):

    
    for step in range(iterations):
        
        # Train the discriminator & generator on one batch of real images.
        d_loss, g_loss = train_step()
        
        
        if (step +1) % evals == 0:
            # Print metrics
            print("Epoch #{}: Generative Loss: {}, Discriminator Loss: {}"
                  .format((step+1), g_loss, d_loss))
            # n = int(1e5)
            # # Make the plots
            # noise = tf.random.normal((n,latent_dim))
            # true, true_weight = sample_circle_weighted(n) # sample_camel_hybrid
            # gen = generator(noise)
            # plot_2d("gan_2048_{}".format(step + 1), true.numpy(), true_weight.numpy(), gen.numpy())
            
    #Final_plot
    n = int(1e6)
    # Make the plots
    noise = tf.random.normal((n,latent_dim))
    true, true_weight = sample_circle_weighted(n) # sample_camel_hybrid
    gen = generator(noise)
    plot_2d("gan_2048", true.numpy(), true_weight.numpy(), gen.numpy())
    plot_weights(true.numpy(), true_weight.numpy(), gen.numpy())
    

if TRAIN:
    start_time = time.time()
    print("Start training...")   
    train_online(max_iter)
    print("--- Run time: %s mins ---" % ((time.time() - start_time)/60))
    print("--- Run time: %s secs ---" % ((time.time() - start_time)))
    
    print("--- Save Weights ---")
    save_dir = "1d_results"
    generator.save_weights("weights_weighted.h5")
else:
    print("Load weights and plot...")
    save_dir = "1d_results"
    generator.load_weights("weights_weighted.h5")
    #Final_plot
    n = int(1e6)
    # Make the plots
    noise = tf.random.normal((n,latent_dim))
    true, true_weight = sample_circle_weighted(n)
    gen = generator(noise)
#    weights = np.exp(discriminator.predict(gen))
    plot_2d("gan_2048", true.numpy(), true_weight.numpy(), gen.numpy())
    plot_weights(true.numpy(), true_weight.numpy(), gen.numpy())







