######################
# Choose dataset: #
######################

dataset = 'eight'

#########
# Data: #
#########

weighted = False
scaler   = 1.

##############
# Training:  #
##############

lr = 1e-3
batch_size = 2000
gamma = 0.999
weight_decay = 0.
betas = (0.5, 0.9)

n_epochs = 200
n_its_per_epoch = 1

n_disc_updates = 4

#################
# Architecture: #
#################

n_units  = 10
n_layers = 5

latent_dim_gen = 2

####################
# Logging/preview: #
####################

show_interval = 5
save_interval = 5

###################
# Loading/saving: #
###################

test = False
train = True

save_model = True
load_model = False

save_dir = './experiments'
checkpoint_on_error = False
