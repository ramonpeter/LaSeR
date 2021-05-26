######################
# Choose dataset: #
######################

dataset = '2d_3_holes_weighted'

#########
# Data: #
#########

weighted = True
on_shell = 0
mom_cons = 0
scaler   = 1.

##############
# Training:  #
##############

lr = 1e-3
batch_size = 2000
gamma = 0.999
weight_decay = 0.
betas = (0.5, 0.9)

do_rev = False
do_fwd = True

n_epochs = 400
n_its_per_epoch = 1

n_disc_updates = 4

mmd = False

#################
# Architecture: #
#################

n_blocks = 8
n_units  = 100
n_layers = 5

latent_dim_gen = 2

####################
# Logging/preview: #
####################

loss_names = ['L', 'L_rev']
progress_bar = True                         # Show a progress bar of each epoch

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
