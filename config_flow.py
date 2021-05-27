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
batch_size = 2048
gamma = 0.995
weight_decay = 1e-5
betas = (0.9, 0.999)

do_rev = False
do_fwd = True

n_epochs = 200
n_its_per_epoch = 1

#################
# Architecture: #
#################

n_blocks = 12
n_units  = 48
n_layers = 3

####################
# Logging/preview: #
####################

show_interval = 10
save_interval = 10

###################
# Loading/saving: #
###################

test = False
train = True

save_model = True
load_model = False

save_dir = './experiments'
checkpoint_on_error = False
