import torch.nn as nn

from utils.train_utils import *
from utils.observables import *
from utils.plotting.distributions import *
from utils.plotting.plots import *
from load_data import *

from gan_models import *
from inn_model import INN

import sys, os

import config_FLOW as c
import opts
opts.parse(sys.argv)
config_str = ""
config_str += "==="*30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
	if v[0]=='_': continue
	s=eval('c.%s'%(v))
	config_str += " {:25}\t{}\n".format(v,s)

config_str += "==="*30 + "\n"

print(config_str)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

train_loader, validate_loader, dataset_size, data_shape, scales = Loader(c.dataset, c.batch_size, c.test, c.scaler, c.on_shell, c.mom_cons, c.weighted)

if c.weighted:
	data_shape -= 1

Flow = INN(num_coupling_layers=c.n_blocks, in_dim=data_shape, num_layers=c.n_layers, internal_size=c.n_units)
Flow.define_model_architecture()
Flow.set_optimizer()
D = netD(in_dim=data_shape, num_layers=2*c.n_layers, internal_size=2*c.n_units)
D.define_model_architecture_unreg()
D.set_optimizer()

print("\n" + "==="*30 + "\n")
print(F)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in Flow.params_trainable]))
print("\n" + "==="*30 + "\n")
print(D)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in D.params_trainable]))
print("\n" + "==="*30 + "\n")

data = pd.read_hdf('./data/' + c.dataset + '.h5').values
#data_shape = data.shape[1]
#scales = np.std(data,0)

log_dir = c.save_dir

checkpoint_path_F = log_dir + '/' + c.dataset + '/' + '/n_epochs_200/' + '/checkpoint_F_epoch_100.pth'
checkpoint_path_D = log_dir + '/' + c.dataset + '/' + '/n_epochs_200/' + '/checkpoint_D_epoch_100.pth'

Flow, Flow.optim, init_epoch = load_checkpoint(checkpoint_path_F, Flow, Flow.optim)
Flow.model.eval()
D, D.optim, init_epoch = load_checkpoint(checkpoint_path_D, D, D.optim)
D.model.eval()

size = 100000
noise = torch.randn(size, data_shape).double().to(device)

'''Compute primary generator output'''
fake = Flow.model(noise, rev=True).detach().numpy()
fake = fake.reshape(size, data_shape)
real = get_real_data(c.dataset, c.test, size)

'''Compute weights'''
out_D = D(torch.Tensor(fake).double())
weights = torch.exp(out_D).detach().numpy().flatten()

fake *= scales

distributions = Distribution(real, fake, 'target', log_dir + '/' + c.dataset + '/n_epochs_' + str(c.n_epochs), c.dataset, weights=weights, latent=False)
distributions.plot()
