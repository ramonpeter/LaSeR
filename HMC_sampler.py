import torch
from torch.autograd import Variable, grad

from utils.train_utils import *
from utils.plotting.distributions import *
from utils.plotting.plots import *
from load_data import *

import os, sys
import time

from GAN_models import netD
from model import INN
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
print(Flow.model)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in Flow.params_trainable]))
print("\n" + "==="*30 + "\n")
print(D)
print('Total parameters: %d' % sum([np.prod(p.size()) for p in D.params_trainable]))
print("\n" + "==="*30 + "\n")

data = pd.read_hdf('./data/' + c.dataset + '.h5').values
data_shape = data.shape[1]

log_dir = c.save_dir

'''Load pretrained baseline model and classifier'''
checkpoint_path_F = log_dir + '/' + c.dataset + '/' + '/n_epochs_200/' + '/checkpoint_F_epoch_100.pth'
checkpoint_path_D = log_dir + '/' + c.dataset + '/' + '/n_epochs_200/' + '/checkpoint_D_epoch_100.pth'

Flow, Flow.optim, init_epoch = load_checkpoint(checkpoint_path_F, Flow, Flow.optim)
Flow.model.eval()
D, D.optim, init_epoch = load_checkpoint(checkpoint_path_D, D, D.optim)
D.model.eval()

class HamiltonMCMC():

	'''pytorch adaptation of Ramon's version'''

	def __init__(self, generator, classifier, latent_dim: int, M = None, L: int = 100, eps: float=1e-2, n_chains=1):

		super(HamiltonMCMC, self).__init__()

		self.generator = generator
		self.classifier = classifier
		self.latent_dim = latent_dim

		if M == None:
			self.M = torch.diag(torch.Tensor([1] * self.latent_dim))
		else:
			self.M = M
	
		self.L = L
		self.eps = eps
		self.n_chains = n_chains

	def U(self, q):
		'''Compute potential defined via the classifier's weights'''
		sq_norm = torch.sum(torch.square(q), dim=-1, keepdim=True)

		return sq_norm / 2 - self.classifier(self.generator.model(q, rev=True).view(self.n_chains,2))

	def grad_U(self, q):
		'''Compute gradient of the potential'''
		
		q.requires_grad = True

		grad_ = grad(self.U(q).sum(), q)[0]

		q = q.detach()

		return grad_

	def leapfrog_step(self, q_init):
		'''Compute trajectories using the leapfrom algorithm'''

		q = q_init
		p_init = torch.randn(q.shape).detach().to(device)
		p = p_init.detach()

		# Make half a step for momentum at the beginning
		p = p - self.eps * self.grad_U(q) / 2

		q=q.detach()
		q_init=q_init.detach()

		# Alternate full steps for position and momentum
		for i in range(self.L):
			# full step position
			with torch.no_grad():
				q = q + self.eps * p
			# make full step momentum, except at end of trajectory
			if i != self.L -1:
				p = p - self.eps * self.grad_U(q)

		# Make half step for momentum at the end
		p = p - self.eps * self.grad_U(q) / 2
		# Negate momentum at and of trajectory to make proposal symmetric
		p = p * -1

		q=q.detach()
		
		# Evaluate potential and kinetic energies 
		with torch.no_grad():
			U_init = self.U(q_init)
			K_init = torch.sum(torch.square(p_init), dim=-1, keepdim=True) / 2
			U_proposed = self.U(q)
			K_proposed = torch.sum(torch.square(p), dim=-1, keepdim=True) / 2

		u = torch.rand(self.n_chains,1).to(device)
		mask = (u < torch.exp(U_init - U_proposed + K_init - K_proposed)).flatten()

		q[~mask] = q_init[~mask]

		return q, torch.sum(mask).detach().numpy()

	def sample(self, latent_dim, n_samples):
		q = torch.normal(0,1.,(self.n_chains, latent_dim)).double().detach().to(device)
		sample = []
		accepted = 0
		
		# Burn in
		for _ in range(1000):
			q, _ = self.leapfrog_step(q)
		print('end burn in')

		for i in range(n_samples):
			q, acc = self.leapfrog_step(q)
			#print(q)
			accepted += acc
			sample.append(q)

			if i % 100 == 0:
				print(accepted)

		acc_rate = accepted/(self.n_chains * n_samples)
		return torch.cat(sample), acc_rate

hamilton = HamiltonMCMC(Flow.model, D, latent_dim=data_shape, L=50, eps=0.004, n_chains=100)
z, rate = hamilton.sample(data_shape, 1000)

print('rate = ', rate)

z = z[:200000]

inv = Flow.model(z, rev=True).view(z.shape[0],2)
inv = inv.detach().numpy() * scales

z = z.detach().numpy()

s1=pd.HDFStore('HMC_latent.h5')
s2=pd.HDFStore('HMC_refined.h5')
s1.append('data', pd.DataFrame(z))
s2.append('data', pd.DataFrame(inv))
s1.close()
s2.close()
