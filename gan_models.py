import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np

import config_gan as c

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#----------------------------------------------------------------------
# Generators
#----------------------------------------------------------------------

class netG(nn.Module):
	def __init__(self, in_dim=2, internal_size=16, num_layers=1, init_zeros=False):
		super(netG, self).__init__()

		self.in_dim = in_dim
		self.internal_size = internal_size
		self.num_layers = num_layers
		self.device = device

		self.params_trainable = list(filter(lambda p: p.requires_grad, self.parameters()))

	def define_model_architecture(self):

		model = nn.ModuleList()

		model.append(nn.Linear(c.latent_dim_gen * self.in_dim, self.internal_size))
		#model.append(nn.ReLU())
		model.append(nn.LeakyReLU(0.1))

		for layer in range(self.num_layers):
			model.append(nn.Linear(self.internal_size, self.internal_size))
			#model.append(nn.ReLU())
			model.append(nn.LeakyReLU(0.1))
	
		model.append(nn.Linear(self.internal_size, self.in_dim))

		self.model = model.double().to(device)
		self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))

	def forward(self, x):
		for l in self.model:
			x = l(x)
		return x

	def set_optimizer(self):
		'''Set optimizer for training'''
		self.optim = torch.optim.Adam(
			self.params_trainable,
			lr=c.lr,
			betas=c.betas,
			eps=1e-6,
			weight_decay=c.weight_decay
		)
		self.scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer=self.optim,
			step_size=1,
			gamma = c.gamma
		)

	def save(self, name):
		torch.save({'opt': self.optim.state_dict(),
					'net': self.state_dict()}, name)

#----------------------------------------------------------------------
# Discriminators
#----------------------------------------------------------------------

class netD(nn.Module):
	def __init__(self, in_dim=2, internal_size=16, num_layers=1, init_zeros=False):
		super(netD, self).__init__()

		self.in_dim = in_dim
		self.internal_size = internal_size
		self.num_layers = num_layers
		self.device = device

		self.params_trainable = list(filter(lambda p: p.requires_grad, self.parameters()))

	def define_model_architecture(self):
		'''define model with spectral normalization regulariazation'''
		model = nn.ModuleList()

		model.append(spectral_norm(nn.Linear(self.in_dim, self.internal_size), n_power_iterations=2))
		#model.append(nn.ReLU())
		model.append(nn.LeakyReLU(0.1))

		for layer in range(self.num_layers):
			model.append(spectral_norm(nn.Linear(self.internal_size, self.internal_size), n_power_iterations=2))
			#model.append(nn.ReLU())
			model.append(nn.LeakyReLU(0.1))
	
		model.append(spectral_norm(nn.Linear(self.internal_size, 1), n_power_iterations=2))

		self.model = model.double().to(device)
		self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))

	def define_model_architecture_unreg(self):
		model = nn.ModuleList()

		model.append(nn.Linear(self.in_dim, self.internal_size))
		#model.append(nn.ReLU())
		model.append(nn.LeakyReLU(0.1))

		for layer in range(self.num_layers):
			model.append(nn.Linear(self.internal_size, self.internal_size))
			#model.append(nn.ReLU())
			model.append(nn.LeakyReLU(0.1))
	
		model.append(nn.Linear(self.internal_size, 1))

		self.model = model.double().to(device)
		self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))

	def forward(self, x):
		for l in self.model:
			x = l(x)
		return x

	def set_optimizer(self):
		'''Set optimizer for training'''
		self.optim = torch.optim.Adam(
			self.params_trainable,
			lr=c.lr,
			betas=c.betas,
			eps=1e-6,
			weight_decay=c.weight_decay
		)
		self.scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer=self.optim,
			step_size=1,
			gamma = c.gamma
		)

	def save(self, name):
		torch.save({'opt': self.optim.state_dict(),
					'net': self.state_dict()}, 
					#'epoch': self.init_epoch}, 
					name)
