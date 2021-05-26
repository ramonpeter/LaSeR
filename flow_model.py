from FrEIA.framework import *
from FrEIA.modules import *

from coupling_layer import Block
import config_FLOW as c

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class BlockConstructor(nn.Module):
	def __init__(self, num_layers, size_in, size_out,  internal_size=None, dropout=0.):
		super().__init__()
		if internal_size is None:
			internal_size = size_out * 2
		if num_layers < 1:
			raise(ValueError("Subnet size has to be 1 or greater"))
		self.layers = []
		for n in range(num_layers):
			input_dim, output_dim = internal_size, internal_size
			if n == 0:
				input_dim = size_in
			if n == num_layers -1:
				output_dim = size_out
			self.layers.append(nn.Linear(input_dim, output_dim))
			if n < num_layers -1:
				self.layers.append(nn.Dropout(p=dropout))
				self.layers.append(nn.LeakyReLU(negative_slope=0.1))
		self.layers = nn.ModuleList(self.layers)

	def forward(self, x):
		for l in self.layers:
			x = l(x)
		return x

class INN:
	def __init__(self, in_dim=2, num_coupling_layers=1, internal_size=16, num_layers=1, init_zeros=False,dropout=False):

		self.in_dim = in_dim
		self.n_blocks = num_coupling_layers
		self.internal_size = internal_size
		self.num_layers = num_layers
		self.clamping = 1
		self.device = device

	def define_model_architecture(self): 
	
		input_dim = (self.in_dim,1)

		nodes = [InputNode(*input_dim, name='input')]

		nodes.append(Node([nodes[-1].out0], Flatten, {}, name='flatten'))
		
		for i in range(self.n_blocks):
			nodes.append(
				Node(
					[nodes[-1].out0], 
					PermuteRandom,
					{'seed':i}, 
					name=F'permute_{i}'
				)
			)
			nodes.append(
				Node(
					[nodes[-1].out0], 
					Block,
					{
						'clamp' : self.clamping,
						'subnet_constructor' : BlockConstructor,
						'internal_size' : self.internal_size,
						'num_layers' : self.num_layers,
					},
					name = F'block_{i}'
				)
			)
		
		
		nodes.append(OutputNode([nodes[-1].out0], name='out'))

		self.model = ReversibleGraphNet(nodes, verbose=False).double().to(self.device)
		self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
	
	def set_optimizer(self):
		
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
		"""
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			self.optim,
			factor = 0.4,
			patience=50,
			cooldown=150,
			threshold=5e-5,
			threshold_mode='rel',
			verbose=True
		)
		"""

	def initialize_train_loader(self, data):
		self.train_loader = torch.utils.data.DataLoader(
			torch.from_numpy(data),
			batch_size=c.batch_size, 
			shuffle=True, 
			drop_last=True,
		)


	def save(self, name):
		torch.save({'opt':self.optim.state_dict(),
					'net':self.model.state_dict()}, name)
	
	def load(self, name):
		state_dicts = torch.load(name, map_location=self.device)
		self.model.load_state_dict(state_dicts['net'])
		try:
			self.optim.load_state_dict(state_dicts['opt'])
		except ValueError:
			print('Cannot load optimizer for some reason or other')
