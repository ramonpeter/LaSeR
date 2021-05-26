############################################
# Imports
############################################

from utils.plotting.plots import *
from utils.observables import Observable

import numpy as np

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

############################################
# Plotting
############################################

class Distribution(Observable):
	"""Custom Distribution class.

	Defines which Observables will be plotted depending on the
	specified dataset.
	"""
	def __init__(self,
				 real_data,
				 gen_data,
				 name,
				 log_dir,
				 dataset,
				 mean=[],
				 std=[],
				 latent=False,
				 weights=[],
				 extra_data=[]):
		super(Distribution, self).__init__()
		self.real_data = real_data
		self.gen_data = gen_data
		self.name = name
		self.log_dir = log_dir
		self.dataset = dataset
		self.latent = latent
		self.mean = mean
		self.std = std
		self.weights = weights
		self.extra_data = extra_data

	def plot(self):
		if self.latent == True:
			self.latent_distributions()
		else:
			if self.dataset == 'Drell_Yan':
				self.drell_yan_distributions()
			if self.dataset == 'w_2jets':
				self.w_2jets_distributions()
			elif self.dataset == '2d_ring_gaussian':
				self.basic_2d_distributions()
			elif self.dataset == '2d_eight':
				self.eight_distributions()
			elif self.dataset == '2d_4_gaussian_mixture':
				self.four_gaussian_distributions()
			elif self.dataset == '2d_3_holes':
				self.three_holes_distributions()
			elif self.dataset == '2d_linear':
				self.basic_2d_distributions()
			elif self.dataset == '1d_camel':
				self.basic_1d_distributions()
			else:
				self.basic_2d_distributions()

		if True:
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')
			plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

		if False:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(1)
					plot_distribution(fig, axs, self.real_data, self.gen_data, self.args[observable])
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()

		if True:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_ratio.pdf') as pp:
				for observable in self.args.keys():
					fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1], 'hspace' : 0.00})
					plot_distribution_ratio(fig, axs, self.real_data, self.gen_data,  self.args[observable], self.weights, self.extra_data)
					fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
					plt.close()

		if True:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1,3, figsize=(20,5))
						#fig, axs = plt.subplots(1)
						plot_2d_distribution(fig, axs, self.real_data, self.gen_data, self.args2[observable], self.args2[observable2], self.weights, self.extra_data)
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()

		if False:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1,2, figsize=(15,5))
						plot_2d_distribution_2(fig, axs, self.real_data, self.gen_data, self.args2[observable], self.args2[observable2], self.weights, self.extra_data)
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()

		if False:
			with PdfPages(self.log_dir + '/' + self.dataset + '_' + self.name + '_2d.pdf') as pp:
				for i, observable in enumerate(list(self.args2.keys())):
					for observable2 in list(self.args2.keys())[i+1:]:
						fig, axs = plt.subplots(1)
						plot_2d_distribution_single(fig, axs, self.real_data, self.gen_data, self.args2[observable], self.args2[observable2], self.weights)
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()

	def basic_1d_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.roll_0, 50, (-7.2,7.2) ,r'$x$', r'x',False),
		}	 

		self.args = args

	def basic_2d_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.roll_0, 100, (-10.2,10.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.roll_0, 100, (-10.2,10.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def four_gaussian_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.roll_0, 100, (-10.2,10.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 100, (-10.2,10.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.roll_0, 100, (-8.2,8.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def eight_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.roll_0, 100, (-12.2,12.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.roll_0, 200, (-12.2,12.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 200, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def three_holes_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.roll_0, 100, (-20.2,20.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.roll_0, 200, (-20.2,20.2) ,r'$x$', r'x',False),
			'y' : ([0], self.roll_1, 200, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def drell_yan_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'pte1' : ([0], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^-}$ [GeV]', r'p_{T, e^-}',False),
		 	'pxe1' : ([0], self.x_momentum, 50, (-80,80), r'$p_{\mathrm{x}, e^-}$ [GeV]', r'p_{x, e^-}',False),
			'pye1' : ([0], self.y_momentum, 50, (-80,80), r'$p_{\mathrm{y}, e^-}$ [GeV]', r'p_{y, e^-}',False),
			'pze1' : ([0], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, e^-}$ [GeV]', r'p_{z, e^-}',False),
			'Ee1'  : ([0], self.energy, 40, (0,300), r'$E_{e^-}$ [GeV]', r'E_{e^-}',False),
			#---------------------#		
			#'pte2' : ([1], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^+}$ [GeV]', r'p_{T, e^+}',False),
		 	#'pxe2' : ([1], self.x_momentum, 40, (-80,80), r'$p_{\mathrm{x}, e^+}$ [GeV]', r'p_{x, e^+}',False),
			#'pye2' : ([1], self.y_momentum, 40, (-80,80), r'$p_{\mathrm{y}, e^+}$ [GeV]', r'p_{y, e^+}',False),
			#'pze2' : ([1], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, e^+}$ [GeV]', r'p_{z, e^+}',False),
			#'Ee2'  : ([1], self.energy, 40, (0,300), r'$E_{e^+}$ [GeV]', r'E_{e^+}',False),
			#---------------------#			
			'ptZ' : ([0,1], self.transverse_momentum, 40, (-2,2) ,r'$p_{T, e^- e^+}$ [GeV]', r'p_{T, e^- e^+}',False),
			#'pxZ' : ([0,1], self.x_momentum, 40, (-2,2), r'$p_{\mathrm{x}, e^- e^+}$ [GeV]', r'p_{x, e^- e^+}',False),
			#'pyZ' : ([0,1], self.y_momentum, 40, (-200,200), r'$p_{\mathrm{y}, e^- e^+}$ [GeV]', r'p_{y, e^- e^+}',False),
			#'pzZ' : ([0,1], self.z_momentum, 40, (-750,750), r'$p_{\mathrm{z}, e^- e^+}$ [GeV]', r'p_{z, e^- e^+}',False),
			'EZ'  : ([0,1], self.energy, 40, (40,500), r'$E_{e^- e^+}$ [GeV]', r'E_{e^- e^+}',False),
			'MZ'  : ([0,1], self.invariant_mass, 40, (82,100), r'$m_{e^- e^+}$ [GeV]', r'm_{e^- e^+}',False),
			#---------------------#			
		}	 

		args2 = {			 
			'pte1' : ([0], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^-}$ [GeV]', r'p_{T, e^-}',False),
		 	'pxe1' : ([0], self.x_momentum, 40, (-80,80), r'$p_{\mathrm{x}, e^-}$ [GeV]', r'p_{x, e^-}',False),
			'pye1' : ([0], self.y_momentum, 40, (-80,80), r'$p_{\mathrm{y}, e^-}$ [GeV]', r'p_{y, e^-}',False),
			'pze1' : ([0], self.z_momentum, 40, (-400,400), r'$p_{\mathrm{z}, e^-}$ [GeV]', r'p_{z, e^-}',False),
			'Ee1'  : ([0], self.energy, 40, (0,300), r'$E_{e^-}$ [GeV]', r'E_{e^-}',False),
			#---------------------#		
			#'pte2' : ([1], self.transverse_momentum, 40, (0,60) ,r'$p_{T, e^+}$ [GeV]', r'p_{T, e^+}',False),
		 	#'pxe2' : ([1], self.x_momentum, 40, (-80,80), r'$p_{\mathrm{x}, e^+}$ [GeV]', r'p_{x, e^+}',False),
			#'pye2' : ([1], self.y_momentum, 40, (-80,80), r'$p_{\mathrm{y}, e^+}$ [GeV]', r'p_{y, e^+}',False),
			'pze2' : ([1], self.z_momentum, 40, (-350,350), r'$p_{\mathrm{z}, e^+}$ [GeV]', r'p_{z, e^+}',False),
			#'Ee2'  : ([1], self.energy, 40, (0,300), r'$E_{e^+}$ [GeV]', r'E_{e^+}',False),
			#---------------------#			
			'ptZ' : ([0,1], self.transverse_momentum, 40, (-2,2) ,r'$p_{T, e^- e^+}$ [GeV]', r'p_{T, e^- e^+}',False),
			#'pxZ' : ([0,1], self.x_momentum, 40, (-2,2), r'$p_{\mathrm{x}, e^- e^+}$ [GeV]', r'p_{x, e^- e^+}',False),
			#'pyZ' : ([0,1], self.y_momentum, 40, (-200,200), r'$p_{\mathrm{y}, e^- e^+}$ [GeV]', r'p_{y, e^- e^+}',False),
			#'pzZ' : ([0,1], self.z_momentum, 40, (-750,750), r'$p_{\mathrm{z}, e^- e^+}$ [GeV]', r'p_{z, e^- e^+}',False),
			#'EZ'  : ([0,1], self.energy, 40, (40,500), r'$E_{e^- e^+}$ [GeV]', r'E_{e^- e^+}',False),
			'MZ'  : ([0,1], self.invariant_mass, 40, (82,100), r'$m_{e^- e^+}$ [GeV]', r'm_{e^- e^+}',False),
			#---------------------#			
		}	 
	 
		self.args = args
		self.args2 = args2

	def w_2jets_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{T, W}$ [GeV]', r'p_{T, W}',False),
		 	'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{\mathrm{x}, W}$ [GeV]', r'p_{x, W}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{\mathrm{y}, W}$ [GeV]', r'p_{y, W}',False),
			'pzW' : ([0], self.z_momentum, 50, (-600,600), r'$p_{\mathrm{z}, W}$ [GeV]', r'p_{z, W}',False),
			'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{W}$ [GeV]', r'E_{W}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j1}$ [GeV]', r'p_{T, j1}',False),
		 	'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j1}$ [GeV]', r'p_{x, j1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j1}$ [GeV]', r'p_{y, j1}',False),
			'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j1}$ [GeV]', r'p_{z, j1}',False),
			'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{j1}$ [GeV]', r'E_{j1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j2}$ [GeV]', r'p_{T, j2}',False),
		 	'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j2}$ [GeV]', r'p_{x, j2}',False),
			'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j2}$ [GeV]', r'p_{y, j2}',False),
			'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j2}$ [GeV]', r'p_{z, j2}',False),
			'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{j2}$ [GeV]', r'E_{j2}',False),
			#---------------------#			
		}	 

		args2 = {			 
			'ptW' : ([0], self.transverse_momentum, 40, (0,300) ,r'$p_{T, W}$ [GeV]', r'p_{T, W}',False),
		 	'pxW' : ([0], self.x_momentum, 50, (-160,160), r'$p_{\mathrm{x}, W}$ [GeV]', r'p_{x, W}',False),
			'pyW' : ([0], self.y_momentum, 50, (-160,160), r'$p_{\mathrm{y}, W}$ [GeV]', r'p_{y, W}',False),
			#'pzW' : ([0], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, W}$ [GeV]', r'p_{z, W}',False),
			#'EW'  : ([0], self.energy, 40, (0,1000), r'$E_{W}$ [GeV]', r'E_{W}',False),
			#---------------------#		
			'ptj1' : ([1], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j1}$ [GeV]', r'p_{T, j1}',False),
		 	'pxj1' : ([1], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j1}$ [GeV]', r'p_{x, j1}',False),
			'pyj1' : ([1], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j1}$ [GeV]', r'p_{y, j1}',False),
			#'pzj1' : ([1], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j1}$ [GeV]', r'p_{z, j1}',False),
			#'Ej1'  : ([1], self.energy, 40, (0,600), r'$E_{j1}$ [GeV]', r'E_{j1}',False),
			#---------------------#			
			'ptj2' : ([2], self.transverse_momentum, 40, (0,180) ,r'$p_{T, j2}$ [GeV]', r'p_{T, j2}',False),
		 	'pxj2' : ([2], self.x_momentum, 40, (-120,120), r'$p_{\mathrm{x}, j2}$ [GeV]', r'p_{x, j2}',False),
			'pyj2' : ([2], self.y_momentum, 40, (-120,120), r'$p_{\mathrm{y}, j2}$ [GeV]', r'p_{y, j2}',False),
			#'pzj2' : ([2], self.z_momentum, 50, (-400,400), r'$p_{\mathrm{z}, j2}$ [GeV]', r'p_{z, j2}',False),
			#'Ej2'  : ([2], self.energy, 40, (0,600), r'$E_{j2}$ [GeV]', r'E_{j2}',False),
			#---------------------#			
		}	 
	 
		self.args = args
		self.args2 = args2

	def latent_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'z0' : ([0], self.roll_0, 60, (-4,4) ,r'$z_0$', r'z_0',False),
			'z1' : ([0], self.roll_1, 60, (-4,4) ,r'$z_1$', r'z_1',False),
			#'z2' : ([0], self.roll_2, 60, (-4,4) ,r'$z_2$', r'z_2',False),
			#'z3' : ([0], self.roll_3, 60, (-4,4) ,r'$z_3$', r'z_3',False),
			#'z4' : ([0], self.roll_4, 60, (-4,4) ,r'$z_4$', r'z_4',False),
			#'z5' : ([0], self.roll_5, 60, (-4,4) ,r'$z_5$', r'z_5',False),
			#'z6' : ([0], self.roll_6, 60, (-4,4) ,r'$z_6$', r'z_6',False),
			#'z7' : ([0], self.roll_7, 60, (-4,4) ,r'$z_7$', r'z_7',False),
			#'z8' : ([0], self.roll_8, 60, (-4,4) ,r'$z_8$', r'z_8',False),
			#'z9' : ([0], self.roll_9, 60, (-4,4) ,r'$z_9$', r'z_9',False),
			#'z10' : ([0], self.roll_10, 60, (-4,4) ,r'$z_{10}$', r'z_{10}',False),
			#'z11' : ([0], self.roll_11, 60, (-4,4) ,r'$z_{11}$', r'z_{11}',False),
		}	 

		args2 = {			 
			'z0' : ([0], self.roll_0, 200, (-3,3) ,r'$z_0$', r'z_0',False),
			'z1' : ([0], self.roll_1, 200, (-3,3) ,r'$z_1$', r'z_1',False),
			#'z2' : ([0], self.roll_2, 200, (-4,4) ,r'$z_2$', r'z_2',False),
			#'z3' : ([0], self.roll_3, 200, (-4,4) ,r'$z_3$', r'z_3',False),
			#'z4' : ([0], self.roll_4, 200, (-4,4) ,r'$z_4$', r'z_4',False),
			#'z5' : ([0], self.roll_5, 120, (-4,4) ,r'$z_5$', r'z_5',False),
			#'z6' : ([0], self.roll_6, 120, (-4,4) ,r'$z_6$', r'z_6',False),
			#'z7' : ([0], self.roll_7, 120, (-4,4) ,r'$z_7$', r'z_7',False),
			#'z8' : ([0], self.roll_8, 120, (-4,4) ,r'$z_8$', r'z_8',False),
			#'z9' : ([0], self.roll_9, 120, (-4,4) ,r'$z_9$', r'z_9',False),
			#'z10' : ([0], self.roll_10, 120, (-4,4) ,r'$z_{10}$', r'z_{10}',False),
			#'z11' : ([0], self.roll_11, 120, (-4,4) ,r'$z_{11}$', r'z_{11}',False),
		}	 
	 
		self.args = args
		self.args2 = args2
