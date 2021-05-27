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
		self.weights = weights
		self.extra_data = extra_data

	def plot(self):
		if self.latent == True:
			self.latent_distributions()
		else:
			if self.dataset == 'eight':
				self.eight_distributions()
			elif self.dataset == '4_gaussians':
				self.four_gaussian_distributions()
			elif self.dataset == '3_rings':
				self.three_rings_distributions()
			else:
				self.basic_2d_distributions()

		if True:
			plt.rc('text', usetex=True)
			plt.rc('font', family='serif')
			plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

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
						plot_2d_distribution(fig, axs, self.real_data, self.gen_data, self.args2[observable], self.args2[observable2], self.weights, self.extra_data)
						plt.subplots_adjust(wspace=0.45, hspace=0.25)
						fig.savefig(pp, bbox_inches='tight', format='pdf', pad_inches=0.05)
						plt.close()

	def basic_2d_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coordinate_0, 100, (-10.2,10.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coordinate_0, 100, (-10.2,10.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def four_gaussian_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coordinate_0, 100, (-10.2,10.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 100, (-10.2,10.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coordinate_0, 100, (-8.2,8.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def eight_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coordinate_0, 100, (-12.2,12.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coordinate_0, 200, (-12.2,12.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 200, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def three_rings_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'x' : ([0], self.coordinate_0, 100, (-20.2,20.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 100, (-8.2,8.2) ,r'$y$', r'y',False),
		}	 

		args2 = {			 
			'x' : ([0], self.coordinate_0, 200, (-20.2,20.2) ,r'$x$', r'x',False),
			'y' : ([0], self.coordinate_1, 200, (-8.2,8.2) ,r'$y$', r'y',False),
		}
	 
		self.args = args
		self.args2 = args2

	def latent_distributions(self):
		 # Particle_id, observable, bins, range, x_label, log_scale

		args = {			 
			'z0' : ([0], self.coordinate_0, 60, (-4,4) ,r'$z_0$', r'z_0',False),
			'z1' : ([0], self.coordinate_1, 60, (-4,4) ,r'$z_1$', r'z_1',False),
			#'z2' : ([0], self.coordinate_2, 60, (-4,4) ,r'$z_2$', r'z_2',False),
			#'z3' : ([0], self.coordinate_3, 60, (-4,4) ,r'$z_3$', r'z_3',False),
			#'z4' : ([0], self.coordinate_4, 60, (-4,4) ,r'$z_4$', r'z_4',False),
			#'z5' : ([0], self.coordinate_5, 60, (-4,4) ,r'$z_5$', r'z_5',False),
			#'z6' : ([0], self.coordinate_6, 60, (-4,4) ,r'$z_6$', r'z_6',False),
			#'z7' : ([0], self.coordinate_7, 60, (-4,4) ,r'$z_7$', r'z_7',False),
			#'z8' : ([0], self.coordinate_8, 60, (-4,4) ,r'$z_8$', r'z_8',False),
		}	 

		args2 = {			 
			'z0' : ([0], self.coordinate_0, 200, (-3,3) ,r'$z_0$', r'z_0',False),
			'z1' : ([0], self.coordinate_1, 200, (-3,3) ,r'$z_1$', r'z_1',False),
			#'z2' : ([0], self.coordinate_2, 200, (-4,4) ,r'$z_2$', r'z_2',False),
			#'z3' : ([0], self.coordinate_3, 200, (-4,4) ,r'$z_3$', r'z_3',False),
			#'z4' : ([0], self.coordinate_4, 200, (-4,4) ,r'$z_4$', r'z_4',False),
			#'z5' : ([0], self.coordinate_5, 120, (-4,4) ,r'$z_5$', r'z_5',False),
			#'z6' : ([0], self.coordinate_6, 120, (-4,4) ,r'$z_6$', r'z_6',False),
			#'z7' : ([0], self.coordinate_7, 120, (-4,4) ,r'$z_7$', r'z_7',False),
			#'z8' : ([0], self.coordinate_8, 120, (-4,4) ,r'$z_8$', r'z_8',False),
		}	 
	 
		self.args = args
		self.args2 = args2
