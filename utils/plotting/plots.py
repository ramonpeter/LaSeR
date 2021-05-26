import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import kde
import sys
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter

import matplotlib.colors as mcolors
import matplotlib.cm as cm

from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

gcolor = '#3b528b'
dcolor = '#e41a1c'
teal = '#10AB75'

FONTSIZE = 20 #18, (20 2d) 12 (if latex is false in distributions)

############################################
# Functions
############################################

class ScalarFormatterForceFormat(ScalarFormatter):
	def _set_format(self):	# Override function that finds format to use.
		self.format = "%1.1f"  # Give format her

class AnyObjectHandler(HandlerBase):
	def create_artists(self, legend, orig_handle,
					   x0, y0, width, height, fontsize, trans):
		l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
						   linestyle=orig_handle[1], color=orig_handle[0])
		l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
						   linestyle = orig_handle[2], color = orig_handle[0])
		return [l1, l2]

def plot_distribution_ratio(fig, axs, y_train, y_predict, args, weights, extra):
	# Particle_id, observable, bins, range, x_label, log_scale

	y_train = args[1](y_train, args[0])
	y_predict = args[1](y_predict, args[0])

	#y_extra = args[1](extra, args[0])

	if args[6]:
		axs[0].set_yscale('log')
	
	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])#, weights=weights)
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])#, weights=weights)

	#y_w, x_w = np.histogram(y_predict, args[2], density=True, range=args[3], weights=weights)
	#y_e, x_e = np.histogram(y_extra, args[2], density=True, range=args[3])

	axs[0].step(x_t[:args[2]], y_t, gcolor, label='Truth', linewidth=1.0, where='mid')
	axs[0].step(x_t[:args[2]], y_p, dcolor, label='DCTRFLOW', linewidth=1.0, where='mid')
	#axs[0].step(x_e[:args[2]], y_e, teal, label='LSRFLOW', linewidth=1.0, where='mid')
	#axs[0].step(x_t[:args[2]], y_w, teal, label='DCTRGAN', linewidth=1.0, where='mid')

	for j in range(2):
		for label in ( [axs[j].yaxis.get_offset_text()] +
					  axs[j].get_xticklabels() + axs[j].get_yticklabels()):
			label.set_fontsize(FONTSIZE)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs[0].yaxis.set_major_formatter(yfmt)

	axs[0].set_ylabel('Normalized', fontsize = FONTSIZE)
	axs[0].legend(loc='upper center', prop={'size': int(FONTSIZE-6)}, frameon=False)

	axs[1].set_ylabel(r'$\frac{\text{Truth}}{\text{INN}}$', fontsize = FONTSIZE-2)

	y_r = (y_p)/y_t
	y_r [np.isnan(y_r )==True]=1
	y_r [y_r==np.inf]=1

	#y_r3 = (y_e)/y_t
	#y_r3 [np.isnan(y_r3)==True]=1
	#y_r3 [y_r3==np.inf]=1

	#y_r2 = (y_w)/y_t
	#y_r2 [np.isnan(y_r2)==True]=1
	#y_r2 [y_r2==np.inf]=1

	axs[1].step(x_t[:args[2]], y_r, dcolor, linewidth=1.0, where='mid')
	#axs[1].step(x_t[:args[2]], y_r3, teal, linewidth=1.0, where='mid')
	#axs[1].step(x_t[:args[2]], y_r2, teal, linewidth=1.0, where='mid')
	axs[1].set_ylim((0.82,1.18))
	axs[1].set_yticks([0.9, 1.0, 1.1])
	axs[1].set_yticklabels([r'$0.9$', r'$1.0$', "$1.1$"])

	axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')
	axs[1].axhline(y=2,linewidth=1, linestyle='--', color='grey')
	#axs[1].axhline(y=1.1,linewidth=1, linestyle='--', color='grey')
	#axs[1].axhline(y=0.9,linewidth=1, linestyle='--', color='grey')
	axs[1].set_xlabel(args[4], fontsize = FONTSIZE)

def plot_distribution(fig, axs, y_train, y_predict, args):
	"""Plot the distributions"""
	# Particle_id, observable, bins, range, x_label, log_scale
	y_train = args[1](y_train, args[0])

	y_predict = args[1](y_predict, args[0])

	if args[6]:
		axs.set_yscale('log')

	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])
	#y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3], weights=w)

	axs.step(x_t[:args[2]], y_t, '#f03b20', label='Truth', linewidth=1.0, where='mid')
	axs.step(x_p[:args[2]], y_p, '#2c7fb8', label='INN', linewidth=1.0, where='mid')

	for label in ( [axs.yaxis.get_offset_text()] +
				  axs.get_xticklabels() + axs.get_yticklabels()):
		label.set_fontsize(FONTSIZE)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	axs.yaxis.set_major_formatter(yfmt)

	axs.set_ylabel('Normalized', fontsize = FONTSIZE)
	axs.legend(loc='upper right', prop={'size':(FONTSIZE-4)}, frameon=False)
	axs.set_xlabel(r'$%s$' %(args[5]), fontsize = FONTSIZE)

def save_distribution(y_train, y_predict, args):
	"""Plot the distributions"""
	# Particle_id, observable, bins, range, x_label, log_scale

	y_train = args[1](y_train, args[0])
	y_predict = args[1](y_predict, args[0])

	y_t, x_t = np.histogram(y_train, args[2], density=True, range=args[3])
	y_p, x_p = np.histogram(y_predict, args[2], density=True, range=args[3])
	x_p = x_p[0:args[2]]

	histo = np.vstack((x_p,y_t,y_p)).T

	return histo

def plot_2d_distribution(fig, axs, y_train, y_predict, args1, args2, weights, extra):
	"""Plot the distributions"""
	fontsize=FONTSIZE+2
	data = [[0.,0.], [0.,0.]]
	h = [[0.,0.], [0.,0.]]

	# Fill (x,y) with data
	data[0][0] = args1[1](y_train, args1[0])
	data[1][0] = args2[1](y_train, args2[0])

	data[0][1] = args1[1](y_predict, args1[0])
	data[1][1] = args2[1](y_predict, args2[0])

	#e1 = args1[1](extra, args1[0])
	#e2 = args2[1](extra, args1[0])

	h[0][0], xedges, yedges = np.histogram2d(data[0][0], data[1][0], bins=args1[2], range=(args1[3],args2[3]), density=True)#, weights=weights)
	h[0][1], xedges, yedges = np.histogram2d(data[0][1], data[1][1], bins=args1[2], range=(args1[3],args2[3]), density=True)#, weights=weights)
	#h[1][0], xedges, yedges = np.histogram2d(data[0][1], data[1][1], bins=args1[2], range=(args1[3],args2[3]), density=True, weights=weights)
	#h[1][0], xedges, yedges = np.histogram2d(e1, e2, bins=args1[2], range=(args1[3],args2[3]), density=True)
	
	h[1][0] = (h[0][1]-h[0][0])/(h[0][1]+h[0][0])
	h[1][0][np.isnan(h[1][0])==True]=0 #1
	h[1][0][h[1][0]==np.inf]=0 #1
	#h[1][0][h[1][0]==0]=1

	for j in range(3):
		for label in ( [axs[j].yaxis.get_offset_text()] +
					  axs[j].get_xticklabels() + axs[j].get_yticklabels()):
			label.set_fontsize(fontsize)

	Z = h[0][0].T
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[0].pcolormesh(xedges, yedges, Z, rasterized=True)
	divider = make_axes_locatable(axs[0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[0].set_xlabel(args1[4], fontsize = fontsize)
	axs[0].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)

	Z = h[0][1].T
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[1].pcolormesh(xedges, yedges, Z, rasterized=True)
	divider = make_axes_locatable(axs[1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[1].set_xlabel(args1[4], fontsize = fontsize)
	axs[1].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)



	Z = h[1][0].T
	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[2].pcolormesh(xedges, yedges, Z, rasterized=True)
	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[2].set_xlabel(args1[4], fontsize = fontsize)
	axs[2].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)

	"""

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs[2].pcolormesh(xedges, yedges, h[1][0].T, rasterized=True)
	divider = make_axes_locatable(axs[2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs[2].set_xlabel(args1[4], fontsize = fontsize)
	axs[2].set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)
	"""

def plot_2d_distribution_single(fig, axs, y_train, y_predict, args1, args2, weights):

	fontsize = FONTSIZE#+10
	"""Plot the distributions"""
	data = [[0.,0.], [0.,0.]]
	h    = [[0.,0.], [0.,0.]]

	# Fill (x,y) with data
	data[0][0] = args1[1](y_train, args1[0])
	data[1][0] = args2[1](y_train, args2[0])

	data[0][1] = args1[1](y_predict, args1[0])
	data[1][1] = args2[1](y_predict, args2[0])

	h[0][1], xedges, yedges = np.histogram2d(data[0][0], data[1][0], bins=args1[2], range=(args1[3],args2[3]), density=True, weights=weights)

	if args1[6]:
		axs.set_xscale('log')
	if args2[6]:
		axs.set_yscale('log')

	for label in ([axs.yaxis.get_offset_text()] + axs.get_xticklabels() + axs.get_yticklabels()):
		label.set_fontsize(fontsize)

	yfmt = ScalarFormatterForceFormat()
	yfmt.set_powerlimits((0,0))
	im = axs.pcolormesh(xedges, yedges, h[0][1].T, rasterized=True)
	divider = make_axes_locatable(axs)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	axs.set_xlabel(args1[4], fontsize = fontsize)
	axs.set_ylabel(args2[4], fontsize = fontsize)
	cbar = fig.colorbar(im, cax=cax, orientation='vertical')
	cbar.ax.tick_params(labelsize=fontsize)
	cbar.ax.yaxis.set_major_formatter(yfmt)
	for label in ([cbar.ax.yaxis.get_offset_text()]):
		label.set_fontsize(fontsize)

