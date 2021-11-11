import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

class ScalarFormatterForceFormat(ScalarFormatter):
    # Remove vmin, vmax, not needed for mathplotlib 3
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format her
        
def plot_2d(name, true, true_weight, gen):
    
    # generate random noise
    gcolor = '#3b528b'
    dcolor = '#e41a1c'
    
    # Use Latex to compile
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    
    # generate random noise
    fig, axs = plt.subplots(2,2, figsize=(12,9))
    
    
    FONTSIZE = 16   
    fontsize = FONTSIZE
    
    truth_weight = true_weight[:,0]
    gen_weight = truth_weight.mean() * np.ones_like(truth_weight)
    
    data = [[0.,0.], [0.,0.]]
    h = [[0.,0.], [0.,0.]]
    
    # Truth
    data[0][0] = true[:,0]
    data[1][0] = true[:,1]

    # Pred
    data[0][1] = gen[:,0]
    data[1][1] = gen[:,1]
    
    h[0][0], xedges, yedges = np.histogram2d(data[0][0], data[1][0], weights = truth_weight,  bins=50, range=([0,1],[0,1]), density=True)
    h[0][1], xedges, yedges = np.histogram2d(data[0][1], data[1][1], weights = gen_weight, bins=50, range=([0,1],[0,1]), density=True)
#    ratio = h[0][1]/ h[0][0]
    h[1][0] = (h[0][0]-h[0][1])/(h[0][1]+h[0][0])
    h_max = np.max([h[0][0].max(),h[0][1].max()])

    delta = 4/50
    mean = 0.5
    lim_max = mean+delta
    lim_min = mean-delta

#    sys.exit()

    truth = true[:,0]
    truth = truth[data[1][0]<=lim_max]
    selector1 = data[1][0][data[1][0]<=lim_max]
    truth = truth[selector1>=lim_min]
    
    weight_1d = truth_weight
    weight_1d = weight_1d[data[1][0]<=lim_max]
    weight_1d = weight_1d[selector1>=lim_min]

    pred = gen[:,0]
    pred = pred[data[1][1]<=lim_max]
    selector2 = data[1][1][data[1][1]<=lim_max]
    pred = pred[selector2>=lim_min]

    weight_1d_mean = truth_weight.mean() * np.ones_like(pred)
        
    w_t = 1/len(truth)

    bins1d = 50
    
    y_t, x_t = np.histogram(truth, bins1d, weights=weight_1d, range=[0,1])
    y_p, x_p = np.histogram(pred, bins1d, weights=weight_1d_mean, range=[0,1])

    h[1][0][np.isnan(h[1][0])==True]=0 #1
    h[1][0][h[1][0]==1]=0 #1
#    h[1][0][h[1][0]==0]=1

    yfmt = ScalarFormatterForceFormat()
    yfmt.set_powerlimits((0,0))

    fig, axs = plt.subplots(1)
    plt.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.15)
    
    for label in ( [axs.yaxis.get_offset_text()] +
                            axs.get_yticklabels() + axs.get_xticklabels()):
                label.set_fontsize(fontsize)
    
    im = axs.pcolormesh(xedges, yedges, h[0][0].T, rasterized=True, vmin=0, vmax=h_max)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axs.set_xlabel(r'$x$', fontsize = fontsize)
    axs.set_ylabel(r'$y$', fontsize = fontsize)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.set_major_formatter(yfmt)
    for label in ( [cbar.ax.yaxis.get_offset_text()]):
        label.set_fontsize(fontsize)
        
    fig.savefig("truth_2d.pdf", format='pdf')
    plt.close()
    
    # plot GAN
    
    fig, axs = plt.subplots(1)
    plt.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.15)
    
    for label in ( [axs.yaxis.get_offset_text()] +
                            axs.get_yticklabels() + axs.get_xticklabels()):
                label.set_fontsize(fontsize)
    
    im = axs.pcolormesh(xedges, yedges, h[0][1].T, rasterized=True, vmin=0, vmax=h_max)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axs.set_xlabel(r'$x$', fontsize = fontsize)
    axs.set_ylabel(r'$y$', fontsize = fontsize)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.set_major_formatter(yfmt)
    for label in ( [cbar.ax.yaxis.get_offset_text()]):
        label.set_fontsize(fontsize)
        
    fig.savefig("gen_2d.pdf", format='pdf')
    plt.close()
        
    
    # plot asymmetrie
    
    fig, axs = plt.subplots(1)
    plt.subplots_adjust(left=0.12, right=0.92, top=0.92, bottom=0.15)
    
    for label in ( [axs.yaxis.get_offset_text()] +
                            axs.get_yticklabels() + axs.get_xticklabels()):
                label.set_fontsize(fontsize)
    
    im = axs.pcolormesh(xedges, yedges, h[1][0].T, rasterized=True, vmin=-1, vmax=1)
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    axs.set_xlabel(r'$x$', fontsize = fontsize)
    axs.set_ylabel(r'$y$', fontsize = fontsize)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.set_major_formatter(yfmt)
    for label in ( [cbar.ax.yaxis.get_offset_text()]):
        label.set_fontsize(fontsize)
        
    fig.savefig("asymmetry_2d.pdf", format='pdf')
    plt.close()
    
    
    # plot sliced
    
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios' : [4, 1], 'hspace' : 0.00})
    plt.subplots_adjust(left=0.12, right=0.925, top=0.92, bottom=0.15)
    
    for j in range(2):
        for label in ( [axs[j].yaxis.get_offset_text()] +
                      axs[j].get_xticklabels() + axs[j].get_yticklabels()):
            label.set_fontsize(FONTSIZE-2)
    
    axs[0].text(0.015,0.11,
                    r'slice at $y=0.5$',
                    fontsize = fontsize
                )
    axs[0].yaxis.set_major_formatter(yfmt)
    axs[0].step(x_p[:bins1d], w_t*y_t, dcolor, label='Truth', linewidth=1.0, where='mid')
    axs[0].step(x_p[:bins1d], w_t*y_p, gcolor, label='GAN', linewidth=1.0, where='mid')
    axs[0].set_ylabel(r'Normalized', fontsize = fontsize)
    axs[0].legend(loc='upper right', prop={'size':(fontsize-2)}, frameon=False)
    axs[0].set_ylim((-0.005,0.125))
    
    # middle panel

    axs[1].set_ylabel(r'$\frac{\mathrm{Truth}}{\mathrm{GAN}}$', fontsize = FONTSIZE)
    
    y_r = (y_t)/(y_p)

    #statistic
    r_stat = np.sqrt(y_t * (y_p + y_t)/((y_p)**3))
    r_statp = y_r + r_stat
    r_statm = y_r - r_stat
    

    axs[1].step(x_t[:bins1d], y_r, 'black', linewidth=1.0, where='mid')
    axs[1].step(x_t[:bins1d], r_statp, color='grey', label='$+- stat$', linewidth=0.5, where='mid')
    axs[1].step(x_t[:bins1d], r_statm, color='grey', linewidth=0.5, where='mid')
    axs[1].fill_between(x_t[:bins1d], r_statm, r_statp, facecolor='grey', alpha = 0.5, step = 'mid')
    
    axs[1].set_ylim((0.35,1.65))
    axs[1].set_yticks([0.5, 1.0, 1.5])
    axs[1].set_yticklabels([r'$0.5$', r'$1.0$', "$1.5$"])
    axs[1].axhline(y=1,linewidth=1, linestyle='--', color='grey')
    
    axs[1].set_xlabel(r'$x$', fontsize = FONTSIZE)
        
    fig.savefig("slice_2d.pdf", format='pdf')
    plt.close()
    
def plot_weights(true, true_weight, gen):
    gcolor = '#3b528b'
    dcolor = '#e41a1c'
    FONTSIZE=16
    
    # Use Latex to compile
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    
    # generate random noise
    bins = 50
    fig, axs = plt.subplots(1)
    plt.subplots_adjust(left=0.12, right=0.925, top=0.92, bottom=0.15)
    
    truth_weight = true_weight[:,0]
    gen_weight = truth_weight.mean() * np.ones_like(truth_weight)
    
    data = [[0.,0.], [0.,0.]]
    
    # Truth
    data[0][0] = true[:,0]
    data[1][0] = true[:,1]

    # Pred
    data[0][1] = gen[:,0]
    data[1][1] = gen[:,1]
    
    truth, xedges, yedges = np.histogram2d(data[0][0], data[1][0], weights = truth_weight,  bins=bins, range=([0,1],[0,1]), density=True)
    gan, xedges, yedges = np.histogram2d(data[0][1], data[1][1], weights = gen_weight, bins=bins, range=([0,1],[0,1]), density=True)
    # vegas, xedges, yedges = np.histogram2d(dataset2[:,0], dataset2[:,1], weights = gen_weight, bins=bins, range=([0,1],[0,1]), density=True)
#    ratio = h[0][1]/ h[0][0]
    
    
#    ratio_gan = truth/gan
#    ratio_vegas = truth/vegas
    
    ratio_gan = truth/np.maximum(gan,truth.min())
    # ratio_vegas = truth/np.maximum(vegas,truth.min())

    axs.set_yscale('log')
#    yfmt = ScalarFormatterForceFormat()
#    yfmt.set_powerlimits((0,0))
#    axs.yaxis.set_major_formatter(yfmt)

    
#    truth = np.ones_like(truth)

    r_g, x_g = np.histogram(ratio_gan, 50, weights= gan, range=[0, 3], density= True)
    # r_v, x_v = np.histogram(ratio_vegas, 50, weights= vegas, range=[0, 3], density= True)

    line_gan, = axs.step(x_g[:50], r_g, gcolor, label='GAN', linewidth=1.0, where='mid')
    
    for j in range(1):
        for label in ( [axs.yaxis.get_offset_text()] +
                      axs.get_xticklabels() + axs.get_yticklabels()):
            label.set_fontsize(FONTSIZE-2)

    axs.set_ylabel(r'Normalized', fontsize = FONTSIZE)
    
    plt.legend(
            [line_gan],
            ['GAN'],
            #title = "GAN vs Data",
            loc='upper right',
            prop={'size':(FONTSIZE-2)},
            frameon=False)
    
    axs.set_xlabel(r'bin ratios', fontsize = FONTSIZE)
    
                    
    fig.savefig("weights_2d_50.pdf", format='pdf')
    plt.close()