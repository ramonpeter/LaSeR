###############################################################################
# Imports
###############################################################################

# import standard libraries
import numpy as np
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# import python optimal transport library
import ot

class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f"  # Give format her

###############################################################################
# Load all data
###############################################################################


def load_data(path: str):
    data = pd.read_hdf(path)
    data = data.iloc[:, :].values
    return data

# baseline and refined output
flow = load_data('base.h5')
laser = load_data('refined.h5')
hmc = load_data('HMC_refined_2.h5')

# truth and independet test sample
truth = load_data('truth.h5')
test = load_data('test.h5')


# latent space and refined latent space
latent_space = load_data('noise.h5')
hmc_latent = load_data('HMC_latent_2.h5')
laser_latent = load_data('GAN.h5')

# classifier weights
weights = load_data('weights.h5')

###############################################################################
# Plotting routine
###############################################################################

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20.0)
plt.rc('axes', labelsize='large')
plt.rc('pdf', compression=9)
cmap = plt.get_cmap('viridis')

def js_divergence(p, q):
    eps = 3e-16
    M = 0.5*(p + q)
    return 1/2 * np.sum(p * (np.log(p+eps) - np.log(M+eps))) + 1/2 * np.sum(q * (np.log(q+eps) - np.log(M+eps)))

def plot_and_scores(datasets: list, name, ranges):
    data = datasets[0]
    test = datasets[1]

    flow = datasets[2]
    hmc= datasets[3]
    laser = datasets[4]
    weights = datasets[5]

    latent = datasets[6]
    hmc_latent = datasets[7]
    lsr_latent = datasets[8]

    # get the histograms
    bins = 100
    fig, axes = plt.subplots(2, 5, figsize=(12,5), gridspec_kw={'hspace' : 0.1, 'wspace' : 0.1})
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
    h1, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=ranges, density=True)
    h1b, xedges, yedges = np.histogram2d(test[:, 0], test[:, 1], bins=bins, range=ranges, density=True)
    h2, xedges, yedges = np.histogram2d(flow[:, 0], flow[:, 1], bins=bins, range=ranges, density=True)
    h3, xedges, yedges = np.histogram2d(hmc[:, 0], hmc[:, 1], bins=bins, range=ranges, density=True)
    h4, xedges, yedges = np.histogram2d(laser[:, 0], laser[:, 1], bins=bins, range=ranges, density=True)
    h5, xedges, yedges = np.histogram2d(flow[:, 0], flow[:, 1], bins=bins, weights=weights[:,0], range=ranges, density=True)


    # calculate EMD
    #
    xs = np.array([ (xedges[i] + xedges[i+1])/2 for i in range(len(xedges)-1)])
    ys = np.array([ (yedges[i] + yedges[i+1])/2 for i in range(len(yedges)-1)])
    entries = []
    for i in range(bins):
        for j in range(bins):
            entries.append([xs[i], ys[j]])

    entries = np.array(entries)
    M1 = ot.dist(entries, entries, metric='euclidean')
    M1 /= M1.max()

    test_emd = ot.emd2(h1.flatten(), h1b.flatten(), M1, numItermax=5000000)
    flow_emd = ot.emd2(h1.flatten(), h2.flatten(), M1, numItermax=5000000)
    hmc_emd = ot.emd2(h1.flatten(), h3.flatten(), M1, numItermax=5000000)
    laser_emd = ot.emd2(h1.flatten(), h4.flatten(), M1, numItermax=5000000)
    weighted_emd = ot.emd2(h1.flatten(), h5.flatten(), M1, numItermax=5000000)


    print(f"------------------------------------------------------")
    print(f"Test EMD     : {test_emd:.5g}")
    print(f"Flow EMD     : {flow_emd:.5g}")
    print(f"HMCFlow EMD  : {hmc_emd:.5g}")
    print(f"LSRFlow EMD  : {laser_emd:.5g}")
    print(f"Weighted EMD : {weighted_emd:.5g}")
    print(f"------------------------------------------------------")

    # calculate Jenson-Shannon Divergence
    #
    test_kl = js_divergence(h1.flatten(), h1b.flatten())
    flow_kl = js_divergence(h1.flatten(), h2.flatten())
    hmc_kl = js_divergence(h1.flatten(), h3.flatten())
    laser_kl = js_divergence(h1.flatten(), h4.flatten())
    weighted_kl = js_divergence(h1.flatten(), h5.flatten())

    print(f"------------------------------------------------------")
    print(f"Test JS     : {test_kl:.5g}")
    print(f"Flow JS     : {flow_kl:.5g}")
    print(f"HMCFlow JS  : {hmc_kl:.5g}")
    print(f"LSRFlow JS  : {laser_kl:.5g}")
    print(f"Weighted JS : {weighted_kl:.5g}")
    print(f"------------------------------------------------------")

    h = [h1,h2,h3,h4,h5]
    titles = ["Truth", "Baseline", "HMC", r"\textsc{LaSeR}", r"\textsc{Dctr}"]
    vmax = np.max(h)
    vmin = np.min(h)

    #plot0
    for i in range(5):
        axes[0][i].set_title(titles[i])
        axes[0][i].pcolormesh(xedges, yedges, h[i].T, vmin=vmin, vmax=vmax, rasterized=True, cmap=cmap)
        #axes[0][i].set_aspect('equal', adjustable='box')
        # if i > 1:
        #     axes[0][i].tick_params(left=False, labelleft=False)
        axes[0][i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        
    axes[0][0].set_ylabel(f"Feature Space")

    l1, xedges, yedges = np.histogram2d(latent[:, 0], latent[:, 1], bins=100, range=([-3,3],[-3,3]), density=True)
    l2, xedges, yedges = np.histogram2d(hmc_latent[:, 0], hmc_latent[:, 1], bins=100, range=([-3,3],[-3,3]), density=True)
    l3, xedges, yedges = np.histogram2d(lsr_latent[:, 0], lsr_latent[:, 1], bins=100, range=([-3,3],[-3,3]), density=True)
    l4, xedges, yedges = np.histogram2d(latent[:, 0], latent[:, 1], bins=100, weights=weights[:,0], range=([-3,3],[-3,3]), density=True)

    l = [l1,l2,l3,l4]
    vmaxl = np.max(l)
    vminl = np.min(l)

    axes[1][0].set_aspect('equal', adjustable='box')
    axes[1][0].axis('off')
    for i in [1,2,3,4]:
        axes[1][i].pcolormesh(xedges, yedges, l[i-1].T, vmin=vminl, vmax=vmaxl, rasterized=True, cmap=cmap)
        axes[1][i].set_aspect('equal', adjustable='box')
        # if i > 1:
        #     axes[1][i].tick_params(left=False, labelleft=False)
        axes[1][i].tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    axes[1][1].set_ylabel(f"Latent Space")
    
    fig.savefig(f"{name}.pdf", format="pdf")
    plt.close()

plot_and_scores([truth, test, flow, hmc, laser, weights, latent_space, hmc_latent, laser_latent], "double_donut", ranges=([-11,11],[-7,7]))
