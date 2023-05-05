"""
Plot halo swap results
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})


# Load data
nprocs_1 = np.load('outputs/halo_results/nprocs_1.npy').flatten()
nprocs_2 = np.load('outputs/halo_results/nprocs_8.npy').flatten()
nprocs_2_nohalo = np.load('outputs/halo_results/nprocs_8_nohalo.npy').flatten()

# histogram 
bins = 50

# reference:
nprocs_1_hist = np.histogram(nprocs_1, bins=bins, normed=True)

bins = nprocs_1_hist[1]

nprocs_2_hist = np.histogram(nprocs_2, bins=bins, normed=True)
nprocs_2_nohalo_hist = np.histogram(nprocs_2_nohalo, bins=bins, normed=True)


# Bins: 
fig, ax = plt.subplots()
ax.plot(bins[:-1], nprocs_1_hist[0]/nprocs_1_hist[0].sum(), color='black', lw=3, label='Unpartitioned')
ax.plot(bins[:-1], nprocs_2_hist[0]/nprocs_2_hist[0].sum(), color='lime', ls='--', lw=2, label='Partitioned, with Halo')
ax.plot(bins[:-1], nprocs_2_nohalo_hist[0]/ nprocs_2_nohalo_hist[0].sum(), color='red', ls='--', lw=2, label='Partitioned, w/o Halo')
ax.set_ylim([0, 0.06])
ax.grid(False)
ax.legend(fancybox=False, framealpha=1, edgecolor='black')
plt.show(block=False)


# # Scatter plot of pdf
# ms=10
# exact = [nprocs_1_hist[0].min(), nprocs_1_hist[0].max()]
# fig, ax = plt.subplots()
# ax.plot(nprocs_1_hist[0], nprocs_2_hist[0], color='red', lw=0, marker='s', fillstyle='none', ms=ms, label='With Halo Swap')
# ax.plot(nprocs_1_hist[0], nprocs_2_nohalo_hist[0], color='blue', lw=0, marker='o', fillstyle='none', ms=ms, label='No Halo Swap')
# ax.plot(exact, exact, color='black', lw=2, label='Exact')
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.show(block=False)


