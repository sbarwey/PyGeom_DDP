"""
Plot halo swap results
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 14})


# Load data
nprocs_1 = np.load('outputs/halo_results/nprocs_1.npy').flatten()
nprocs_4 = np.load('outputs/halo_results/nprocs_4.npy').flatten()
nprocs_4_nohalo = np.load('outputs/halo_results/nprocs_4_nohalo.npy').flatten()
nprocs_8 = np.load('outputs/halo_results/nprocs_8.npy').flatten()
nprocs_8_nohalo = np.load('outputs/halo_results/nprocs_8_nohalo.npy').flatten()



# 4 procs 
ms = 10
exact = [nprocs_1.min(), nprocs_1.max()]
fig, ax = plt.subplots()
ax.plot(nprocs_1, nprocs_8_nohalo, color='red', lw=0, marker='o', fillstyle='none', ms=ms, label='No Halo Swap')
ax.plot(nprocs_1, nprocs_8, color='blue', lw=0, marker='s', fillstyle='none', ms=ms, label='With Halo Swap')
ax.plot(exact, exact, color='black', lw=2, label='Exact')
ax.set_aspect('equal')
ax.set_xlabel('Full Graph')
ax.set_ylabel('Partitioned Graph')
ax.set_title('8 Ranks')
ax.legend(fancybox=False, framealpha=1)
plt.show(block=False)
