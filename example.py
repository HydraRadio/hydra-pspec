
import numpy as np
import hydra_pspec as hp
import pylab as plt
import scipy.special

from pyuvdata import UVData
from hera_pspec.data import DATA_PATH
import os

# Load example data from HERA PSpec
dfiles = [
    'zen.2458042.12552.xx.HH.uvXAA',
    'zen.2458042.12552.xx.HH.uvXAA'
]
uvd = UVData()
uvd.read_miriad(os.path.join(DATA_PATH, dfiles[0]))
antpairpols = uvd.get_antpairpols()

# Get data for single baseline
print("Baseline:", antpairpols[0])
d = uvd.get_data(antpairpols[0])
w = ~uvd.get_flags(antpairpols[0])

# Get no. of frequencies/times
Ntimes, Nfreqs = d.shape
print("Nfreqs:", np.unique(uvd.freq_array).size)
print("Ntimes:", np.unique(uvd.lst_array).size, Ntimes)

# Plot data and flags
#plt.matshow(d.real)
#plt.colorbar()
#plt.show()

# Initial guess at EoR power spectrum
S_initial = np.eye(Nfreqs)

# Simple guess for noise variance
Ninv = np.eye(Nfreqs)

# Load example foreground modes (120 frequencies, 8 modes)
#fgmodes = np.load("foreground_matrix.npy")
#fgmodes = fgmodes[:Nfreqs,:] # FIXME: Trim the frequencies. This is horribly wrong!

# Generate approximate set of FG modes from Legendre polynomials
fgmodes = np.array([scipy.special.legendre(i)(np.linspace(-1., 1., Nfreqs)) 
					for i in range(8)]).T

# Power spectrum prior
# This has shape (2, Ndelays). The first dimension is for the upper and 
# lower prior bounds respectively. If the prior for a given delay is 
# set to zero, no prior is applied. Otherwise, the solution is restricted 
# to be within the range ps_prior[1] < soln < ps_prior[0]. 
ps_prior = np.zeros((2,Nfreqs))
ps_prior[0,Nfreqs//2-3:Nfreqs//2+3] = 0.1
ps_prior[1,Nfreqs//2-3:Nfreqs//2+3] = -0.1

# Run Gibbs sampler
signal_cr, signal_S, signal_ps, fg_amps \
  = hp.pspec.gibbs_sample_with_fg(
    d,
    w[0], # FIXME
    S_initial,
    fgmodes, # FIXME
    Ninv,
    ps_prior,
    Niter=100,
    seed=None,
    verbose=True,
    nproc=2,
)

plt.subplot(111)
plt.plot(signal_ps.T, color='r', alpha=0.3)
plt.show()