import neurodiffeq
from utils import load
import numpy as np
import matplotlib, torch
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from astropy.modeling.models import BlackBody
from astropy import units as unit

neurodiffeq.utils.set_tensor_type('cpu')

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

def S_lam(wav):
    wav = 10**wav # undo log10
    #try:
    #    if wav.dtype == torch.float32: wav = to_numpy(wav)
    #except: 
    #    pass
    wav = wav*unit.AA
    S_lambda = BlackBody(temperature=5800*unit.K, scale=1*unit.erg / (unit.cm**2 * unit.s * unit.AA * unit.sr))
    S_lambda = S_lambda(wav)
    return np.log10(S_lambda.value)
    #return torch.log10(torch.Tensor(np.array([S_lambda.value])))

solver = load("model_params.pt")
solution = solver.get_solution(best=True)

# Convert our scalar wavelength to a vector of 10 wavelengths

wav = np.logspace(np.log10(4000), np.log10(7000), 10)

## Define our analytic solution for a blackbody source function starting at some intensity I_0 = 2*S_\lambda

# Physical constants

h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1

# Analytic blackbody function calculation

wav_analytic = np.logspace(np.log10(4000), np.log10(7000), len(wav))*1e-8 # 1000 to 100,000 Angstroms (100 nm to 10 microns) in units of cm
T = 5800 # temperature, in Kelvin
S_lam_analytic = (2*h*c**2/wav_analytic**5) * 1/(np.exp(h*c/(wav_analytic*k*T)) - 1) # erg / s / cm2 / cm
S_lam_analytic *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A

# Initial condition

I_0 = 10*S_lam_analytic # intensity at r=0

I_0, S_lam_analytic = np.log10(I_0), np.log10(S_lam_analytic)

taus = np.linspace(0, 10, 100)
I_nu = np.zeros((len(taus), len(wav)))

for j in range(len(wav_analytic)):
    for i in range(len(taus)):
        I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
        + I_0[j]*np.exp(-taus[i])

I_nu_nn = np.zeros((len(taus), len(wav)))

for l, lmd_value in enumerate(wav):
    lmd = np.log10(lmd_value) * np.ones_like(taus)
    u0 = np.log10(2)+S_lam(np.log10(lmd_value)) * np.ones_like(taus)
    taus = taus.astype(np.float32)
    lmd = lmd.astype(np.float32)
    u0 = u0.astype(np.float32)
    #u0_value = to_numpy(np.log10(2)+S_lam(lmd_value))
    #lmd = to_numpy(lmd_value) * np.ones_like(R)
    #u0 = u0_value * np.ones_like(taus)
    #I_nu_nn[:, l] = solution(taus, lmd, u0, to_numpy=True)  # network solution takes in three inputs
    I_nu_nn[:, l] = solution(taus, lmd, to_numpy=True)  # network solution takes in three inputs

plt.rc('font', size=20)
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
ref_tau = np.array([[0, 2], [5, 10]])
for i in [0, 1]:
    for j in [0, 1]:
        plt.rc('font', size=18)
        tau_idx = np.argmin(np.abs(taus-ref_tau[i, j]))
        axs[i, j].plot(wav, I_nu_nn[tau_idx, :], c='r', label='NN')
        axs[i, j].plot(wav, I_nu[tau_idx, :], c='k', label='analytic')
        axs[i, j].plot(wav_analytic*1e8, S_lam_analytic, c='blue', ls='--', label=r'$S_\lambda$') # source function reference
        axs[i, j].set_title(r'$\tau = {}$'.format(taus[tau_idx]))
        axs[i, j].set_xscale('log')
        axs[i, j].set_xlabel(r'$\lambda \ [\AA]$')
        axs[i, j].set_ylabel(r'$\log_{10} I_\nu \ [\frac{erg}{s cm^2 A}]$')
        #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0f'))
        plt.legend(fontsize=14)
        plt.tight_layout()
plt.savefig('I_vs_lambda.png')

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(I_nu.shape[1]):
    ax.plot(taus, I_nu_nn[:, i], c='r')
    ax.plot(taus, I_nu[:, i], label=wav[i])
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\log_{10} I_\nu$')
plt.savefig('I_vs_tau.png')
