import torch, glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq.networks import FCNN       # fully-connected neural network (MRP)
from neurodiffeq import diff
from scipy.integrate import simpson

# physical constants
h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1

# placeholder kappa / rho / R values for now, spanning 0 <= tau <= 10
rho = 0.1 # bulk density, in g/cm^3
R = 333 # maximum distance in cm 
T = 5800 # temperature, in Kelvin

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

def kappa(wav):
    wav = 10**wav
    return 0.3 # opacity, in cm^2/g

def S_lam(wav):
    wav = 10**wav
    S_nu = (2*h*c**2/wav**5) * 1/(torch.exp(h*c/(wav*k*T)) - 1) # erg / s / cm2 / cm
    S_nu *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A
    #return S_nu
    return torch.log10(S_nu)

R_MIN, R_MAX = 0, 333
wav = torch.logspace(3, 5, 20) # 1000 to 100,000 Angstroms
wav *= 1e-8 # convert Angstroms to cm
wav = torch.log10(wav) # log for nicer training
LAMBDA_MIN, LAMBDA_MAX = wav.min(), wav.max()
print(LAMBDA_MIN, LAMBDA_MAX)
#U0_MIN, U0_MAX = S_lam(wav).min(), S_lam(wav).max()
U0_MIN, U0_MAX = np.log10(1/20)+S_lam(wav).min(), np.log10(20)+S_lam(wav).max()
#U0_MIN, U0_MAX = 1/20e5*S_lam(wav).min(), 20e5*S_lam(wav).max()

diff_eq = lambda u, r, lmd: [-u - (1/kappa(lmd)/rho)*diff(u, r) + S_lam(lmd)]

conditions = [
    BundleIVP(t_0=0, u_0=None, bundle_param_lookup={'u_0': 1})   # we refer to u_0 as parameter 1; correspondingly lambda will be parameter 0 below
]

nets = [
    FCNN(n_input_units=3, hidden_units=(512,), n_output_units=1)
]

solver = BundleSolver1D(
    ode_system=diff_eq,
    conditions=conditions,
    t_min=R_MIN,
    t_max=R_MAX, 
    theta_min=[LAMBDA_MIN, U0_MIN],  # 0: lambda, 1: u_0
    theta_max=[LAMBDA_MAX, U0_MAX],  # 0: lambda, 1: u_0
    eq_param_index=(0,),  # we refer to lambda as parameter 0; correspondingly u_0 is parameter 1 (as in conditions above)
    n_batches_valid=10,
)

solver.fit(max_epochs=500)

solution = solver.get_solution(best=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

R = np.linspace(0, 333)

I_nu_nn = np.zeros((len(R), len(wav)))

#for l, lmd_value in enumerate(to_numpy(wav)):
for l, lmd_value in enumerate(wav):
    #for u0_value in [-2, 0, 2]:
    u0_value = to_numpy(np.log10(2)+S_lam(lmd_value))
    lmd = to_numpy(lmd_value) * np.ones_like(R)
    u0 = u0_value * np.ones_like(R)
    I_nu_nn[:, l] = solution(R, lmd, u0, to_numpy=True)  # network solution takes in three inputs
    #ax.plot(t, u, label=f'$\\lambda={lmd_value}$, $u_0={u0_value}$')

I_nu = np.zeros((len(R), len(wav)))

wav_analytic = np.logspace(3, 5, len(wav))*1e-8 # 1000 to 100,000 Angstroms (100 nm to 10 microns) in units of cm
S_lam_analytic = (2*h*c**2/wav_analytic**5) * 1/(np.exp(h*c/(wav_analytic*k*T)) - 1) # erg / s / cm2 / cm
S_lam_analytic *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A
I_0 = 2*S_lam_analytic # intensity at r=0

I_0, S_lam_analytic = np.log10(I_0), np.log10(S_lam_analytic)

taus = 0.3*rho*R
for j in range(len(wav)):
    for i in range(len(R)):
        #I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(Rs[i] - Rs[:i+1])), x=Rs[:i+1]) \
        #+ I_0[j]*np.exp(-Rs[i])
        I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
        + I_0[j]*np.exp(-taus[i])

print(I_nu_nn.shape, I_nu.shape)

wav = to_numpy(10**wav*1e8) # Angstroms

plt.rc('font', size=18)
tau_idx = np.argmin(np.abs(taus-0.2))
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(wav, I_nu_nn[tau_idx, :], c='r', label='NN')
ax.plot(wav, I_nu[tau_idx, :], c='k', label='analytic')
ax.plot(wav, S_lam_analytic, c='blue', ls='--', label=r'$S_\lambda$') # source function reference
ax.set_title(r'$\tau = {}$'.format(taus[tau_idx]))
ax.set_xscale('log')
ax.set_xlabel(r'$\lambda \ [\AA]$')
ax.set_ylabel(r'$\log_{10} I_\nu \ [\frac{erg}{s cm^2 A}]$')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0f'))
filters = np.array(glob.glob('filters/*'))
wavelengths = 'grizyJHKS'
colors = {"g": "blue", "r": "cyan", "i": "lime", "z": "green", "y": "greenyellow", "J": "gold",
 "H": "orange", "K": "red", "S": "darkred"}
text_locs = {"g": 0.09, "r": 0.25, "i": 0.35, "z": 0.44, "y": 0.51, "J": 0.61, "H": 0.75, "K": 0.92}
for fltr in range(len(filters)):
    filter_wavs = np.loadtxt(filters[fltr])
    filter_wavs = filter_wavs[:, 0] # Angstroms
    wav_low, wav_upp = filter_wavs[0], filter_wavs[-1]
    fltr_band = filters[fltr].split('/')[-1][0]
    if fltr_band == "S": continue
    text_loc = text_locs[fltr_band]
    fltr_indx = wavelengths.find(fltr_band)
    plt.axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('I_vs_lambda.png')

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(I_nu.shape[1]):
    ax.plot(taus, I_nu_nn[:, i], c='r')
    ax.plot(taus, I_nu[:, i], label=wav[i])
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\log_{10} I_\nu$')
#plt.legend()
#ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0f'))
plt.savefig('I_vs_tau.png')
counter = 1
for (a, b) in zip(solver.metrics_history['train_loss'], solver.metrics_history['valid_loss']):
    if (counter % 10) == 0: print(a, b)
    counter += 1
