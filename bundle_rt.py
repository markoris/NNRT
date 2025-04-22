import torch, glob
import numpy as np
#from neurodiffeq import diff                # differentiation operator
from neurodiffeq.neurodiffeq import unsafe_diff as diff
from neurodiffeq.networks import FCNN       # fully-connected neural network (MRP)
from neurodiffeq.solvers import BundleSolver1D    # 2-D solver
from neurodiffeq.monitors import Monitor1D  # 2-D monitor
from neurodiffeq.generators import Generator1D # 2-D data generator
from neurodiffeq.conditions import EnsembleCondition, BundleIVP   # initial-boundary value problem in 1-D
from neurodiffeq.utils import set_tensor_type # allow training on GPU
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from astropy.modeling.models import BlackBody
from astropy import units as u

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

set_tensor_type(device="cuda")

### Source function parameters

# physical constants
h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1

# placeholder kappa / rho / R values for now, spanning 0 <= tau <= 10
rho = 0.1 # bulk density, in g/cm^3
R = 333 # maximum distance in cm 
T = 5800 # temperature, in Kelvin
wav = torch.log10(torch.logspace(3, 5, 20)*1e-8) # 1000 to 100,000 Angstroms (100 nm to 10 microns)
# for training purposes, normalized between -5 and -3

def kappa(wav):
    wav = 10**wav
    return 0.3 # opacity, in cm^2/g

def S_lam(wav):
    wav = 10**wav
    S_nu = (2*h*c**2/wav**5) * 1/(torch.exp(h*c/(wav*k*T)) - 1) # erg / s / cm2 / cm
    S_nu *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A
    return torch.log10(S_nu)

### 

# radiation transport equation in 1-D (using a 1-D solver)
# 0 = -I - dI/dtau + S_lam
#rt = lambda I, tau, lam : [
#    -I - diff(I, tau) + S_lam(lam, tau)
#]

rt = lambda I, r, lam : [
    -I - (1 / kappa(wav) / rho)*diff(I, r) + S_lam(lam)
]

# initial conditions
# solving only for one function, thus we have a single condition object
conditions = [BundleIVP(t_0=0.0, u_0=None, bundle_param_lookup={'u_0': 1})]

# define the neural network architecture (basic, for now)
nets = [
    FCNN(n_input_units=3, hidden_units=(16, 16), n_output_units=1)
]
# instantiate the solver

solver = BundleSolver1D(
    ode_system=rt,
    conditions = conditions,
    nets = nets,
    t_min = 0,
    t_max = R,
    theta_min=[wav.min(), S_lam(wav.min())],
    theta_max=[wav.max(), S_lam(wav.max())],
    eq_param_index=(0,),
    n_batches_valid=10,
)

# train neural network
solver.fit(max_epochs=1000)

# recover NN solution
solution_nn_rt = solver.get_solution()

# plot solution on grid
Rs = torch.linspace(0, R, 101)
I_nu_nn = np.zeros((len(Rs), len(wav)))
for l in range(len(wav)):
    lmd_val = to_numpy(wav[l])*np.ones_like(to_numpy(Rs))
    u0 = np.log10(15)+to_numpy(S_lam(wav[l])) * np.ones_like(to_numpy(Rs))
    I_nu_nn[:, l] = solution_nn_rt(Rs, lmd_val, u0, to_numpy=True)
        
print(I_nu_nn.shape)
Rs = to_numpy(Rs)

# analytic solution
I_nu = np.zeros((len(Rs), len(wav)))

wav_analytic = np.logspace(3, 5, 20)*1e-8 # 1000 to 100,000 Angstroms (100 nm to 10 microns) in units of cm
S_lam_analytic = (2*h*c**2/wav_analytic**5) * 1/(np.exp(h*c/(wav_analytic*k*T)) - 1) # erg / s / cm2 / cm
S_lam_analytic *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A
I_0 = 15*S_lam_analytic # intensity at r=0

I_0, S_lam_analytic = np.log10(I_0), np.log10(S_lam_analytic)

taus = 0.3*rho*Rs
for j in range(len(wav)):
    for i in range(len(Rs)):
        #I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(Rs[i] - Rs[:i+1])), x=Rs[:i+1]) \
        #+ I_0[j]*np.exp(-Rs[i])
        I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
        + I_0[j]*np.exp(-taus[i])

print(I_nu_nn.shape, I_nu.shape)

wav = to_numpy(10**wav*1e8) # Angstroms

plt.rc('font', size=18)
print(taus)
tau_idx = np.argmin(np.abs(taus-2))
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


#bb = BlackBody(temperature=5800*u.K, scale=1*u.erg / (u.cm**2 * u.s * u.AA * u.sr))
#S_nu = bb(wav*u.AA)
#S_nu = torch.Tensor(S_nu).to('cuda:0')
###

#def S_nu(wav):
#    bb = BlackBody(temperature=5800*u.K, scale=1*u.erg / (u.cm**2 * u.s * u.AA * u.sr))
#    S_nu = bb(to_numpy(wav)*u.AA)
#    return torch.log10(torch.Tensor(S_nu)).to('cuda:0')

