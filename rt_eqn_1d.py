import torch, glob
import numpy as np
#from neurodiffeq import diff                # differentiation operator
from neurodiffeq.neurodiffeq import unsafe_diff as diff
from neurodiffeq.networks import FCNN       # fully-connected neural network (MRP)
from neurodiffeq.solvers import Solver1D    # 2-D solver
from neurodiffeq.monitors import Monitor1D  # 2-D monitor
from neurodiffeq.generators import Generator1D # 2-D data generator
from neurodiffeq.conditions import IVP   # initial-boundary value problem in 1-D
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

# from Iglesia & Rogers 1996, for rho = 100 kg/m^-3 and log10(T) = 6.5, log10(kappa) = -0.5

kappa = 0.3 # opacity, in cm^2/g
rho = 0.1 # bulk density, in g/cm^3
R = 333 # maximum distance in cm 
T = 5800 # temperature, in Kelvin
#wav = 5e-5 # 500 nm to cm
wav = np.logspace(3, 5, 20) # 1000 to 100,000 Angstroms (100 nm to 10 microns)
wav_old = np.logspace(3, 5, 20)*1e-8 # 1000 to 100,000 Angstroms (100 nm to 10 microns) in units of cm
h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1
S_nu_old = (2*h*c**2/wav_old**5) * 1/(np.exp(h*c/(wav_old*k*T)) - 1) # erg / s / cm2 / cm
S_nu_old *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A
S_nu_old = np.log10(S_nu_old)

bb = BlackBody(temperature=5800*u.K, scale=1*u.erg / (u.cm**2 * u.s * u.AA * u.sr))
S_nu = bb(wav*u.AA)
S_nu = torch.Tensor(S_nu).to('cuda:0')
#S_nu = 0.42 # artifically set S_nu = val to get simple exponential decay behavior
### 

I_0 = 15*S_nu # intensity at r=0

I_0, S_nu = torch.log10(I_0), torch.log10(S_nu)

# radiation transport equation in 1-D (using a 1-D solver)
# 0 = - dI/dtau - I_nu + S_nu
rt = lambda I, tau : [
    -I - diff(I, tau) + S_nu 
]

#I_0 = torch.Tensor(I_0).to('cuda')

# initial conditions
# solving only for one function, thus we have a single condition object
conditions = [IVP(t_0=0.0, u_0 = I_0) for _ in range(len(wav))]

# define the neural network architecture (basic, for now)
nets = [
    FCNN(n_input_units=1, hidden_units=(16, 16), n_output_units=len(wav))
]

# instantiate the solver

solver = Solver1D(
    ode_system=rt,
    conditions = conditions,
    nets = nets,
    train_generator=Generator1D(32, 0, kappa*rho*R, method="equally-spaced-noisy"),
    valid_generator=Generator1D(32, 0, kappa*rho*R, method="equally-spaced")
)

# train neural network
solver.fit(max_epochs=1000)

# recover NN solution
solution_nn_rt = solver.get_solution()

# plot solution on grid
taus = torch.linspace(0, kappa*rho*R, 101)
I_nu_nn = solution_nn_rt(taus)

taus, I_nu_nn, S_nu = to_numpy(taus), to_numpy(I_nu_nn), to_numpy(S_nu)

# analytic solution
I_nu = np.zeros((len(taus), len(wav)))

for j in range(I_nu.shape[1]):
    for i in range(I_nu.shape[0]):
        I_nu[i, j] = to_numpy(simpson(S_nu[j]*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
        + I_0[j]*np.exp(-taus[i]))

print(I_nu_nn.shape, I_nu.shape)

plt.rc('font', size=18)
tau_idx = np.argmin(np.abs(taus-3))
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(wav, I_nu_nn[tau_idx, :], c='r', label='NN')
ax.plot(wav, I_nu[tau_idx, :], c='k', label='analytic')
ax.plot(wav, S_nu_old, c='blue', ls='--', label=r'$S_\lambda$') # source function reference
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
    ax.plot(taus, I_nu[:, i], label=wav[i]*1e7)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\log_{10} I_\nu$')
plt.legend()
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0f'))
plt.savefig('I_vs_tau.png')
