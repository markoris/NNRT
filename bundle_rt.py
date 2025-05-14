import matplotlib, neurodiffeq, torch, dill, glob
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import BlackBody
from astropy import units as unit
from neurodiffeq import diff
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq.generators import GeneratorND
from scipy.integrate import simpson
from utils import save, train

neurodiffeq.utils.set_tensor_type('cpu')

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
    
        self.lin1 = torch.nn.Linear(2, 32)
        self.lin2 = torch.nn.Linear(32, 32)
        self.lin3 = torch.nn.Linear(32, 32)
        self.lin4 = torch.nn.Linear(32, 32)
        self.lin5 = torch.nn.Linear(32, 32)
        self.lin6 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        x = torch.tanh(self.lin4(x))
        x = torch.tanh(self.lin5(x))
        x = self.lin6(x)
        return x

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

def S_lam(wav):
    wav = 10**wav # undo log10
    try:
        if wav.dtype == torch.float32: wav = to_numpy(wav)
    except: 
        pass
    wav = wav*unit.AA
    S_lambda = BlackBody(temperature=5800*unit.K, scale=1*unit.erg / (unit.cm**2 * unit.s * unit.AA * unit.sr))
    S_lambda = S_lambda(wav)
    return torch.log10(torch.Tensor(np.array([S_lambda.value])))

TAU_MIN, TAU_MAX = 0, 10

#-- We change the limits of \lambda to be between 100 and 1000 nm, in log-space
LAMBDA_MIN, LAMBDA_MAX = np.log10(1000), np.log10(10000)

#-- Assume a single initial value of I0, evolve it to a wavelength-dependent source function
I0 = np.log10(10)+S_lam(np.log10(5500))

diff_eq = lambda I, t, lmd: [-I - diff(I, t) + S_lam(lmd)]

# Learning only S_lambda as a function of lambda, fix I_0 to some value
conditions = [
    BundleIVP(t_0=0, u_0=I0)   # we refer to u_0 as parameter 1; correspondingly lambda will be parameter 0 below
]

nets = [
    #FCNN(n_input_units=3, hidden_units=(512,), n_output_units=1, actv=torch.nn.LeakyReLU)
    NN()
]

#train_gen = GeneratorND(grid=[16, 32], r_min=[TAU_MIN, LAMBDA_MIN], r_max=[TAU_MAX, LAMBDA_MAX],\
#                        methods=['uniform', 'log-spaced', 'uniform'])
#valid_gen = GeneratorND(grid=[16, 32], r_min=[TAU_MIN, LAMBDA_MIN], r_max=[TAU_MAX, LAMBDA_MAX],\
#                        methods=['uniform', 'log-spaced'])

solver = BundleSolver1D(
    ode_system=diff_eq,
    conditions=conditions,
#    optimizer=torch.optim.Adam(nets[0].parameters(), lr=0.001),
    t_min=TAU_MIN,
    t_max=TAU_MAX, 
#    train_generator=train_gen,
#    valid_generator=valid_gen,
    theta_min=[LAMBDA_MIN],  # 0: lambda, 1: u_0
    theta_max=[LAMBDA_MAX],  # 0: lambda, 1: u_0
    eq_param_index=(0,),  # we refer to lambda as parameter 0; correspondingly u_0 is parameter 1 (as in conditions above)
    n_batches_valid=1,
    loss_fn=torch.nn.modules.loss.MSELoss(),
    nets=nets,
)

solver.fit(max_epochs=40000)
solution = solver.get_solution(best=True)

save(solver, "model_params.pt")

fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(solver.metrics_history['train_loss'], label='train loss')
plt.plot(solver.metrics_history['valid_loss'], label='valid loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('loss.png')

# Convert our scalar wavelength to a vector of 10 wavelengths

wav = np.logspace(LAMBDA_MIN, LAMBDA_MAX, 10)

## Define our analytic solution for a blackbody source function starting at some intensity I_0 = 2*S_\lambda

# Physical constants

h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1

# Analytic blackbody function calculation

wav_analytic = np.logspace(LAMBDA_MIN, LAMBDA_MAX, len(wav))*1e-8 # 1000 to 100,000 Angstroms (100 nm to 10 microns) in units of cm
T = 5800 # temperature, in Kelvin
S_lam_analytic = (2*h*c**2/wav_analytic**5) * 1/(np.exp(h*c/(wav_analytic*k*T)) - 1) # erg / s / cm2 / cm
S_lam_analytic *= 1e-8 # erg / s / cm2 / cm to erg / s / cm2 / A

# Initial condition

S_lam_analytic = np.log10(S_lam_analytic)

taus = np.linspace(0, 10, 100)
I_nu = np.zeros((len(taus), len(wav)))

for j in range(len(wav_analytic)):
    for i in range(len(taus)):
        I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
        + I0*np.exp(-taus[i])

I_nu_nn = np.zeros((len(taus), len(wav)))

for l, lmd_value in enumerate(wav):
    lmd = torch.Tensor(np.log10(lmd_value) * np.ones_like(taus))
    I_nu_nn[:, l] = solution(torch.Tensor(taus), lmd, to_numpy=True)  # network solution takes in two inputs

plt.rc('font', size=20)
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
ref_tau = np.array([[0, 2], [4, 10]])
for i in [0, 1]:
    for j in [0, 1]:
        plt.rc('font', size=18)
        tau_idx = np.argmin(np.abs(taus-ref_tau[i, j]))
        axs[i, j].plot(wav, I_nu_nn[tau_idx, :], c='r', label='NN')
        axs[i, j].plot(wav, I_nu[tau_idx, :], c='k', ls='-.', alpha=0.5, label='analytic')
        axs[i, j].plot(wav_analytic*1e8, S_lam_analytic, c='blue', ls='--', alpha=0.5, label=r'$S_\lambda$') # source function reference
        axs[i, j].set_title(r'$\tau = {}$'.format(taus[tau_idx]))
        axs[i, j].set_xscale('log')
        axs[i, j].set_xlabel(r'$\lambda \ [\AA]$')
        axs[i, j].set_ylabel(r'$\log_{10} I_\nu \ [\frac{erg}{s cm^2 A}]$')
        plt.tight_layout()
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
            axs[i, j].axvspan(wav_low, wav_upp, alpha=0.5, color=colors[wavelengths[fltr_indx]])
            print(wav, wav_analytic*1e8, wav_low, wav_upp)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center')
plt.savefig('I_vs_lambda.png')

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(I_nu.shape[1]):
    ax.plot(taus, I_nu_nn[:, i], c='r')
    ax.plot(taus, I_nu[:, i], label=wav[i])
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\log_{10} I_\nu$')
plt.savefig('I_vs_tau.png')
