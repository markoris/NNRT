import matplotlib, dill
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import math, torch, neurodiffeq
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP, IVP
from neurodiffeq import diff
from astropy.modeling.models import BlackBody
from astropy import units as unit
from neurodiffeq.networks import FCNN
from neurodiffeq.generators import GeneratorND
from scipy.integrate import simpson
from utils import save, train

neurodiffeq.utils.set_tensor_type('cpu')

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
    
        #self.lin1 = torch.nn.Linear(3, 32)
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

#-- We change the limits of \lambda to be between 400 and 700 nm, in log-space
LAMBDA_MIN, LAMBDA_MAX = math.log10(4000), math.log10(7000)

#-- We change the limits of I_0 to be scalar multiples of S_lam
#-- This requires a wavelength input, so let's use 550 nm for now
I0_MIN, I0_MAX = math.log10(0.1)+S_lam(math.log10(5500)), math.log10(10)+S_lam(math.log10(5500))

print(I0_MIN[0], I0_MAX[0])

diff_eq = lambda I, t, lmd: [-I - diff(I, t) + S_lam(lmd)]

#conditions = [
#    BundleIVP(t_0=0, u_0=None, bundle_param_lookup={'u_0': 1})   # we refer to u_0 as parameter 1; correspondingly lambda will be parameter 0 below
#]

# Learning only S_lambda as a function of lambda, fix I_0 to some value
conditions = [
    BundleIVP(t_0=0, u_0=I0_MAX)   # we refer to u_0 as parameter 1; correspondingly lambda will be parameter 0 below
]

nets = [
    #FCNN(n_input_units=3, hidden_units=(512,), n_output_units=1, actv=torch.nn.LeakyReLU)
    NN()
]

train_gen = GeneratorND(grid=[16, 32, 16], r_min=[TAU_MIN, LAMBDA_MIN, I0_MIN], r_max=[TAU_MAX, LAMBDA_MAX, I0_MAX],\
                        methods=['uniform', 'log-spaced', 'uniform'])
valid_gen = GeneratorND(grid=[16, 32, 16], r_min=[TAU_MIN, LAMBDA_MIN, I0_MIN], r_max=[TAU_MAX, LAMBDA_MAX, I0_MAX],\
                        methods=['uniform', 'log-spaced', 'uniform'])

solver = BundleSolver1D(
    ode_system=diff_eq,
    conditions=conditions,
#    optimizer=torch.optim.Adam(nets[0].parameters(), lr=0.001),
    t_min=TAU_MIN,
    t_max=TAU_MAX, 
#    train_generator=train_gen,
#    valid_generator=valid_gen,
#    theta_min=[LAMBDA_MIN, I0_MIN[0]],  # 0: lambda, 1: u_0
#    theta_max=[LAMBDA_MAX, I0_MAX[0]],  # 0: lambda, 1: u_0
    theta_min=[LAMBDA_MIN],  # 0: lambda, 1: u_0
    theta_max=[LAMBDA_MAX],  # 0: lambda, 1: u_0
    eq_param_index=(0,),  # we refer to lambda as parameter 0; correspondingly u_0 is parameter 1 (as in conditions above)
    n_batches_valid=1,
    loss_fn=torch.nn.modules.loss.MSELoss(),
    nets=nets,
)

#train(solver, epochs=[10, 10, 10], lrs=[1e-3, 5e-4, 1e-4])

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

I_0 = 2*S_lam_analytic # intensity at r=0

I_0, S_lam_analytic = np.log10(I_0), np.log10(S_lam_analytic)

taus = torch.linspace(0, 10, 100)
I_nu = np.zeros((len(taus), len(wav)))

for j in range(len(wav_analytic)):
    for i in range(len(taus)):
        I_nu[i, j] = simpson(S_lam_analytic[j]*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
        + I_0[j]*np.exp(-taus[i])

I_nu_nn = np.zeros((len(taus), len(wav)))

for l, lmd_value in enumerate(wav):
    lmd = math.log10(lmd_value) * torch.ones_like(taus)
    u0 = np.log10(10)+S_lam(torch.log10(torch.tensor([lmd_value]))) * torch.ones_like(taus)
    #I_nu_nn[:, l] = solution(taus, lmd, u0, to_numpy=True)  # network solution takes in three inputs
    I_nu_nn[:, l] = solution(taus, lmd, to_numpy=True)  # network solution takes in two inputs
#    print(not np.any(solution(taus, lmd, u0, to_numpy=True)-new_solution(taus, lmd, u0, to_numpy=True)))

plt.rc('font', size=18)
tau_idx = np.argmin(np.abs(taus-2.0))
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(wav, I_nu_nn[tau_idx, :], c='r', label='NN')
ax.plot(wav, I_nu[tau_idx, :], c='k', label='analytic')
ax.plot(wav_analytic*1e8, S_lam_analytic, c='blue', ls='--', label=r'$S_\lambda$') # source function reference
ax.set_title(r'$\tau = {}$'.format(taus[tau_idx]))
ax.set_xscale('log')
ax.set_xlabel(r'$\lambda \ [\AA]$')
ax.set_ylabel(r'$\log_{10} I_\nu \ [\frac{erg}{s cm^2 A}]$')
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
