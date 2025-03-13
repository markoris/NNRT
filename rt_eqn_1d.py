import torch
import numpy as np
from neurodiffeq import diff                # differentiation operator
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

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

set_tensor_type(device="cuda")

### Source function parameters

# from Iglesia & Rogers 1996, for rho = 100 kg/m^-3 and log10(T) = 6.5, log10(kappa) = -0.5

kappa = 0.3 # opacity, in cm^2/g
rho = 0.1 # bulk density, in g/cm^3
R = 333 # maximum distance in cm 
T = 10**(6.5) # temperature, in Kelvin
wav = 5e-5 # 500 nm to cm
h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1
S_nu = (2*h*c**2/wav**5) * 1/(np.exp(h*c/(wav*k*T)) -1)
#S_nu = 0.42 # artifically set S_nu = val to get simple exponential decay behavior
### 

I_0 = 1.5*S_nu # intensity at r=0

I_0, S_nu = np.log10(I_0), np.log10(S_nu)

# radiation transport equation in 1-D (using a 1-D solver)
# 0 = - dI/dtau - I_nu + S_nu
rt = lambda I, tau : [
    -I - diff(I, tau) + S_nu 
]

# initial conditions
# solving only for one function, thus we have a single condition object
conditions = [IVP(t_0=0.0, u_0 = I_0)]

# define the neural network architecture (basic, for now)
nets = [
    FCNN(n_input_units=1, hidden_units=(16, 16))
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

taus, I_nu_nn = to_numpy(taus), to_numpy(I_nu_nn)

# analytic solution
I_nu =  [ \
            simpson(S_nu*np.exp(-(taus[i] - taus[:i+1])), x=taus[:i+1]) \
            + I_0*np.exp(-taus[i]) \
        for i in range(len(taus)) \
        ]

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(taus, I_nu_nn, c='r')
ax.plot(taus, I_nu, c='k')
ax.set_yscale('log')
ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$\log_{10} I_\nu$')
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%0.0f'))
plt.savefig('rt_1d.png')

