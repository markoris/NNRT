import torch
import numpy as np
from neurodiffeq import diff                # differentiation operator
from neurodiffeq.networks import FCNN       # fully-connected neural network (MRP)
from neurodiffeq.solvers import Solver1D    # 2-D solver
from neurodiffeq.monitors import Monitor1D  # 2-D monitor
from neurodiffeq.generators import Generator1D # 2-D data generator
from neurodiffeq.conditions import IVP   # initial-boundary value problem in 1-D
from neurodiffeq.pde import make_animation
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
R = 33 # maximum distance in cm 
#T = 10**(6.5) # temperature, in Kelvin
#wav = 5e-5 # 500 nm to cm
#h = 6.626e-27 # cm^2 g s^-1
#c = 2.9979e10 # cm/s
#k = 1.3807e-16 # cm^2 g s^-2 K^-1
#S_nu = (2*h*c**2/wav**5) * 1/(np.exp(h*c/(wav*k*T)) -1)
S_nu = 0.42 # artifically set S_nu = 0 to get simple exponential decay behavior
### 

I_0 = 1 # intensity at r=0

# radiation transport equation in 1-D (using a 1-D solver)
# 0 = - dI/dtau - I_nu + S_nu
# 0 = -1/(kappa*rho) * dI/dr - I_nu + S_nu
rt = lambda I, r : [
    -I - 1/(kappa*rho)*diff(I, r) + S_nu 
]

# initial conditions
# solving only for one function, thus we have a single condition object
conditions = [IVP(t_0=0.0, u_0 = I_0)]

# define the neural network architecture (basic, for now)
nets = [
    FCNN(n_input_units=1, hidden_units=(64, 64))
]

# instantiate the solver

solver = Solver1D(
    ode_system=rt,
    conditions = conditions,
    nets = nets,
    train_generator=Generator1D(32, 0, R, method="equally-spaced-noisy"),
    valid_generator=Generator1D(32, 0, R, method="equally-spaced")
)

# train neural network
solver.fit(max_epochs=1000)

# recover NN solution
solution_nn_rt = solver.get_solution()

# plot solution on grid
rs = torch.linspace(0, R, 101)
I_nu_nn = solution_nn_rt(rs)

rs, I_nu_nn = to_numpy(rs), to_numpy(I_nu_nn)

# analytic solution
#tau_nu = kappa*rho*R
t_nu = kappa*rho*rs

I_nu =  [ \
            simpson(S_nu*np.exp(-(t_nu[i] - t_nu[:i+1])), x=t_nu[:i+1]) \
            + I_0*np.exp(-t_nu[i]) \
        for i in range(len(rs)) \
        ]
#I_nu = [simpson(I_nu[:i+1], x=t_nu[:i+1]) for i in range(len(t_nu))]
#term2 = I_0*np.exp(-kappa*rho*
#I_nu += I_0*np.exp(-kappa*rho*rs)

plt.close() # close all previous plots
plt.plot(t_nu, I_nu_nn, c='r')
plt.plot(t_nu, I_nu, c='k')
plt.savefig('rt_1d.png')

