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

# from Iglesia & Rogers 1996, for rho = 100 kg/m^-3 and log10(T) = 6.5, log10(kappa) = -0.5

kappa = 10**(-0.5) # opacity, in cm^2/g
rho = 0.1 # bulk density, in g/cm^3
T = 10**(6.5) # temperature, in Kelvin
#R = 6.957e8 # Solar radius
R = 100 # 
I_0 = 1 # intensity at r=0
wav = 5e-5 # 500 nm to cm
h = 6.626e-27 # cm^2 g s^-1
c = 2.9979e10 # cm/s
k = 1.3807e-16 # cm^2 g s^-2 K^-1
S_nu = (2*h*c**2/wav**5) * 1/(np.exp(h*c/(wav*k*T)) -1)

# radiation transport equation in 1-D (using a 1-D solver)
# du/dt - k*d^2u/dx^2 = 0
rt = lambda I, r : [
    -I - 1/(kappa*rho)*diff(I, r) + S_nu 
]

# initial conditions
# solving only for one function, thus we have a single condition object

conditions = [IVP(t_0=0.0, u_0 = I_0)]

# define the neural network architecture (basic, for now)

nets = [
    FCNN(n_input_units=1, hidden_units=(32, 32))
]

#monitor = Monitor1D(check_every=10, t_min=0, t_max=R) # what determines order of dimensions? i.e. why not (T, R)? perhaps order of variables in the definition of the heat function on line 14?
#monitor_callback = monitor.to_callback()

# instantiate the solver

solver = Solver1D(
    ode_system=rt,
    conditions = conditions,
    nets = nets,
    train_generator=Generator1D(32, 0, R, method="equally-spaced-noisy"),
    valid_generator=Generator1D(32, 0, R, method="equally-spaced")
)

# train neural network
solver.fit(max_epochs=200)#, callbacks=[monitor_callback])

print(solver.loss_fn)

# recover solution
solution_nn_rt = solver.get_solution()

# plot solution on grid

rs = torch.linspace(0, R, 101)
I_nu_nn = solution_nn_rt(rs)

rs, I_nu_nn = to_numpy(rs), to_numpy(I_nu_nn)

# analytic solution

I_nu = S_nu*np.exp(-kappa*rho*(R - rs)) + I_0*np.exp(-kappa*rho*R)

#print(I_nu)
#print(S_nu)

plt.close() # close all previous plots
plt.plot(rs, I_nu_nn, c='r')
plt.plot(rs, I_nu, c='k')
plt.yscale('log')
plt.savefig('rt_1d.png')

