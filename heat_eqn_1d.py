import torch
import numpy as np
from neurodiffeq import diff                # differentiation operator
from neurodiffeq.networks import FCNN       # fully-connected neural network (MLP)
from neurodiffeq.solvers import Solver2D    # 2-D solver
from neurodiffeq.monitors import Monitor2D  # 2-D monitor
from neurodiffeq.generators import Generator2D # 2-D data generator
from neurodiffeq.conditions import IBVP1D   # initial-boundary value problem in 1-D
from neurodiffeq.pde import make_animation
from neurodiffeq.utils import set_tensor_type # allow training on GPU
import matplotlib.pyplot as plt

def to_numpy(x):
    ''' helper function to convert PyTorch tensor to NumPy array '''
    return x.detach().cpu().numpy()

set_tensor_type(device="cuda")

k = 0.2 # thermal diffusivity
L = 2 # maximum value of length dimension
T = 3 # maximum value of time dimension

# heat equation in 1-D (using a 2-D solver, since we are 1-D in space and 1-D in time for a total of 2-D)
# du/dt - k*d^2u/dx^2 = 0
heat = lambda u, x, t: [
    diff(u, t) - k * diff(u, x, order=2)
]

# initial and boundary conditions
# solving only for one function, thus we have a single condition object

conditions = [
    IBVP1D(
        t_min = 0, t_min_val = lambda x: torch.sin(np.pi * x / L), # u(x, t=0) = sin(pi*x)
        x_min = 0, x_min_prime = lambda t: np.pi/L * torch.exp(-k * np.pi**2 * t / L**2), # u(x=0, t) = pi/L * exp( -k * pi^2 * t / L^2)
        x_max = L, x_max_prime = lambda t: -np.pi/L * torch.exp(-k * np.pi**2 * t / L**2) # u(x=L, t) = -pi/L * exp( -k * pi^2 * t / L^2)
    )
]

# define the neural network architecture (basic, for now)

nets = [
    FCNN(n_input_units=2, hidden_units=(32, 32))
]

monitor = Monitor2D(check_every=10, xy_min=(0, 0), xy_max=(L, T)) # what determines order of dimensions? i.e. why not (T, L)? perhaps order of variables in the definition of the heat function on line 14?
monitor_callback = monitor.to_callback()

# instantiate the solver

solver = Solver2D(
    pde_system=heat,
    conditions = conditions,
    xy_min = (0, 0),
    xy_max = (L, T),
    nets = nets,
    train_generator=Generator2D((32, 32), (0, 0), (L, T), method="equally-spaced-noisy"),
    valid_generator=Generator2D((32, 32), (0, 0), (L, T), method="equally-spaced")
)

# train neural network
solver.fit(max_epochs=200, callbacks=[monitor_callback])

print(solver.loss_fn)

# recover solution
solution_nn_heat = solver.get_solution()

# plot solution on grid

xs = torch.linspace(0, L, 101)
ts = torch.linspace(0, T, 101)
xx, tt = torch.meshgrid(xs, ts, indexing='xy')
u = solution_nn_heat(xx, tt)

xx, tt, u = to_numpy(xx), to_numpy(tt), to_numpy(u)


plt.close() # close all previous plots
ax = plt.axes(projection='3d')
ax.plot_surface(xx, tt, u)
plt.savefig('heat_1d.png')

#make_animation(solution_nn_heat, xs, ts)
