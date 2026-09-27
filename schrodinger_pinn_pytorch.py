# %% [markdown]
# # Solve the 1D Linear Schrödinger equation using a Physics Informed Neural Network
#
# PyTorch translation of the original TensorFlow implementation.

# %% [markdown]
# In the following, we will solve the dimensionless 1D Schrödinger equation with inhomogeneous Dirichlet boundary conditions using a Physics Informed Neural Network (PINN).
# This boundary values problem can be stated as
#
# $$ i \hbar \partial_t \psi(x, t) = \left(-\frac{\hbar^2}{2m}\partial_x^2 + V(x, t)\right) \psi(x, t) $$
# with
# $$ \psi(x, t = t_0) \equiv \psi_0(x) \qquad \psi(x=x_l, t) \equiv \psi_L(t) \qquad \psi(x=x_r, t) \equiv \psi_R(t) $$
# for $t \in [t_0, t_1]$ and $x \in [x_l, x_r]$.

# %% [markdown]
# ### Includes

# %%
import os
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time

dtype  = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %% [markdown]
# ## Make training reproducible

# %%
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # required for deterministic CUDA ops

# %% [markdown]
# ### Define the test problem

# %%
t0   = torch.tensor(0.0,      dtype=dtype, device=device)
t1   = torch.tensor(np.pi,    dtype=dtype, device=device)
xl   = torch.tensor(0.0,      dtype=dtype, device=device)
xr   = torch.tensor(1.0,      dtype=dtype, device=device)
hbar = torch.tensor(1.0,      dtype=dtype, device=device)
mass = torch.tensor(1.0,      dtype=dtype, device=device)


def plane_wave(x, t):
    k     = 1.0
    omega = 0.5 / mass * k**2
    re    = torch.cos(k * x - omega * t)
    im    = torch.sin(k * x - omega * t)
    return torch.cat([re, im], dim=1)


def gaussian(x, t):
    xc    = 0.5
    alpha = 0.01
    cx    = x.to(torch.complex64)
    ct    = t.to(torch.complex64)
    chbar = hbar.to(torch.complex64)
    cmass = mass.to(torch.complex64)
    psi   = torch.sqrt(1 / (alpha + 1j * ct * chbar / cmass)) * torch.exp(
        -((cx - xc) ** 2) / (2 * (alpha + 1j * ct * chbar / cmass))
    )
    return torch.cat([psi.real, psi.imag], dim=1)


def moving_gaussian(x, t):
    xc    = torch.tensor(0.5,      dtype=torch.complex64, device=device)
    alpha = torch.tensor(1.0/100,  dtype=torch.complex64, device=device)
    k, s  = 1.0, 0.1
    cx    = x.to(torch.complex64)
    ct    = t.to(torch.complex64)
    s2    = s**2
    psi   = torch.exp(-0.25 * (cx - xc - 2j * k * s2)**2 / (s2 + 1j * ct))
    psi   = psi * torch.exp(1j * k * xc - k**2 * s2) / torch.sqrt(s2 + 1j * ct)
    psi   = (0.5 * s2 / np.pi)**0.25 * psi
    return torch.cat([psi.real, psi.imag], dim=1)


def psi(x, t):
    return plane_wave(x, t)


def get_density(p): return p[:, 0]**2 + p[:, 1]**2
def get_real(p):    return p[:, 0]
def get_imag(p):    return p[:, 1]

# %% [markdown]
# Generate collocation points as well as points of initial and boundary data.

# %%
n_initial     = 50
n_boundary    = 50
n_collocation = 10000

# Initial data at t0
t_initial   = torch.ones((n_initial, 1),   dtype=dtype, device=device) * t0
x_initial   = torch.rand((n_initial, 1),   dtype=dtype, device=device) * (xr - xl) + xl
xt_initial  = torch.cat([x_initial, t_initial], dim=1)
psi_initial = psi(x_initial, t_initial)

# Boundary data (left/right via Bernoulli)
t_boundary   = torch.rand((n_boundary, 1), dtype=dtype, device=device) * (t1 - t0) + t0
bernoulli    = (torch.rand((n_boundary, 1), device=device) > 0.5).to(dtype)
x_boundary   = xl + (xr - xl) * bernoulli
xt_boundary  = torch.cat([x_boundary, t_boundary], dim=1)
psi_boundary = psi(x_boundary, t_boundary)

# Collocation points
t_collocation  = torch.rand((n_collocation, 1), dtype=dtype, device=device) * (t1 - t0) + t0
x_collocation  = torch.rand((n_collocation, 1), dtype=dtype, device=device) * (xr - xl) + xl
xt_collocation = torch.cat([x_collocation, t_collocation], dim=1)

print(xt_initial.shape, psi_initial.shape, xt_boundary.shape, xt_collocation.shape)

# %% [markdown]
# ## Visualise initial and boundary data

# %%
n_linear  = 100
xl_linear = torch.ones((n_linear, 1), device=device) * xl
xr_linear = torch.ones((n_linear, 1), device=device) * xr
t0_linear = torch.ones((n_linear, 1), device=device) * t0
t1_linear = torch.ones((n_linear, 1), device=device) * t1
x_linear  = torch.linspace(xl.item(), xr.item(), n_linear, device=device).reshape(n_linear, 1)
t_linear  = torch.linspace(t0.item(), t1.item(), n_linear, device=device).reshape(n_linear, 1)

psi_0_linear = psi(x_linear, t0_linear)
psi_1_linear = psi(x_linear, t1_linear)
psi_l_linear = psi(xl_linear, t_linear)
psi_r_linear = psi(xr_linear, t_linear)

# %%
def to_np(t): return t.detach().cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, getter, label in zip(axes,
    [get_density, get_real, get_imag],
    ['Density $|\\psi|^2$', 'Real part', 'Imag part']):
    ax.set_title(f'Initial data — {label}')
    ax.scatter(to_np(x_initial), to_np(getter(psi_initial)), marker='X', label='PINN input at $t_0$')
    ax.plot(to_np(x_linear), to_np(getter(psi_0_linear)), label='Analytical $t_0$')
    ax.plot(to_np(x_linear), to_np(getter(psi_1_linear)), label='Analytical $t_1$')
    ax.set_xlabel('$x$'); ax.legend()
plt.tight_layout(); plt.show(); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, getter, label in zip(axes,
    [get_density, get_real, get_imag],
    ['Density $|\\psi|^2$', 'Real part', 'Imag part']):
    ax.set_title(f'Boundary data — {label}')
    ax.scatter(to_np(t_boundary), to_np(getter(psi_boundary)), marker='X', label='PINN input')
    ax.plot(to_np(t_linear), to_np(getter(psi_l_linear)), label='Analytical $x=x_l$')
    ax.plot(to_np(t_linear), to_np(getter(psi_r_linear)), label='Analytical $x=x_r$')
    ax.set_xlabel('$t$'); ax.legend()
plt.tight_layout(); plt.show(); plt.close()

# %% [markdown]
# ## Create a neural network
#
# Fully connected feedforward model with hidden layers of `tanh` activations and
# a normalisation layer that maps inputs to $[-1, 1]$.

# %%
lower_bounds = torch.stack([xl, t0]).to(device)
upper_bounds = torch.stack([xr, t1]).to(device)


class PINN(nn.Module):
    def __init__(self, num_hidden_layers=4, num_neurons_per_layer=20):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(num_hidden_layers):
            linear = nn.Linear(in_dim, num_neurons_per_layer)
            nn.init.xavier_normal_(linear.weight)   # Glorot normal, same as TF default
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.Tanh()]
            in_dim  = num_neurons_per_layer
        out = nn.Linear(in_dim, 2)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        # Normalise inputs to [-1, 1]
        x_norm = 2.0 * (xt - lower_bounds) / (upper_bounds - lower_bounds) - 1.0
        return self.net(x_norm)


model = PINN().to(device)
print(model(xt_collocation).shape)

# %% [markdown]
# Compute gradients with `torch.autograd.grad`.
#
# Because each batch element $i$ only depends on its own $(x_i, t_i)$, summing
# before differentiating gives the same per-element derivative as `tape.gradient`
# in TensorFlow.

# %%
def compute_residual(model, xt):
    # Detach from any existing graph and re-attach with requires_grad
    x = xt[:, 0:1].detach().requires_grad_(True)
    t = xt[:, 1:2].detach().requires_grad_(True)

    psi_pred = model(torch.cat([x, t], dim=1))
    re = psi_pred[:, 0]
    im = psi_pred[:, 1]

    # First-order derivatives
    re_x  = torch.autograd.grad(re.sum(), x, create_graph=True)[0]
    im_x  = torch.autograd.grad(im.sum(), x, create_graph=True)[0]
    re_t  = torch.autograd.grad(re.sum(), t, create_graph=True)[0]
    im_t  = torch.autograd.grad(im.sum(), t, create_graph=True)[0]

    # Second-order spatial derivatives
    re_xx = torch.autograd.grad(re_x.sum(), x, create_graph=True)[0]
    im_xx = torch.autograd.grad(im_x.sum(), x, create_graph=True)[0]

    # R = i hbar psi_t + hbar²/(2m) psi_xx  →  residual should vanish
    residual_re = re_t + 0.5 * im_xx
    residual_im = im_t - 0.5 * re_xx

    return torch.cat([residual_re, residual_im], dim=1)


# Sanity check
print(torch.mean(compute_residual(PINN().to(device), xt_collocation)**2))

# %% [markdown]
# Compute total loss (PDE residual + initial + boundary).

# %%
def compute_loss(model, xt_collocation, xt_initial, psi_initial, xt_boundary, psi_boundary):
    loss  = torch.mean(compute_residual(model, xt_collocation)**2)
    loss += torch.mean((model(xt_initial)  - psi_initial)**2)
    loss += torch.mean((model(xt_boundary) - psi_boundary)**2)
    return loss

# %% [markdown]
# ## Train
#
# Piecewise constant learning rate schedule matching the original:
# - steps 0–999:    lr = 1e-2
# - steps 1000–2999: lr = 1e-3
# - steps 3000+:    lr = 5e-4

# %%
model     = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def lr_lambda(step):
    if step < 1000:  return 1.0   # 1e-2
    if step < 3000:  return 0.1   # 1e-3
    return 0.05                    # 5e-4

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

N    = 5000
hist = []

current_time = time()

for i in range(N + 1):
    optimizer.zero_grad()
    loss = compute_loss(model, xt_collocation, xt_initial, psi_initial, xt_boundary, psi_boundary)
    loss.backward()
    optimizer.step()
    scheduler.step()

    hist.append(loss.item())

    if i % 50 == 0:
        print(f'It {i:05d}: loss = {loss.item():.8e}')

print(f'\nComputation time: {time() - current_time:.2f} seconds')

# %% [markdown]
# ## Check what model has learnt
#
# ### Check initial conditions

# %%
with torch.no_grad():
    psi_initial_pred = model(xt_initial)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, getter, label in zip(axes,
    [get_density, get_real, get_imag],
    ['Density $|\\psi|^2$', 'Real part', 'Imag part']):
    ax.set_title(f'Initial — {label}')
    ax.scatter(to_np(x_initial), to_np(getter(psi_initial_pred)), marker='X', label='PINN prediction at $t_0$')
    ax.plot(to_np(x_linear), to_np(getter(psi_0_linear)), label='Analytical $t_0$')
    ax.set_xlabel('$x$'); ax.legend()
plt.tight_layout(); plt.show(); plt.close()

# %% [markdown]
# ### Check boundary conditions

# %%
with torch.no_grad():
    psi_boundary_pred = model(xt_boundary)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, getter, label in zip(axes,
    [get_density, get_real, get_imag],
    ['Density $|\\psi|^2$', 'Real part', 'Imag part']):
    ax.set_title(f'Boundary — {label}')
    ax.scatter(to_np(t_boundary), to_np(getter(psi_boundary_pred)), marker='X', label='PINN prediction')
    ax.plot(to_np(t_linear), to_np(getter(psi_l_linear)), label='Analytical $x=x_l$')
    ax.plot(to_np(t_linear), to_np(getter(psi_r_linear)), label='Analytical $x=x_r$')
    ax.set_xlabel('$t$'); ax.legend()
plt.tight_layout(); plt.show(); plt.close()

# %% [markdown]
# ### Check inside domain

# %%
with torch.no_grad():
    psi_collocation_pred = model(xt_collocation)

fig = plt.figure(figsize=(9, 6))
plt.title('Positions of collocation points and boundary data')
plt.scatter(to_np(t_initial),     to_np(x_initial),     c=to_np(get_real(psi_initial)),          marker='X', vmin=-1, vmax=1)
plt.scatter(to_np(t_boundary),    to_np(x_boundary),    c=to_np(get_real(psi_boundary)),          marker='X', vmin=-1, vmax=1)
plt.scatter(to_np(t_collocation), to_np(x_collocation), c=to_np(get_real(psi_collocation_pred)), vmin=-1, vmax=1, marker='.', alpha=0.1)
plt.xlabel('$t$'); plt.ylabel('$x$'); plt.show()

psi_collocation_ana = psi(x_collocation, t_collocation)
print("L1 error on collocation points:",
      torch.mean(torch.abs(psi_collocation_ana - psi_collocation_pred)).item())

# %%
n_test = 10000
t_test  = torch.rand((n_test, 1), dtype=dtype, device=device) * (t1 - t0) + t0
x_test  = torch.rand((n_test, 1), dtype=dtype, device=device) * (xr - xl) + xl
xt_test = torch.cat([x_test, t_test], dim=1)

with torch.no_grad():
    psi_test_ana  = psi(x_test, t_test)
    psi_test_pred = model(xt_test)

print("L1 error on test points:",
      torch.mean(torch.abs(psi_test_ana - psi_test_pred)).item())

# %%
xlin   = np.linspace(xl.item(), xr.item(), n_linear)
tlin   = np.linspace(t0.item(), t1.item(), n_linear)
xx, tt = np.meshgrid(xlin, tlin)
xxtt   = torch.tensor(np.vstack([xx.flatten(), tt.flatten()]).T, dtype=dtype, device=device)

with torch.no_grad():
    psi_pred_grid = model(xxtt).cpu().numpy()
    psi_ana_grid  = psi(xxtt[:, 0:1], xxtt[:, 1:2]).cpu().numpy()

re_pred = psi_pred_grid[:, 0]
im_pred = psi_pred_grid[:, 1]
re_ana  = psi_ana_grid[:, 0]

print("L1 error (grid):", np.mean(np.abs(re_ana - re_pred)))
print("Avg density of prediction:", np.mean(re_pred**2 + im_pred**2))

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(9, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, tt, re_pred.reshape(n_linear, n_linear), cmap='viridis')
ax.view_init(35, 35)
ax.set_xlabel('$x$'); ax.set_ylabel('$t$')
ax.set_zlabel('$\\Re(\\psi(x, t))$')
ax.set_title('Real part of solution of 1D Schrödinger equation')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(range(len(hist)), hist, 'k-')
ax.set_xlabel('$n_{epoch}$')
ax.set_ylabel('$\\phi_{n_{epoch}}$')
plt.show()
