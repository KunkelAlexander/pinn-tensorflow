"""Neon GIF of a discrete-time PINN learning a two-frequency wave: low frequencies first (spectral bias).

Solves  i psi_t = -1/2 psi_xx  on the periodic domain [0, 1] for one Gauss-Legendre IRK step, as in
2_linear_schroedinger_1d_discrete_time.ipynb, with the initial condition
    psi0 = exp(2 pi i 2 x) + 0.5 exp(2 pi i 16 x).
A single unlabelled panel: the real part of the network's psi^{n+1} during training in cyan (exact solution
as a faint ghost), and below it the mismatch Re(psi_theta - psi_exact) in pink.

Usage:  python 4_spectral_bias_gif.py      ->  figures/spectral_bias.gif
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# ---- problem: two plane waves, one time step in which the fast one advances a third of a period --------
WAVES, AMPS = np.array([2, 16]), np.array([1.0, 0.5])
K = 2 * np.pi * WAVES
OMEGA = 0.5 * K**2
DT = 2.0 / OMEGA[1]
N, Q, WIDTH, DEPTH, STEPS, FRAMES = 256, 16, 64, 4, 10000, 100


def exact(x, t):
    return sum(a * np.exp(1j * (k * x - w * t)) for a, k, w in zip(AMPS, K, OMEGA))


def gauss_legendre(q):
    """Butcher tableau (A, b) of the q-stage Gauss-Legendre scheme, stacked as rows [A; b]."""
    x, _ = np.polynomial.legendre.leggauss(q)
    c = 0.5 * (x + 1.0)
    P = np.polynomial.legendre.legvander(x, q)
    integrals = np.column_stack([c] + [(P[:, k + 1] - P[:, k - 1]) / (2 * (2 * k + 1)) for k in range(1, q)])
    A = np.linalg.solve(P[:, :q].T, integrals.T).T
    b = np.linalg.solve(P[:, :q].T, np.eye(q)[0])
    return torch.tensor(np.vstack([A, b]))


# ---- network: outputs Re and Im of the q stages and psi^{n+1}, with analytic x-derivatives -------------
class MLP(torch.nn.Module):
    def __init__(self, n_out):
        super().__init__()
        sizes = [1] + [WIDTH] * DEPTH
        self.hidden = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = torch.nn.Linear(WIDTH, n_out)

    def forward(self, x):
        """Return psi, psi_x, psi_xx; the input x in [0, 1] is scaled to [-1, 1]."""
        h, hx, hxx = 2 * x - 1, torch.full_like(x, 2.0), torch.zeros_like(x)
        for layer in self.hidden:
            W = layer.weight
            z, zx, zxx = layer(h), hx @ W.T, hxx @ W.T
            a = torch.tanh(z)
            d1 = 1 - a**2
            h, hx, hxx = a, d1 * zx, d1 * zxx - 2 * a * d1 * zx**2
        W = self.out.weight
        return self.out(h), hx @ W.T, hxx @ W.T


weights = gauss_legendre(Q)                                  # (q+1, q)
x = torch.tensor((np.arange(N) + 0.5) / N).reshape(-1, 1)
psi0 = exact(x.numpy(), 0.0)
re0, im0 = torch.tensor(psi0.real), torch.tensor(psi0.imag)
x_bnd = torch.tensor([[0.0], [1.0]])
model = MLP(2 * (Q + 1))


def loss_fn():
    psi, _, psi_xx = model(x)
    re, im = psi[:, :Q + 1], psi[:, Q + 1:]
    N_re, N_im = -0.5 * psi_xx[:, Q + 1:2 * Q + 1], 0.5 * psi_xx[:, :Q]   # N[psi] = i/2 psi_xx on the stages
    # undo the IRK step: every stage and psi^{n+1} must lead back to psi0
    loss = torch.mean((re - DT * N_re @ weights.T - re0)**2 + (im - DT * N_im @ weights.T - im0)**2)
    # periodic boundary conditions for value and slope
    p, px, _ = model(x_bnd)
    return loss + torch.mean((p[0] - p[1])**2) + torch.mean((px[0] - px[1])**2) / K[1]**2


# ---- training, recording psi^{n+1} on a fine grid at log-spaced steps ---------------------------------
x_plot = np.linspace(0, 1, 1000)
x_plot_t = torch.tensor(x_plot).reshape(-1, 1)
psi_exact = exact(x_plot, DT)
record = set(np.unique(np.geomspace(1, STEPS, FRAMES).astype(int)))
snapshots = []

optim = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.MultiStepLR(optim, [int(0.6 * STEPS)], gamma=0.3)
for step in range(1, STEPS + 1):
    optim.zero_grad()
    loss = loss_fn()
    loss.backward()
    optim.step()
    sched.step()
    if step in record:
        with torch.no_grad():
            out = model(x_plot_t)[0].numpy()
        snapshots.append((step, out[:, Q] + 1j * out[:, 2 * Q + 1]))
    if step % 500 == 0:
        print(f'step {step:5d}: loss = {loss.item():.3e}', flush=True)

# ---- neon animation ---------------------------------------------------------------------------------
plt.style.use('dark_background')
CYAN, PINK = '#08F7FE', '#FE53BB'


def glow(ax, x, y, color, lw=2.0):
    ax.plot(x, y, color=color, lw=lw)
    for w, a in zip(np.logspace(-1, 3.5, 12, base=2), np.linspace(0.35, 0.02, 12)):
        ax.plot(x, y, color=color, lw=lw + w, alpha=a * 0.5, solid_capstyle='round')


fig = plt.figure(figsize=(12, 6.75), facecolor='black')
ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
err_max = max(np.abs((p - psi_exact).real).max() for _, p in snapshots)  # largest mismatch, at the start
top = 1.1 * np.abs(psi_exact.real).max()     # output centred at +top, mismatch at -err_max
y_out, y_err = top, -1.1 * err_max


def draw(i):
    _, psi = snapshots[i]
    err = (psi - psi_exact).real
    ax.cla()
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(y_err - 1.15 * err_max, y_out + 1.15 * top)
    ax.plot(x_plot, y_out + psi_exact.real, color='white', lw=1, alpha=0.25, ls='--')
    glow(ax, x_plot, y_out + psi.real, CYAN)
    ax.fill_between(x_plot, y_err, y_err + err, color=PINK, alpha=0.15, lw=0)
    glow(ax, x_plot, y_err + err, PINK)


# hold the last frame for a moment before the loop restarts
frames = list(range(len(snapshots))) + [len(snapshots) - 1] * 20
os.makedirs('figures', exist_ok=True)
FuncAnimation(fig, draw, frames=frames).save('figures/spectral_bias.gif', writer=PillowWriter(fps=12),
                                             dpi=80, savefig_kwargs={'facecolor': 'black'})
print('wrote figures/spectral_bias.gif')
