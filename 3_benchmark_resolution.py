"""Resolution benchmark: PINNs vs. finite differences and FFT for the 1D linear Schroedinger equation.

Solves  i psi_t = -1/2 psi_xx  (hbar = m = 1)  for a plane wave  psi = exp(i (k x - omega t)),
omega = k^2 / 2, on x in [0, 1] with `--waves` wavelengths in the domain (default 16, so that
N = 2^5 corresponds to two points per wavelength). The final time is chosen such that the wave
travels a third of the domain, i.e. t1 = 2 / (3 k).

Methods
-------
- 4th- and 6th-order central finite differences (periodic), explicit RK4 in time at half the
  stability limit
- Pseudospectral FFT solver with exact time integration
- Continuous-time PINN (as in 1_linear_schroedinger_1d.ipynb): psi_theta(x, t), residual enforced on
  an N x N grid of collocation points
- Discrete-time PINN (as in 2_linear_schroedinger_1d_discrete_time.ipynb): one Gauss-Legendre IRK step
  with q = 256 stages, psi_theta(x) enforced on N points at t0

"Resolution" N means grid points for FD/FFT and collocation points per dimension for the PINNs.
The PINNs are additionally benchmarked against network width at fixed N.

Error metric: mean L1 error of real and imaginary part at t1, 0.5 * mean(|dRe| + |dIm|).

Step-size benchmark
-------------------
A single time step of size dt for a plane wave in the periodic potential V(x) = V0 cos(2 pi x),
for omega dt = 1, ..., 128 (omega = k^2 / 2). Without a potential the Fourier propagator would be
exact for any dt, so the potential is what makes the classical solvers step-size dependent, as in
Schroedinger-Poisson. The exact solution follows from the eigen-decomposition of the Hamiltonian in a
truncated Fourier basis. Compared are
- the discrete-time PINN as in the notebook (q = 256, psi^{n+1} as a network output)
- an improved discrete-time PINN: q adapted to the step (smallest Gauss-Legendre scheme that is
  accurate to 1e-8 for this dt) and psi^{n+1} computed from the stages, psi0 + dt sum_k b_k N[psi_k]
- the continuous-time PINN on [0, dt] with 16 collocation points per period in t
- single-step split-step Fourier (Strang) and Crank-Nicolson, and the Gauss-Legendre IRK scheme with the
  same q solved exactly, i.e. what a perfectly trained discrete-time PINN would achieve

Usage
-----
    python 3_benchmark_resolution.py              # both benchmarks (~2.5 h on a 4-core CPU), resumes from cache
    python 3_benchmark_resolution.py --benchmarks step  # only the step-size benchmark (~1 h)
    python 3_benchmark_resolution.py --quick      # smoke test, a few minutes
    python 3_benchmark_resolution.py --plot-only  # replot from cached results

Results are cached in results/benchmark_plane_wave_<waves>waves.json and results/benchmark_step_size.json
after every run, so an interrupted benchmark resumes where it left off. Figures are written to
figures/benchmark_*.png.
"""

import argparse
import json
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import torch

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------------------------
# Neon plot style (same as the notebooks)
# ----------------------------------------------------------------------------------------------

plt.style.use('dark_background')

colors = [
    '#08F7FE',  # teal/cyan
    '#FE53BB',  # pink
    '#F5D300',  # yellow
    '#00ff41',  # matrix green
    '#FF00FF',  # magenta
    '#FFA500',  # orange
    '#00FFFF',  # cyan
]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
mpl.rcParams['image.cmap'] = 'magma'

# layers of increasingly wide, increasingly transparent lines create the glow
linewidths     = np.logspace(-5, 5, 20, base=2)
transparencies = np.linspace(1, 0, 20)


def glow(x, y, ax=None, **kwargs):
    """Plot a line with a neon-glow effect; kwargs go to the main line."""
    ax = ax or plt.gca()
    line, = ax.plot(x, y, **kwargs)
    for lw, alpha in zip(linewidths, transparencies):
        ax.plot(x, y, lw=lw, c=line.get_color(), alpha=alpha * 0.3)
    return line


def save(name, fig=None, tight=True):
    """Save the figure as figures/<name>.png."""
    os.makedirs('figures', exist_ok=True)
    (fig or plt.gcf()).savefig(f'figures/{name}.png', dpi=200, bbox_inches='tight' if tight else None)


# publication-ready fonts: large, bold labels and ticks, no titles (same as the notebooks)
mpl.rcParams.update({
    'font.size':          16,
    'font.weight':        'bold',
    'axes.labelsize':     20,
    'axes.labelweight':   'bold',
    'axes.labelpad':      8,
    'axes.linewidth':     1.5,
    'axes.edgecolor':     '#CCCCCC',
    'xtick.labelsize':    16,
    'ytick.labelsize':    16,
    'xtick.major.width':  1.5,
    'ytick.major.width':  1.5,
    'xtick.major.size':   6,
    'ytick.major.size':   6,
    'legend.fontsize':    14,
    'legend.frameon':     True,
    'legend.framealpha':  0.8,
    'legend.facecolor':   'black',
    'legend.edgecolor':   '#CCCCCC',
    'mathtext.fontset':   'dejavusans',
    'mathtext.default':   'bf',      # bold upright math so labels match the bold text
    'savefig.facecolor':  'black',
})
# the same fixed size for all benchmark figures (saved without tight cropping so that they have
# identical pixel sizes)
FIGSIZE = (11, 14)


def panel_label(ax, text, loc='upper right'):
    """Bold label inside a panel, in place of a title."""
    x, ha = (0.98, 'right') if loc.endswith('right') else (0.02, 'left')
    y, va = {'upper': (0.96, 'top'), 'center': (0.5, 'center'), 'lower': (0.04, 'bottom')}[loc.split()[0]]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=18, fontweight='bold',
            bbox=dict(facecolor='black', edgecolor='none', alpha=0.7, pad=3))


# one colour per method, used in all figures
STYLE = {
    'fd4':        dict(label='4th-order FD',          color='#08F7FE', marker='o'),
    'fd6':        dict(label='6th-order FD',          color='#F5D300', marker='o'),
    'fft':        dict(label='FFT',                   color='#00ff41', marker='o'),
    'pinn_cont':  dict(label='PINN continuous time',  color='#FE53BB', marker='X'),
    'pinn_disc':  dict(label='PINN discrete time',    color='#FFA500', marker='X'),
    # step-size benchmark
    'strang':       dict(label='Split-step Fourier (Strang)',                 color='#08F7FE', marker='o'),
    'cn':           dict(label='Crank-Nicolson (Fourier)',                    color='#F5D300', marker='o'),
    'irk_exact':    dict(label='Gauss IRK, same $q$, exact solve',            color='#00ff41', marker='o'),
    'pinn_disc_nb': dict(label=r'PINN discrete time (notebook: $q = 256$, $\psi^{n+1}$ learned)',
                         color='#FFA500', marker='X'),
    'pinn_disc_q':  dict(label=r'PINN discrete time ($q$ adapted, $\psi^{n+1}$ from stages)',
                         color='#FFFFFF', marker='X'),
}


# ----------------------------------------------------------------------------------------------
# Test problem
# ----------------------------------------------------------------------------------------------

class PlaneWave:
    def __init__(self, waves):
        self.xl, self.xr = 0.0, 1.0
        self.k     = 2 * np.pi * waves / (self.xr - self.xl)
        self.omega = 0.5 * self.k**2
        self.t0    = 0.0
        # the wave travels a third of the domain (phase velocity omega/k = k/2); avoids omega t1 being a
        # multiple of 2 pi, for which the aliased solution at N = waves would be exact by accident
        self.t1    = (2.0 / 3.0) / self.k

    def exact(self, x, t):
        """Complex exact solution, works for numpy arrays."""
        return np.exp(1j * (self.k * x - self.omega * t))

    def exact_torch(self, x, t):
        """Exact solution as [Re, Im] columns, x and t of shape (n, 1)."""
        phase = self.k * x - self.omega * t
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)

    def potential_torch(self, x):
        return torch.zeros_like(x)


def l1_error(psi_pred, psi_ana):
    """Mean L1 error of real and imaginary part."""
    return 0.5 * np.mean(np.abs(psi_pred.real - psi_ana.real) + np.abs(psi_pred.imag - psi_ana.imag))


# ----------------------------------------------------------------------------------------------
# Reference solvers: finite differences and FFT on a periodic grid
# ----------------------------------------------------------------------------------------------

# central stencils for the second derivative, coefficients for offsets 0, 1, 2, 3
FD_STENCILS = {
    4: np.array([-5/2, 4/3, -1/12]),
    6: np.array([-49/18, 3/2, -3/20, 1/90]),
}


def solve_fd(problem, n, order):
    h = (problem.xr - problem.xl) / n
    x = problem.xl + h * np.arange(n)
    c = FD_STENCILS[order] / h**2

    # periodic stencil as a sparse matrix (the diagonals at +-(n - j) wrap around), rhs = i/2 psi_xx
    L = sum(c[j] * (scipy.sparse.eye(n, k=j) + scipy.sparse.eye(n, k=-j) +
                    scipy.sparse.eye(n, k=n - j) + scipy.sparse.eye(n, k=-(n - j)))
            for j in range(1, len(c))) + c[0] * scipy.sparse.eye(n)
    L = (0.5j * L).tocsr()

    def rhs(psi):
        return L @ psi

    # RK4 is stable for imaginary eigenvalues |z| <= 2 sqrt(2); largest eigenvalue of the stencil at kh = pi
    lam_max = 0.5 * abs(c[0] + 2 * np.sum(c[1:] * (-1.0)**np.arange(1, len(c))))
    n_steps = int(np.ceil((problem.t1 - problem.t0) / (0.5 * 2 * np.sqrt(2) / lam_max)))
    dt      = (problem.t1 - problem.t0) / n_steps

    psi = problem.exact(x, problem.t0)
    start = time.perf_counter()
    for _ in range(n_steps):
        k1 = rhs(psi)
        k2 = rhs(psi + 0.5 * dt * k1)
        k3 = rhs(psi + 0.5 * dt * k2)
        k4 = rhs(psi + dt * k3)
        psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    wall = time.perf_counter() - start

    return l1_error(psi, problem.exact(x, problem.t1)), wall


def solve_fft(problem, n):
    h = (problem.xr - problem.xl) / n
    x = problem.xl + h * np.arange(n)
    kk = 2 * np.pi * np.fft.fftfreq(n, d=h)

    start = time.perf_counter()
    psi_hat = np.fft.fft(problem.exact(x, problem.t0))
    psi_hat *= np.exp(-0.5j * kk**2 * (problem.t1 - problem.t0))  # exact propagator in Fourier space
    psi = np.fft.ifft(psi_hat)
    wall = time.perf_counter() - start

    return l1_error(psi, problem.exact(x, problem.t1)), wall


# ----------------------------------------------------------------------------------------------
# Neural network with analytic input derivatives
# ----------------------------------------------------------------------------------------------

class MLP(torch.nn.Module):
    """Fully connected tanh network like init_model() in the notebooks.

    forward_with_derivatives() propagates first and second derivatives w.r.t. x (and the first
    derivative w.r.t. t) through the layers alongside the values. This yields the derivatives of
    all outputs at once, which matters for the 2(q+1) outputs of the discrete-time PINN, and is
    much cheaper than nested autograd.
    """

    def __init__(self, n_in, n_out, width, depth, lower, upper):
        super().__init__()
        self.register_buffer('lower', torch.tensor(lower))
        self.register_buffer('upper', torch.tensor(upper))
        sizes = [n_in] + [width] * depth
        self.hidden = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = torch.nn.Linear(width, n_out)
        for layer in list(self.hidden) + [self.out]:
            torch.nn.init.xavier_normal_(layer.weight)  # glorot_normal as in the notebooks
            torch.nn.init.zeros_(layer.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def scale(self, X):
        # normalise input to [-1, 1] in all dimensions
        return 2.0 * (X - self.lower) / (self.upper - self.lower) - 1.0

    def forward(self, X):
        h = self.scale(X)
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)

    def forward_with_derivatives(self, X, with_t=False):
        """Return psi, psi_x, psi_xx and (if with_t) psi_t. Input column 0 is x, column 1 is t."""
        n, n_in = X.shape
        ds = 2.0 / (self.upper - self.lower)  # derivative of the scaling layer

        h = self.scale(X)
        hx = torch.zeros(n, n_in); hx[:, 0] = ds[0]
        hxx = torch.zeros(n, n_in)
        if with_t:
            ht = torch.zeros(n, n_in); ht[:, 1] = ds[1]

        for layer in self.hidden:
            W = layer.weight
            z, zx, zxx = layer(h), hx @ W.T, hxx @ W.T
            a = torch.tanh(z)
            d1 = 1.0 - a**2           # tanh'
            h   = a
            hx  = d1 * zx
            hxx = d1 * zxx - 2.0 * a * d1 * zx**2
            if with_t:
                ht = d1 * (ht @ W.T)

        W = self.out.weight
        derivs = [self.out(h), hx @ W.T, hxx @ W.T]
        if with_t:
            derivs.append(ht @ W.T)
        return derivs


# ----------------------------------------------------------------------------------------------
# Training: Adam followed by L-BFGS
# ----------------------------------------------------------------------------------------------

def train(model, loss_fn, adam_steps, lbfgs_steps, log_prefix=''):
    # piecewise constant learning rate as in the notebooks, milestones at 20 % and 60 % of the steps
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    milestones = [int(0.2 * adam_steps), int(0.6 * adam_steps)]
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones, gamma=0.1)

    for i in range(adam_steps):
        optim.zero_grad()
        loss = loss_fn()
        loss.backward()
        optim.step()
        sched.step()
        if i % 1000 == 0:
            print(f'{log_prefix} Adam  {i:6d}: loss = {loss.item():.3e}', flush=True)

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_steps, history_size=50,
                                  tolerance_grad=1e-12, tolerance_change=1e-16,
                                  line_search_fn='strong_wolfe')

        def closure():
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        lbfgs.step(closure)

    with torch.no_grad():
        final = loss_fn().item()
    print(f'{log_prefix} final loss = {final:.3e}', flush=True)
    return final


def midpoints(a, b, n):
    return a + (b - a) * (np.arange(n) + 0.5) / n


def eval_grid(problem, n_eval=4096):
    x = midpoints(problem.xl, problem.xr, n_eval)
    return x, problem.exact(x, problem.t1)


def run_pinn_continuous(problem, n, width, depth, adam_steps, lbfgs_steps, seed=0, n_t=None):
    torch.manual_seed(seed)
    col = lambda a: torch.tensor(a).reshape(-1, 1)
    n_t = n_t or n

    # collocation points on an N x N_t grid, initial data on N points, boundary data on N_t points per side
    xg, tg = np.meshgrid(midpoints(problem.xl, problem.xr, n), midpoints(problem.t0, problem.t1, n_t))
    xt_col = torch.cat([col(xg.ravel()), col(tg.ravel())], dim=1)
    V_col = problem.potential_torch(xt_col[:, 0])

    x_ini = col(midpoints(problem.xl, problem.xr, n))
    t_ini = torch.full_like(x_ini, problem.t0)
    xt_ini, psi_ini = torch.cat([x_ini, t_ini], dim=1), problem.exact_torch(x_ini, t_ini)

    t_bnd = col(np.tile(midpoints(problem.t0, problem.t1, n_t), 2))
    x_bnd = col(np.repeat([problem.xl, problem.xr], n_t))
    xt_bnd, psi_bnd = torch.cat([x_bnd, t_bnd], dim=1), problem.exact_torch(x_bnd, t_bnd)

    model = MLP(2, 2, width, depth, [problem.xl, problem.t0], [problem.xr, problem.t1])

    def loss_fn():
        psi, psi_x, psi_xx, psi_t = model.forward_with_derivatives(xt_col, with_t=True)
        # residual of i psi_t + 1/2 psi_xx - V psi = 0, divided by omega to make it O(1) (non-dimensionalisation)
        res_re = psi_t[:, 0] + 0.5 * psi_xx[:, 1] - V_col * psi[:, 1]
        res_im = psi_t[:, 1] - 0.5 * psi_xx[:, 0] + V_col * psi[:, 0]
        loss  = torch.mean(res_re**2 + res_im**2) / problem.omega**2
        loss += torch.mean((model(xt_ini) - psi_ini)**2)
        loss += torch.mean((model(xt_bnd) - psi_bnd)**2)
        return loss

    start = time.perf_counter()
    loss = train(model, loss_fn, adam_steps, lbfgs_steps, log_prefix=f'[cont N={n} Nt={n_t} w={width}]')
    wall = time.perf_counter() - start

    x_eval, psi_eval = eval_grid(problem)
    with torch.no_grad():
        xt = torch.cat([col(x_eval), torch.full((len(x_eval), 1), problem.t1)], dim=1)
        out = model(xt).numpy()
    return l1_error(out[:, 0] + 1j * out[:, 1], psi_eval), loss, wall, model.n_params()


def load_irk(q):
    tmp = np.loadtxt(f'utilities/IRK_weights/Butcher_IRK{q}.txt', ndmin=2)
    assert tmp.shape[0] == q**2 + 2 * q
    weights = np.reshape(tmp[0:q**2 + q], (q + 1, q))  # rows 0, ..., q-1 contain A, row q contains b
    times = tmp[q**2 + q:]                              # c
    return torch.tensor(weights), times


def run_pinn_discrete(problem, n, width, depth, adam_steps, lbfgs_steps, q=256, seed=0, tableau=None,
                      final_from_stages=False):
    """Discrete-time PINN with one q-stage Gauss-Legendre IRK step.

    tableau: (A, b, c) of the IRK scheme, by default loaded from utilities/IRK_weights.
    final_from_stages: if False (as in the notebook), psi^{n+1} is an additional network output that is
    trained with the b row of the IRK scheme. If True, the network only outputs the q stages and
    psi^{n+1} = psi0 + dt sum_k b_k N[psi_k] is computed from them.
    """
    torch.manual_seed(seed)
    col = lambda a: torch.tensor(a).reshape(-1, 1)
    if tableau is None:
        irk_weights, irk_times = load_irk(q)
    else:
        A, b, c = tableau
        q = len(b)
        irk_weights, irk_times = torch.tensor(np.vstack([A, b])), c
    dt = problem.t1 - problem.t0

    # functions of x the network outputs: the q stages and, unless computed from them, psi^{n+1}
    n_t = q if final_from_stages else q + 1
    t_out = col(problem.t0 + dt * (np.ravel(irk_times) if final_from_stages else np.append(irk_times, 1.0)))
    weights = irk_weights[:n_t]  # rows of A, plus b if psi^{n+1} is an output

    # initial data on N points; every one of the n_t outputs has to reproduce it
    # output ordering: [Re(psi^{n+c_1}), ..., Re(psi^{n+1}), Im(psi^{n+c_1}), ..., Im(psi^{n+1})]
    x_ini = col(midpoints(problem.xl, problem.xr, n))
    psi_ini = problem.exact_torch(x_ini, torch.full_like(x_ini, problem.t0))
    psi_ini_stages = torch.cat([psi_ini[:, 0:1].expand(-1, n_t), psi_ini[:, 1:2].expand(-1, n_t)], dim=1)

    # boundary data at both boundaries for all output times, one row per boundary
    x_bnd = col([problem.xl, problem.xr])
    psi_l = problem.exact_torch(torch.full_like(t_out, problem.xl), t_out)
    psi_r = problem.exact_torch(torch.full_like(t_out, problem.xr), t_out)
    psi_bnd = torch.cat([torch.stack([psi_l[:, 0], psi_r[:, 0]]), torch.stack([psi_l[:, 1], psi_r[:, 1]])], dim=1)

    model = MLP(1, 2 * n_t, width, depth, [problem.xl], [problem.xr])

    def irk_rhs(x, psi, psi_xx):
        # N[psi] = i/2 psi_xx - i V psi, only the q stages enter - not the final time t1
        re, im = psi[:, :q], psi[:, n_t:n_t + q]
        V = problem.potential_torch(x)
        N_re = -0.5 * psi_xx[:, n_t:n_t + q] + V * im
        N_im =  0.5 * psi_xx[:, :q] - V * re
        return N_re, N_im

    def loss_fn():
        psi, _, psi_xx = model.forward_with_derivatives(x_ini)
        N_re, N_im = irk_rhs(x_ini, psi, psi_xx)
        # undo the IRK step to reconstruct psi^n from every output
        re0 = psi[:, :n_t] - dt * N_re @ weights.T
        im0 = psi[:, n_t:] - dt * N_im @ weights.T
        loss  = torch.mean((torch.cat([re0, im0], dim=1) - psi_ini_stages)**2)
        loss += torch.mean((model(x_bnd) - psi_bnd)**2)
        return loss

    start = time.perf_counter()
    loss = train(model, loss_fn, adam_steps, lbfgs_steps, log_prefix=f'[disc N={n} q={q} w={width}]')
    wall = time.perf_counter() - start

    x_eval, psi_eval = eval_grid(problem)
    x_eval_t = col(x_eval)
    with torch.no_grad():
        if final_from_stages:
            psi, _, psi_xx = model.forward_with_derivatives(x_eval_t)
            N_re, N_im = irk_rhs(x_eval_t, psi, psi_xx)
            psi0 = problem.exact_torch(x_eval_t, torch.full_like(x_eval_t, problem.t0))
            b_t = irk_weights[q]
            re1 = (psi0[:, 0] + dt * N_re @ b_t).numpy()
            im1 = (psi0[:, 1] + dt * N_im @ b_t).numpy()
        else:
            out = model(x_eval_t).numpy()
            re1, im1 = out[:, q], out[:, 2 * q + 1]
    return l1_error(re1 + 1j * im1, psi_eval), loss, wall, model.n_params()


# ----------------------------------------------------------------------------------------------
# Benchmark driver with a result cache
# ----------------------------------------------------------------------------------------------

class Cache:
    def __init__(self, path):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=1)


def run_benchmark(args, problem, cache, width_sweep):
    grid_ns = 2**np.arange(4, args.max_exp_grid + 1)
    pinn_cont_ns = 2**np.arange(4, args.max_exp_cont + 1)
    pinn_disc_ns = 2**np.arange(4, args.max_exp_disc + 1)

    # reference solvers
    for n in grid_ns:
        for method in ['fd4', 'fd6', 'fft']:
            key = f'{method}/N={n}'
            if cache.get(key) is None:
                err, wall = solve_fft(problem, n) if method == 'fft' else solve_fd(problem, n, int(method[2]))
                cache.put(key, dict(method=method, N=int(n), l1=err, wall=wall, dof=2 * int(n)))
            print(f'{key:24s} L1 = {cache.get(key)["l1"]:.3e}', flush=True)

    runners = {'pinn_cont': run_pinn_continuous, 'pinn_disc': run_pinn_discrete}
    default_width = {'pinn_cont': args.width_cont, 'pinn_disc': args.width_disc}

    # list of PINN runs: resolution sweep at default width, then width sweep at fixed resolution
    jobs = [(m, int(n), default_width[m]) for m, ns in
            [('pinn_disc', pinn_disc_ns), ('pinn_cont', pinn_cont_ns)] for n in ns]
    if width_sweep:
        jobs += [('pinn_disc', args.width_sweep_n, w) for w in args.widths_disc]
        jobs += [('pinn_cont', args.width_sweep_n, w) for w in args.widths_cont]

    for method, n, width in jobs:
        key = f'{method}/N={n}/width={width}/depth={args.depth}/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}'
        if cache.get(key) is None:
            err, loss, wall, n_params = runners[method](problem, n, width, args.depth, args.adam_steps, args.lbfgs_steps)
            cache.put(key, dict(method=method, N=n, width=width, depth=args.depth, l1=err, loss=loss,
                                wall=wall, n_params=n_params, adam_steps=args.adam_steps,
                                lbfgs_steps=args.lbfgs_steps))
        r = cache.get(key)
        print(f'{key:60s} L1 = {r["l1"]:.3e}  params = {r["n_params"]}  time = {r["wall"]:.0f} s', flush=True)


def select(cache, method, **match):
    rows = [r for r in cache.data.values() if r['method'] == method and all(r.get(k) == v for k, v in match.items())]
    return sorted(rows, key=lambda r: (r['N'], r.get('width', 0)))


def resolution_series(args, cache):
    """Rows of the resolution sweep for every method, PINNs at their default width."""
    train_match = dict(depth=args.depth, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)
    return {
        'fd4': select(cache, 'fd4'), 'fd6': select(cache, 'fd6'),
        'pinn_cont': select(cache, 'pinn_cont', width=args.width_cont, **train_match),
        'pinn_disc': select(cache, 'pinn_disc', width=args.width_disc, **train_match),
        'fft': select(cache, 'fft'),
    }


def plural(waves):
    return f'{waves} wavelength' + ('s' if waves > 1 else '')


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def plot_resolution(args, cases):
    """One panel per number of wavelengths, stacked vertically, like Fig. 10 of Kunkel et al. (2025)."""
    fig, axes = plt.subplots(len(cases), 1, figsize=FIGSIZE, sharex=True, squeeze=False, layout='constrained')
    axes = axes[:, 0]

    for ax, (waves, cache) in zip(axes, cases):
        panel_label(ax, f'Plane wave, {plural(waves)}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

        series = resolution_series(args, cache)
        for method, rows in series.items():
            if rows:
                label = STYLE[method]['label']
                if method.startswith('pinn'):
                    label += f' ({rows[0]["depth"]}x{rows[0]["width"]}, {rows[0]["n_params"]:,} params)'
                glow([r['N'] for r in rows], [max(r['l1'], 1e-16) for r in rows], ax=ax, lw=2.5, markersize=9,
                     color=STYLE[method]['color'], marker=STYLE[method]['marker'], label=label)

        # guide lines N^-4 and N^-6, anchored at the finest FD resolution that is not yet at round-off
        for method, p, ls in [('fd4', 4, '--'), ('fd6', 6, ':')]:
            rows = [r for r in series[method] if r['l1'] > 1e-11]
            if rows:
                n0, e0 = rows[-1]['N'], rows[-1]['l1']
                ns = np.array([max(n0 / 16, 16), min(n0 * 4, 2048)], dtype=float)
                ax.plot(ns, e0 * (ns / n0)**(-p), color='grey', ls=ls, lw=1.5, label=f'$N^{{-{p}}}$')

        # two points per wavelength
        if 2 * waves >= 16:
            ax.axvline(2 * waves, color='white', lw=0.8, alpha=0.5)
            ax.text(2 * waves * 1.07, 0.25, '2 points / wavelength', rotation=90, va='bottom', color='white',
                    alpha=0.7, fontsize=14, transform=ax.get_xaxis_transform())

        ax.set_ylim(1e-16, 1e4)   # headroom for the panel label
        ax.set_yticks(10.0**np.arange(-16, 5, 4))
        ax.set_xticks(2.0**np.arange(4, 12))
        ax.set_ylabel('$L_1$ error at $t_1$')

    axes[-1].set_xlabel('Points $N$ per dimension')
    # one legend for all panels below the plots, without duplicates
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=2, frameon=True)
    save('benchmark_1_error_vs_resolution', fig, tight=False)
    plt.close(fig)


def plot_cost(args, waves, cache):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, layout='constrained')
    train_match = dict(depth=args.depth, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)

    # top: error vs. degrees of freedom - trainable parameters for PINNs, 2N real unknowns for grid methods
    panel_label(ax1, f'{plural(waves)}\nwidth sweep at $N = {args.width_sweep_n}$', loc='center right')
    for method in ['fd4', 'fd6', 'fft']:
        rows = select(cache, method)
        glow([r['dof'] for r in rows], [max(r['l1'], 1e-16) for r in rows], ax=ax1, lw=2.5, markersize=9,
             color=STYLE[method]['color'], marker=STYLE[method]['marker'], label=STYLE[method]['label'])
    for method in ['pinn_cont', 'pinn_disc']:
        rows = sorted(select(cache, method, N=args.width_sweep_n, **train_match), key=lambda r: r['n_params'])
        if rows:
            glow([r['n_params'] for r in rows], [r['l1'] for r in rows], ax=ax1, lw=2.5, markersize=9,
                 color=STYLE[method]['color'], marker=STYLE[method]['marker'], label=STYLE[method]['label'])
            # label every point with its width
            for r in rows:
                ax1.annotate(f'w={r["width"]}', (r['n_params'], r['l1']), textcoords='offset points',
                             xytext=(0, -22), ha='center', fontsize=14, color=STYLE[method]['color'])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Degrees of freedom')
    ax1.set_ylabel('$L_1$ error at $t_1$')

    # bottom: error vs. wall-clock time of the resolution sweep
    panel_label(ax2, 'Resolution sweep', loc='lower right')
    for method, rows in resolution_series(args, cache).items():
        if rows:
            glow([max(r['wall'], 1e-6) for r in rows], [max(r['l1'], 1e-16) for r in rows], ax=ax2, lw=2.5,
                 markersize=9, color=STYLE[method]['color'], marker=STYLE[method]['marker'])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Wall-clock time [s]')
    ax2.set_ylabel('$L_1$ error at $t_1$')

    # one legend for both panels below the plots
    fig.legend(*ax1.get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=True)
    save('benchmark_2_error_vs_cost', fig, tight=False)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Step-size benchmark: a single time step in a periodic potential
# ----------------------------------------------------------------------------------------------

class PlaneWaveInPotential:
    """Plane wave with `waves` wavelengths in V(x) = V0 cos(2 pi x) on the periodic domain [0, 1].

    The exact solution is computed in a truncated Fourier basis psi = sum_m c_m exp(2 pi i m x) from the
    eigen-decomposition of the Hamiltonian H = -1/2 d_xx + V, in which V couples the modes m and m +- 1.
    """

    def __init__(self, waves, V0, dt, n_modes=128):
        self.xl, self.xr = 0.0, 1.0
        self.t0, self.t1 = 0.0, dt
        self.waves, self.V0 = waves, V0
        self.k     = 2 * np.pi * waves
        self.omega = 0.5 * self.k**2  # frequency of the initial plane wave without potential

        self.m = np.fft.fftfreq(n_modes, 1.0 / n_modes)  # integer mode numbers, FFT ordering
        H = np.diag(0.5 * (2 * np.pi * self.m)**2) + 0.5 * V0 * (np.abs(self.m[:, None] - self.m[None, :]) == 1)
        self.E, self.U = np.linalg.eigh(H)
        self.c0 = (self.m == waves).astype(complex)
        self.a0 = self.U.T @ self.c0  # initial state in the eigenbasis (H is real symmetric)

    def coefficients(self, t, propagator=None):
        """Fourier coefficients at time t; propagator(E) replaces the exact exp(-i E t)."""
        g = np.exp(-1j * self.E * t) if propagator is None else propagator(self.E)
        return self.U @ (g * self.a0)

    def evaluate(self, c, x):
        return np.exp(2j * np.pi * np.outer(x, self.m)) @ c

    def exact(self, x, t):
        x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
        out = np.empty(x.shape, dtype=complex)
        for tu in np.unique(t):
            sel = t == tu
            out[sel] = self.evaluate(self.coefficients(tu), x[sel])
        return out

    def exact_torch(self, x, t):
        psi = self.exact(x.numpy().ravel(), t.numpy().ravel())
        return torch.tensor(np.stack([psi.real, psi.imag], axis=1))

    def potential_torch(self, x):
        return self.V0 * torch.cos(2 * np.pi * x)


def gauss_legendre_tableau(q):
    """Butcher tableau (A, b, c) of the q-stage Gauss-Legendre IRK scheme of order 2q.

    The collocation conditions are written in the Legendre basis instead of monomials, which keeps the
    linear systems well conditioned also for large q.
    """
    x, _ = np.polynomial.legendre.leggauss(q)
    c = 0.5 * (x + 1.0)
    P = np.polynomial.legendre.legvander(x, q)  # P[j, k] = P_k(x_j), k = 0, ..., q
    # integrals of P_k(2 tau - 1) from 0 to c_j: c_j for k = 0, (P_{k+1} - P_{k-1}) / (2 (2k + 1)) otherwise
    integrals = np.empty((q, q))
    integrals[:, 0] = c
    for k in range(1, q):
        integrals[:, k] = (P[:, k + 1] - P[:, k - 1]) / (2 * (2 * k + 1))
    V = P[:, :q]
    A = np.linalg.solve(V.T, integrals.T).T      # sum_j a_ij P_k(x_j) = int_0^{c_i} P_k
    b = np.linalg.solve(V.T, np.eye(q)[0])       # sum_j b_j P_k(x_j) = int_0^1 P_k = delta_k0
    return A, b, c


def irk_propagator(A, b, dt):
    """Stability function R(-i E dt) = 1 + z b^T (I - z A)^{-1} 1 of the IRK scheme for every energy E."""
    q = len(b)
    def propagator(E):
        return np.array([1 + z * b @ np.linalg.solve(np.eye(q) - z * A, np.ones(q)) for z in -1j * E * dt])
    return propagator


Q_CANDIDATES = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def required_q(problem, tol=1e-8):
    """Smallest Gauss-Legendre scheme whose temporal error (RMS over x, via Parseval) is below tol."""
    exact = problem.coefficients(problem.t1)
    for q in Q_CANDIDATES:
        A, b, c = gauss_legendre_tableau(q)
        if np.linalg.norm(problem.coefficients(None, irk_propagator(A, b, problem.t1)) - exact) < tol:
            break
    return q, (A, b, c)


def solve_strang(problem):
    """One Strang split step exp(-i V dt/2) exp(i dt/2 d_xx) exp(-i V dt/2) on the Fourier grid."""
    n, dt = len(problem.m), problem.t1
    x = np.arange(n) / n
    half_kick = np.exp(-0.5j * dt * problem.V0 * np.cos(2 * np.pi * x))
    start = time.perf_counter()
    psi = half_kick * problem.evaluate(problem.c0, x)
    psi = np.fft.ifft(np.exp(-0.5j * dt * (2 * np.pi * problem.m)**2) * np.fft.fft(psi))
    psi = half_kick * psi
    c = np.fft.fft(psi) / n
    wall = time.perf_counter() - start
    return c, wall


def step_errors_classical(problem):
    """L1 errors and wall-clock times of the classical single-step solvers."""
    x_eval, psi_eval = eval_grid(problem)
    dt = problem.t1
    results = {}

    c, wall = solve_strang(problem)
    results['strang'] = (l1_error(problem.evaluate(c, x_eval), psi_eval), wall, None)

    start = time.perf_counter()
    c = problem.coefficients(None, lambda E: (1 - 0.5j * E * dt) / (1 + 0.5j * E * dt))
    results['cn'] = (l1_error(problem.evaluate(c, x_eval), psi_eval), time.perf_counter() - start, None)

    q, (A, b, _) = required_q(problem)
    start = time.perf_counter()
    c = problem.coefficients(None, irk_propagator(A, b, dt))
    results['irk_exact'] = (l1_error(problem.evaluate(c, x_eval), psi_eval), time.perf_counter() - start, q)
    return results


def run_step_benchmark(args, cache):
    for wdt in args.step_values:
        omega = 0.5 * (2 * np.pi * args.step_waves)**2
        problem = PlaneWaveInPotential(args.step_waves, args.potential, wdt / omega)
        q, tableau = required_q(problem)
        print(f'--- omega dt = {wdt:g}: dt = {problem.t1:.5f}, required q = {q}', flush=True)

        for method, (err, wall, q_used) in step_errors_classical(problem).items():
            key = f'{method}/wdt={wdt:g}'
            cache.put(key, dict(method=method, wdt=wdt, l1=err, wall=wall, q=q_used))
            print(f'{key:40s} L1 = {err:.3e}', flush=True)

        # PINNs; the continuous one gets 16 collocation points per period of omega in t
        n_t = max(args.step_n, int(np.ceil(16 * wdt / (2 * np.pi))))
        runs = {
            'pinn_disc_nb': lambda: run_pinn_discrete(problem, args.step_n, args.width_disc, args.depth,
                                                      args.adam_steps, args.lbfgs_steps, q=256),
            'pinn_disc_q':  lambda: run_pinn_discrete(problem, args.step_n, args.width_disc, args.depth,
                                                      args.adam_steps, args.lbfgs_steps, tableau=tableau,
                                                      final_from_stages=True),
            'pinn_cont':    lambda: run_pinn_continuous(problem, args.step_n, args.width_cont, args.depth,
                                                        args.adam_steps, args.lbfgs_steps, n_t=n_t),
        }
        q_of = {'pinn_disc_nb': 256, 'pinn_disc_q': q, 'pinn_cont': None}
        for method, run in runs.items():
            key = (f'{method}/wdt={wdt:g}/N={args.step_n}/V0={args.potential:g}/depth={args.depth}'
                   f'/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}')
            if cache.get(key) is None:
                err, loss, wall, n_params = run()
                cache.put(key, dict(method=method, wdt=wdt, l1=err, loss=loss, wall=wall, n_params=n_params,
                                    q=q_of[method], n_t=n_t if method == 'pinn_cont' else None,
                                    N=args.step_n, V0=args.potential, depth=args.depth,
                                    adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps))
            r = cache.get(key)
            print(f'{key:70s} L1 = {r["l1"]:.3e}  time = {r["wall"]:.0f} s', flush=True)


def plot_step_size(args, cache):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout='constrained')
    pinn_match = dict(N=args.step_n, V0=args.potential, depth=args.depth,
                      adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)

    for method in ['strang', 'cn', 'irk_exact', 'pinn_cont', 'pinn_disc_nb', 'pinn_disc_q']:
        match = pinn_match if method.startswith('pinn') else {}
        rows = sorted([r for r in cache.data.values() if r['method'] == method and r['wdt'] in args.step_values
                       and all(r.get(k) == v for k, v in match.items())], key=lambda r: r['wdt'])
        if not rows:
            continue
        wdt = [r['wdt'] for r in rows]
        glow(wdt, [max(r['l1'], 1e-16) for r in rows], ax=ax1, lw=2.5, markersize=9,
             color=STYLE[method]['color'], marker=STYLE[method]['marker'], label=STYLE[method]['label'])
        glow(wdt, [max(r['wall'], 1e-6) for r in rows], ax=ax2, lw=2.5, markersize=9,
             color=STYLE[method]['color'], marker=STYLE[method]['marker'])
        if method == 'pinn_disc_q':
            # number of stages used for each step, in a row above the curves
            for r in rows:
                ax1.text(r['wdt'], 0.84, f'q={r["q"]}', transform=ax1.get_xaxis_transform(),
                         ha='center', fontsize=14, color=STYLE[method]['color'])

    # stability limit of an explicit scheme at the same spatial resolution, for reference
    c = FD_STENCILS[6]
    lam_max = 0.5 * abs(c[0] + 2 * np.sum(c[1:] * (-1.0)**np.arange(1, len(c)))) * args.step_n**2
    wdt_explicit = 0.5 * (2 * np.pi * args.step_waves)**2 * 2 * np.sqrt(2) / lam_max
    ax1.text(0.02, 0.03, f'Explicit RK4 + 6th-order FD at $N = {args.step_n}$ is stable only for '
             f'$\\omega\\Delta t \\leq {wdt_explicit:.3f}$', transform=ax1.transAxes, fontsize=14, alpha=0.8)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-16, 1e4)   # headroom for the panel label
    ax1.set_yticks(10.0**np.arange(-16, 5, 4))
    ax1.set_ylabel('$L_1$ error at $t_1 = \\Delta t$')
    panel_label(ax1, f'Single step, {plural(args.step_waves)}, $V = {args.potential:g}\\cos(2\\pi x)$', loc='upper left')
    ax2.set_yscale('log')
    ax2.set_ylabel('Wall-clock time [s]')
    panel_label(ax2, 'Cost of one step', loc='upper left')
    ax2.set_xlabel(r'Step size $\omega \Delta t$')
    fig.legend(*ax1.get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=True)
    save('benchmark_3_error_vs_step_size', fig, tight=False)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--waves', type=int, nargs='+', default=[16, 2], help='wavelengths in the domain, one panel each')
    p.add_argument('--width-sweep-waves', type=int, default=2, help='test case for the width sweep')
    p.add_argument('--depth', type=int, default=4, help='hidden layers of both PINNs')
    p.add_argument('--width-cont', type=int, default=20, help='neurons per layer, continuous PINN (notebook: 20)')
    p.add_argument('--width-disc', type=int, default=50, help='neurons per layer, discrete PINN (notebook: 50)')
    p.add_argument('--widths-cont', type=int, nargs='*', default=[10, 20, 40, 80])
    p.add_argument('--widths-disc', type=int, nargs='*', default=[12, 25, 50, 100])
    p.add_argument('--width-sweep-n', type=int, default=64, help='resolution N of the width sweep')
    p.add_argument('--max-exp-cont', type=int, default=7, help='continuous PINN up to N = 2^this')
    p.add_argument('--max-exp-disc', type=int, default=11, help='discrete PINN up to N = 2^this')
    p.add_argument('--max-exp-grid', type=int, default=11, help='FD and FFT up to N = 2^this')
    p.add_argument('--adam-steps', type=int, default=2000)
    p.add_argument('--lbfgs-steps', type=int, default=1500)
    p.add_argument('--benchmarks', nargs='+', choices=['resolution', 'step'], default=['resolution', 'step'])
    p.add_argument('--step-waves', type=int, default=2, help='wavelengths in the step-size benchmark')
    p.add_argument('--potential', type=float, default=20.0, help='V0 of V = V0 cos(2 pi x), step-size benchmark')
    p.add_argument('--step-n', type=int, default=64, help='collocation points per dimension, step-size benchmark')
    p.add_argument('--step-values', type=float, nargs='*', default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help='step sizes omega dt of the step-size benchmark')
    p.add_argument('--quick', action='store_true', help='smoke test with few steps and a small sweep')
    p.add_argument('--plot-only', action='store_true', help='only replot cached results')
    args = p.parse_args()

    if args.quick:
        args.adam_steps, args.lbfgs_steps = 200, 50
        args.max_exp_cont, args.max_exp_disc, args.max_exp_grid = 5, 6, 9
        args.widths_cont, args.widths_disc = [10, 20], [25, 50]
        args.width_sweep_n = 32
        args.step_values = [1, 8, 64]
    suffix = '_quick' if args.quick else ''

    if 'resolution' in args.benchmarks:
        cases = []
        for waves in args.waves:
            cache = Cache(f'results/benchmark_plane_wave_{waves}waves{suffix}.json')
            problem = PlaneWave(waves)
            print(f'--- {plural(waves)}: k = {problem.k:.3f}, omega = {problem.omega:.3f}, t1 = {problem.t1:.5f}')
            if not args.plot_only:
                run_benchmark(args, problem, cache, width_sweep=(waves == args.width_sweep_waves))
            cases.append((waves, cache))

        plot_resolution(args, cases)
        for waves, cache in cases:
            if waves == args.width_sweep_waves:
                plot_cost(args, waves, cache)

    if 'step' in args.benchmarks:
        cache = Cache(f'results/benchmark_step_size{suffix}.json')
        if not args.plot_only:
            run_step_benchmark(args, cache)
        plot_step_size(args, cache)


if __name__ == '__main__':
    main()
