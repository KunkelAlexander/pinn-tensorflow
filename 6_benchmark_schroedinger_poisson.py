"""Schroedinger-Poisson benchmark: discrete-time PINN vs. split-step Fourier on a periodic problem with random modes.

Scripts 3 and 5 used a plane wave (in a given potential), for which the PINN needs to represent a single Fourier
mode. Here the problem is the 1D Schroedinger-Poisson system on the periodic domain [0, 1],

    i psi_t = -1/2 psi_xx + V psi,      V_xx = kappa (|psi|^2 - 1),      int |psi|^2 dx = 1,

with an initial state of a few random Fourier modes, psi_0 = sum_m a_m exp(2 pi i m x): nonzero integer modes m drawn
from -k_max ... k_max with random complex amplitudes. The potential is the self-gravity of the wave (kappa > 0 attracts),
and is what makes the problem nonlinear and the classical solvers step-size dependent.

One time step of size dt, with omega dt = `--wdt` and omega = 1/2 (2 pi k_max)^2 as in script 5, is solved with
- split-step Fourier (Strang, kick-drift-kick) with 1, 2, 4, ... substeps on the same N-point grid,
- the Gauss-Legendre IRK scheme with the smallest q accurate to 1e-8, solved exactly (Newton-Krylov in Fourier
  space): what a perfectly trained discrete-time PINN would achieve,
- the discrete-time PINN of script 3 (one IRK step, psi^{n+1} computed from the stages), as baseline and with the
  improvements of script 5. The potential of every stage is obtained from |psi_k|^2 on the collocation grid with a
  differentiable FFT Poisson solver. Periodicity is enforced with a loss term for psi and psi_x at x = 0 and 1,
  since the exact solution is unknown at the boundary.
The error is measured against a converged reference: 4th-order (Yoshida) split-step on 1024 points with many steps.
Same metric as in script 3, mean L1 error of real and imaginary part at t1.

Variants of the PINN (`--variants`, combined with '+'): baseline; ff, aw, rwf as in script 5; and
  per  periodic features [sin(2 pi m x), cos(2 pi m x)], m = 1 ... 16, which make the network exactly periodic with
       integer modes, matching this problem; the periodicity loss is dropped (the one thing script 5 did not test)

Usage
-----
    python 6_benchmark_schroedinger_poisson.py                 # one problem, three PINN variants, resumes from cache
    python 6_benchmark_schroedinger_poisson.py --variants baseline ff+aw+rwf
    python 6_benchmark_schroedinger_poisson.py --quick         # smoke test
    python 6_benchmark_schroedinger_poisson.py --plot-only     # replot from cached results

Results are cached in results/benchmark_sp.json (the problem is part of the key, so other parameters can be added
later). Figures are written to figures/benchmark_8_* and benchmark_9_*.png.
"""

import argparse
import importlib.util
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch

# 5_benchmark_strategies.py starts with a digit and can therefore not be imported by name; it brings script 3 along
spec = importlib.util.spec_from_file_location(
    'strategies', os.path.join(os.path.dirname(os.path.abspath(__file__)), '5_benchmark_strategies.py'))
b5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b5)
b3 = b5.b3

STRATEGIES = dict(b5.STRATEGIES, per=dict(periodic=True))
LABELS = dict(b5.LABELS, per='periodic features')
COLORS = {**b5.COLORS, ('per',): '#F5D300'}
N_PLOT = 512   # points of the solution stored in the cache for the figures


def combo_of(variant):
    return () if variant == 'baseline' else tuple(variant.split('+'))


def label(combo):
    return 'baseline' if not combo else ' + '.join(LABELS[s] for s in combo)


def color(combo):
    return COLORS.get(combo, '#08F7FE')


def method_name(combo):
    return 'pinn_sp' + ''.join('+' + s for s in combo)


def kwargs(combo):
    return {k: v for s in combo for k, v in STRATEGIES[s].items()}


# ----------------------------------------------------------------------------------------------
# Problem and classical solvers
# ----------------------------------------------------------------------------------------------

def inverse_laplacian(n):
    """-1 / k^2 for the modes of an n-point periodic grid on [0, 1], zero for the mean."""
    k = 2 * np.pi * np.fft.fftfreq(n, 1.0 / n)
    out = np.zeros(n)
    out[1:] = -1.0 / k[1:]**2
    return out


class SchroedingerPoisson:
    def __init__(self, n_modes, k_max, coupling, wdt, seed):
        rng = np.random.default_rng(seed)
        nonzero = np.delete(np.arange(-k_max, k_max + 1), k_max)
        self.modes = np.sort(rng.choice(nonzero, n_modes, replace=False))
        amps = rng.normal(size=n_modes) + 1j * rng.normal(size=n_modes)
        self.amps = amps / np.linalg.norm(amps)    # mean density 1
        self.xl, self.xr, self.t0 = 0.0, 1.0, 0.0
        self.kappa = coupling
        self.omega = 0.5 * (2 * np.pi * k_max)**2
        self.t1 = wdt / self.omega
        self.signature = f'modes={n_modes}/kmax={k_max}/seed={seed}/kappa={coupling:g}/wdt={wdt:g}'
        self._reference = None

    def psi0(self, x):
        return np.exp(2j * np.pi * np.outer(x, self.modes)) @ self.amps

    def potential(self, psi):
        """Periodic Poisson solve on the grid along the last axis: V_xx = kappa (|psi|^2 - mean)."""
        return np.fft.ifft(self.kappa * inverse_laplacian(psi.shape[-1]) * np.fft.fft(abs(psi)**2, axis=-1),
                           axis=-1).real

    def potential_torch(self, rho):
        """The same for rho of shape (n, q), differentiable."""
        g = torch.tensor(self.kappa * inverse_laplacian(rho.shape[0]))[:, None]
        return torch.fft.ifft(g * torch.fft.fft(rho, dim=0), dim=0).real

    def strang(self, psi, dt):
        """Kick-drift-kick split step; the second potential is from the density after the drift."""
        k = 2 * np.pi * np.fft.fftfreq(psi.shape[-1], 1.0 / psi.shape[-1])
        psi = np.exp(-0.5j * dt * self.potential(psi)) * psi
        psi = np.fft.ifft(np.exp(-0.5j * dt * k**2) * np.fft.fft(psi))
        return np.exp(-0.5j * dt * self.potential(psi)) * psi

    def yoshida(self, psi, dt):
        """4th-order composition of three Strang steps."""
        s = 2.0**(1.0 / 3.0)
        w1, w0 = 1.0 / (2.0 - s), -s / (2.0 - s)
        for w in (w1, w0, w1):
            psi = self.strang(psi, w * dt)
        return psi

    def split_step(self, n, steps, step=None):
        """psi(t1) on an n-point grid after `steps` substeps."""
        psi = self.psi0(np.arange(n) / n)
        for _ in range(steps):
            psi = (step or self.strang)(psi, (self.t1 - self.t0) / steps)
        return psi

    def evaluate(self, psi_grid, x):
        """Fourier interpolation of a solution given on a uniform grid [0, 1) to the points x."""
        n = len(psi_grid)
        return np.exp(2j * np.pi * np.outer(x, np.fft.fftfreq(n, 1.0 / n))) @ (np.fft.fft(psi_grid) / n)

    def reference(self, x, n=1024, steps=None):
        """Converged solution at the points x (64 substeps per unit of omega dt, at least 256); also returns the
        difference to half the steps."""
        if self._reference is None:
            steps = steps or max(256, int(round(64 * self.t1 * self.omega)))
            fine = self.split_step(n, steps, self.yoshida)
            coarse = self.split_step(n, steps // 2, self.yoshida)
            self._reference = (fine, np.max(abs(fine - coarse)), np.max(abs(np.fft.fft(fine)[n // 4:3 * n // 4] / n)))
        fine, difference, tail = self._reference
        return self.evaluate(fine, x), difference, tail


def eval_points(n_eval=4096):
    return b3.midpoints(0.0, 1.0, n_eval)


def ssf_results(problem, n, steps_list, repeats=3):
    """L1 error and wall-clock time of Strang split-step Fourier with every number of substeps."""
    x = eval_points()
    ref = problem.reference(x)[0]
    out = {}
    for steps in steps_list:
        wall = min(timed(problem.split_step, n, steps) for _ in range(repeats))
        out[steps] = dict(l1=b3.l1_error(problem.evaluate(problem.split_step(n, steps), x), ref), wall=wall)
    return out


def timed(f, *args):
    start = time.perf_counter()
    f(*args)
    return time.perf_counter() - start


def solve_irk(problem, tableau, n=128, return_stages=False):
    """One step of the q-stage Gauss-Legendre scheme, stage equations solved exactly (Newton-Krylov).

    The residual is preconditioned with the linear part of the equation, which is diagonal in Fourier space,
    (I - dt lambda_j A)^{-1} for every mode j. Returns psi(t1) on an n-point grid, with return_stages also the
    stages, shape (q, n).
    """
    A, b, _ = tableau
    q, dt = len(b), problem.t1 - problem.t0
    k = 2 * np.pi * np.fft.fftfreq(n, 1.0 / n)
    lam = -0.5j * k**2    # i/2 d_xx in Fourier space
    precond = np.linalg.inv(np.eye(q) - dt * lam[:, None, None] * A)   # (n, q, q)
    psi0 = problem.psi0(np.arange(n) / n)

    def rhs(psi):
        return np.fft.ifft(lam * np.fft.fft(psi, axis=-1), axis=-1) - 1j * problem.potential(psi) * psi

    def residual(z):
        psi = (z[:q * n] + 1j * z[q * n:]).reshape(q, n)
        r = np.fft.fft(psi - psi0 - dt * A @ rhs(psi), axis=-1)
        g = np.fft.ifft(np.einsum('jab,bj->aj', precond, r), axis=-1)
        return np.concatenate([g.real.ravel(), g.imag.ravel()])

    z0 = np.tile(psi0, (q, 1))
    z = scipy.optimize.newton_krylov(residual, np.concatenate([z0.real.ravel(), z0.imag.ravel()]), f_tol=1e-12)
    psi = (z[:q * n] + 1j * z[q * n:]).reshape(q, n)
    psi1 = psi0 + dt * b @ rhs(psi)
    return (psi1, psi) if return_stages else psi1


Q_CANDIDATES = [4, 6, 8, 12, 16, 24, 32, 48, 64]


def irk_exact(problem, tol=1e-8):
    """Smallest q of Q_CANDIDATES for which the IRK step is within tol (RMS) of the reference, and its error."""
    x = eval_points()
    ref = problem.reference(x)[0]
    for q in Q_CANDIDATES:
        tableau = b3.gauss_legendre_tableau(q)
        start = time.perf_counter()
        psi = solve_irk(problem, tableau)
        wall = time.perf_counter() - start
        pred = problem.evaluate(psi, x)
        print(f'    IRK q = {q}: rms difference to reference {np.sqrt(np.mean(abs(pred - ref)**2)):.2e}', flush=True)
        if np.sqrt(np.mean(abs(pred - ref)**2)) < tol:
            break
    return dict(l1=b3.l1_error(pred, ref), wall=wall, q=q)


# ----------------------------------------------------------------------------------------------
# Discrete-time PINN
# ----------------------------------------------------------------------------------------------

def run_pinn_sp(problem, n, width, depth, adam_steps, lbfgs_steps, tableau, seed=0, fourier=None, rwf=False,
                adaptive=None, periodic=False):
    """Discrete-time PINN with one q-stage Gauss-Legendre IRK step for Schroedinger-Poisson.

    As run_pinn_discrete() of script 3 with psi^{n+1} computed from the stages, except for the potential of every stage
    (Poisson solve of |psi_k|^2 on the N collocation points) and the boundary term, which is periodicity of psi
    and psi_x instead of exact data. periodic: network of integer Fourier modes that is periodic by construction, the
    boundary term is dropped.
    """
    torch.manual_seed(seed)
    col = lambda a: torch.tensor(a).reshape(-1, 1)
    A, b, _ = tableau
    q, dt = len(b), problem.t1 - problem.t0
    A_t, b_t = torch.tensor(A), torch.tensor(b)

    x_col = col(b3.midpoints(0.0, 1.0, n))
    psi0 = problem.psi0(x_col.numpy().ravel())
    psi0_re, psi0_im = torch.tensor(psi0.real)[:, None], torch.tensor(psi0.imag)[:, None]
    x_bnd = col([0.0, 1.0])
    k_max = float(abs(problem.modes).max()) or 1.0

    model = b3.MLP(1, 2 * q, width, depth, [0.0], [1.0], fourier_scales=(1,) if periodic else fourier, rwf=rwf)
    if periodic:
        model.freqs = torch.arange(1.0, b3.FOURIER_PER_SCALE + 1).reshape(-1, 1)   # cycles per domain

    def stages(x):
        """Stages psi_k and N[psi_k] = i/2 psi_xx - i V psi_k, real and imaginary part, shape (len(x), q) each."""
        out, _, out_xx = model.forward_with_derivatives(x)
        re, im = out[:, :q], out[:, q:]
        V = problem.potential_torch(re**2 + im**2)
        return re, im, -0.5 * out_xx[:, q:] + V * im, 0.5 * out_xx[:, :q] - V * re

    def terms_fn():
        re, im, N_re, N_im = stages(x_col)
        # undo the IRK step to reconstruct psi^n from every stage
        terms = [torch.mean(torch.cat([re - dt * N_re @ A_t.T - psi0_re, im - dt * N_im @ A_t.T - psi0_im], dim=1)**2)]
        if not periodic:
            out, out_x, _ = model.forward_with_derivatives(x_bnd)
            terms.append(torch.mean((out[0] - out[1])**2) + torch.mean((out_x[0] - out_x[1])**2) / (2 * np.pi * k_max)**2)
        return torch.stack(terms)

    start = time.perf_counter()
    loss = b3.train(model, terms_fn, adam_steps, lbfgs_steps, adaptive=adaptive,
                    log_prefix=f'[sp N={n} q={q} w={width}]')
    wall = time.perf_counter() - start

    x_eval = eval_points()
    x_eval_t = col(x_eval)
    with torch.no_grad():
        re, im, N_re, N_im = stages(x_eval_t)
    psi0_eval = problem.psi0(x_eval)
    psi1 = psi0_eval + dt * (N_re @ b_t).numpy() + 1j * dt * (N_im @ b_t).numpy()
    ref = problem.reference(x_eval)[0]
    return b3.l1_error(psi1, ref), loss, wall, model.n_params(), psi1[::len(x_eval) // N_PLOT]


# ----------------------------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------------------------

def train_match(args):
    return dict(N=args.n, width=args.width, depth=args.depth, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)


def run_benchmark(args, problem, cache):
    sig = problem.signature
    key = f'irk_exact/{sig}'
    if cache.get(key) is None:
        r = irk_exact(problem)
        cache.put(key, dict(method='irk_exact', problem=sig, l1=r['l1'], wall=r['wall'], q=r['q']))
    r = cache.get(key)
    print(f'{key:70s} L1 = {r["l1"]:.3e}  q = {r["q"]}', flush=True)

    for variant in args.variants:
        combo = combo_of(variant)
        method = method_name(combo)
        key = (f'{method}/{sig}/N={args.n}/width={args.width}/depth={args.depth}'
               f'/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}')
        if cache.get(key) is None:
            tableau = b3.gauss_legendre_tableau(cache.get(f'irk_exact/{sig}')['q'])
            err, loss, wall, n_params, psi = run_pinn_sp(problem, args.n, args.width, args.depth, args.adam_steps,
                                                         args.lbfgs_steps, tableau, **kwargs(combo))
            cache.put(key, dict(method=method, problem=sig, l1=err, loss=loss, wall=wall, n_params=n_params,
                                q=len(tableau[1]), psi_re=psi.real.tolist(), psi_im=psi.imag.tolist(),
                                **train_match(args)))
        r = cache.get(key)
        print(f'{key:100s} L1 = {r["l1"]:.3e}  loss = {r["loss"]:.1e}  time = {r["wall"]:.0f} s', flush=True)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def pinn_rows(args, problem, cache):
    """Cached PINN results of the requested variants, {combo: row}."""
    rows = {}
    for variant in args.variants:
        combo = combo_of(variant)
        r = b3.select(cache, method_name(combo), problem=problem.signature, **train_match(args))
        if r:
            rows[combo] = r[0]
    return rows


def describe(problem, args):
    psi0 = problem.psi0(np.arange(args.n) / args.n)
    _, difference, tail = problem.reference(eval_points())
    print(f'--- {problem.signature}')
    print(f'modes m = {problem.modes.tolist()}, |a_m|^2 = {np.round(abs(problem.amps)**2, 3).tolist()}')
    print(f'omega = {problem.omega:.1f}, dt = {problem.t1:.5f}, V rms = {np.std(problem.potential(psi0)):.1f}, '
          f'dt V rms = {problem.t1 * np.std(problem.potential(psi0)):.2f}')
    print(f'reference: max difference to half the steps {difference:.1e}, largest coefficient of the upper half of '
          f'the spectrum {tail:.1e}', flush=True)


def title(problem, args):
    modes = ', '.join(f'{m:+d}'.replace('-', '−') for m in problem.modes)
    return (f'Schrödinger-Poisson, $m$ = {modes}, $\\omega\\Delta t = {args.wdt:g}$, $\\kappa = {args.coupling:g}$')


def plot_solution(args, problem, rows, ssf):
    """Density at t1 and pointwise error of one split step and the PINNs."""
    x = eval_points()[::len(eval_points()) // N_PLOT]
    ref = problem.reference(x)[0]
    one_step = problem.evaluate(problem.split_step(args.n, 1), x)
    initial = problem.psi0(x)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=b3.FIGSIZE, sharex=True, layout='constrained')
    ax1.plot(x, abs(initial)**2, color='#888888', ls=':', lw=2.5, label='initial state')
    curves = [('Strang, 1 step', one_step, b3.STYLE['strang']['color'])]
    curves += [(f'PINN: {label(c)}', np.array(r['psi_re']) + 1j * np.array(r['psi_im']), color(c))
               for c, r in rows.items()]
    for text, psi, col in curves:
        b3.glow(x, abs(psi)**2, ax=ax1, lw=2.5, color=col, label=text)
        b3.glow(x, np.maximum(abs(psi - ref), 1e-16), ax=ax2, lw=2.5, color=col)
    ax1.plot(x, abs(ref)**2, color='white', ls='--', lw=2.5, zorder=5, label='reference')
    ax1.set_ylabel('Density $|\\psi|^2$ at $t_1$')
    b3.panel_label(ax1, title(problem, args), loc='upper left')
    ax2.set_yscale('log')
    ax2.set_ylabel('$|\\psi - \\psi_\\mathrm{ref}|$ at $t_1$')
    ax2.set_xlabel('$x$')
    b3.panel_label(ax2, 'Error of one step', loc='upper left')
    ax2.set_ylim(1e-6, 1e1)
    fig.legend(*ax1.get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=True)
    b3.save('benchmark_8_sp_solution', fig, tight=False)
    plt.close(fig)


def plot_cost(args, problem, rows, ssf, irk):
    """Error of the PINNs against split-step Fourier with a growing number of substeps, by steps and by wall-clock."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=b3.FIGSIZE, layout='constrained')
    steps = list(ssf)
    style = dict(lw=2.5, markersize=9)
    ssf_color = b3.STYLE['strang']['color']
    b3.glow(steps, [ssf[s]['l1'] for s in steps], ax=ax1, color=ssf_color, marker='o', label='Split-step Fourier (Strang)',
            **style)
    b3.glow([ssf[s]['wall'] for s in steps], [ssf[s]['l1'] for s in steps], ax=ax2, color=ssf_color, marker='o', **style)
    for s in steps[::2]:
        ax2.annotate(f'{s}', (ssf[s]['wall'], ssf[s]['l1']), textcoords='offset points', xytext=(0, 12), ha='center',
                     fontsize=14, color=ssf_color)
    if irk:
        ax1.axhline(irk['l1'], color=b3.STYLE['irk_exact']['color'], ls='--', lw=2,
                    label=f'Gauss IRK, $q = {irk["q"]}$, exact solve')
        ax2.scatter(irk['wall'], irk['l1'], s=250, marker='o', color=b3.STYLE['irk_exact']['color'], zorder=3)
    for c, r in rows.items():
        ax1.axhline(r['l1'], color=color(c), lw=2.5, label=f'PINN: {label(c)}')
        ax2.scatter(r['wall'], r['l1'], s=300, marker='X', color=color(c), zorder=3)
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Substeps of the split-step solver')
    ax2.set_xscale('log')
    ax2.set_xlabel('Wall-clock time [s] (training for the PINNs)')
    errors = [r['l1'] for r in ssf.values()] + [r['l1'] for r in rows.values()] + ([irk['l1']] if irk else [])
    for ax in (ax1, ax2):
        ax.set_yscale('log')
        ax.set_ylim(min(errors) / 10, max(errors) * 300)   # headroom for the panel label
        ax.set_ylabel('$L_1$ error at $t_1$')
    b3.panel_label(ax1, title(problem, args), loc='upper right')
    b3.panel_label(ax2, f'Same grid, $N = {args.n}$; numbers: substeps', loc='upper right')
    fig.legend(*ax1.get_legend_handles_labels(), loc='outside lower center', ncol=1, frameon=True)
    b3.save('benchmark_9_sp_error_vs_cost', fig, tight=False)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modes', type=int, default=3, help='number of random initial modes')
    p.add_argument('--k-max', type=int, default=4, help='nonzero modes are drawn from -k_max ... k_max')
    p.add_argument('--seed', type=int, default=0, help='seed of the random modes and amplitudes')
    p.add_argument('--coupling', type=float, default=1000.0, help='kappa of V_xx = kappa (|psi|^2 - 1)')
    p.add_argument('--wdt', type=float, default=4.0, help='step size omega dt, omega = 1/2 (2 pi k_max)^2')
    p.add_argument('--n', type=int, default=256, help='collocation points of the PINN and grid of the split-step solver')
    p.add_argument('--depth', type=int, default=4, help='hidden layers')
    p.add_argument('--width', type=int, default=50, help='neurons per layer')
    p.add_argument('--adam-steps', type=int, default=2000)
    p.add_argument('--lbfgs-steps', type=int, default=1500)
    p.add_argument('--variants', nargs='+', default=['baseline', 'ff', 'per'],
                   help="PINN variants, '+' combines: baseline, ff, aw, rwf, per")
    p.add_argument('--ssf-steps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help='substeps of the split-step solver')
    p.add_argument('--quick', action='store_true', help='smoke test with few steps and a small grid')
    p.add_argument('--plot-only', action='store_true', help='only replot cached results')
    args = p.parse_args()

    if args.quick:
        args.adam_steps, args.lbfgs_steps, args.n, args.wdt = 200, 50, 64, 1.0
    suffix = '_quick' if args.quick else ''

    problem = SchroedingerPoisson(args.modes, args.k_max, args.coupling, args.wdt, args.seed)
    cache = b3.Cache(f'results/benchmark_sp{suffix}.json')
    describe(problem, args)
    if not args.plot_only:
        run_benchmark(args, problem, cache)

    ssf = ssf_results(problem, args.n, args.ssf_steps)
    for s, r in ssf.items():
        print(f'Strang, {s:4d} substeps: L1 = {r["l1"]:.3e}  time = {r["wall"] * 1e3:.2f} ms')
    irk = (b3.select(cache, 'irk_exact', problem=problem.signature) or [None])[0]
    rows = pinn_rows(args, problem, cache)
    plot_solution(args, problem, rows, ssf)
    plot_cost(args, problem, rows, ssf, irk)


if __name__ == '__main__':
    main()
