"""Improving the discrete-time PINN for Schroedinger-Poisson: optimiser, loss form and time integrator.

Script 6 found that the discrete-time PINN stalls at a final loss of about 1e-5 and an error of 1e-2 to 1e-3, while
the same Gauss-Legendre equations solved exactly give 1e-9: the PINN is limited by training, not by the scheme. This
script tests, one at a time and on the problem of script 6 (same arguments, same cache signature), what removes that
limit. Experiments (`--experiments`):

  regression  Can the network represent the solution? The exact IRK stages (Newton-Krylov solution) are fitted by
              plain regression of the stages and their second derivatives, and psi^{n+1} is formed from the fit with
              the same formula as in the PINN. Separates representation error from optimisation error.
  lm          Levenberg-Marquardt (damped Gauss-Newton with a dense Jacobian) instead of L-BFGS after the Adam phase,
              on the loss of script 6 (variant `lm`; Gauss-Newton for PINNs: Mueller & Zeinhofer 2023, Jnini et al.
              2024, Wang et al. 2025).
  kform       Stage-derivative form (variant `k`): the network outputs K_i = N[psi_i] / omega and the stages are
              psi_i = psi_0 + dt omega sum_j a_ij K_j. The initial data is then exact, the residual is K_i - N[psi_i]
              / omega, and psi^{n+1} = psi_0 + dt omega sum_i b_i K_i needs no derivative of the network.
  ip          Interaction picture (variant `ip`, Lawson-type exponential IRK): the free propagation exp(i/2 dt
              d_xx) is applied exactly in Fourier space and the network represents phi_i = exp(-i/2 c_i dt d_xx)
              psi_i. The needed stages q scale with dt V instead of omega dt. All derivatives are spectral, so the loss
              is evaluated on the collocation grid only (a spectral collocation method with a network as the
              parametrisation, not a PINN in the strict sense).

Variants are named by '+'-joined tokens: k | ip (form, default: the stages of script 6), ff | per | rwf | aw
(as in script 6), lm (optimiser), e.g. `k+lm`, `ip+lm`. Results are cached in results/improvements_sp.json.

Usage
-----
    python 7_improve_schroedinger_poisson.py --experiments regression
    python 7_improve_schroedinger_poisson.py --experiments lm kform ip
    python 7_improve_schroedinger_poisson.py --plot-only
"""

import argparse
import importlib.util
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
from torch.func import functional_call, jacrev

spec = importlib.util.spec_from_file_location(
    'sp', os.path.join(os.path.dirname(os.path.abspath(__file__)), '6_benchmark_schroedinger_poisson.py'))
b6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b6)
b3 = b6.b3

N_PLOT = b6.N_PLOT


def parse(variant):
    """Tokens of a variant name -> (form, optimiser, strategy kwargs of script 6)."""
    tokens = [] if variant == 'baseline' else variant.split('+')
    form = 'ip' if 'ip' in tokens else 'k' if 'k' in tokens else 'stage'
    strategies = tuple(t for t in tokens if t in b6.STRATEGIES)
    return form, 'lm' if 'lm' in tokens else 'lbfgs', strategies


def midpoint_interpolation(psi_grid, x, derivative=0, offset=0.5):
    """Fourier interpolation (and derivative) of values on the grid (j + offset) / n to the points x."""
    n = len(psi_grid)
    m = np.fft.fftfreq(n, 1.0 / n)
    c = np.fft.fft(psi_grid) / n * np.exp(-2j * np.pi * m * offset / n) * (2j * np.pi * m)**derivative
    return np.exp(2j * np.pi * np.outer(x, m)) @ c


# ----------------------------------------------------------------------------------------------
# Discretisations: network plus the residual of one of the forms
# ----------------------------------------------------------------------------------------------

class Discretisation(torch.nn.Module):
    """Network plus the residual of one of the forms; forward() returns the residual vector for functional_call.

    The residual vector r is built from blocks such that sum(r**2) is the sum of the mean-squared blocks, i.e. the
    unweighted loss of script 6, so that Adam/L-BFGS (terms()) and Levenberg-Marquardt (forward()) minimise the same.
    """

    def __init__(self, problem, tableau, n, width, depth, form='stage', fourier=None, rwf=False, periodic=False,
                 seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.problem, self.form, self.periodic = problem, form, periodic
        A, b, c = tableau
        self.q, self.dt = len(b), problem.t1 - problem.t0
        self.omega = problem.omega
        self.s = self.dt * self.omega                      # non-dimensional step omega dt
        self.A, self.b = torch.tensor(A), torch.tensor(b)
        self.k_max = float(abs(problem.modes).max())
        col = lambda a: torch.tensor(a).reshape(-1, 1)
        self.x, self.x_bnd = col(b3.midpoints(0.0, 1.0, n)), col([0.0, 1.0])
        self.p0 = self.initial(self.x)
        self.model = b3.MLP(1, 2 * self.q, width, depth, [0.0], [1.0], fourier_scales=(1,) if periodic else fourier,
                            rwf=rwf)
        if periodic:
            self.model.freqs = torch.arange(1.0, b3.FOURIER_PER_SCALE + 1).reshape(-1, 1)   # cycles per domain
        if form == 'ip':
            k = torch.tensor(2 * np.pi * np.fft.fftfreq(n, 1.0 / n))[:, None]
            self.phase = torch.exp(-0.5j * k**2 * torch.tensor(np.ravel(c))[None, :] * self.dt)   # exp(L c_j dt)
            self.phase1 = torch.exp(-0.5j * k**2 * self.dt)
            self.A_c, self.b_c = self.A.to(torch.complex128), self.b.to(torch.complex128)

    def initial(self, x):
        """Initial state and its second derivative at the points x (n, 1): Re, Im, Re_xx, Im_xx, each (n, 1)."""
        pr = self.problem
        e = np.exp(2j * np.pi * np.outer(x.numpy().ravel(), pr.modes))
        psi, psi_xx = e @ pr.amps, e @ (-(2 * np.pi * pr.modes)**2 * pr.amps)
        return tuple(torch.tensor(a)[:, None] for a in (psi.real, psi.imag, psi_xx.real, psi_xx.imag))

    def rhs(self, re, im, xx_re, xx_im):
        """N[psi] = i/2 psi_xx - i V psi, V from the Poisson solve of |psi|^2 along the grid axis."""
        V = self.problem.potential_torch(re**2 + im**2)
        return -0.5 * xx_im + V * im, 0.5 * xx_re - V * re

    def fields(self, x, p0):
        """'stage' and 'k' forms: network output and stages, with N[stages], at the points x."""
        q = self.q
        out, _, out_xx = self.model.forward_with_derivatives(x)
        if self.form == 'stage':
            re, im, xx_re, xx_im = out[:, :q], out[:, q:], out_xx[:, :q], out_xx[:, q:]
        else:
            lin = lambda y, base: base + self.s * y @ self.A.T
            re, im = lin(out[:, :q], p0[0]), lin(out[:, q:], p0[1])
            xx_re, xx_im = lin(out_xx[:, :q], p0[2]), lin(out_xx[:, q:], p0[3])
        return out, re, im, self.rhs(re, im, xx_re, xx_im)

    def ip_stages(self, out, phi0):
        """Interaction picture: stages psi_j and G_j = exp(-L c_j dt) F[psi_j], F = -i V psi, from the phi_j."""
        q = self.q
        phi = out[:, :q] + 1j * out[:, q:]
        psi = torch.fft.ifft(self.phase * torch.fft.fft(phi, dim=0), dim=0)
        V = self.problem.potential_torch(psi.real**2 + psi.imag**2)
        return phi, torch.fft.ifft(self.phase.conj() * torch.fft.fft(-1j * V * psi, dim=0), dim=0)

    def blocks(self):
        q, dt, A, p0 = self.q, self.dt, self.A, self.p0
        if self.form == 'ip':
            phi, G = self.ip_stages(self.model(self.x), p0[0] + 1j * p0[1])
            res = phi - (p0[0] + 1j * p0[1]) - dt * G @ self.A_c.T
            return [torch.cat([res.real, res.imag], dim=1)]
        out, re, im, (N_re, N_im) = self.fields(self.x, p0)
        if self.form == 'stage':   # undo the IRK step to reconstruct psi^n from every stage
            res = torch.cat([re - dt * N_re @ A.T - p0[0], im - dt * N_im @ A.T - p0[1]], dim=1)
        else:
            res = torch.cat([out[:, :q] - N_re / self.omega, out[:, q:] - N_im / self.omega], dim=1)
        blocks = [res]
        if not self.periodic:      # periodicity of the network output and its derivative
            o, ox, _ = self.model.forward_with_derivatives(self.x_bnd)
            blocks.append(2**0.5 * torch.cat([o[0] - o[1], (ox[0] - ox[1]) / (2 * np.pi * self.k_max)]))
        return blocks

    def terms(self):
        return torch.stack([torch.mean(b**2) for b in self.blocks()])

    def forward(self):
        return torch.cat([b.flatten() / b.numel()**0.5 for b in self.blocks()])

    @torch.no_grad()
    def predict(self, x_eval):
        """psi^{n+1} at the points x_eval (complex numpy); the 'ip' form is evaluated on the grid and interpolated."""
        q, dt, p0 = self.q, self.dt, self.p0
        if self.form == 'ip':
            phi0 = p0[0] + 1j * p0[1]
            _, G = self.ip_stages(self.model(self.x), phi0)
            psi1 = torch.fft.ifft(self.phase1[:, 0] * torch.fft.fft(phi0[:, 0] + dt * G @ self.b_c))
            return midpoint_interpolation(psi1.numpy(), x_eval)
        x = torch.tensor(x_eval).reshape(-1, 1)
        p0 = self.initial(x)
        if self.form == 'k':
            K = self.model(x)
            return ((p0[0][:, 0] + self.s * K[:, :q] @ self.b) + 1j * (p0[1][:, 0] + self.s * K[:, q:] @ self.b)).numpy()
        _, _, _, (N_re, N_im) = self.fields(x, p0)
        return ((p0[0][:, 0] + dt * N_re @ self.b) + 1j * (p0[1][:, 0] + dt * N_im @ self.b)).numpy()


# ----------------------------------------------------------------------------------------------
# Exact solution of the interaction-picture IRK equations (to choose q)
# ----------------------------------------------------------------------------------------------

def solve_irk_ip(problem, tableau, n=128):
    """Interaction-picture IRK step, stage equations solved by Newton-Krylov (no stiff linear part is left, so no
    preconditioner is needed). Returns psi(t1) on the grid j / n."""
    A, b, c = tableau
    q, dt = len(b), problem.t1 - problem.t0
    k = 2 * np.pi * np.fft.fftfreq(n, 1.0 / n)
    phase = np.exp(-0.5j * k[None, :]**2 * np.ravel(c)[:, None] * dt)         # (q, n)
    phi0 = problem.psi0(np.arange(n) / n)

    def G(phi):
        psi = np.fft.ifft(phase * np.fft.fft(phi, axis=-1), axis=-1)
        return np.fft.ifft(phase.conj() * np.fft.fft(-1j * problem.potential(psi) * psi, axis=-1), axis=-1)

    def residual(z):
        phi = (z[:q * n] + 1j * z[q * n:]).reshape(q, n)
        r = phi - phi0 - dt * A @ G(phi)
        return np.concatenate([r.real.ravel(), r.imag.ravel()])

    z0 = np.tile(phi0, (q, 1))
    z = scipy.optimize.newton_krylov(residual, np.concatenate([z0.real.ravel(), z0.imag.ravel()]), f_tol=1e-12)
    phi = (z[:q * n] + 1j * z[q * n:]).reshape(q, n)
    return np.fft.ifft(np.exp(-0.5j * k**2 * dt) * np.fft.fft(phi0 + dt * b @ G(phi)))


def required_q_ip(problem, tol=1e-8):
    """Smallest Gauss-Legendre scheme for which the interaction-picture step is within tol (RMS) of the reference."""
    x = b6.eval_points()
    ref = problem.reference(x)[0]
    for q in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]:
        tableau = b3.gauss_legendre_tableau(q)
        err = np.sqrt(np.mean(abs(problem.evaluate(solve_irk_ip(problem, tableau), x) - ref)**2))
        print(f'    IP IRK q = {q}: rms difference to reference {err:.2e}', flush=True)
        if err < tol:
            break
    return q


# ----------------------------------------------------------------------------------------------
# Training: Levenberg-Marquardt
# ----------------------------------------------------------------------------------------------

def train_lm(disc, steps, log_prefix='', chunk=256):
    """Levenberg-Marquardt on the residual vector of disc, dense Jacobian by reverse mode.

    The step solves (J J^T + mu I) y = r, delta = -J^T y (equivalent to the damped normal equations for fewer residuals
    than parameters, otherwise those are solved). mu is lowered after a successful step and raised after a failed one.
    """
    names = [k for k, _ in disc.named_parameters()]
    shapes = [p.shape for p in disc.parameters()]
    sizes = [p.numel() for p in disc.parameters()]
    theta = torch.cat([p.detach().flatten() for p in disc.parameters()])

    def residual(flat):
        parts = [t.reshape(s) for t, s in zip(torch.split(flat, sizes), shapes)]
        return functional_call(disc, dict(zip(names, parts)), ())

    jac = jacrev(residual, chunk_size=chunk)
    r = residual(theta).detach()
    loss, mu = (r**2).sum().item(), None
    for i in range(steps):
        J = jac(theta).detach()
        dual = J.shape[0] <= J.shape[1]
        gram = J @ J.T if dual else J.T @ J
        mu = 1e-3 * gram.diagonal().mean().item() if mu is None else mu
        for _ in range(12):                    # raise mu until the loss decreases
            damped = gram + mu * torch.eye(len(gram))
            step = -J.T @ torch.linalg.solve(damped, r) if dual else -torch.linalg.solve(damped, J.T @ r)
            r_new = residual(theta + step).detach()
            loss_new = (r_new**2).sum().item()
            if loss_new < loss:
                theta, r, loss, mu = theta + step, r_new, loss_new, mu / 3
                break
            mu *= 4
        else:
            print(f'{log_prefix} LM stalled at step {i}', flush=True)
            break
        if i % 10 == 0:
            print(f'{log_prefix} LM    {i:6d}: loss = {loss:.3e}  mu = {mu:.1e}', flush=True)
    with torch.no_grad():
        for p, t in zip(disc.parameters(), torch.split(theta, sizes)):
            p.copy_(t.reshape(p.shape))
        final = disc.terms().sum().item()
    print(f'{log_prefix} final loss = {final:.3e}', flush=True)
    return final


# ----------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------

NETWORK_KEYS = ('fourier', 'rwf', 'periodic')


def build(problem, tableau, args, form, kw):
    return Discretisation(problem, tableau, args.n, args.width, args.depth, form,
                          **{k: v for k, v in kw.items() if k in NETWORK_KEYS})


def finish(disc, problem, loss, wall):
    x_eval = b6.eval_points()
    psi = disc.predict(x_eval)
    return (b3.l1_error(psi, problem.reference(x_eval)[0]), loss, wall, disc.model.n_params(),
            psi[::len(x_eval) // N_PLOT])


def run_variant(problem, tableau, args, variant):
    form, optimiser, strategies = parse(variant)
    kw = b6.kwargs(strategies)
    disc = build(problem, tableau, args, form, kw)
    prefix = f'[{variant} N={args.n} q={len(tableau[1])}]'
    start = time.perf_counter()
    loss = b3.train(disc, disc.terms, args.adam_steps, args.lbfgs_steps if optimiser == 'lbfgs' else 0,
                    adaptive=kw.get('adaptive'), log_prefix=prefix)
    if optimiser == 'lm':
        loss = train_lm(disc, args.lm_steps, prefix)
    return finish(disc, problem, loss, time.perf_counter() - start)


def run_regression(problem, tableau, args, variant):
    """Fit the exact stages (values and second derivatives) by regression; error of psi^{n+1} formed from the fit."""
    _, _, strategies = parse(variant)
    kw = b6.kwargs(strategies)
    disc = build(problem, tableau, args, 'stage', kw)
    _, stages = b6.solve_irk(problem, tableau, return_stages=True)
    x = disc.x.numpy().ravel()
    target = lambda d: torch.tensor(np.stack([midpoint_interpolation(st, x, d, offset=0.0) for st in stages], 1))
    t_val, t_xx = torch.cat([target(0).real, target(0).imag], 1), torch.cat([target(2).real, target(2).imag], 1)

    def terms():
        out, _, out_xx = disc.model.forward_with_derivatives(disc.x)
        return torch.stack([torch.mean((out - t_val)**2), torch.mean((0.5 * disc.dt * (out_xx - t_xx))**2)])

    start = time.perf_counter()
    loss = b3.train(disc, terms, args.adam_steps, args.lbfgs_steps, adaptive=kw.get('adaptive'),
                    log_prefix=f'[regression {variant} N={args.n} q={disc.q}]')
    return finish(disc, problem, loss, time.perf_counter() - start)


def cached(cache, key, run, **meta):
    if cache.get(key) is None:
        err, loss, wall, n_params, psi = run()
        cache.put(key, dict(l1=err, loss=loss, wall=wall, n_params=n_params, psi_re=psi.real.tolist(),
                            psi_im=psi.imag.tolist(), **meta))
    r = cache.get(key)
    print(f'{key:120s} L1 = {r["l1"]:.3e}  loss = {r["loss"]:.1e}  time = {r["wall"]:.0f} s', flush=True)


def run_experiments(args, problem, cache, q):
    sig = problem.signature
    for experiment in args.experiments:
        for variant in args.variants_of[experiment]:
            form, _, _ = parse(variant)
            tableau = b3.gauss_legendre_tableau(q['ip' if form == 'ip' else 'std'])
            method = f'{experiment}:{variant}'
            key = (f'{method}/{sig}/q={len(tableau[1])}/N={args.n}/width={args.width}/depth={args.depth}'
                   f'/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}/lm={args.lm_steps}')
            run = run_regression if experiment == 'regression' else run_variant
            cached(cache, key, lambda: run(problem, tableau, args, variant), method=method, problem=sig,
                   q=len(tableau[1]), N=args.n, width=args.width, depth=args.depth, adam_steps=args.adam_steps,
                   lbfgs_steps=args.lbfgs_steps, lm_steps=args.lm_steps)


# ----------------------------------------------------------------------------------------------

GROUPS = {   # (form, optimiser) -> label, colour
    ('stage', 'lbfgs'): ('stages (script 6), Adam + L-BFGS', '#FFA500'),
    ('stage', 'lm'):    ('stages (script 6), Adam + LM',      '#FE53BB'),
    ('k', 'lbfgs'):     ('stage derivatives K, Adam + L-BFGS', '#B967FF'),
    ('k', 'lm'):        ('stage derivatives K, Adam + LM',    '#08F7FE'),
    ('ip', 'lbfgs'):    ('interaction picture, Adam + L-BFGS', '#F5D300'),
    ('ip', 'lm'):       ('interaction picture, Adam + LM',    '#00ff41'),
}
NAMES = dict(per='periodic', ff='FF', rwf='RWF', aw='AW', lm='', k='', ip='')


def short(variant):
    return ' + '.join(NAMES[t] for t in variant.split('+') if NAMES.get(t)) or 'baseline'


def improvement_rows(args, problem, cache):
    """Rows of the experiments at the requested N and LM budget, in the order of GROUPS: [(form, optimiser, variant, row)]."""
    rows = []
    for r in cache.data.values():
        if r.get('problem') != problem.signature or r.get('N') != args.n or r['method'].startswith(('regression', 'irk')):
            continue
        if (r['lm_steps'], r['adam_steps'], r['lbfgs_steps']) != (args.lm_steps, args.adam_steps, args.lbfgs_steps):
            continue
        variant = r['method'].split(':')[1]
        form, optimiser, _ = parse(variant)
        rows.append((form, optimiser, variant, r))
    order = list(GROUPS)
    return sorted(rows, key=lambda t: (order.index(t[:2]), t[2].count('+'), t[2]))


def plot_improvements(args, problem, cache, ssf):
    rows = improvement_rows(args, problem, cache)
    if not rows:
        return
    irk = (b3.select(b3.Cache('results/benchmark_sp.json'), 'irk_exact', problem=problem.signature) or [None])[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=b3.FIGSIZE, layout='constrained')

    ys = {i: len(rows) - 1 - i for i in range(len(rows))}
    for i, (form, optimiser, variant, r) in enumerate(rows):
        col = GROUPS[(form, optimiser)][1]
        ax1.scatter(r['l1'], ys[i], s=250, marker='X', color=col, zorder=3, label=GROUPS[(form, optimiser)][0])
        ax1.text(r['l1'] * 1.4, ys[i], f'{r["l1"]:.1e}', va='center', fontsize=14)
        ax2.scatter(r['loss'], r['l1'], s=250, marker='X', color=col, zorder=3)
        ax2.annotate(short(variant), (r['loss'], r['l1']), textcoords='offset points', xytext=(8, 6), fontsize=12, color=col)
    for s in (1, 4, 16, 64):
        ax1.axvline(ssf[s]['l1'], color=b3.STYLE['strang']['color'], ls='--', lw=1.5, alpha=0.8)
        ax1.text(ssf[s]['l1'] * 1.1, 0.97, f'Strang, {s}', rotation=90, va='top', fontsize=12,
                 color=b3.STYLE['strang']['color'], transform=ax1.get_xaxis_transform())
    if irk:
        ax1.axvline(irk['l1'], color=b3.STYLE['irk_exact']['color'], ls='--', lw=1.5, alpha=0.8)
        ax1.text(irk['l1'] * 1.1, 0.97, f'Gauss IRK, $q = {irk["q"]}$', rotation=90, va='top', fontsize=12,
                 color=b3.STYLE['irk_exact']['color'], transform=ax1.get_xaxis_transform())
    ax1.set_xscale('log')
    ax1.set_xlim(1e-10, 20)
    ax1.set_ylim(-0.7, len(rows) + 1.5)
    ax1.set_yticks(list(ys.values()))
    ax1.set_yticklabels([short(v) for _, _, v, _ in rows])
    ax1.grid(axis='y', alpha=0.2)
    ax1.set_xlabel('$L_1$ error at $t_1$')
    b3.panel_label(ax1, f'Single step, $N = {args.n}$', loc='upper left')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Final training loss')
    ax2.set_ylabel('$L_1$ error at $t_1$')
    b3.panel_label(ax2, 'Low loss is not low error', loc='upper left')
    handles = {}
    for h, l in zip(*ax1.get_legend_handles_labels()):
        handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=2, frameon=True)
    b3.save('benchmark_10_sp_improvements', fig, tight=False)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--modes', type=int, default=3)
    p.add_argument('--k-max', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--coupling', type=float, default=1000.0)
    p.add_argument('--wdt', type=float, default=4.0)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--width', type=int, default=50)
    p.add_argument('--adam-steps', type=int, default=2000)
    p.add_argument('--lbfgs-steps', type=int, default=1500)
    p.add_argument('--lm-steps', type=int, default=100, help='Levenberg-Marquardt iterations after the Adam phase')
    p.add_argument('--experiments', nargs='+', default=[], choices=['regression', 'lm', 'kform', 'ip'])
    p.add_argument('--regression-variants', nargs='+', default=['baseline', 'ff', 'per'])
    p.add_argument('--lm-variants', nargs='+', default=['lm', 'ff+lm', 'per+lm'])
    p.add_argument('--k-variants', nargs='+', default=['k', 'k+ff', 'k+per', 'k+lm', 'k+per+lm'])
    p.add_argument('--ip-variants', nargs='+', default=['ip', 'ip+per', 'ip+lm', 'ip+per+lm'])
    p.add_argument('--quick', action='store_true')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args()
    args.variants_of = dict(regression=args.regression_variants, lm=args.lm_variants, kform=args.k_variants,
                            ip=args.ip_variants)
    if args.quick:
        args.adam_steps, args.lbfgs_steps, args.lm_steps, args.n, args.wdt = 100, 20, 5, 64, 1.0
    suffix = '_quick' if args.quick else ''

    problem = b6.SchroedingerPoisson(args.modes, args.k_max, args.coupling, args.wdt, args.seed)
    cache6 = b3.Cache(f'results/benchmark_sp{suffix}.json')
    cache = b3.Cache(f'results/improvements_sp{suffix}.json')
    b6.describe(problem, args)
    if cache6.get(f'irk_exact/{problem.signature}') is None:
        cache6.put(f'irk_exact/{problem.signature}', dict(method='irk_exact', problem=problem.signature,
                                                         **{k: v for k, v in b6.irk_exact(problem).items()}))
    q = dict(std=cache6.get(f'irk_exact/{problem.signature}')['q'])
    key = f'irk_ip/{problem.signature}'
    if cache.get(key) is None:
        cache.put(key, dict(method='irk_ip', problem=problem.signature, q=required_q_ip(problem)))
    q['ip'] = cache.get(key)['q']
    print(f'q = {q["std"]} (stages), {q["ip"]} (interaction picture)')
    if not args.plot_only:
        run_experiments(args, problem, cache, q)
    plot_improvements(args, problem, cache, b6.ssf_results(problem, args.n, [1, 4, 16, 64]))


if __name__ == '__main__':
    main()
