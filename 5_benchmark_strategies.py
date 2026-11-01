"""Strategy benchmark: do scalable improvements fix the failures of the discrete-time PINN?

Script 3 found three problems of the discrete-time PINN (one Gauss-Legendre IRK step, as in
2_linear_schroedinger_1d_discrete_time.ipynb):
  1. for 16 wavelengths it fails at every resolution (spectral bias),
  2. for 2 wavelengths its error stagnates at about 5e-3, independent of N (training plateau),
  3. in the step-size benchmark it fails for large steps omega dt (the training loss grows with the step).

This script runs the same two benchmarks (same problems, same N and omega dt, same error metric, same 4x50 network,
2000 Adam + 1500 L-BFGS steps, one seed) for two changes of the discrete-time PINN that need neither a uniform grid nor
the solution, so that they could carry over to a 3D cosmological box:
  per  periodic features [sin(2 pi m x), cos(2 pi m x)], m = 1 ... 16, as input of the network. The network is exactly
       periodic with integer modes, which replaces the boundary term (a periodicity loss without them). The features
       do not depend on N or the wave number, but contain the wave numbers of the problems (2 and 16, the latter the
       highest feature), so for these the result tests the optimisation, not the discovery of the frequency. m up to
       32 does not train (2 waves, N = 256: loss of the regression 5e-2 instead of 9e-5 for m up to 16): the second
       derivative of a feature grows as m^2.
  k    stage-derivative form: the network outputs K_i = N[psi_i] / omega and the stages are
       psi_i = psi_0 + dt omega sum_j a_ij K_j. The initial data is then exact, the residual is K_i - N[psi_i] / omega,
       and psi^{n+1} = psi_0 + dt omega sum_i b_i K_i needs no derivative of the network.
and their combination. The baseline is the network of script 3 with psi^{n+1} computed from the stages (the notebook
version has it as an output; the README shows that this makes no difference) and a periodicity loss for psi and psi_x at
x = 0, 1 instead of exact boundary data, which a simulation would not have. The variant with exact boundary data shows
what that costs ('data'); the notebook version of script 3 is plotted as a grey reference.

Regression diagnostic. For every network (stage form, K form, with and without periodic features) the exact IRK
stages (values and second derivatives, K and its second derivative for the K form) are fitted by plain regression with
the same network, optimiser and budget, and psi^{n+1} is formed from the fit as in the PINN. The error of the fit is
the representation floor of that network (dashed lines); the difference to the PINN is the optimisation error.

Not included, since they would not scale: Levenberg-Marquardt (dense Jacobian with 2 q N^d rows and one column per
parameter, full batch; 140 GB at q = 256, N = 1024), and the interaction picture (a spectral collocation loss on a
uniform grid, where a PINN is not needed). Random Fourier features, adaptive loss weights and random weight factorisation
of earlier versions of this script (git history) are not included either: only features that match the solution helped.

Everything else is as in 3_benchmark_resolution.py, which this script builds on: q = 256 for the resolution
benchmark and q adapted to the step for the step-size benchmark, there at N = 256 instead of 64, since periodic features
up to m = 16 are only 4 points per period on a grid with N = 64 (the network can then satisfy the equations between the
collocation points in many ways). Caches are shared with script 3; the key of the baseline with exact data is the one
of the old runs of this script (width is not part of the key of the step-size benchmark, as in scripts 3 and 5). Error
metric as there.

Usage
-----
    python 5_benchmark_strategies.py                          # all benchmarks, resumes from cache
    python 5_benchmark_strategies.py --waves 16 --benchmarks resolution   # one cache file per process, so
    python 5_benchmark_strategies.py --waves 2 --benchmarks resolution    # these can run in parallel
    python 5_benchmark_strategies.py --benchmarks step
    python 5_benchmark_strategies.py --seeds 5                # add seeds 1-4 to the ablation (default: 1)
    python 5_benchmark_strategies.py --quick                  # smoke test
    python 5_benchmark_strategies.py --plot-only              # replot from cached results

Figures are written to figures/benchmark_4_* ... benchmark_6_*.png.
"""

import argparse
import importlib.util
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

# 3_benchmark_resolution.py starts with a digit and can therefore not be imported by name
spec = importlib.util.spec_from_file_location(
    'benchmark', os.path.join(os.path.dirname(os.path.abspath(__file__)), '3_benchmark_resolution.py'))
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)

PERIODIC_MODES = 16    # integer modes m = 1 ... 16 of the periodic features (a multiple of b3.FOURIER_PER_SCALE)


# ----------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------

VARIANTS = {   # in the order of the figures; 'data' is run by script 3
    'data':  dict(label='baseline, exact boundary data', color='#FFA500'),
    'stage': dict(label='baseline', form='stage', periodic=False, color='#FE53BB'),
    'per':   dict(label='periodic features', form='stage', periodic=True, color='#08F7FE'),
    'k':     dict(label='K form', form='k', periodic=False, color='#B967FF'),
    'k_per': dict(label='K form + periodic features', form='k', periodic=True, color='#00ff41'),
}
FLOORS = ['stage', 'per', 'k', 'k_per']                                      # networks of the regression
FLOOR_OF = dict(data='stage', stage='stage', per='per', k='k', k_per='k_per')  # 'data' has the network of 'stage'
REFERENCE = dict(label='baseline of script 3 (notebook form, exact boundary data)', color='#888888')
RUNS = [('reg', v) for v in FLOORS] + [('pinn', v) for v in VARIANTS]        # regression first


def method(kind, variant):
    """Method name in the caches; the baseline with exact data keeps the name of the old runs."""
    return 'pinn_disc_q' if (kind, variant) == ('pinn', 'data') else f'{kind}_{variant}'


# ----------------------------------------------------------------------------------------------
# Discrete-time PINN with the stage or K form, and the regression of the exact stages
# ----------------------------------------------------------------------------------------------

def exact_stages(problem, tableau, x):
    """Exact solution of the (linear) stage equations of one IRK step as functions of x.

    Returns the stages psi_i and K_i = N[psi_i] / omega with their second derivatives, complex, (len(x), q) each:
    psi, psi_xx, K, K_xx. In the eigenbasis of H the stages decouple, psi_i = psi_0 [(I + i E dt A)^-1 1]_i."""
    A, b, _ = tableau
    q, dt, omega = len(b), problem.t1 - problem.t0, problem.omega
    if hasattr(problem, 'E'):      # plane wave in a potential: Fourier coefficients from the eigen-decomposition
        ctilde = np.stack([problem.a0[j] * np.linalg.solve(np.eye(q) + 1j * E * dt * A, np.ones(q))
                           for j, E in enumerate(problem.E)])                      # (modes, q) in the eigenbasis
        coeff = problem.U @ ctilde
        k_coeff = -1j / omega * problem.U @ (problem.E[:, None] * ctilde)          # N[psi] = -i H psi
        wave = 2 * np.pi * problem.m
    else:                          # plane wave: a single eigenmode with H = omega
        coeff = np.linalg.solve(np.eye(q) + 1j * omega * dt * A, np.ones(q))[None, :]
        k_coeff = -1j * coeff
        wave = np.array([problem.k])
    basis = np.exp(1j * np.outer(x, wave))
    d2 = -wave[:, None]**2
    return basis @ coeff, basis @ (d2 * coeff), basis @ k_coeff, basis @ (d2 * k_coeff)


def run_discrete(problem, n, width, depth, adam_steps, lbfgs_steps, tableau, form='stage', periodic=False, seed=0,
                 regression=False):
    """Discrete-time PINN with one q-stage Gauss-Legendre IRK step, psi^{n+1} computed from the stages.

    form 'stage': the network outputs the stages, the loss reconstructs psi^n from every stage (as in script 3).
    form 'k': the network outputs K_i = N[psi_i] / omega, the stages are psi_0 + dt omega A K, the loss is K - N[psi]/omega.
    periodic: periodic features, no boundary term; otherwise a periodicity loss for the network output and its derivative.
    regression: fit the exact stages (values and second derivatives, K and K_xx for the form 'k') instead.
    Both problems start from psi_0 = exp(i k x).
    """
    torch.manual_seed(seed)
    col = lambda a: torch.tensor(a).reshape(-1, 1)
    A, b, _ = tableau
    q, dt, omega = len(b), problem.t1 - problem.t0, problem.omega
    s = dt * omega                       # non-dimensional step
    A_t, b_t = torch.tensor(A), torch.tensor(b)

    def initial(x):
        """psi_0 and its second derivative at x: Re, Im, Re_xx, Im_xx, each (len(x), 1)."""
        cos, sin = torch.cos(problem.k * x), torch.sin(problem.k * x)
        return cos, sin, -problem.k**2 * cos, -problem.k**2 * sin

    x_col, x_bnd = col(b3.midpoints(problem.xl, problem.xr, n)), col([problem.xl, problem.xr])
    p0 = initial(x_col)
    scales = (1,) * (PERIODIC_MODES // b3.FOURIER_PER_SCALE) if periodic else None
    model = b3.MLP(1, 2 * q, width, depth, [problem.xl], [problem.xr], fourier_scales=scales)
    if periodic:
        model.freqs = torch.arange(1.0, PERIODIC_MODES + 1).reshape(-1, 1)    # cycles per domain

    def stages(x, p0):
        """Network output and its derivatives, the stages and N[stages] = (N_re, N_im) at the points x."""
        out, out_x, out_xx = model.forward_with_derivatives(x)
        if form == 'stage':
            re, im, xx_re, xx_im = out[:, :q], out[:, q:], out_xx[:, :q], out_xx[:, q:]
        else:
            lin = lambda y, base: base + s * y @ A_t.T
            re, im = lin(out[:, :q], p0[0]), lin(out[:, q:], p0[1])
            xx_re, xx_im = lin(out_xx[:, :q], p0[2]), lin(out_xx[:, q:], p0[3])
        V = problem.potential_torch(x)
        return out, out_x, out_xx, (-0.5 * xx_im + V * im, 0.5 * xx_re - V * re)

    if regression:
        psi, psi_xx, K, K_xx = exact_stages(problem, tableau, x_col.numpy().ravel())
        val, xx = (psi, psi_xx) if form == 'stage' else (K, K_xx)
        t_val, t_xx = [torch.tensor(np.concatenate([a.real, a.imag], 1)) for a in (val, xx)]

        def terms_fn():
            out, _, out_xx, _ = stages(x_col, p0)
            return torch.stack([torch.mean((out - t_val)**2), torch.mean((0.5 * dt * (out_xx - t_xx))**2)])
    else:
        def terms_fn():
            out, _, _, (N_re, N_im) = stages(x_col, p0)
            if form == 'stage':   # undo the IRK step to reconstruct psi^n from every stage
                re, im = out[:, :q], out[:, q:]
                res = torch.cat([re - dt * N_re @ A_t.T - p0[0], im - dt * N_im @ A_t.T - p0[1]], dim=1)
            else:
                res = torch.cat([out[:, :q] - N_re / omega, out[:, q:] - N_im / omega], dim=1)
            terms = [torch.mean(res**2)]
            if not periodic:
                o, ox, _ = model.forward_with_derivatives(x_bnd)
                terms.append(torch.mean((o[0] - o[1])**2) + torch.mean((ox[0] - ox[1])**2) / problem.k**2)
            return torch.stack(terms)

    start = time.perf_counter()
    loss = b3.train(model, terms_fn, adam_steps, lbfgs_steps,
                    log_prefix=f"[{'reg' if regression else 'pinn'} {form}{'+per' if periodic else ''} N={n} q={q}]")
    wall = time.perf_counter() - start

    x_eval, psi_eval = b3.eval_grid(problem)
    x_eval_t = col(x_eval)
    p0_eval = initial(x_eval_t)
    with torch.no_grad():
        out, _, _, (N_re, N_im) = stages(x_eval_t, p0_eval)
        if form == 'stage':
            re1, im1 = p0_eval[0][:, 0] + dt * N_re @ b_t, p0_eval[1][:, 0] + dt * N_im @ b_t
        else:
            re1, im1 = p0_eval[0][:, 0] + s * out[:, :q] @ b_t, p0_eval[1][:, 0] + s * out[:, q:] @ b_t
    return b3.l1_error(re1.numpy() + 1j * im1.numpy(), psi_eval), loss, wall, model.n_params()


def run_variant(problem, n, args, tableau, kind, variant, seed):
    if (kind, variant) == ('pinn', 'data'):   # script 3 with psi^{n+1} from the stages
        return b3.run_pinn_discrete(problem, n, args.width, args.depth, args.adam_steps, args.lbfgs_steps,
                                    tableau=tableau, final_from_stages=True, seed=seed)
    v = VARIANTS[variant]
    return run_discrete(problem, n, args.width, args.depth, args.adam_steps, args.lbfgs_steps, tableau,
                        v['form'], v['periodic'], seed, regression=kind == 'reg')


# ----------------------------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------------------------

def cached_run(cache, key, run, **meta):
    if cache.get(key) is None:
        err, loss, wall, n_params = run()
        cache.put(key, dict(l1=err, loss=loss, wall=wall, n_params=n_params, **meta))
    r = cache.get(key)
    print(f'{key:96s} L1 = {r["l1"]:.3e}  loss = {r["loss"]:.1e}  time = {r["wall"]:.0f} s', flush=True)


def seed_suffix(seed):
    return f'/seed={seed}' if seed else ''     # seed 0 keeps the keys of the older runs


def run_resolution(args, problem, cache):
    """Every run for N = 2^4 ... 2^max_exp with seed 0, then the other seeds at the fixed N of the ablation."""
    tableau = b3.gauss_legendre_tableau(256)
    jobs = [(kind, v, int(n), 0) for n in 2**np.arange(4, args.max_exp + 1) for kind, v in RUNS]
    jobs += [(kind, v, args.fixed_n, seed) for seed in range(args.seeds) for kind, v in RUNS]
    for kind, v, n, seed in jobs:
        m = method(kind, v)
        key = (f'{m}/N={n}/width={args.width}/depth={args.depth}/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}'
               + seed_suffix(seed))
        cached_run(cache, key, lambda: run_variant(problem, n, args, tableau, kind, v, seed), method=m, N=n,
                   width=args.width, depth=args.depth, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps,
                   seed=seed)


def run_step(args, cache):
    """Single step in the periodic potential, q adapted. Step sizes with seed 0, then the other seeds at fixed_wdt."""
    omega = 0.5 * (2 * np.pi * args.step_waves)**2
    jobs = [(wdt, 0) for wdt in args.step_values] + [(args.fixed_wdt, seed) for seed in range(args.seeds)]
    problems = {}
    for wdt, seed in jobs:
        if wdt not in problems:
            problem = b3.PlaneWaveInPotential(args.step_waves, args.potential, wdt / omega)
            q, tableau = b3.required_q(problem)
            problems[wdt] = problem, q, tableau
            print(f'--- omega dt = {wdt:g}: dt = {problem.t1:.5f}, required q = {q}', flush=True)
            key = f'irk_exact/wdt={wdt:g}'
            if cache.get(key) is None:
                err, wall, q_used = b3.step_errors_classical(problem)['irk_exact']
                cache.put(key, dict(method='irk_exact', wdt=wdt, l1=err, wall=wall, q=q_used))
        problem, q, tableau = problems[wdt]
        for kind, v in RUNS:
            m = method(kind, v)
            key = (f'{m}/wdt={wdt:g}/N={args.step_n}/V0={args.potential:g}/depth={args.depth}'
                   f'/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}' + seed_suffix(seed))
            cached_run(cache, key, lambda: run_variant(problem, args.step_n, args, tableau, kind, v, seed),
                       method=m, wdt=wdt, q=q, N=args.step_n, V0=args.potential, depth=args.depth,
                       adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps, seed=seed)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def resolution_match(args):
    return dict(width=args.width, depth=args.depth, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)


def step_match(args, n=None):
    return dict(N=n or args.step_n, V0=args.potential, depth=args.depth, adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps)


def select(cache, m, seed=0, **match):
    return [r for r in b3.select(cache, m, **match) if r.get('seed', 0) == seed]


def floor_proxy():
    return Line2D([], [], color='white', ls='--', marker='o', markersize=6, lw=2,
                  label='regression floor of the same network (dashed)')


def plot_resolution(args, cases):
    """One panel per number of wavelengths, stacked vertically, like Fig. 10 of Kunkel et al. (2025)."""
    fig, axes = plt.subplots(len(cases), 1, figsize=b3.FIGSIZE, sharex=True, squeeze=False, layout='constrained')
    for ax, (waves, cache) in zip(axes[:, 0], cases):
        b3.panel_label(ax, f'Plane wave, {b3.plural(waves)}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        style = dict(lw=2.5, markersize=9)
        series = {m: b3.select(cache, m) for m in ['fd4', 'fd6', 'fft']}
        for m, rows in series.items():
            if rows:
                b3.glow([r['N'] for r in rows], [max(r['l1'], 1e-16) for r in rows], ax=ax,
                        color=b3.STYLE[m]['color'], marker=b3.STYLE[m]['marker'], label=b3.STYLE[m]['label'], **style)
        rows = [r for r in b3.select(cache, 'pinn_disc', **resolution_match(args)) if r['N'] <= 2**args.max_exp]
        if rows:
            b3.glow([r['N'] for r in rows], [r['l1'] for r in rows], ax=ax, color=REFERENCE['color'], marker='X',
                    label=REFERENCE['label'], **style)
        for v, spec in VARIANTS.items():
            rows = [r for r in select(cache, method('pinn', v), **resolution_match(args)) if r['N'] <= 2**args.max_exp]
            if rows:
                b3.glow([r['N'] for r in rows], [r['l1'] for r in rows], ax=ax, color=spec['color'], marker='X',
                        label=spec['label'], **style)
        for v in FLOORS:
            rows = [r for r in select(cache, method('reg', v), **resolution_match(args)) if r['N'] <= 2**args.max_exp]
            if rows:
                ax.plot([r['N'] for r in rows], [r['l1'] for r in rows], color=VARIANTS[v]['color'], ls='--', lw=2,
                        marker='o', markersize=6)

        # guide lines N^-4 and N^-6, anchored at the finest FD resolution that is not yet at round-off
        for m, p, ls in [('fd4', 4, '--'), ('fd6', 6, ':')]:
            rows = [r for r in series[m] if r['l1'] > 1e-11]
            if rows:
                n0, e0 = rows[-1]['N'], rows[-1]['l1']
                ns = np.array([max(n0 / 16, 16), min(n0 * 4, 2048)], dtype=float)
                ax.plot(ns, e0 * (ns / n0)**(-p), color='grey', ls=ls, lw=1.5, label=f'$N^{{-{p}}}$')
        same = 2 * waves == 2 * PERIODIC_MODES     # both lines at the same N: one line, one label
        if 2 * waves >= 16:     # two points per wavelength
            ax.axvline(2 * waves, color='white', lw=0.8, alpha=0.5)
            ax.text(2 * waves * 1.07, 0.25,
                    'Nyquist: 2 points / wavelength' if same else '2 points / wavelength',
                    rotation=90, va='bottom', color='white', alpha=0.7, fontsize=14,
                    transform=ax.get_xaxis_transform())
        if not same:
            ax.axvline(2 * PERIODIC_MODES, color=VARIANTS['per']['color'], lw=0.8, alpha=0.5)
            ax.text(2 * PERIODIC_MODES * 1.07, 0.25, 'Nyquist of the periodic features', rotation=90, va='bottom',
                    color=VARIANTS['per']['color'], alpha=0.7, fontsize=14, transform=ax.get_xaxis_transform())
        ax.set_ylim(1e-16, 1e4)   # headroom for the panel label
        ax.set_yticks(10.0**np.arange(-16, 5, 4))
        ax.set_xticks(2.0**np.arange(4, 12))
        ax.set_ylabel('$L_1$ error at $t_1$')
    axes[-1, 0].set_xlabel('Points $N$ per dimension')
    handles = {}
    for ax in axes[:, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    handles[floor_proxy().get_label()] = floor_proxy()
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=1, frameon=True)
    b3.save('benchmark_4_strategies_vs_resolution', fig, tight=False)
    plt.close(fig)


def plot_step_size(args, cache):
    """Error and final training loss of one step vs. step size."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=b3.FIGSIZE, sharex=True, layout='constrained')
    style = dict(lw=2.5, markersize=9)
    rows = [r for r in b3.select(cache, 'irk_exact') if r['wdt'] in args.step_values]
    if rows:
        b3.glow([r['wdt'] for r in rows], [max(r['l1'], 1e-16) for r in rows], ax=ax1, color=b3.STYLE['irk_exact']['color'],
                marker='o', label=b3.STYLE['irk_exact']['label'], **style)
    rows = [r for r in select(cache, 'pinn_disc_q', **step_match(args, 64)) if r['wdt'] in args.step_values]
    if rows:    # the baseline of script 3 on its own grid
        wdt = [r['wdt'] for r in rows]
        b3.glow(wdt, [r['l1'] for r in rows], ax=ax1, color=REFERENCE['color'], marker='X',
                label='baseline, exact boundary data, $N = 64$ (script 3)', **style)
        b3.glow(wdt, [r['loss'] for r in rows], ax=ax2, color=REFERENCE['color'], marker='X', **style)
    for v, spec in VARIANTS.items():
        rows = [r for r in select(cache, method('pinn', v), **step_match(args)) if r['wdt'] in args.step_values]
        if rows:
            wdt = [r['wdt'] for r in rows]
            b3.glow(wdt, [r['l1'] for r in rows], ax=ax1, color=spec['color'], marker='X',
                    label=spec['label'] + f', $N = {args.step_n}$', **style)
            b3.glow(wdt, [r['loss'] for r in rows], ax=ax2, color=spec['color'], marker='X', **style)
    for v in FLOORS:
        rows = [r for r in select(cache, method('reg', v), **step_match(args)) if r['wdt'] in args.step_values]
        if rows:
            ax1.plot([r['wdt'] for r in rows], [r['l1'] for r in rows], color=VARIANTS[v]['color'], ls='--', lw=2,
                     marker='o', markersize=6)

    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-16, 1e4)   # headroom for the panel label
    ax1.set_yticks(10.0**np.arange(-16, 5, 4))
    ax1.set_ylabel('$L_1$ error at $t_1 = \\Delta t$')
    b3.panel_label(ax1, f'Single step, {b3.plural(args.step_waves)}, $V = {args.potential:g}\\cos(2\\pi x)$',
                   loc='upper left')
    ax2.set_yscale('log')
    ax2.set_ylabel('Final training loss')
    b3.panel_label(ax2, 'Unweighted sum of the loss terms', loc='upper left')
    ax2.set_xlabel(r'Step size $\omega \Delta t$')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles + [floor_proxy()], labels + [floor_proxy().get_label()], loc='outside lower center', ncol=1,
               frameon=True)
    b3.save('benchmark_5_strategies_vs_step_size', fig, tight=False)
    plt.close(fig)


def plot_ablation(args, cases, step_cache):
    """Error of every variant at fixed N (plane waves) and fixed step size: median and range over the seeds, with the
    median regression floor of the network."""
    panels = [(f'Plane wave, {b3.plural(waves)}, $N = {args.fixed_n}$',
               lambda m, s, cache=cache: select(cache, m, s, N=args.fixed_n, **resolution_match(args)))
              for waves, cache in cases]
    panels.append((f'Single step, $\\omega\\Delta t = {args.fixed_wdt:g}$',
                   lambda m, s: select(step_cache, m, s, wdt=args.fixed_wdt, **step_match(args))))

    fig, axes = plt.subplots(len(panels), 1, figsize=b3.FIGSIZE, layout='constrained')
    for ax, (title, rows_of) in zip(axes, panels):
        def stats(m):
            e = [r['l1'] for r in (next(iter(rows_of(m, s)), None) for s in range(args.seeds)) if r]
            return (np.median(e), min(e), max(e), len(e)) if e else None
        results = {v: (stats(method('pinn', v)), stats(method('reg', FLOOR_OF[v]))) for v in VARIANTS}
        print(f'{title}: ' + ', '.join(f'{VARIANTS[v]["label"]} {r[0][0]:.2e} [{r[0][1]:.1e}, {r[0][2]:.1e}] '
                                       f'(floor {r[1][0]:.1e})' if r[0] and r[1] else f'{v}: incomplete'
                                       for v, r in results.items()))
        ys = {v: len(VARIANTS) - 1 - i for i, v in enumerate(VARIANTS)}    # baseline on top
        values = [x for r, f in results.values() for x in ((r[0], r[1], r[2]) if r else ()) + ((f[0],) if f else ())]
        if not values:
            continue
        for v, (r, floor) in results.items():
            color = VARIANTS[v]['color']
            if r:
                ax.hlines(ys[v], r[1], r[2], color=color, lw=3, alpha=0.8)
                ax.scatter(r[0], ys[v], s=250, marker='X', color=color, zorder=3)
                ax.text(r[2] * 1.25, ys[v], f'{r[0]:.1e}' + (f' ($n = {r[3]}$)' if r[3] > 1 else ''), va='center',
                        fontsize=14)
            if floor:
                ax.scatter(floor[0], ys[v], s=120, marker='o', facecolor='none', edgecolor=color, lw=2.5, zorder=3)
        ax.grid(axis='y', alpha=0.2)
        ax.set_xscale('log')
        ax.set_xlim(min(values) / 3, 40 * max(values))
        ax.set_ylim(-0.6, len(VARIANTS) + 0.7)   # headroom for the panel label
        ax.set_yticks(list(ys.values()))
        ax.set_yticklabels([VARIANTS[v]['label'].replace(', exact', ',\nexact').replace(' + ', ' +\n') for v in ys],
                           fontsize=13)
        b3.panel_label(ax, title, loc='upper right')
        ax.set_xlabel('$L_1$ error at $t_1$')
    handles = [Line2D([], [], color='white', marker='X', markersize=12, ls='-', lw=3,
                      label='PINN: median and range over the seeds'),
               Line2D([], [], color='white', marker='o', markersize=10, ls='', markerfacecolor='none', mew=2.5,
                      label='regression floor of the network (median)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=1, frameon=True)
    b3.save('benchmark_6_strategy_ablation', fig, tight=False)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--waves', type=int, nargs='+', default=[16, 2], help='wavelengths in the domain, one panel each')
    p.add_argument('--depth', type=int, default=4, help='hidden layers')
    p.add_argument('--width', type=int, default=50, help='neurons per layer (notebook: 50)')
    p.add_argument('--max-exp', type=int, default=10, help='resolution sweep up to N = 2^this')
    p.add_argument('--fixed-n', type=int, default=256, help='resolution N of the ablation')
    p.add_argument('--fixed-wdt', type=float, default=16, help='step size of the ablation')
    p.add_argument('--seeds', type=int, default=1, help='seeds of the ablation (the sweeps use seed 0), more than 1 adds runs')
    p.add_argument('--adam-steps', type=int, default=2000)
    p.add_argument('--lbfgs-steps', type=int, default=1500)
    p.add_argument('--benchmarks', nargs='+', choices=['resolution', 'step'], default=['resolution', 'step'])
    p.add_argument('--step-waves', type=int, default=2, help='wavelengths in the step-size benchmark')
    p.add_argument('--potential', type=float, default=20.0, help='V0 of V = V0 cos(2 pi x), step-size benchmark')
    p.add_argument('--step-n', type=int, default=256,
                   help='collocation points, step-size benchmark (script 3: 64; the periodic features up to m = 16 '
                        'need a grid that resolves them)')
    p.add_argument('--step-values', type=float, nargs='*', default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help='step sizes omega dt of the step-size benchmark')
    p.add_argument('--quick', action='store_true', help='smoke test with few steps and a small sweep')
    p.add_argument('--plot-only', action='store_true', help='only replot cached results')
    args = p.parse_args()

    if args.quick:
        args.adam_steps, args.lbfgs_steps = 200, 50
        args.max_exp, args.fixed_n, args.seeds = 5, 32, 2
        args.step_values, args.fixed_wdt, args.step_n = [1, 16], 16, 64
    suffix = '_quick' if args.quick else ''

    cases = [(waves, b3.Cache(f'results/benchmark_plane_wave_{waves}waves{suffix}.json')) for waves in args.waves]
    step_cache = b3.Cache(f'results/benchmark_step_size{suffix}.json')
    if not args.plot_only:
        if 'resolution' in args.benchmarks:
            for waves, cache in cases:
                print(f'--- {b3.plural(waves)}', flush=True)
                run_resolution(args, b3.PlaneWave(waves), cache)
        if 'step' in args.benchmarks:
            run_step(args, step_cache)

    plot_resolution(args, cases)
    plot_step_size(args, step_cache)
    plot_ablation(args, cases, step_cache)


if __name__ == '__main__':
    main()
