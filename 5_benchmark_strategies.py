"""Strategy benchmark: do improvements from the literature fix the failures of the discrete-time PINN?

Script 3 found three problems of the discrete-time PINN (one Gauss-Legendre IRK step, as in
2_linear_schroedinger_1d_discrete_time.ipynb):
  1. for 16 wavelengths it fails at every resolution (spectral bias),
  2. for 2 wavelengths its error stagnates at about 5e-3, independent of N (training plateau),
  3. in the step-size benchmark it fails for large steps omega dt (the training loss grows with the step).

Three improvements are tested, every one alone and in all combinations (2^3 variants per problem):
  ff   random Fourier features against spectral bias: 16 frequencies from each of three Gaussians with
       standard deviations 1, 4, 16 cycles per domain length (multi-scale, the same for all problems);
       Tancik et al. 2020, Wang, Wang & Perdikaris 2021
  aw   adaptive loss weights against the plateau: relative loss balancing of the IRK reconstruction term and
       the boundary term during Adam, frozen for L-BFGS; simplified from Bischof & Kraus 2021
  rwf  random weight factorisation W = diag(exp(s)) V of all layers, which speeds up convergence;
       Wang et al. 2022
and, outside the factorial, gn: gradient-norm balancing of the same two terms (Wang, Teng & Perdikaris 2021), the
most cited adaptive weighting. It diverges here, because the boundary term is satisfied almost immediately, so its
gradient vanishes and its weight grows without bound (see README).

Everything else is as in 3_benchmark_resolution.py, which this script builds on: 4x50 network, 2000 Adam + 1500
L-BFGS steps, one seed, q = 256 for the resolution benchmark and q adapted to the step with psi^{n+1} computed
from the stages for the step-size benchmark, there at N = 256 instead of 64 since Fourier features alias on a grid
that does not resolve their frequencies. Baselines of the resolution benchmark are read from the caches of
script 3 (same keys), so the caches are shared. Error metric as there.

Usage
-----
    python 5_benchmark_strategies.py                          # all benchmarks, resumes from cache
    python 5_benchmark_strategies.py --waves 16 --benchmarks resolution   # one cache file per process, so
    python 5_benchmark_strategies.py --waves 2 --benchmarks resolution    # these can run in parallel
    python 5_benchmark_strategies.py --benchmarks step
    python 5_benchmark_strategies.py --quick                  # smoke test
    python 5_benchmark_strategies.py --plot-only              # replot from cached results

Figures are written to figures/benchmark_4_* ... benchmark_7_*.png.
"""

import argparse
import importlib.util
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

# 3_benchmark_resolution.py starts with a digit and can therefore not be imported by name
spec = importlib.util.spec_from_file_location(
    'benchmark', os.path.join(os.path.dirname(os.path.abspath(__file__)), '3_benchmark_resolution.py'))
b3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b3)


# ----------------------------------------------------------------------------------------------
# Variants: all subsets of the three strategies
# ----------------------------------------------------------------------------------------------

STRATEGIES = {
    'ff':  dict(fourier=(1, 4, 16)),
    'aw':  dict(adaptive='loss'),
    'rwf': dict(rwf=True),
    'gn':  dict(adaptive='gradient'),   # the alternative to aw from the literature; diverges, so only run alone
}
FULL = ('ff', 'aw', 'rwf')
COMBOS = [c for r in range(len(FULL) + 1) for c in itertools.combinations(FULL, r)] + [('gn',)]
MAIN = [c for c in COMBOS if len(c) in (0, 1, len(FULL)) and c != ('gn',)]   # in the resolution and step-size figures
RUN_ORDER = MAIN + [c for c in COMBOS if c not in MAIN]                         # the others only in the ablation
LABELS = {'ff': 'Fourier features', 'aw': 'adaptive weights', 'rwf': 'RWF', 'gn': 'gradient-norm weights'}
SHORT = {'ff': 'FF', 'aw': 'AW', 'rwf': 'RWF', 'gn': 'GN'}
COLORS = {(): '#FFA500', ('ff',): '#FE53BB', ('aw',): '#B967FF', ('rwf',): '#FF3131', FULL: '#FFFFFF'}


def label(combo):
    return 'baseline' if not combo else 'all three' if combo == FULL else ' + '.join(LABELS[s] for s in combo)


def name(base, combo):
    """Method name in the caches; the baseline keeps the name of script 3."""
    return base + ''.join('+' + s for s in combo)


def kwargs(combo):
    return {k: v for s in combo for k, v in STRATEGIES[s].items()}


for combo in MAIN:
    b3.STYLE[name('pinn_disc', combo)] = dict(label='PINN discrete time: ' + label(combo), color=COLORS[combo], marker='X')


# ----------------------------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------------------------

def cached_run(cache, key, run, **meta):
    if cache.get(key) is None:
        err, loss, wall, n_params = run()
        cache.put(key, dict(l1=err, loss=loss, wall=wall, n_params=n_params, **meta))
    r = cache.get(key)
    print(f'{key:76s} L1 = {r["l1"]:.3e}  loss = {r["loss"]:.1e}  time = {r["wall"]:.0f} s', flush=True)


def run_resolution(args, problem, cache):
    """Every variant for N = 2^4 ... 2^max_exp, then single-scale Fourier features at fixed N (sigma sweep)."""
    jobs = [(name('pinn_disc', c), kwargs(c), int(n)) for c in RUN_ORDER for n in 2**np.arange(4, args.max_exp + 1)]
    jobs += [(f'pinn_disc+ffsigma{s:g}', dict(fourier=(s,) * 3), args.fixed_n) for s in args.sigmas]
    for method, kw, n in jobs:
        # same keys as in script 3
        key = f'{method}/N={n}/width={args.width}/depth={args.depth}/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}'
        cached_run(cache, key, lambda: b3.run_pinn_discrete(problem, n, args.width, args.depth, args.adam_steps,
                                                            args.lbfgs_steps, **kw),
                   method=method, N=n, width=args.width, depth=args.depth, adam_steps=args.adam_steps,
                   lbfgs_steps=args.lbfgs_steps)


def run_step(args, cache):
    """Single step in the periodic potential, discrete-time PINN with q adapted and psi^{n+1} from the stages."""
    omega = 0.5 * (2 * np.pi * args.step_waves)**2
    for wdt in args.step_values:
        problem = b3.PlaneWaveInPotential(args.step_waves, args.potential, wdt / omega)
        q, tableau = b3.required_q(problem)
        print(f'--- omega dt = {wdt:g}: dt = {problem.t1:.5f}, required q = {q}', flush=True)

        key = f'irk_exact/wdt={wdt:g}'
        if cache.get(key) is None:
            err, wall, q_used = b3.step_errors_classical(problem)['irk_exact']
            cache.put(key, dict(method='irk_exact', wdt=wdt, l1=err, wall=wall, q=q_used))

        for c in RUN_ORDER:
            method = name('pinn_disc_q', c)
            key = (f'{method}/wdt={wdt:g}/N={args.step_n}/V0={args.potential:g}/depth={args.depth}'
                   f'/adam={args.adam_steps}/lbfgs={args.lbfgs_steps}')
            cached_run(cache, key, lambda: b3.run_pinn_discrete(problem, args.step_n, args.width, args.depth,
                                                                args.adam_steps, args.lbfgs_steps, tableau=tableau,
                                                                final_from_stages=True, **kwargs(c)),
                       method=method, wdt=wdt, q=q, N=args.step_n, V0=args.potential, depth=args.depth,
                       adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def resolution_match(args):
    return dict(width=args.width, depth=args.depth, adam_steps=args.adam_steps, lbfgs_steps=args.lbfgs_steps)


def step_match(args):
    return dict(N=args.step_n, V0=args.potential, depth=args.depth, adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps)


def resolution_series(args, cache):
    series = {m: b3.select(cache, m) for m in ['fd4', 'fd6']}
    for c in MAIN:
        series[name('pinn_disc', c)] = [r for r in b3.select(cache, name('pinn_disc', c), **resolution_match(args))
                                        if r['N'] <= 2**args.max_exp]
    series['fft'] = b3.select(cache, 'fft')
    return series


def plot_step_size(args, cache):
    """Error and final training loss of one step vs. step size."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=b3.FIGSIZE, sharex=True, layout='constrained')
    rows = [r for r in b3.select(cache, 'irk_exact') if r['wdt'] in args.step_values]
    if rows:
        b3.glow([r['wdt'] for r in rows], [max(r['l1'], 1e-16) for r in rows], ax=ax1, lw=2.5, markersize=9,
                color=b3.STYLE['irk_exact']['color'], marker='o', label=b3.STYLE['irk_exact']['label'])
    for c in MAIN:
        rows = [r for r in b3.select(cache, name('pinn_disc_q', c), **step_match(args)) if r['wdt'] in args.step_values]
        if rows:
            wdt = [r['wdt'] for r in rows]
            style = dict(lw=2.5, markersize=9, color=COLORS[c], marker='X')
            b3.glow(wdt, [max(r['l1'], 1e-16) for r in rows], ax=ax1, label=label(c), **style)
            b3.glow(wdt, [r['loss'] for r in rows], ax=ax2, **style)

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
    fig.legend(*ax1.get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=True)
    b3.save('benchmark_5_strategies_vs_step_size', fig, tight=False)
    plt.close(fig)


def plot_ablation(args, cases, step_cache):
    """Error of all 2^3 variants at fixed N (plane waves) and fixed step size, as horizontal bars."""
    panels = [(f'Plane wave, {b3.plural(waves)}, $N = {args.fixed_n}$',
               {c: b3.select(cache, name('pinn_disc', c), N=args.fixed_n, **resolution_match(args)) for c in COMBOS})
              for waves, cache in cases]
    panels.append((f'Single step, $\\omega\\Delta t = {args.fixed_wdt:g}$',
                   {c: b3.select(step_cache, name('pinn_disc_q', c), wdt=args.fixed_wdt, **step_match(args))
                    for c in COMBOS}))

    fig, axes = plt.subplots(len(panels), 1, figsize=b3.FIGSIZE, layout='constrained')
    for ax, (title, rows) in zip(axes, panels):
        errors = {c: r[0]['l1'] for c, r in rows.items() if r}
        print(f'{title}: ' + ', '.join(f'{label(c)} {e:.2e}' for c, e in errors.items()))
        if not errors:
            continue
        ys = {c: len(COMBOS) - 1 - i for i, c in enumerate(COMBOS)}    # baseline on top
        for c, e in errors.items():
            ax.scatter(e, ys[c], s=250, marker='X', color=COLORS.get(c, '#08F7FE'), zorder=3)
            ax.text(e * 1.25, ys[c], f'{e:.1e}', va='center', fontsize=14)
        if () in errors:
            ax.axvline(errors[()], color=COLORS[()], ls='--', lw=1.5)
        ax.grid(axis='y', alpha=0.2)
        ax.set_xscale('log')
        ax.set_xlim(min(errors.values()) / 3, 6 * max(errors.values()))
        ax.set_ylim(-0.6, len(COMBOS) + 0.7)   # headroom for the panel label
        ax.set_yticks(list(ys.values()))
        ax.set_yticklabels(['baseline' if not c else ' + '.join(SHORT[s] for s in c) for c in ys])
        b3.panel_label(ax, title, loc='upper right')
        ax.set_xlabel('$L_1$ error at $t_1$')
    b3.save('benchmark_6_strategy_ablation', fig, tight=False)
    plt.close(fig)


def plot_sigma(args, cases):
    """Error vs. the scale of single-scale Fourier features, with the baseline and the multi-scale set for reference."""
    fig, axes = plt.subplots(len(cases), 1, figsize=b3.FIGSIZE, sharex=True, squeeze=False, layout='constrained')
    for ax, (waves, cache) in zip(axes[:, 0], cases):
        rows = {s: b3.select(cache, f'pinn_disc+ffsigma{s:g}', N=args.fixed_n, **resolution_match(args))
                for s in args.sigmas}
        rows = {s: r[0] for s, r in rows.items() if r}
        if rows:
            b3.glow(list(rows), [r['l1'] for r in rows.values()], ax=ax, lw=2.5, markersize=9, color=COLORS[('ff',)],
                    marker='X', label='single-scale Fourier features')
        for c, ls, text in [((), '--', 'baseline'), (('ff',), ':', 'multi-scale Fourier features, $\\sigma$ = 1, 4, 16')]:
            r = b3.select(cache, name('pinn_disc', c), N=args.fixed_n, **resolution_match(args))
            if r:
                ax.axhline(r[0]['l1'], color=COLORS[c], ls=ls, lw=2, label=text)
        ax.axvline(waves, color='white', lw=0.8, alpha=0.5)
        ax.text(waves * 1.07, 0.04, 'wavenumber of the solution', rotation=90, va='bottom', color='white', alpha=0.7,
                fontsize=14, transform=ax.get_xaxis_transform())
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_ylabel(f'$L_1$ error at $t_1$')
        b3.panel_label(ax, f'Plane wave, {b3.plural(waves)}, $N = {args.fixed_n}$', loc='upper left')
    axes[-1, 0].set_xlabel('Scale $\\sigma$ of the Fourier features [cycles per domain]')
    handles = {}
    for ax in axes[:, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=2, frameon=True)
    b3.save('benchmark_7_fourier_scale', fig, tight=False)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--waves', type=int, nargs='+', default=[16, 2], help='wavelengths in the domain, one panel each')
    p.add_argument('--depth', type=int, default=4, help='hidden layers')
    p.add_argument('--width', type=int, default=50, help='neurons per layer (notebook: 50)')
    p.add_argument('--max-exp', type=int, default=10, help='resolution sweep up to N = 2^this')
    p.add_argument('--fixed-n', type=int, default=256, help='resolution N of the sigma sweep and the ablation')
    p.add_argument('--fixed-wdt', type=float, default=16, help='step size of the ablation')
    p.add_argument('--sigmas', type=float, nargs='*', default=[1, 2, 4, 8, 16, 32, 64], help='sigma sweep, [] to skip')
    p.add_argument('--adam-steps', type=int, default=2000)
    p.add_argument('--lbfgs-steps', type=int, default=1500)
    p.add_argument('--benchmarks', nargs='+', choices=['resolution', 'step'], default=['resolution', 'step'])
    p.add_argument('--step-waves', type=int, default=2, help='wavelengths in the step-size benchmark')
    p.add_argument('--potential', type=float, default=20.0, help='V0 of V = V0 cos(2 pi x), step-size benchmark')
    p.add_argument('--step-n', type=int, default=256,
                   help='collocation points, step-size benchmark (script 3: 64; Fourier features need the grid to '
                        'resolve their frequencies, see the resolution benchmark)')
    p.add_argument('--step-values', type=float, nargs='*', default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help='step sizes omega dt of the step-size benchmark')
    p.add_argument('--quick', action='store_true', help='smoke test with few steps and a small sweep')
    p.add_argument('--plot-only', action='store_true', help='only replot cached results')
    args = p.parse_args()

    if args.quick:
        args.adam_steps, args.lbfgs_steps = 200, 50
        args.max_exp, args.fixed_n, args.sigmas = 5, 32, [4, 16]
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

    b3.plot_resolution(args, cases, resolution_series, 'benchmark_4_strategies_vs_resolution', ncol=1)
    plot_step_size(args, step_cache)
    plot_ablation(args, cases, step_cache)
    plot_sigma(args, cases)


if __name__ == '__main__':
    main()
