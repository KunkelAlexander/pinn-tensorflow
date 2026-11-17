"""Relative profile error rho_method(r)/rho_ref(r) - 1 on a log-r axis (mid/outer halo), time-averaged, with a per-range table.

Reads soliton_gravity_results/halo_n256<tag>.json (halo_gravity.py). Profiles have 48 bins of width 0.25 out to r=12.
Because the outer halo is the diagnostic region in the 3D runs, the error is relative to the local reference density, not to the peak.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT

RANGES = [('core r<1', 0., 1.), ('mid 1-3', 1., 3.), ('outer 3-6', 3., 6.), ('far 6-10', 6., 10.)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='_stationary')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--t0', type=float, default=4.)
    ap.add_argument('--groups', default='grid,patch_r4,patch_r8')
    a = ap.parse_args()
    d = json.loads((OUT / f'halo_n{a.n}{a.tag}.json').read_text())
    t = np.array(d['times'])
    sel = t >= a.t0
    edges = np.array(d['profile_r_edges'])
    r = .5 * (edges[1:] + edges[:-1])
    ref = np.array(d['reference_profile_t'])[sel].mean(0)
    rows = {x['name']: x for x in d['rows']}
    rel = {n: np.array(x['profile_t'])[sel].mean(0) / ref - 1 for n, x in rows.items() if 'profile_t' in x}
    print(f"time-averaged over t >= {a.t0}; signed mean relative error of the time-averaged profile (bins weighted equally) per radial range")
    print(f"{'method':24s}" + ''.join(f'{n:>14s}' for n, _, _ in RANGES))
    for n, e in rel.items():
        if n.startswith('floor'):
            continue
        print(f'{n:24s}' + ''.join(f'{np.mean(e[(r >= lo) & (r < hi)]):+14.3f}' for _, lo, hi in RANGES))
    groups = {'grid': ['fd2', 'coarse_r2_fd2', 'coarse_r4_fd2', 'coarse_r8_fd2', 'coarse_r8_spectral', 'fd4'],
              'patch_r4': [f'{k}_r4_half{h}' for h in (16, 32, 48) for k in ('patch', 'mlc')],
              'patch_r8': [f'{k}_r8_half{h}' for h in (16, 32, 48) for k in ('patch', 'mlc')]}
    cmap = plt.get_cmap('tab10')
    for g in a.groups.split(','):
        names = [n for n in groups[g] if n in rel]
        fig, ax = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1.6]))
        ax[0].loglog(r, ref, 'k', lw=2.4)
        ax[0].set_ylabel(f'<rho>(r), t>={a.t0:g}')
        ax[0].set_ylim(ref[ref > 0].min() * .5, ref.max() * 2)
        for i, n in enumerate(names):
            ax[1].semilogx(r, rel[n], color=cmap(i % 10), ls='--' if n.startswith('mlc') else '-', lw=1.4, label=n)
        ax[1].axhline(0, color='k', lw=.6)
        ax[1].set_ylim(-.6, .6)
        ax[1].set_ylabel('rho_method / rho_ref - 1')
        ax[1].set_xlabel('r')
        ax[1].grid(alpha=.3, which='both')
        ax[1].legend(fontsize=8, ncol=2)
        fig.suptitle(f'Relative profile error, time-averaged ({a.tag}, group {g})')
        fig.tight_layout()
        path = OUT / f'halo_n{a.n}{a.tag}_relative_{g}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print('wrote', path)


if __name__ == '__main__':
    main()
