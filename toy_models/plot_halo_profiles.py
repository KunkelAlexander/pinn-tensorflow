"""Overlay azimuthal density profiles of the halo runs for the different gravity treatments.

Reads soliton_gravity_results/halo_n256<tag>.json (written by halo_gravity.py) and writes two figures:
  *_profiles_grid.png   spectral reference vs fd2/fd4/coarse (whole domain at lower resolution)
  *_profiles_patch.png  spectral reference vs inherited-boundary patch and local-corrections patch, matched sizes
Top row: rho(r) at several times; bottom row: |rho - rho_ref| / peak(rho_ref) at that time.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT


def draw(d, names, times, title, path):
    edges = np.array(d['profile_r_edges'])
    r = .5 * (edges[1:] + edges[:-1])
    rows = {row['name']: row for row in d['rows']}
    names = [n for n in names if n in rows]
    ts = np.array(d['times'])
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(2, len(times), figsize=(4.2 * len(times), 6.4), sharex='col', gridspec_kw=dict(height_ratios=[2, 1.3]))
    for c, t in enumerate(times):
        k = int(np.argmin(abs(ts - t)))
        ref = np.array(d['reference_profile_t'][k])
        ax[0, c].plot(r, ref, 'k', lw=2.4, label='spectral (reference)', zorder=10)
        for i, nme in enumerate(names):
            p = np.array(rows[nme]['profile_t'][k])
            ax[0, c].plot(r, p, color=cmap(i % 10), lw=1.3, ls='--' if nme.startswith('mlc') else '-', label=nme)
            ax[1, c].plot(r, abs(p - ref) / ref.max() + 1e-12, color=cmap(i % 10), lw=1.3, ls='--' if nme.startswith('mlc') else '-')
        ax[0, c].set_yscale('log')
        ax[0, c].set_ylim(max(ref.min(), 1e-6) * .5, ref.max() * 3)
        ax[0, c].set_title(f't = {ts[k]:.1f}')
        ax[1, c].set_yscale('log')
        ax[1, c].set_xlabel('r')
        ax[1, c].set_ylim(1e-6, 2)
        ax[1, c].grid(alpha=.3)
    ax[0, 0].set_ylabel('azimuthal mean density')
    ax[1, 0].set_ylabel('|rho - rho_ref| / peak rho_ref')
    ax[0, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print('wrote', path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='_virial_mlc')
    ap.add_argument('--times', type=float, nargs='+', default=[2., 6., 10., 16.])
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--halves', type=int, nargs='+', default=[16, 32, 48])
    ap.add_argument('--grid', nargs='+', default=['fd2', 'fd4', 'coarse_r4_fd2', 'coarse_r8_fd2'])
    args = ap.parse_args()
    d = json.loads((OUT / f'halo_n{args.n}{args.tag}.json').read_text())
    a = d['args']
    sub = f"mass {a['mass']:g}, N={d['n']}" + (f", R {a['R']:g}, theta {a['theta']:g}" if not a.get('psi0') else ', stationary halo')
    base = OUT / f'halo_n{args.n}{args.tag}'
    draw(d, args.grid, args.times, f'Halo profiles: gravity grid ({sub})', f'{base}_profiles_grid.png')
    for r in (4, 8):
        draw(d, ['fd2'] + [f'{k}_r{r}_half{h}' for h in args.halves for k in ('patch', 'mlc')], args.times,
             f'Halo profiles: inherited-boundary patch (solid) vs local corrections (dashed), parent ratio {r} ({sub})',
             f'{base}_profiles_patch_r{r}.png')


if __name__ == '__main__':
    main()
