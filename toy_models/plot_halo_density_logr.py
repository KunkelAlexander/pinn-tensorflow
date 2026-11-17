"""Absolute time-averaged radial density <rho>(r) on log-log axes (spectral vs gravity treatments), with relative error underneath.

Reads halo_long.py output (profiles every 0.5 time units, 48 annuli of width 0.25 about the grid origin).
Columns = groups of methods; top row = <rho>(r) in the late window, bottom row = <rho>/<rho>_spec - 1.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT

GROUPS = [('finite-difference & coarse grid (cell mean + Lagrange)', ['fd2', 'fd4', 'coarse_r2_fd2', 'coarse_r4_fd2', 'coarse_r8_fd2']),
          ('smooth TSC coarse grid / inherited-boundary patch', ['coarse_r4_tsc_fd2', 'coarse_r8_tsc_fd2', 'patch_r8_half32', 'patch_r8_half48']),
          ('local corrections / composite Galerkin', ['mlc_r8_half32', 'mlc_r8_half48', 'galerkin_r8_tsc_half32', 'galerkin_r8_tsc_half48'])]


def load(tags, n=256):
    cases, t = {}, None
    for tag in tags:
        p = OUT / f'halo_n{n}{tag}.json'
        if p.exists():
            d = json.loads(p.read_text())
            t = np.array(d['times'])
            for k, v in d['cases'].items():
                cases.setdefault(k, v)
    return t, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--late', type=float, default=24.)
    a = ap.parse_args()
    r = (np.arange(48) + .5) * .25
    for label, tags in (('A', ['_long', '_long2', '_long3', '_longC']), ('B', ['_longB'])):
        t, c = load(tags)
        late = t >= a.late
        early = (t >= 2) & (t < 8)
        ref = np.array(c['spectral']['profile'])
        fig, ax = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.3]))
        cm = plt.get_cmap('tab10')
        for g, (title, names) in enumerate(GROUPS):
            ax[0, g].loglog(r, ref[early].mean(0), color='.6', lw=2, ls=':', label=f'spectral, t in [2,8] (start)')
            ax[0, g].loglog(r, ref[late].mean(0), 'k', lw=2.5, label=f'spectral, t >= {a.late:g}', zorder=10)
            for i, nme in enumerate(names):
                if nme not in c:
                    continue
                pr = np.array(c[nme]['profile'])[late].mean(0)
                ax[0, g].loglog(r, pr, color=cm(i), lw=1.4, label=nme)
                ax[1, g].semilogx(r, pr / ref[late].mean(0) - 1, color=cm(i), lw=1.4, label=nme)
            ax[1, g].axhline(0, color='k', lw=.6)
            ax[1, g].set_ylim(-.6, .6); ax[1, g].set_xlabel('r'); ax[1, g].grid(alpha=.3, which='both')
            ax[0, g].grid(alpha=.3, which='both'); ax[0, g].set_title(title, fontsize=10)
            ax[0, g].set_ylim(max(ref[late].mean(0).min(), 1e-5) * .5, ref[late].mean(0).max() * 2)
            ax[0, g].legend(fontsize=8, loc='lower left')
        ax[0, 0].set_ylabel('time-averaged azimuthal density <rho>(r)')
        ax[1, 0].set_ylabel('<rho>/<rho>_spec - 1')
        fig.suptitle(f'Halo {label}: radial density profile on log-log axes, late window t >= {a.late:g} (profile about the grid origin)')
        fig.tight_layout()
        path = OUT / f'halo_n256_{label}_density_logr.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print('wrote', path)


if __name__ == '__main__':
    main()
