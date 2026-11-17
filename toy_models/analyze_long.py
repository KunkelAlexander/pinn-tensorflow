"""Tables and figures for halo_long.py output: secular drift, spurious self-force, granule statistics, relative profile errors (log r)."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT

RANGES = [('core<1', 0., 1.), ('mid1-3', 1., 3.), ('out3-6', 3., 6.), ('far6-10', 6., 10.)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', default=['_long', '_long2', '_long3'])
    ap.add_argument('--out', default='_long_all')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--late', type=float, default=24.)
    a = ap.parse_args()
    c = {}
    for tag in a.tags:
        p = OUT / f'halo_n{a.n}{tag}.json'
        if p.exists():
            dd = json.loads(p.read_text())
            t = np.array(dd['times'])
            for k, v in dd['cases'].items():
                c.setdefault(k, v)
    ref = c['spectral']
    r = (np.arange(48) + .5) * .25
    late = t >= a.late
    pref_early, pref_late = (np.array(ref['profile'])[m].mean(0) for m in (t < a.late, late))
    print(f"{'method':18s} {'|centroid|':>10s} {'|p| end':>9s} {'E drift':>9s} {'rms_c end':>9s} {'core late':>9s} {'contrast':>8s}   late-time (t>={a.late:g}) relative error of <rho> in:  " + ' '.join(f'{n:>8s}' for n, _, _ in RANGES))
    rows = {}
    for name, rec in c.items():
        p_late = np.array(rec['profile'])[late].mean(0)
        e = p_late / pref_late - 1
        rows[name] = e
        print(f"{name:18s} {np.hypot(rec['cx'][-1], rec['cy'][-1]):10.3f} {np.hypot(rec['px'][-1], rec['py'][-1]):9.2e} {rec['E'][-1] / rec['E'][0] - 1:+9.1e} "
              f"{np.mean(rec['rms_c'][-10:]):9.3f} {np.mean(np.array(rec['core'])[late]):9.2f} {np.mean(rec['contrast']):8.3f}   {' ' * 47}" +
              ' '.join(f'{np.mean(e[(r >= lo) & (r < hi)]):+8.3f}' for _, lo, hi in RANGES))
    names = list(c)
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    for i, nme in enumerate(names):
        rec = c[nme]
        kw = dict(color=cmap(i % 20), lw=2.5 if nme == 'spectral' else 1.2, label=nme, ls='--' if nme.startswith('mlc') else '-')
        ax[0, 0].plot(t, rec['rms_c'], **kw)
        ax[0, 1].plot(t, np.hypot(rec['cx'], rec['cy']), **kw)
        ax[0, 2].semilogy(t, np.hypot(rec['px'], rec['py']) + 1e-9, **kw)
        ax[1, 0].plot(t, rec['core'], **kw) if False else ax[1, 0].plot(t, np.convolve(rec['core'], np.ones(9) / 9, 'same'), **kw)
        ax[1, 1].plot(t, np.array(rec['E']) / rec['E'][0] - 1, **kw)
        ax[1, 2].semilogx(r, rows[nme], **kw)
    for x, tt in zip(ax.ravel(), ['rms about centroid', '|centroid| (spurious drift)', '|net momentum|', 'core density (9-pt smoothed)',
                                  'spectral-functional energy drift', f'<rho> / <rho>_spec - 1, t >= {a.late:g}']):
        x.set_title(tt)
    ax[1, 2].set_ylim(-.6, .6); ax[1, 2].axhline(0, color='k', lw=.5); ax[1, 2].set_xlabel('r'); ax[1, 0].set_xlabel('t'); ax[1, 1].set_xlabel('t')
    ax[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle('Long-time fidelity of gravity treatments on the stationary halo')
    fig.tight_layout()
    path = OUT / f'halo_n{a.n}{a.out}_summary.png'
    fig.savefig(path, dpi=125)
    print('wrote', path)
    focus = ['fd2', 'fd4', 'coarse_r4_fd2', 'coarse_r8_fd2', 'coarse_r4_tsc_fd2', 'coarse_r8_tsc_fd2', 'patch_r8_half32', 'patch_r8_half48', 'mlc_r8_half48',
             'galerkin_r8_tsc_half32', 'galerkin_r8_tsc_half48']
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    for k, (lo, hi) in enumerate(((a.late, 1e9), (0., a.late))):
        m = (t >= lo) & (t < hi)
        base = np.array(ref['profile'])[m].mean(0)
        for i, nme in enumerate(focus):
            if nme in c:
                ax[k].semilogx(r, np.array(c[nme]['profile'])[m].mean(0) / base - 1, color=plt.get_cmap('tab20')(i), lw=1.5, ls='--' if nme.startswith('mlc') else '-', label=nme)
        ax[k].axhline(0, color='k', lw=.6); ax[k].set_ylim(-.6, .6); ax[k].grid(alpha=.3, which='both'); ax[k].set_xlabel('r')
        ax[k].set_title(f'time-averaged <rho>/<rho>_spectral - 1, t in [{lo:g}, {min(hi, t[-1]):g}]')
    ax[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = OUT / f'halo_n{a.n}{a.out}_relative_logr.png'
    fig.savefig(path, dpi=130)
    print('wrote', path)


if __name__ == '__main__':
    main()
