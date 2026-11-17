"""Static force test: what do the gravity treatments do to a smooth stationary halo, without any wave evolution (seconds per method)?

For the self-consistent target density rho(r) of a stationary halo placed at an offset from the grid origin we compare each
treatment's force -grad(phi) (fourth-order central differences, identical for all) with the spectral force:
  * net self-force  F_net = -int rho grad(phi): zero for the spectral solve (translation invariance), non-zero when the solver
    breaks it; the dynamic runs show it as halo drift (momentum non-conservation);
  * local vector force error rms_annulus |F - F_spec| / mean_annulus |F_spec| on a log-r axis (mid/outer halo).
The patch is centred on the grid origin; the halo is shifted by (ox, oy) in units of the coarse cell to scan the grid phase.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import Gravity, grid, L, OUT
import stationary_halo as sh

METHODS = {
    'fd2': dict(kind='fd2'), 'fd4': dict(kind='fd4'),
    'coarse_r2': dict(kind='coarse', ratio=2, parent='fd2'),
    'coarse_r4': dict(kind='coarse', ratio=4, parent='fd2'),
    'coarse_r8': dict(kind='coarse', ratio=8, parent='fd2'),
    'coarse_r4_fd4': dict(kind='coarse', ratio=4, parent='fd4'),
    'coarse_r8_tsc_fd4': dict(kind='coarse', ratio=8, parent='fd4', adjoint=2),
    'galerkin_r4_tsc': dict(kind='galerkin', ratio=4, adjoint=2),
    'galerkin_r8_tsc': dict(kind='galerkin', ratio=8, adjoint=2),
    'galerkin_r8_tsc_half48': dict(kind='galerkin', ratio=8, half=48, adjoint=2),
    'coarse_r4_tsc_fd2': dict(kind='coarse', ratio=4, parent='fd2', adjoint=2),
    'coarse_r8_tsc_fd2': dict(kind='coarse', ratio=8, parent='fd2', adjoint=2),
    'patch_r4_half32_tsc': dict(kind='patch', ratio=4, parent='fd2', half=32, adjoint=2),
    'patch_r8_half32_tsc': dict(kind='patch', ratio=8, parent='fd2', half=32, adjoint=2),
    'patch_r8_half48_tsc': dict(kind='patch', ratio=8, parent='fd2', half=48, adjoint=2),
    'patch_r4_half32': dict(kind='patch', ratio=4, parent='fd2', half=32),
    'patch_r8_half32': dict(kind='patch', ratio=8, parent='fd2', half=32),
    'patch_r8_half48': dict(kind='patch', ratio=8, parent='fd2', half=48),
    'mlc_r8_half48': dict(kind='mlc', ratio=8, parent='fd2', half=48),
}


def grad4(f, h):
    gx = (-np.roll(f, -2, 0) + 8 * np.roll(f, -1, 0) - 8 * np.roll(f, 1, 0) + np.roll(f, 2, 0)) / (12 * h)
    gy = (-np.roll(f, -2, 1) + 8 * np.roll(f, -1, 1) - 8 * np.roll(f, 1, 1) + np.roll(f, 2, 1)) / (12 * h)
    return gx, gy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=float, default=5.)
    ap.add_argument('--rc', type=float, default=8.)
    ap.add_argument('--mass', type=float, default=80.)
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--tag', default='A')
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--phases', type=int, default=3, help='grid phases scanned per axis (0 .. coarse cell)')
    a = ap.parse_args()
    n, h = a.n, L / a.n
    sol = sh.solve(n, a.mass, a.T, a.rc, mmax=64, verbose=False)
    r1, _ = sh.radial_grid()
    x = grid(n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    spec = Gravity(n, 'spectral')
    edges = np.linspace(0, 12, 49)
    rc = .5 * (edges[1:] + edges[:-1])
    out = dict(args=vars(a), r=rc.tolist(), methods={})
    grav = {}
    for name in a.methods:
        kw = METHODS[name]
        grav[name] = Gravity(n, **kw)
        out['methods'][name] = dict(net=[], err_profile=None, ratio=kw.get('ratio', 1))
    for pi in range(a.phases):
        for pj in range(a.phases):
            for name in a.methods:
                ratio = out['methods'][name]['ratio']
                ox, oy = pi / a.phases * ratio * h, pj / a.phases * ratio * h
                rr = np.hypot(xx - ox, yy - oy)
                rho = np.interp(rr, r1, sol['rho_r'], right=0.)
                fs = grad4(spec(rho), h)
                f = grad4(grav[name](rho), h)
                net = -np.array([(rho * f[0]).sum(), (rho * f[1]).sum()]) * h * h
                out['methods'][name]['net'].append(float(np.hypot(*net)))
                if pi == 0 and pj == 0:
                    ids = np.minimum((rr / 12 * 48).astype(int), 48).ravel()
                    d2 = ((f[0] - fs[0]) ** 2 + (f[1] - fs[1]) ** 2).ravel()
                    m2 = np.hypot(*fs).ravel()
                    cnt = np.maximum(np.bincount(ids, minlength=49), 1)
                    err = np.sqrt(np.bincount(ids, weights=d2, minlength=49) / cnt) / (np.bincount(ids, weights=m2, minlength=49) / cnt)
                    out['methods'][name]['err_profile'] = err[:48].tolist()
    print(f"halo {a.tag}: T={a.T} r_c={a.rc}   |F_net| (max over {a.phases ** 2} grid phases) and local force error at r=1,3,6,10")
    for name, m in out['methods'].items():
        e = np.array(m['err_profile'])
        pick = [e[np.searchsorted(rc, v)] for v in (1, 3, 6, 10)]
        print(f"{name:24s} |F_net| max {max(m['net']):.2e}  mean {np.mean(m['net']):.2e}   force err " + ' '.join(f'{v:.1e}' for v in pick))
    (OUT / f'static_force_{a.tag}.json').write_text(json.dumps(out) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    cm = plt.get_cmap('tab20')
    for i, (name, m) in enumerate(out['methods'].items()):
        ax[0].loglog(rc, np.maximum(m['err_profile'], 1e-8), color=cm(i % 20), ls='--' if name.startswith('mlc') else '-', label=name)
    ax[0].set_xlabel('r'); ax[0].set_ylabel('rms |F - F_spec| / <|F_spec|> in annulus'); ax[0].set_ylim(1e-6, 2); ax[0].grid(alpha=.3, which='both'); ax[0].legend(fontsize=7, ncol=2)
    ax[0].set_title('local force error (halo at grid origin)')
    names = list(out['methods'])
    ax[1].barh(names, [max(out['methods'][k]['net']) + 1e-12 for k in names]); ax[1].set_xscale('log')
    ax[1].set_xlabel('max over grid phases of |net self-force|'); ax[1].set_title('net self-force (momentum non-conservation)')
    fig.tight_layout()
    fig.savefig(OUT / f'static_force_{a.tag}.png', dpi=125)
    print('wrote', OUT / f'static_force_{a.tag}.png')


if __name__ == '__main__':
    main()
