"""Research direction 2: does subtracting the mass-weighted mean force (Newton's-3rd-law projection) cure the coarse-grid halo drift?

Same harness as halo_1d.py (L=32 default, N=1024). For each coarse H we run: spectral, coarse, coarse+meanforce.
Meanforce: phi -> phi + Fbar*x with Fbar = -int rho phi' / M, so the mass-weighted net force of the potential used in the
wave solver is exactly zero. (x is box-centred; the linear ramp only adds a phase jump at the box edge where rho ~ 0.)
Reports centroid drift and the relative profile error rho/rho_spec - 1 in core/mid/outer bins (mean over seeds), plus
the realisation noise floor (perturbed spectral).
"""
import argparse
import json
import numpy as np
import halo_1d as hd
from halo_1d import x, h, N, k, EDGES, profile, evolve, solve, build_psi, make_grav
from soliton_gravity import OUT


def with_meanforce(grav):
    def g(rho):
        phi = grav(rho)
        dphi = np.fft.ifft(1j * k * np.fft.fft(phi)).real
        fbar = -(rho * dphi).sum() / rho.sum()
        return phi + fbar * x
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mass', type=float, default=32.)
    ap.add_argument('--T', type=float, default=4.)
    ap.add_argument('--rc', type=float, default=8.)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--tend', type=float, default=60.)
    ap.add_argument('--tavg', type=float, default=20.)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--Hs', type=int, nargs='+', default=[8, 16, 32])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    sol = solve(a.mass, a.T, a.rc, verbose=False)
    every = round(.25 / a.dt)
    runs_g = {'spectral': hd.spec}
    for R in a.Hs:
        c = make_grav('coarse', R)
        runs_g[f'H={R * h:.3f}'] = c
        runs_g[f'H={R * h:.3f}+mf'] = with_meanforce(c)
    res = {}
    for seed in a.seeds:
        psi0 = build_psi(sol, seed, a.mass)
        pert = psi0 * (1 + 1e-6 * np.random.default_rng(100 + seed).standard_normal(N))
        jobs = {'spectral': (psi0, hd.spec), 'pert': (pert, hd.spec)}
        jobs.update({n: (psi0, g) for n, g in runs_g.items() if n != 'spectral'})
        res[seed] = {}
        for name, (w, g) in jobs.items():
            sn = evolve(w, g, a.tend, a.dt, every)
            ts = np.arange(len(sn)) * every * a.dt
            sel = ts >= a.tend - a.tavg
            prof = np.mean([profile(s) for s in sn[sel]], axis=0)
            cen = np.array([(s * x).sum() / s.sum() for s in sn])
            res[seed][name] = dict(profile=prof.tolist(), maxcen=float(np.abs(cen).max()))
            print(f'seed {seed} {name:16s} max|centroid| {np.abs(cen).max():.3f}', flush=True)
    # summary: relative profile error vs spectral, in thirds of the bin range (core / mid / outer)
    nb = len(EDGES) - 1
    zones = {'core': slice(0, 4), 'mid': slice(8, 16), 'outer': slice(16, nb)}
    print('\nmean over seeds of relative profile error vs spectral (zone-summed density ratio) (and mean max|centroid|)')
    names = [n for n in res[a.seeds[0]] if n != 'spectral']
    for n in names:
        row = []
        for z, s in zones.items():
            e = np.mean([np.sum(np.array(res[sd][n]['profile'])[s]) / np.sum(np.array(res[sd]['spectral']['profile'])[s]) - 1
                         for sd in a.seeds])
            row.append(f'{z} {e:+.1%}')
        cen = np.mean([res[sd][n]['maxcen'] for sd in a.seeds])
        print(f'{n:16s} ' + '  '.join(row) + f'  drift {cen:.3f}')
    (OUT / f'halo_1d_meanforce{a.tag}.json').write_text(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
