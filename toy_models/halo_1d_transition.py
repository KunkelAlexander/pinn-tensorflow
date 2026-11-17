"""Transition probe: halos from compact (T=4) to extended (T=12); global coarse (cell-mean/B3, +mf) at H=0.5,1,2; 4 seeds, plus
perturbed-spectral noise floor per zone (|pert/spec-1|, mean over seeds)."""
import json, sys
import numpy as np
import halo_1d as hd
from halo_1d import x, h, EDGES, profile, evolve, solve, build_psi
from halo_1d_meanforce import with_meanforce
from halo_1d_stages import make
from halo_1d_filters import make_filt
from soliton_gravity import OUT
zones = {'core': slice(0, 4), 'mid': slice(8, 16), 'outer': slice(16, len(EDGES) - 1)}
seeds = [1, 2, 3, 4]; tend, tavg, dt = 60., 20., .005; every = round(.25 / dt)
out = {}
for T, rc in [(6, 8.5), (8, 9.), (10, 9.5)]:
    sol = solve(32., T, rc, verbose=False); r = sol['rho']
    print(f'T={T} rc={rc}: far mass |x|>2: {r[abs(x)>2].sum()/r.sum():.3f}, >4: {r[abs(x)>4].sum()/r.sum():.4f}', flush=True)
    res = {}
    for seed in seeds:
        psi0 = build_psi(sol, seed, 32.)
        pert = psi0 * (1 + 1e-6 * np.random.default_rng(100 + seed).standard_normal(len(psi0)))
        jobs = {'spectral': (psi0, hd.spec), 'pert': (pert, hd.spec)}
        for R in [16, 32, 64]:
            jobs[f'H={R*h:.2f} mean+mf'] = (psi0, with_meanforce(make(R, 'mean', 'fd2', 'lag4')))
            jobs[f'H={R*h:.2f} B3+mf'] = (psi0, with_meanforce(make_filt(R, 3, 0)))
        res[seed] = {}
        for name, (w, g) in jobs.items():
            sn = evolve(w, g, tend, dt, every); ts = np.arange(len(sn)) * every * dt
            res[seed][name] = np.mean([profile(s) for s in sn[ts >= tend - tavg]], axis=0).tolist()
    print(f'  zone error (mean over seeds; +/- = scatter of per-seed error)')
    for n in [n for n in res[1] if n != 'spectral']:
        row = []
        for z, s in zones.items():
            e = np.array([np.sum(np.array(res[sd][n])[s]) / np.sum(np.array(res[sd]['spectral'])[s]) - 1 for sd in seeds])
            row.append(f'{z} {e.mean():+.1%}±{e.std():.1%}')
        print(f'  {n:18s} ' + '  '.join(row), flush=True)
    out[f'T{T}'] = res
(OUT / 'halo_1d_transition.json').write_text(json.dumps(out) + '\n')
