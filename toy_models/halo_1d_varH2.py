"""Direction 4 on an EXTENDED halo (mass in the envelope): global coarse vs composite (fine spectral core + coarse far part)."""
import json, sys
import numpy as np
import halo_1d as hd
from halo_1d import x, h, EDGES, profile, evolve, solve, build_psi
from halo_1d_meanforce import with_meanforce
from halo_1d_stages import make
from halo_1d_filters import make_filt
from halo_1d_varH import composite
from soliton_gravity import OUT
T, rc = float(sys.argv[1]), float(sys.argv[2]); tag = sys.argv[3]
seeds = [1, 2]; tend, tavg, dt = 60., 20., .005
sol = solve(32., T, rc, verbose=False); every = round(.25 / dt)
res = {}
for seed in seeds:
    psi0 = build_psi(sol, seed, 32.)
    pert = psi0 * (1 + 1e-6 * np.random.default_rng(100 + seed).standard_normal(len(psi0)))
    jobs = {'spectral': (psi0, hd.spec), 'pert': (pert, hd.spec)}
    for R in [16, 32, 64]:
        H = R * h
        jobs[f'H={H:.2f} mean+mf'] = (psi0, with_meanforce(make(R, 'mean', 'fd2', 'lag4')))
        jobs[f'H={H:.2f} B3+mf'] = (psi0, with_meanforce(make_filt(R, 3, 0)))
        for rw in [2., 4.]:
            jobs[f'H={H:.2f} comp-mean rw={rw:g}+mf'] = (psi0, with_meanforce(composite(R, rw)))
            jobs[f'H={H:.2f} comp-B3 rw={rw:g}+mf'] = (psi0, with_meanforce(composite(R, rw, far=make_filt(R, 3, 0))))
    res[seed] = {}
    for name, (w, g) in jobs.items():
        sn = evolve(w, g, tend, dt, every); ts = np.arange(len(sn)) * every * dt
        prof = np.mean([profile(s) for s in sn[ts >= tend - tavg]], axis=0)
        cen = np.array([(s * x).sum() / s.sum() for s in sn])
        res[seed][name] = dict(profile=prof.tolist(), maxcen=float(np.abs(cen).max()))
        print(f'seed {seed} {name:34s} drift {np.abs(cen).max():.3f}', flush=True)
zones = {'core': slice(0, 4), 'mid': slice(8, 16), 'outer': slice(16, len(EDGES) - 1)}
print('\nzone error vs spectral')
for n in [n for n in res[seeds[0]] if n != 'spectral']:
    row = []
    for z, s in zones.items():
        e = np.mean([np.sum(np.array(res[sd][n]['profile'])[s]) / np.sum(np.array(res[sd]['spectral']['profile'])[s]) - 1 for sd in seeds])
        row.append(f'{z} {e:+.1%}')
    print(f'{n:34s} ' + '  '.join(row) + f"  drift {np.mean([res[sd][n]['maxcen'] for sd in seeds]):.3f}")
(OUT / f'halo_1d_varH2{tag}.json').write_text(json.dumps(res) + '\n')
