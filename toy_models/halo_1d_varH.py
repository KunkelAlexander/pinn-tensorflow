"""Direction 4: (a) H sweep at fixed halo for cell-mean / B3 / fourier restriction (+mean force);
(b) spatially varying resolution: rho = w*rho + (1-w)*rho, near part (|x|<rw, smooth taper) solved at fine h (spectral),
far part solved with coarse cell-mean fd2 at H; mean-force subtraction on the total. Metric as halo_1d_stages.py."""
import json, sys
import numpy as np
import halo_1d as hd
from halo_1d import x, h, N, EDGES, profile, evolve, solve, build_psi
from halo_1d_meanforce import with_meanforce
from halo_1d_stages import make
from halo_1d_filters import make_filt
from soliton_gravity import OUT

def composite(R, rw, wid=2., far=None):
    far = far or make(R, 'mean', 'fd2', 'lag4')
    w = np.clip(.5 * (1 + np.cos(np.pi * (np.abs(x) - rw) / wid)), 0, 1)
    w[np.abs(x) <= rw] = 1; w[np.abs(x) >= rw + wid] = 0
    def g(rho):
        return hd.spec(rho * w) + far(rho * (1 - w))
    return g

if __name__ == '__main__':
    seeds = [1, 2]; tend, tavg, dt = 60., 20., .005
    sol = solve(32., 4., 8., verbose=False); every = round(.25 / dt)
    res = {}
    for seed in seeds:
        psi0 = build_psi(sol, seed, 32.)
        jobs = {'spectral': hd.spec}
        for R in [8, 16, 32, 64]:
            H = R * h
            jobs[f'H={H:.2f} mean'] = make(R, 'mean', 'fd2', 'lag4')
            jobs[f'H={H:.2f} mean+mf'] = with_meanforce(make(R, 'mean', 'fd2', 'lag4'))
            jobs[f'H={H:.2f} B3'] = make_filt(R, 3, 0)
            jobs[f'H={H:.2f} B3+mf'] = with_meanforce(make_filt(R, 3, 0))
            jobs[f'H={H:.2f} fourier'] = make(R, 'fourier', 'fd2', 'fourier')
        for R in [16, 32]:
            for rw in [2., 4., 6.]:
                jobs[f'H={R*h:.2f} comp rw={rw:g}+mf'] = with_meanforce(composite(R, rw))
                jobs[f'H={R*h:.2f} comp rw={rw:g}'] = composite(R, rw)
        res[seed] = {}
        for name, g in jobs.items():
            sn = evolve(psi0, g, tend, dt, every); ts = np.arange(len(sn)) * every * dt
            prof = np.mean([profile(s) for s in sn[ts >= tend - tavg]], axis=0)
            cen = np.array([(s * x).sum() / s.sum() for s in sn])
            res[seed][name] = dict(profile=prof.tolist(), maxcen=float(np.abs(cen).max()))
            print(f'seed {seed} {name:30s} drift {np.abs(cen).max():.3f}', flush=True)
    zones = {'core': slice(0, 4), 'mid': slice(8, 16), 'outer': slice(16, len(EDGES) - 1)}
    print('\nzone error vs spectral')
    for n in [n for n in res[seeds[0]] if n != 'spectral']:
        row = []
        for z, s in zones.items():
            e = np.mean([np.sum(np.array(res[sd][n]['profile'])[s]) / np.sum(np.array(res[sd]['spectral']['profile'])[s]) - 1 for sd in seeds])
            row.append(f'{z} {e:+.1%}')
        print(f'{n:30s} ' + '  '.join(row) + f"  drift {np.mean([res[sd][n]['maxcen'] for sd in seeds]):.3f}")
    (OUT / 'halo_1d_varH.json').write_text(json.dumps(res) + '\n')
