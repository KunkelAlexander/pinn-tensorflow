"""Research directions 3/4 screen: which stage of the coarse Poisson solver creates the halo error?

Stages (coarse cell H = R*h):  restriction  {mean: cell mean, fourier: band-limited sampling at coarse centres}
                               parent solve {fd2: 3-point stencil, spec: exact Fourier}
                               prolongation {lag4: quartic Lagrange, fourier: zero-padded FFT interpolation}
Fourier/spec/fourier is translation covariant (a pure low-pass of the force) -> expect no drift, only softening.
Each variant is run plain and with mean-force subtraction. Metric as in halo_1d_meanforce.py.
"""
import argparse
import json
import numpy as np
import halo_1d as hd
from halo_1d import x, h, N, k, L, EDGES, profile, evolve, solve, build_psi
from halo_1d_meanforce import with_meanforce
from soliton_gravity import interp_matrix, OUT


def make(R, restr, parent, prol):
    m = N // R
    H = L / m
    xc = -L / 2 + (np.arange(m) + .5) * H
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H)
    g2 = np.zeros(m); g2[1:] = -H ** 2 / (4 * np.sin(kc[1:] * H / 2) ** 2)
    gs = np.zeros(m); gs[1:] = -1 / kc[1:] ** 2
    gc = g2 if parent == 'fd2' else gs
    T = interp_matrix(xc, x, 4)
    shift_c = np.exp(1j * kc * (xc[0] - x[0]))        # not used for mean-restriction; see below
    kf = k
    lo = np.abs(kf) < np.pi / H - 1e-9                # coarse band (drop Nyquist)
    idx = np.round(kf[lo] / (2 * np.pi / L)).astype(int) % m
    idxf = np.round(kf[lo] / (2 * np.pi / L)).astype(int) % N

    def grav(rho):
        if restr == 'mean':
            rc = rho.reshape(m, R).mean(1)
            Fc = np.fft.fft(rc) * gc
        else:                                         # Fourier: band-limit fine rho, sample at coarse grid points
            F = np.fft.fft(rho)
            Fc = np.zeros(m, complex)
            Fc[idx] = F[idxf] * (m / N)
            Fc *= gc
            # coarse grid points sit at xc = x[R//2 - ...]; fine centre offset handled by phase below
            Fc *= np.exp(-1j * kc * (xc[0] - x[0]) * 0)  # both grids cell-centred over same box: offsets differ by (R-1)h/2
            Fc *= np.exp(1j * kc * ((R - 1) * h / 2) * 0)
        if prol == 'lag4':
            return T @ np.fft.ifft(Fc).real
        G = np.zeros(N, complex)
        G[idxf] = Fc[idx] * (N / m)
        return np.fft.ifft(G).real
    return grav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mass', type=float, default=32.)
    ap.add_argument('--T', type=float, default=4.)
    ap.add_argument('--rc', type=float, default=8.)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--tend', type=float, default=60.)
    ap.add_argument('--tavg', type=float, default=20.)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--Rs', type=int, nargs='+', default=[16, 32])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    sol = solve(a.mass, a.T, a.rc, verbose=False)
    every = round(.25 / a.dt)
    combos = [('mean', 'fd2', 'lag4'), ('mean', 'spec', 'lag4'), ('mean', 'spec', 'fourier'),
              ('fourier', 'fd2', 'fourier'), ('fourier', 'spec', 'fourier')]
    res = {}
    for seed in a.seeds:
        psi0 = build_psi(sol, seed, a.mass)
        jobs = {'spectral': hd.spec}
        for R in a.Rs:
            for c in combos:
                g = make(R, *c)
                nm = f'H={R * h:.2f} {c[0]}/{c[1]}/{c[2]}'
                jobs[nm] = g
                jobs[nm + ' +mf'] = with_meanforce(g)
        res[seed] = {}
        for name, g in jobs.items():
            sn = evolve(psi0, g, a.tend, a.dt, every)
            ts = np.arange(len(sn)) * every * a.dt
            prof = np.mean([profile(s) for s in sn[ts >= a.tend - a.tavg]], axis=0)
            cen = np.array([(s * x).sum() / s.sum() for s in sn])
            res[seed][name] = dict(profile=prof.tolist(), maxcen=float(np.abs(cen).max()))
            print(f'seed {seed} {name:34s} drift {np.abs(cen).max():.3f}', flush=True)
    nb = len(EDGES) - 1
    zones = {'core': slice(0, 4), 'mid': slice(8, 16), 'outer': slice(16, nb)}
    print('\nzone-summed profile error vs spectral (mean over seeds)')
    for n in [n for n in res[a.seeds[0]] if n != 'spectral']:
        row = []
        for z, s in zones.items():
            e = np.mean([np.sum(np.array(res[sd][n]['profile'])[s]) / np.sum(np.array(res[sd]['spectral']['profile'])[s]) - 1
                         for sd in a.seeds])
            row.append(f'{z} {e:+.1%}')
        print(f'{n:34s} ' + '  '.join(row) + f"  drift {np.mean([res[sd][n]['maxcen'] for sd in a.seeds]):.3f}")
    (OUT / f'halo_1d_stages{a.tag}.json').write_text(json.dumps(res) + '\n')


if __name__ == '__main__':
    main()
