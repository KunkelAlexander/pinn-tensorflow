"""Direction 3 (AMR-feasible): local anti-alias restriction. Restriction = B-spline kernel of degree d (width (d+1)H, d=0 is the
cell mean) centred on each coarse cell, row-normalised; optional coarse-grid deconvolution by sinc^(d+1) (gain capped);
parent solve fd2; prolongation quartic Lagrange. Compare to the Fourier/fd2/Fourier reference. Metric as halo_1d_stages.py."""
import argparse, json
import numpy as np
from scipy.interpolate import BSpline
import halo_1d as hd
from halo_1d import x, h, N, L, EDGES, profile, evolve, solve, build_psi
from halo_1d_meanforce import with_meanforce
from halo_1d_stages import make
from soliton_gravity import interp_matrix, OUT


def make_filt(R, d, decon, cap=4.):
    m = N // R; H = L / m
    xc = -L / 2 + (np.arange(m) + .5) * H
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H)
    gc = np.zeros(m); gc[1:] = -H ** 2 / (4 * np.sin(kc[1:] * H / 2) ** 2)
    T = interp_matrix(xc, x, 4)
    if d == 0:
        Rm = np.zeros((m, N))
        for i in range(m): Rm[i, i * R:(i + 1) * R] = 1 / R
    else:
        knots = np.arange(d + 2) - (d + 1) / 2
        b = BSpline.basis_element(knots, extrapolate=False)
        Rm = np.zeros((m, N))
        for i in range(m):
            dx = (x - xc[i] + L / 2) % L - L / 2
            w = np.nan_to_num(b(dx / H)); Rm[i] = w / w.sum()
    s = np.sinc(kc * H / (2 * np.pi)) ** (d + 1)
    dec = np.minimum(1 / np.maximum(s, 1e-9), cap) if decon else np.ones(m)
    def grav(rho):
        Fc = np.fft.fft(Rm @ rho) * dec * gc
        return T @ np.fft.ifft(Fc).real
    return grav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--Rs', type=int, nargs='+', default=[16, 32])
    ap.add_argument('--tend', type=float, default=60.); ap.add_argument('--tavg', type=float, default=20.)
    ap.add_argument('--dt', type=float, default=.005)
    a = ap.parse_args()
    sol = solve(32., 4., 8., verbose=False); every = round(.25 / a.dt)
    res = {}
    for seed in a.seeds:
        psi0 = build_psi(sol, seed, 32.)
        jobs = {'spectral': hd.spec}
        for R in a.Rs:
            for d, dec in [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (3, 1)]:
                jobs[f'H={R*h:.2f} B{d}{"+dec" if dec else ""}'] = make_filt(R, d, dec)
            jobs[f'H={R*h:.2f} B3+dec+mf'] = with_meanforce(make_filt(R, 3, 1))
            jobs[f'H={R*h:.2f} fourier/fd2/fourier'] = make(R, 'fourier', 'fd2', 'fourier')
        res[seed] = {}
        for name, g in jobs.items():
            sn = evolve(psi0, g, a.tend, a.dt, every); ts = np.arange(len(sn)) * every * a.dt
            prof = np.mean([profile(s) for s in sn[ts >= a.tend - a.tavg]], axis=0)
            cen = np.array([(s * x).sum() / s.sum() for s in sn])
            res[seed][name] = dict(profile=prof.tolist(), maxcen=float(np.abs(cen).max()))
            print(f'seed {seed} {name:32s} drift {np.abs(cen).max():.3f}', flush=True)
    nb = len(EDGES) - 1; zones = {'core': slice(0, 4), 'mid': slice(8, 16), 'outer': slice(16, nb)}
    print('\nzone-summed profile error vs spectral')
    for n in [n for n in res[a.seeds[0]] if n != 'spectral']:
        row = [f'{z} {np.mean([np.sum(np.array(res[sd][n]["profile"])[s]) / np.sum(np.array(res[sd]["spectral"]["profile"])[s]) - 1 for sd in a.seeds]):+.1%}' for z, s in zones.items()]
        print(f'{n:32s} ' + '  '.join(row) + f"  drift {np.mean([res[sd][n]['maxcen'] for sd in a.seeds]):.3f}")
    (OUT / 'halo_1d_filters.json').write_text(json.dumps(res) + '\n')

if __name__ == '__main__':
    main()
