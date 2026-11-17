"""2D check of the 1D conclusions: coarse Poisson (cell-mean or cubic-B-spline restriction, fd2 parent solve, quartic Lagrange
prolongation, each +/- mean-force subtraction) vs spectral on a stationary 2D halo. Exact Fourier wave solver for all runs.
Metric: centroid-following radial density profile, zone sums core r<1, mid 2<r<5, outer 5<r<12, ratio to spectral; max centroid drift."""
import json, sys
import numpy as np
from scipy.interpolate import BSpline
import stationary_halo as sh
from soliton_gravity import Gravity, grid, interp_matrix, L, OUT

n = 256; h = L / n; x = grid(n)
xx, yy = np.meshgrid(x, x, indexing='ij')
k = 2 * np.pi * np.fft.fftfreq(n, d=h)
kx, ky = k[:, None], k[None, :]
spec = Gravity(n, 'spectral')

def restr_matrix(R, d):
    m = n // R; H = L / m; xc = -L / 2 + (np.arange(m) + .5) * H
    Rm = np.zeros((m, n))
    if d == 0:
        for i in range(m): Rm[i, i * R:(i + 1) * R] = 1 / R
    else:
        b = BSpline.basis_element(np.arange(d + 2) - (d + 1) / 2, extrapolate=False)
        for i in range(m):
            dx = (x - xc[i] + L / 2) % L - L / 2
            w = np.nan_to_num(b(dx / H)); Rm[i] = w / w.sum()
    return Rm, xc, H

def make(R, d):
    Rm, xc, H = restr_matrix(R, d); m = n // R
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H)
    den = 4 / H ** 2 * (np.sin(kc[:, None] * H / 2) ** 2 + np.sin(kc[None, :] * H / 2) ** 2)
    g = np.zeros_like(den); g[den > 0] = -1 / den[den > 0]
    T = interp_matrix(xc, x, 4)
    def grav(rho):
        rc = Rm @ rho @ Rm.T
        pc = np.fft.ifft2(np.fft.fft2(rc) * g).real
        return T @ pc @ T.T
    return grav

def with_mf(grav):
    def g(rho):
        phi = grav(rho)
        F = np.fft.fft2(phi)
        dx = np.fft.ifft2(1j * kx * F).real; dy = np.fft.ifft2(1j * ky * F).real
        M = rho.sum()
        return phi + (-(rho * dx).sum() / M) * xx + (-(rho * dy).sum() / M) * yy
    return g

EDGES = np.linspace(0, 12, 25)
def profile(rho):
    cx, cy = (rho * xx).sum() / rho.sum(), (rho * yy).sum() / rho.sum()
    r = np.hypot(xx - cx, yy - cy)
    idx = np.digitize(r.ravel(), EDGES) - 1; ok = (idx >= 0) & (idx < len(EDGES) - 1)
    return np.bincount(idx[ok], weights=rho.ravel()[ok], minlength=len(EDGES) - 1) / np.maximum(np.bincount(idx[ok], minlength=len(EDGES) - 1), 1), (cx, cy)

def evolve(w, grav, tend, dt, every):
    dr = np.exp(-.5j * dt * (kx ** 2 + ky ** 2)); out = []
    steps = round(tend / dt)
    for j in range(steps + 1):
        if j % every == 0: out.append(abs(w) ** 2)
        if j == steps: break
        w = w * np.exp(-.5j * dt * grav(abs(w) ** 2)); w = np.fft.ifft2(np.fft.fft2(w) * dr); w = w * np.exp(-.5j * dt * grav(abs(w) ** 2))
    return out

if __name__ == '__main__':
    Tt, rc, mass = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); tag = sys.argv[4]
    tend, tavg, dt = 40., 15., .01; every = round(.5 / dt); seeds = [1, 2]
    sol = sh.solve(n, mass, Tt, rc, verbose=False)
    zones = {'core': (0, 2), 'mid': (4, 10), 'outer': (10, 24)}   # bin indices of EDGES (0.5 wide)
    res = {}
    for seed in seeds:
        psi0, _, _ = sh.build_psi(sol, n, seed); psi0 = sh.centre_and_rest(psi0, xx, yy)
        psi0 *= np.sqrt(mass / (np.sum(abs(psi0) ** 2) * h * h))
        pert = psi0 * (1 + 1e-6 * np.random.default_rng(100 + seed).standard_normal((n, n)))
        jobs = {'spectral': (psi0, spec), 'pert': (pert, spec)}
        for R in [4, 8, 16]:
            jobs[f'H={R*h:.1f} mean+mf'] = (psi0, with_mf(make(R, 0)))
            jobs[f'H={R*h:.1f} B3+mf'] = (psi0, with_mf(make(R, 3)))
        res[seed] = {}
        for name, (w, g) in jobs.items():
            sn = evolve(w, g, tend, dt, every); ts = np.arange(len(sn)) * every * dt
            ps = [profile(s) for s in sn]
            prof = np.mean([p for (p, c), t in zip(ps, ts) if t >= tend - tavg], axis=0)
            drift = max(np.hypot(*c) for p, c in ps)
            res[seed][name] = dict(profile=prof.tolist(), drift=float(drift))
            print(f'seed {seed} {name:18s} drift {drift:.3f}', flush=True)
    print('\nzone error vs spectral')
    for nm in [q for q in res[1] if q != 'spectral']:
        row = []
        for z, (a, b) in zones.items():
            e = np.array([np.sum(np.array(res[s][nm]['profile'])[a:b]) / np.sum(np.array(res[s]['spectral']['profile'])[a:b]) - 1 for s in seeds])
            row.append(f'{z} {e.mean():+.1%}')
        print(f'{nm:18s} ' + '  '.join(row) + f"  drift {np.mean([res[s][nm]['drift'] for s in seeds]):.3f}", flush=True)
    (OUT / f'halo_2d_b3{tag}.json').write_text(json.dumps(res) + '\n')
