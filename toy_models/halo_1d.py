"""1D stationary halo: spectral vs coarse-fd2 Poisson, time-averaged log-r density profile (the 3D mid/outer-halo diagnostic).

Halo = random-phase superposition of eigenstates of its own mean-field potential (lowered isothermal occupations), as in
stationary_halo.py but in 1D (hbar=m=G=1, periodic box, lap phi = rho - mean). Wave solver = exact Fourier Strang for every
run; only the Poisson routine differs. Gravity kinds: spectral, coarse rR (cell-mean restriction, fd2 parent solve, quartic Lagrange).
Noise floor = spectral run with a 1e-6 perturbation of psi0 (the chaos-limited difference of two *equally exact* runs).
"""
import argparse
import json
import numpy as np
from scipy.linalg import eigh, circulant
from soliton_gravity import interp_matrix, OUT

import os
L, N = float(os.environ.get('HALO1D_L', 32.)), int(os.environ.get('HALO1D_N', 1024))
h = L / N
x = -L / 2 + (np.arange(N) + .5) * h
k = 2 * np.pi * np.fft.fftfreq(N, d=h)
gsp = np.zeros(N)
gsp[1:] = -1 / k[1:] ** 2
Hkin = circulant(np.fft.ifft(.5 * k ** 2).real)      # exact Fourier kinetic operator as a matrix


def spec(rho):
    return np.fft.ifft(np.fft.fft(rho) * gsp).real


def make_grav(kind, r=1):
    if kind == 'spectral' or r == 1:
        return spec
    m = N // r
    H = L / m
    xc = -L / 2 + (np.arange(m) + .5) * H
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H)
    gc = np.zeros(m)
    gc[1:] = -H ** 2 / (4 * np.sin(kc[1:] * H / 2) ** 2)
    T = interp_matrix(xc, x, 4)
    return lambda rho: T @ np.fft.ifft(np.fft.fft(rho.reshape(m, r).mean(1)) * gc).real


def solve(mass, T, rc, iters=200, mix=.4, verbose=True, Ne=1024):
    """Self-consistent lowered-isothermal solve on an internal grid of Ne<=N points (eigenstates are smooth), then
    Fourier-interpolate the eigenstates to the N-point wave grid."""
    Ne = min(Ne, N)
    he = L / Ne
    xe = -L / 2 + (np.arange(Ne) + .5) * he
    ke = 2 * np.pi * np.fft.fftfreq(Ne, d=he)
    ge = np.zeros(Ne)
    ge[1:] = -1 / ke[1:] ** 2
    He = circulant(np.fft.ifft(.5 * ke ** 2).real)
    rho = np.exp(-xe ** 2 / (2 * 2. ** 2))
    rho *= mass / (rho.sum() * he)
    for it in range(iters):
        Phi = np.fft.ifft(np.fft.fft(rho) * ge).real
        Ec = np.interp(rc, xe[Ne // 2:], Phi[Ne // 2:])
        E, v = eigh(He + np.diag(Phi), subset_by_value=(Phi.min() - 1., Ec))
        v = v / np.sqrt(he)
        E0 = E[0]
        w = np.maximum(np.exp(-(E - E0) / T) - np.exp(-(Ec - E0) / T), 0.)
        c = mass / w.sum()
        new = (v ** 2) @ (c * w)
        res = np.linalg.norm(new - rho) / np.linalg.norm(new)
        rho = (1 - mix) * rho + mix * new
        if verbose and (it % 10 == 0 or res < 1e-8):
            print(f'  iter {it:3d} residual {res:.2e} peak {rho.max():.3f} states {len(E)}', flush=True)
        if res < 1e-8:
            break
    if Ne < N:                                       # band-limited (Fourier) interpolation of each eigenstate to the wave grid
        kf = 2 * np.pi * np.fft.fftfreq(N, d=h)
        c = np.fft.fft(v, axis=0) * np.exp(1j * ke * (x[0] - xe[0]))[:, None]
        F = np.zeros((N, v.shape[1]), complex)
        lo = np.abs(ke) < np.pi / he - 1e-9           # drop the Nyquist term
        idx = np.round(ke[lo] / (2 * np.pi / L)).astype(int) % N
        F[idx] = c[lo]
        v = (np.fft.ifft(F, axis=0) * N / Ne).real
        v /= np.sqrt((v ** 2).sum(0) * h)
        rho = (v ** 2) @ (mass / w.sum() * w)
        Phi = spec(rho)
    return dict(E=E, v=v, f=mass / w.sum() * w, rho=rho, Phi=Phi, Ec=Ec, res=res)


def build_psi(sol, seed, mass):
    rng = np.random.default_rng(seed)
    psi = sol['v'] @ (np.sqrt(sol['f']) * np.exp(2j * np.pi * rng.random(len(sol['E']))))
    for _ in range(3):                               # Galilean-boost away net momentum, recentre
        f = np.fft.fft(psi)
        p = abs(f) ** 2
        psi = psi * np.exp(-1j * (k * p).sum() / p.sum() * x)
        rho = abs(psi) ** 2
        psi = np.fft.ifft(np.fft.fft(psi) * np.exp(1j * k * (rho * x).sum() / rho.sum()))
    return psi * np.sqrt(mass / (np.sum(abs(psi) ** 2) * h))


def evolve(w, grav, T, dt, every):
    dr = np.exp(-.5j * dt * k ** 2)
    snaps = []
    for j in range(round(T / dt) + 1):
        if j % every == 0:
            snaps.append(abs(w) ** 2)
        if j == round(T / dt):
            break
        w = w * np.exp(-.5j * dt * grav(abs(w) ** 2))
        w = np.fft.ifft(np.fft.fft(w) * dr)
        w = w * np.exp(-.5j * dt * grav(abs(w) ** 2))
    return np.array(snaps)


EDGES = np.concatenate([[0.], np.geomspace(max(.3, 8 * h), L / 2 - .1, 25 if L == 32 else 33)])   # every bin >= one fine cell wide


def profile(rho):
    """Centroid-following, +/- x symmetrised density binned in |x| (log-spaced bins; fine cells assigned by centre)."""
    c = (rho * x).sum() / rho.sum()
    r = abs(x - c)
    idx = np.digitize(r, EDGES) - 1
    ok = (idx >= 0) & (idx < len(EDGES) - 1)
    num = np.bincount(idx[ok], weights=rho[ok], minlength=len(EDGES) - 1)
    cnt = np.bincount(idx[ok], minlength=len(EDGES) - 1)
    return num / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mass', type=float, default=32.)
    ap.add_argument('--T', type=float, default=4.)
    ap.add_argument('--rc', type=float, default=8.)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--tend', type=float, default=60.)
    ap.add_argument('--tavg', type=float, default=20.)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--Hs', type=int, nargs='+', default=[4, 8, 16, 32], help='coarse cell = this many fine... in units of fine cells')
    ap.add_argument('--tag', default='')
    ap.add_argument('--solve_only', action='store_true')
    a = ap.parse_args()
    sol = solve(a.mass, a.T, a.rc)
    rho = sol['rho']
    rms = np.sqrt((rho * x ** 2).sum() * h / a.mass)
    print(f"converged {sol['res']:.1e}; {len(sol['E'])} states; peak {rho.max():.3f}; rms {rms:.2f}; Ec {sol['Ec']:.2f}")
    # granule scale: 1/sqrt-ish; local de Broglie wavelength at r from  k_loc^2/2 = <E>-Phi
    xm = x[N // 2:][np.argmax(sol['Phi'][N // 2:])]
    print(f"Phi max at |x|={xm:.2f} (need r_c < this), edge density {rho[0]:.2e}, mean {a.mass / L:.2f}")
    print('density at |x| = 0,1,2,4,6,8:', [f'{np.interp(r, x[N // 2:], rho[N // 2:]):.3e}' for r in (0, 1, 2, 4, 6, 8)])
    if a.solve_only:
        return
    grs = {'spectral': make_grav('spectral')}
    for R in a.Hs:
        grs[f'coarse H={R * h:.3f}'] = make_grav('coarse', R)
    every = round(.25 / a.dt)
    out = dict(args=vars(a), edges=EDGES.tolist(), rho_target=rho.tolist(), x=x.tolist(), runs={})
    for seed in a.seeds:
        psi0 = build_psi(sol, seed, a.mass)
        pert = psi0 * (1 + 1e-6 * np.random.default_rng(100 + seed).standard_normal(N))
        runs = {'spectral': (psi0, grs['spectral']), 'spectral (perturbed)': (pert, grs['spectral'])}
        for name, g in grs.items():
            if name != 'spectral':
                runs[name] = (psi0, g)
        out['runs'][seed] = {}
        for name, (w, g) in runs.items():
            sn = evolve(w, g, a.tend, a.dt, every)
            ts = np.arange(len(sn)) * every * a.dt
            sel = ts >= a.tend - a.tavg
            prof = np.mean([profile(s) for s in sn[sel]], axis=0)
            cen = [float((s * x).sum() / s.sum()) for s in sn]
            out['runs'][seed][name] = dict(profile=prof.tolist(), centroid=cen, peak=[float(s.max()) for s in sn], times=ts.tolist())
            print(f'seed {seed} {name:24s} core {prof[:4].mean():.3f}  max|centroid| {np.abs(cen).max():.3f}', flush=True)
        (OUT / f'halo_1d{a.tag}.json').write_text(json.dumps(out) + '\n')


if __name__ == '__main__':
    main()
