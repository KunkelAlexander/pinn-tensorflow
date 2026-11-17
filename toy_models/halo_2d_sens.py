"""Sensitivity study, 2D compact halo C, wave and gravity on the SAME grid (n points), vs the n=256 exact-wave/spectral-gravity reference.
Wave kinetic symbol W in {0 exact(proxy for high-order GRAM-FE), 2, 4, 6 = finite-difference dispersion}; gravity symbol G in {0 spectral, 2, 4, 6}
(= converged relaxation of that Laplacian stencil on a periodic box); grids n; time step dt. One json per job (resumable).
Usage: halo_2d_sens.py   (runs all jobs with a process pool)"""
import json, os, sys, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import stationary_halo as sh
from soliton_gravity import L, OUT
import halo_2d_ppw as hp

def ksym(k, h, order):
    c1, c2, c3 = np.cos(k * h), np.cos(2 * k * h), np.cos(3 * k * h)
    return {0: k ** 2, 2: (2 - 2 * c1) / h ** 2, 4: (30 - 32 * c1 + 2 * c2) / (12 * h ** 2),
            6: (490 - 540 * c1 + 54 * c2 - 4 * c3) / (180 * h ** 2)}[order]

def job(args):
    seed, W, G, n, dt = args; name = f'{OUT}/sens/s{seed}_W{W}_G{G}_n{n}_dt{dt}.json'
    if os.path.exists(name): return name
    psi0 = np.load(f'{OUT}/sens/psi0_s{seed}.npy'); psi = psi0 if n == 256 else hp.crop(psi0, n)
    x, h, xx, yy, k = hp.setup(n)
    kw = ksym(k, h, W); kg = ksym(k, h, G); den = kg[:, None] + kg[None, :]
    g = np.zeros_like(den); g[den > 1e-12] = -1 / den[den > 1e-12]
    grav = lambda rho: np.fft.ifft2(np.fft.fft2(rho) * g).real
    dr = np.exp(-.5j * dt * (kw[:, None] + kw[None, :])); w = psi.copy(); tend, tavg, every = 40., 15., round(.5 / dt); out = []; steps = round(tend / dt)
    for j in range(steps + 1):
        if j % every == 0: out.append(abs(w) ** 2)
        if j == steps: break
        w = w * np.exp(-.5j * dt * grav(abs(w) ** 2)); w = np.fft.ifft2(np.fft.fft2(w) * dr); w = w * np.exp(-.5j * dt * grav(abs(w) ** 2))
    ps = []
    for rho in out:
        cx, cy = (rho * xx).sum() / rho.sum(), (rho * yy).sum() / rho.sum(); r = np.hypot(xx - cx, yy - cy)
        idx = np.digitize(r.ravel(), hp.EDGES) - 1; ok = (idx >= 0) & (idx < 24)
        ps.append((np.bincount(idx[ok], weights=rho.ravel()[ok], minlength=24) / np.maximum(np.bincount(idx[ok], minlength=24), 1), np.hypot(cx, cy)))
    ts = np.arange(len(out)) * every * dt
    prof = np.mean([p for (p, c), t in zip(ps, ts) if t >= tend - tavg], axis=0)
    json.dump(dict(profile=prof.tolist(), drift=float(max(c for p, c in ps))), open(name, 'w')); return name

if __name__ == '__main__':
    os.makedirs(f'{OUT}/sens', exist_ok=True); mass = 80.; n0 = 256
    x, h, xx, yy, k = hp.setup(n0)
    for seed in (1, 2):
        if not os.path.exists(f'{OUT}/sens/psi0_s{seed}.npy'):
            sol = sh.solve(n0, mass, 2., 6., verbose=False)
            psi0, _, _ = sh.build_psi(sol, n0, seed); psi0 = sh.centre_and_rest(psi0, xx, yy); psi0 *= np.sqrt(mass / (np.sum(abs(psi0) ** 2) * h * h))
            np.save(f'{OUT}/sens/psi0_s{seed}.npy', psi0)
    jobs = [(s, W, G, n, .01) for n in (256, 192, 128, 96) for W in (0, 2, 4, 6) for G in (0, 2, 4, 6) for s in (1, 2) if not (W == 0 and G == 0 and n == 256)]
    jobs += [(s, 0, 0, 128, dt) for dt in (.005, .02, .04) for s in (1, 2)]
    jobs.sort(key=lambda a: -a[3])   # big grids first
    with ProcessPoolExecutor(7) as ex:
        for nm in ex.map(job, jobs): print('done', os.path.basename(nm), flush=True)
