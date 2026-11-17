"""Self-consistent test with exact (Fourier) wave solver but gravity = periodic Poisson inverted with the symbol of a
finite-difference Laplacian (fd2: 5-point, fd4: 9-point), i.e. what a converged relaxation (SOR/MG) gives on the same grid.
Reference: n=256 spectral from halo_2d_ppw_s{seed}.json. Usage: halo_2d_ppw_grav.py <seed>"""
import json, sys
import numpy as np
import stationary_halo as sh
from soliton_gravity import Gravity, L, OUT
import halo_2d_ppw as hp
import halo_2d_ppw_fd as fd

def fdgrav(n, order):
    x, h, xx, yy, k = hp.setup(n); den = fd.ksym(k, h, order)[:, None] + fd.ksym(k, h, order)[None, :]
    g = np.zeros_like(den); g[den > 1e-12] = -1 / den[den > 1e-12]
    return lambda rho: np.fft.ifft2(np.fft.fft2(rho) * g).real

def run(psi, n, grav, tend=40., tavg=15., dt=.01, every=50):
    x, h, xx, yy, k = hp.setup(n); kx, ky = k[:, None], k[None, :]
    dr = np.exp(-.5j * dt * (kx ** 2 + ky ** 2)); w = psi.copy(); out = []; steps = round(tend / dt)
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
    return np.mean([p for (p, c), t in zip(ps, ts) if t >= tend - tavg], axis=0), max(c for p, c in ps)

if __name__ == '__main__':
    seed = int(sys.argv[1]); mass = 80.; n0 = 256
    x, h, xx, yy, k = hp.setup(n0)
    rr = np.random.default_rng(0).random((64, 64)); assert np.allclose(fdgrav(64, 0)(rr), Gravity(64, 'spectral')(rr)), 'normalisation mismatch'
    sol = sh.solve(n0, mass, 2., 6., verbose=False)
    psi0, _, _ = sh.build_psi(sol, n0, seed); psi0 = sh.centre_and_rest(psi0, xx, yy); psi0 *= np.sqrt(mass / (np.sum(abs(psi0) ** 2) * h * h))
    ref = np.array(json.load(open(f'{OUT}/halo_2d_ppw_s{seed}.json'))['256']['profile'])
    Z = {'core': (0, 2), 'mid': (4, 10), 'outer': (10, 24)}; res = {}
    for order in [2, 4]:
        for n in [256, 192, 128, 96]:
            prof, drift = run(psi0 if n == n0 else hp.crop(psi0, n), n, fdgrav(n, order))
            res[f'o{order}_n{n}'] = dict(profile=prof.tolist(), drift=float(drift))
            print(f'seed {seed} gravity fd{order} n={n} ppw(1.1)={1.1*n/L:.1f}  ' + '  '.join(f'{z} {(prof[a:b].sum()/ref[a:b].sum()-1)*100:+.1f}%' for z, (a, b) in Z.items()), flush=True)
    json.dump(res, open(f'{OUT}/halo_2d_ppw_grav_s{seed}.json', 'w'))
