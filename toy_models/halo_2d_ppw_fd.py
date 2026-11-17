"""Same self-consistent test as halo_2d_ppw.py but the kinetic operator has the dispersion of a finite-difference stencil
(order 2 or 4; time stepping stays exact via exp of the modified symbol). Gravity spectral on the same grid.
Reference: n=256 spectral from halo_2d_ppw_s{seed}.json. Usage: halo_2d_ppw_fd.py <seed>"""
import json, sys
import numpy as np
import stationary_halo as sh
from soliton_gravity import Gravity, L, OUT
import halo_2d_ppw as hp

def ksym(k, h, order):
    if order == 0: return k ** 2
    if order == 2: return (2 / h * np.sin(k * h / 2)) ** 2
    if order == 4: return (30 - 32 * np.cos(k * h) + 2 * np.cos(2 * k * h)) / (12 * h ** 2)

def run(psi, n, order, tend=40., tavg=15., dt=.01, every=50):
    x, h, xx, yy, k = hp.setup(n); g = Gravity(n, 'spectral')
    dr = np.exp(-.5j * dt * (ksym(k, h, order)[:, None] + ksym(k, h, order)[None, :])); w = psi.copy(); out = []; steps = round(tend / dt)
    for j in range(steps + 1):
        if j % every == 0: out.append(abs(w) ** 2)
        if j == steps: break
        w = w * np.exp(-.5j * dt * g(abs(w) ** 2)); w = np.fft.ifft2(np.fft.fft2(w) * dr); w = w * np.exp(-.5j * dt * g(abs(w) ** 2))
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
    sol = sh.solve(n0, mass, 2., 6., verbose=False)
    psi0, _, _ = sh.build_psi(sol, n0, seed); psi0 = sh.centre_and_rest(psi0, xx, yy); psi0 *= np.sqrt(mass / (np.sum(abs(psi0) ** 2) * h * h))
    ref = np.array(json.load(open(f'{OUT}/halo_2d_ppw_s{seed}.json'))['256']['profile'])
    Z = {'core': (0, 2), 'mid': (4, 10), 'outer': (10, 24)}; res = {}
    for order in [2, 4]:
        for n in [192, 128, 96]:
            prof, drift = run(hp.crop(psi0, n), n, order)
            res[f'o{order}_n{n}'] = dict(profile=prof.tolist(), drift=float(drift))
            print(f'seed {seed} order {order} n={n} ppw(1.1)={1.1*n/L:.1f}  ' + '  '.join(f'{z} {(prof[a:b].sum()/ref[a:b].sum()-1)*100:+.1f}%' for z, (a, b) in Z.items()), flush=True)
    json.dump(res, open(f'{OUT}/halo_2d_ppw_fd_s{seed}.json', 'w'))
