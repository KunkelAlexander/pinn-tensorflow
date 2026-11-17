"""Self-consistent resolution test (2D compact halo C): wave solver AND spectral gravity on the SAME grid of n points,
vs the n=256 reference. Same state (Fourier-cropped). Answers: how many points per wavelength does spectral gravity need
when the grid is also the wave grid? Usage: halo_2d_ppw.py <seed>"""
import json, sys
import numpy as np
import stationary_halo as sh
from soliton_gravity import Gravity, grid, L, OUT

EDGES = np.linspace(0, 12, 25)
def setup(n):
    x = grid(n); h = L / n; xx, yy = np.meshgrid(x, x, indexing='ij'); k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    return x, h, xx, yy, k

def crop(psi, ns):
    n = psi.shape[0]; f = np.fft.fftshift(np.fft.fft2(psi)); a = (n - ns) // 2
    return np.fft.ifft2(np.fft.ifftshift(f[a:a + ns, a:a + ns])) * (ns / n) ** 2

def run(psi, n, tend=40., tavg=15., dt=.01, every=50):
    x, h, xx, yy, k = setup(n); kx, ky = k[:, None], k[None, :]; g = Gravity(n, 'spectral')
    dr = np.exp(-.5j * dt * (kx ** 2 + ky ** 2)); w = psi.copy(); out = []; steps = round(tend / dt)
    for j in range(steps + 1):
        if j % every == 0: out.append(abs(w) ** 2)
        if j == steps: break
        w = w * np.exp(-.5j * dt * g(abs(w) ** 2)); w = np.fft.ifft2(np.fft.fft2(w) * dr); w = w * np.exp(-.5j * dt * g(abs(w) ** 2))
    ps = []
    for rho in out:
        cx, cy = (rho * xx).sum() / rho.sum(), (rho * yy).sum() / rho.sum(); r = np.hypot(xx - cx, yy - cy)
        idx = np.digitize(r.ravel(), EDGES) - 1; ok = (idx >= 0) & (idx < 24)
        ps.append((np.bincount(idx[ok], weights=rho.ravel()[ok], minlength=24) / np.maximum(np.bincount(idx[ok], minlength=24), 1), np.hypot(cx, cy)))
    ts = np.arange(len(out)) * every * dt
    return np.mean([p for (p, c), t in zip(ps, ts) if t >= tend - tavg], axis=0), max(c for p, c in ps)

if __name__ == '__main__':
    seed = int(sys.argv[1]); mass = 80.; n0 = 256
    x, h, xx, yy, k = setup(n0)
    sol = sh.solve(n0, mass, 2., 6., verbose=False)
    psi0, _, _ = sh.build_psi(sol, n0, seed); psi0 = sh.centre_and_rest(psi0, xx, yy); psi0 *= np.sqrt(mass / (np.sum(abs(psi0) ** 2) * h * h))
    P = abs(np.fft.fft2(psi0)) ** 2; kr = np.hypot(k[:, None], k[None, :])
    o = np.argsort(kr.ravel()); c = np.cumsum(P.ravel()[o]) / P.sum()
    for q in (.5, .9, .99, .999):
        kq = kr.ravel()[o][np.searchsorted(c, q)]; print(f'psi power: {q*100:g}% below k={kq:.2f} (lambda={2*np.pi/kq:.2f})', flush=True)
    res = {}
    for n in [256, 128, 96, 64, 48]:
        psi = psi0 if n == n0 else crop(psi0, n)
        prof, drift = run(psi, n); res[n] = dict(profile=prof.tolist(), drift=float(drift)); print(f'seed {seed} n={n} h={L/n:.3f} drift {drift:.3f}', flush=True)
    json.dump(res, open(f'{OUT}/halo_2d_ppw_s{seed}.json', 'w'))
