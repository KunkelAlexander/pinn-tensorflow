"""Two-level gravity toy: wave on the fine grid (exact Fourier), gravity = parent level (cell-mean density, periodic FFT with the symbol of a
parent stencil, GAMER-like FD2 or FD4) + a fine patch around the core solved DIRECTLY (= converged relaxation) with Dirichlet BC from the
quadratically interpolated parent potential, fine stencil FD2 (GAMER SOR) or FD4 (needs 2 boundary layers). Outside the patch: interpolated parent.
Reference: n=256 exact/spectral (halo_2d_ppw_s*.json). Usage: halo_2d_patch.py"""
import json, os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from concurrent.futures import ProcessPoolExecutor
from soliton_gravity import L, OUT, interp_matrix
import halo_2d_ppw as hp
from halo_2d_sens import ksym

def d1(N, h, order):
    if order == 2: return sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N), format='csr') / h ** 2
    return sp.diags([-1, 16, -30, 16, -1], [-2, -1, 0, 1, 2], shape=(N, N), format='csr') / (12 * h ** 2)

def make_grav(n, parent, patch, hw):
    x, h, xx, yy, k = hp.setup(n); m = n // 2; H = 2 * h; xc = -L / 2 + (np.arange(m) + .5) * H
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H); kp = ksym(kc, H, parent); den = kp[:, None] + kp[None, :]
    g = np.zeros_like(den); g[den > 1e-12] = -1 / den[den > 1e-12]
    T = interp_matrix(xc, x, 2)
    P = int(round(2 * hw / h)); P += P % 2; a = n // 2 - P // 2; Ne = P + 4
    D = d1(Ne, h, patch); I = sp.identity(Ne, format='csr'); Lx = (sp.kron(D, I) + sp.kron(I, D)).tocsr()
    ii, jj = np.meshgrid(np.arange(2, P + 2), np.arange(2, P + 2), indexing='ij'); inn = (ii * Ne + jj).ravel()
    allidx = np.arange(Ne * Ne); known = np.setdiff1d(allidx, inn)
    A = Lx[inn][:, inn].tocsc(); Bk = Lx[inn][:, known].tocsr(); lu = spl.splu(A)
    sl = slice(a - 2, a + P + 2)
    def grav(rho):
        pc = np.fft.ifft2(np.fft.fft2(rho.reshape(m, 2, m, 2).mean((1, 3))) * g).real
        phi = T @ pc @ T.T
        rhs = (rho - rho.mean())[a:a + P, a:a + P].ravel() - Bk @ phi[sl, sl].ravel()[known]
        out = phi.copy(); out[a:a + P, a:a + P] = lu.solve(rhs).reshape(P, P); return out
    return grav

def job(args):
    seed, parent, patch, hw, n = args; dt = .01
    name = f'{OUT}/patch/s{seed}_Pa{parent}_Pt{patch}_hw{hw}_n{n}.json'
    if os.path.exists(name): return name
    psi = hp.crop(np.load(f'{OUT}/sens/psi0_s{seed}.npy'), n); x, h, xx, yy, k = hp.setup(n); grav = make_grav(n, parent, patch, hw)
    dr = np.exp(-.5j * dt * (k[:, None] ** 2 + k[None, :] ** 2)); w = psi.copy(); every = 50; out = []; steps = 4000
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
    prof = np.mean([p for (p, c), t in zip(ps, ts) if t >= 25], axis=0)
    json.dump(dict(profile=prof.tolist(), drift=float(max(c for p, c in ps))), open(name, 'w')); return name

if __name__ == '__main__':
    os.makedirs(f'{OUT}/patch', exist_ok=True)
    jobs = [(s, pa, pt, hw, n) for n in (192, 128) for hw in (4, 8) for pa, pt in ((2, 2), (2, 4), (4, 4)) for s in (1, 2)]
    with ProcessPoolExecutor(7) as ex:
        for nm in ex.map(job, jobs): print('done', os.path.basename(nm), flush=True)
