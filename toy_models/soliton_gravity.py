"""Stationary bound 2D soliton: does the gravity solver alone distort its density profile?

The wave solver is an exact Fourier Strang split on a uniform grid, so every drift from the
stationary ground state is caused by the gravity treatment. Periodic 2D, G=1, L=32.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse import diags, kron, eye, hstack, csc_matrix, bmat
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'soliton_gravity_results'
L = 32.
G = 1.


def grid(n):
    return -L / 2 + (np.arange(n) + .5) * L / n


def green(n, order):
    """Periodic Poisson kernel; order 0 = spectral, 2/4 = finite-difference symbol."""
    h = L / n
    k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    s = np.sin(k * h / 2) ** 2
    axis = k * k if order == 0 else 4 * s / h ** 2 * (1 + s / 3 if order == 4 else 1)
    den = axis[:, None] + axis[None, :]
    out = np.zeros_like(den)
    out[den > 0] = -G / den[den > 0]
    return out


def interp_matrix(xc, x, degree=4):
    """1D periodic Lagrange interpolation from cell centers xc to points x."""
    h = xc[1] - xc[0]
    n = len(xc)
    out = np.zeros((len(x), n))
    for row, p in enumerate(x):
        j = int(np.floor((p - xc[0]) / h + .5))
        ids = np.arange(j - degree // 2, j + degree // 2 + 1)
        nodes = xc[0] + ids * h
        w = np.ones(len(ids))
        for r in range(len(ids)):
            for s in range(len(ids)):
                if s != r:
                    w[r] *= (p - nodes[s]) / (nodes[r] - nodes[s])
        out[row, ids % n] = w
    return out


def bspline_matrix(xc, x, order):
    """1D periodic B-spline (order 1 = CIC, 2 = TSC) weights from coarse cell centres xc to points x; rows sum to 1."""
    h = xc[1] - xc[0]
    n = len(xc)
    out = np.zeros((len(x), n))
    for row, p in enumerate(x):
        u = (p - xc[0]) / h
        j = int(np.floor(u + .5)) if order == 2 else int(np.floor(u))
        if order == 1:
            f = u - j
            ids, w = [j, j + 1], [1 - f, f]
        else:
            d = u - j
            ids, w = [j - 1, j, j + 1], [.5 * (.5 - d) ** 2, .75 - d * d, .5 * (.5 + d) ** 2]
        for i, wi in zip(ids, w):
            out[row, i % n] += wi
    return out


def lattice_green(nmax, q=2000):
    """Infinite-lattice Green function of the unit 5-point Laplacian, G[a,b] for 0<=a,b<=nmax (G[0,0]=0, ~ln r/2pi)."""
    th, w = np.polynomial.legendre.leggauss(q)
    th, w = (th + 1) * np.pi / 2, w * np.pi / 2
    root = 2 * np.sin(th / 2) * np.sqrt(6 - 2 * np.cos(th))
    lam = (4 - 2 * np.cos(th) - root) / 2
    s = w / root
    a = np.arange(nmax + 1)
    return -(1 / np.pi) * ((np.cos(a[:, None] * th[None, :]) * s) @ np.exp(a[:, None] * np.log(lam)[None, :]).T - s.sum())


def free_kernel_fft(n, h):
    """FFT (on a 2n x 2n zero-padded grid) of the free-space FD2 kernel: phi = h^2 G sum_y Gl(x-y) rho(y)."""
    d = np.abs(np.fft.fftfreq(2 * n, d=1 / (2 * n)).astype(int))
    return np.fft.fft2(G * h * h * lattice_green(n)[d[:, None], d[None, :]])


def free_potential(src, kfft):
    n = src.shape[0]
    return np.fft.ifft2(np.fft.fft2(src, s=(2 * n, 2 * n)) * kfft).real[:n, :n]


class Gravity:
    """phi(rho) variants. All return the potential on the fine N x N wave grid.

    'mlc' is a simplified method of local corrections, all with the FD2 operator: the fine free-space potential of the
    patch density (lattice Green function) plus a smooth correction (coarse periodic solve minus coarse free-space
    solve of the same patch density) that carries the box images, the neutralising background and the far field.
    The patch density is tapered smoothly to zero over `taper` fine cells at the patch edge (default half/2), so the
    coarse part (1-chi) rho has no jump that the coarse grid cannot resolve.
    """

    def __init__(self, n, kind, ratio=2, parent='fd2', half=None, center=(0, 0), degree=4, taper=None, adjoint=0):
        self.n, self.kind = n, kind
        self.h = L / n
        if kind in ('spectral', 'fd2', 'fd4'):
            self.g = green(n, {'spectral': 0, 'fd2': 2, 'fd4': 4}[kind])
        elif kind == 'coarse':  # whole domain solved on a grid `ratio` times coarser, then interpolated
            self.ratio = ratio
            m = n // ratio
            self.g = green(m, {'spectral': 0, 'fd2': 2, 'fd4': 4}[parent])
            self.t = interp_matrix(grid(m), grid(n), degree)
            if adjoint:  # momentum-conserving pair: deposit with W^T / ratio, interpolate the potential with W (same B-spline kernel)
                self.t = bspline_matrix(grid(m), grid(n), adjoint)
                self.tr = self.t.T / ratio
        elif kind == 'galerkin':
            self._init_galerkin(n, ratio, half, center, adjoint)
        elif kind in ('patch', 'mlc'):  # fine patch; boundary inherited from the parent solve ('patch') or locally corrected ('mlc')
            self.ratio = ratio
            m = n // ratio
            if kind == 'mlc':
                # free-space coarse term lives on a grid extended by 2 cells so its interpolation never straddles the periodic seam
                self.kf, self.kc = free_kernel_fft(n, self.h), free_kernel_fft(m + 4, L / m)
                self.te = interp_matrix(-L / 2 + (np.arange(m + 4) - 1.5) * L / m, grid(n), degree)
            self.g = green(m, {'spectral': 0, 'fd2': 2, 'fd4': 4}[parent])
            self.t = interp_matrix(grid(m), grid(n), degree)
            if adjoint and kind == 'patch':  # smooth B-spline deposit / interpolation pair for the parent solve
                self.t = bspline_matrix(grid(m), grid(n), adjoint)
                self.tr = self.t.T / ratio
            cx = int(round((center[0] + L / 2) / self.h - .5))
            cy = int(round((center[1] + L / 2) / self.h - .5))
            # patch aligned to coarse cells
            a = ratio
            self.i0 = (cx - half) // a * a
            self.i1 = (cx + half) // a * a
            self.j0 = (cy - half) // a * a
            self.j1 = (cy + half) // a * a
            nx, ny = self.i1 - self.i0, self.j1 - self.j0
            self.nx, self.ny = nx, ny
            if kind == 'mlc':
                wt = taper if taper is not None else half // 2
                ax = lambda c: np.clip((half - np.abs(np.arange(n) - c + .5)) / max(wt, 1), 0, 1)
                self.chi = np.sin(.5 * np.pi * ax(cx))[:, None] ** 2 * np.sin(.5 * np.pi * ax(cy))[None, :] ** 2
                return
            dx = diags([np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1)], [-1, 0, 1], shape=(nx, nx))
            dy = diags([np.ones(ny - 1), -2 * np.ones(ny), np.ones(ny - 1)], [-1, 0, 1], shape=(ny, ny))
            self.lu = splu((kron(dx, eye(ny)) + kron(eye(nx), dy)).tocsc() / self.h ** 2)
            self.nx, self.ny = nx, ny
        else:
            raise ValueError(kind)

    def _init_galerkin(self, n, ratio, half, center, order):
        """Composite-grid Ritz solve: phi in span{coarse B-splines not wholly inside the patch} + {fine nodal functions in the patch}.

        phi = B u with (B^T L B) u = B^T (G (rho - mean)), L the fine periodic FD2 Laplacian. The map rho -> phi is symmetric by
        construction, so the self-force vanishes and momentum is conserved; there is no interpolated Dirichlet boundary.
        """
        h = self.h
        m = n // ratio
        w1 = bspline_matrix(grid(m), grid(n), order or 1)
        B = kron(csc_matrix(w1), csc_matrix(w1)).tocsc()
        if half is not None:
            cx = int(round((center[0] + L / 2) / h - .5))
            cy = int(round((center[1] + L / 2) / h - .5))
            i0, i1 = (cx - half) // ratio * ratio, (cx + half) // ratio * ratio
            j0, j1 = (cy - half) // ratio * ratio, (cy + half) // ratio * ratio
            inside = np.zeros((n, n), bool)
            inside[i0:i1, j0:j1] = True
            outside_weight = abs(B).T @ (~inside).ravel().astype(float)  # coarse function support reaching outside the patch
            keep = outside_weight > 1e-14
            pid = np.flatnonzero(inside.ravel())
            S = csc_matrix((np.ones(len(pid)), (pid, np.arange(len(pid)))), shape=(n * n, len(pid)))
            B = hstack([B[:, np.flatnonzero(keep)], S]).tocsc()
        d1 = diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tolil()
        d1[0, -1] = d1[-1, 0] = 1
        d1 = d1.tocsc() / h ** 2
        Lap = kron(d1, eye(n)) + kron(eye(n), d1)
        K = (B.T @ Lap @ B).tocsc()
        b = csc_matrix(np.asarray(B.T @ np.ones(n * n)).reshape(-1, 1))
        N = K.shape[0]
        self.B, self.nb = B, N
        self.lu = splu(bmat([[K, b], [b.T, None]], format='csc'))  # bordered: the constant (null) mode is fixed by a multiplier

    def __call__(self, rho):
        n = self.n
        if self.kind == 'galerkin':
            u = self.lu.solve(np.append(self.B.T @ (G * (rho - rho.mean())).ravel(), 0.))[:self.nb]
            phi = (self.B @ u).reshape(n, n)
            return phi - phi.mean()
        if self.kind in ('spectral', 'fd2', 'fd4'):
            return np.fft.ifft2(np.fft.fft2(rho) * self.g).real
        r = self.ratio
        m = n // r
        coarse = self.tr @ rho @ self.tr.T if hasattr(self, 'tr') else rho.reshape(m, r, m, r).mean(axis=(1, 3))
        parent = np.fft.ifft2(np.fft.fft2(coarse) * self.g).real
        i0, i1, j0, j1 = getattr(self, 'i0', 0), getattr(self, 'i1', 0), getattr(self, 'j0', 0), getattr(self, 'j1', 0)
        if self.kind == 'mlc':
            src = self.chi * rho
            cs = np.zeros((m + 4, m + 4))
            cs[2:-2, 2:-2] = src.reshape(m, r, m, r).mean(axis=(1, 3))
            phi = self.t @ parent @ self.t.T - self.te @ free_potential(cs, self.kc) @ self.te.T + free_potential(src, self.kf)
            return phi - phi.mean()
        else:
            phi = self.t @ parent @ self.t.T
        if self.kind == 'coarse':
            return phi
        src = G * (rho[i0:i1, j0:j1] - rho.mean())
        rhs = src.copy()
        # Dirichlet ring values from the interpolated parent potential
        rhs[0, :] -= phi[i0 - 1, j0:j1] / self.h ** 2
        rhs[-1, :] -= phi[i1 % n, j0:j1] / self.h ** 2
        rhs[:, 0] -= phi[i0:i1, j0 - 1] / self.h ** 2
        rhs[:, -1] -= phi[i0:i1, j1 % n] / self.h ** 2
        phi[i0:i1, j0:j1] = self.lu.solve(rhs.ravel()).reshape(self.nx, self.ny)
        return phi - phi.mean()


def ground_state(n, mass, stages=((.01, 3000), (.002, 3000), (.0005, 3000))):
    """Spectral-gravity ground state by imaginary-time propagation (shrinking dt removes the splitting bias)."""
    x = grid(n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    sq = k[:, None] ** 2 + k[None, :] ** 2
    gr = green(n, 0)
    w = np.exp(-(xx ** 2 + yy ** 2) / 1.5) + 0j
    h2 = (L / n) ** 2
    w *= np.sqrt(mass / (np.sum(abs(w) ** 2) * h2))
    for dt, iters in stages:
        drift = np.exp(-.5 * dt * sq)
        for it in range(iters):
            phi = np.fft.ifft2(np.fft.fft2(abs(w) ** 2) * gr).real
            w = w * np.exp(-.5 * dt * phi)
            w = np.fft.ifft2(np.fft.fft2(w) * drift)
            phi = np.fft.ifft2(np.fft.fft2(abs(w) ** 2) * gr).real
            w = w * np.exp(-.5 * dt * phi)
            w *= np.sqrt(mass / (np.sum(abs(w) ** 2) * h2))
    phi = np.fft.ifft2(np.fft.fft2(abs(w) ** 2) * gr).real
    hw = np.fft.ifft2(.5 * sq * np.fft.fft2(w)) + phi * w
    mu = np.sum(w.conj() * hw).real * h2 / mass
    print('ground state residual |Hw-mu w|/|w| =', float(np.linalg.norm(hw - mu * w) / np.linalg.norm(w)), 'mu =', float(mu), flush=True)
    return w.real + 0j, xx, yy


def evolve(w0, phi_fn, T, dt, sample_every, xx, yy):
    n = w0.shape[0]
    h2 = (L / n) ** 2
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    drift = np.exp(-.5j * dt * (k[:, None] ** 2 + k[None, :] ** 2))
    gr0 = green(n, 0)
    w = w0.copy()
    rho0 = abs(w0) ** 2
    steps = round(T / dt)
    hist = []
    phi = phi_fn(abs(w) ** 2)
    for j in range(steps + 1):
        if j % sample_every == 0:
            rho = abs(w) ** 2
            mass = rho.sum() * h2
            cx = (xx * rho).sum() * h2 / mass
            cy = (yy * rho).sum() * h2 / mass
            true_phi = np.fft.ifft2(np.fft.fft2(rho) * gr0).real
            core = (xx ** 2 + yy ** 2) < 4.
            hist.append(dict(time=j * dt, density_error=float(np.linalg.norm(rho - rho0) / np.linalg.norm(rho0)),
                             central_ratio=float(rho.max() / rho0.max()), com_drift=float(np.hypot(cx, cy)),
                             mass_change=float(mass / (rho0.sum() * h2) - 1),
                             phi_core_error=float(np.linalg.norm((phi - phi.mean() - true_phi + true_phi.mean())[core]) /
                                                  np.linalg.norm((true_phi - true_phi.mean())[core]))))
        if j == steps:
            break
        w = w * np.exp(-.5j * dt * phi)
        w = np.fft.ifft2(np.fft.fft2(w) * drift)
        phi = phi_fn(abs(w) ** 2)
        w = w * np.exp(-.5j * dt * phi)
    return w, hist


def radial(rho, xx, yy, bins=40, rmax=4.):
    r = np.hypot(xx, yy)
    ids = np.minimum((r / rmax * bins).astype(int), bins).ravel()
    c = np.bincount(ids, minlength=bins + 1)
    return (np.bincount(ids, weights=rho.ravel(), minlength=bins + 1) / np.maximum(c, 1))[:bins]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--mass', type=float, default=10.)
    ap.add_argument('--T', type=float, default=6.)
    ap.add_argument('--dt', type=float, default=.01)
    ap.add_argument('--offset', type=float, default=0.,
                    help='shift the soliton centre off the coarse-cell lattice (in cells)')
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    n = args.n
    h = L / n
    w0, xx, yy = ground_state(n, args.mass)
    rho0 = abs(w0) ** 2
    rms = float(np.sqrt((rho0 * (xx ** 2 + yy ** 2)).sum() * h * h / args.mass))
    print(f'N={n} h={h:.4f} mass={args.mass} rms radius={rms:.3f} peak rho={rho0.max():.3f} cells per rms radius={rms / h:.1f}', flush=True)
    cases = [('spectral', dict(kind='spectral')), ('fd2', dict(kind='fd2')), ('fd4', dict(kind='fd4'))]
    for r in (2, 4, 8):
        cases.append((f'coarse_r{r}_fd2', dict(kind='coarse', ratio=r, parent='fd2')))
    for half in (16, 24, 32, 48, 64):  # patch half-width in fine cells; rms radius is ~rms/h cells
        for r in (2, 4):
            cases.append((f'patch_r{r}_half{half}', dict(kind='patch', ratio=r, parent='fd2', half=half)))
    cases.append(('patch_r2_half32_spectralparent', dict(kind='patch', ratio=2, parent='spectral', half=32)))
    rows = []
    for name, kw in cases:
        tic = perf_counter()
        g = Gravity(n, **kw)
        w, hist = evolve(w0, g, args.T, args.dt, round(.1 / args.dt), xx, yy)
        rho = abs(w) ** 2
        row = dict(name=name, **{k: v for k, v in kw.items()}, seconds=perf_counter() - tic,
                   max_density_error=max(s['density_error'] for s in hist), final_density_error=hist[-1]['density_error'],
                   central_ratio_min=min(s['central_ratio'] for s in hist), central_ratio_max=max(s['central_ratio'] for s in hist),
                   max_com_drift=max(s['com_drift'] for s in hist), max_mass_change=max(abs(s['mass_change']) for s in hist),
                   max_phi_core_error=max(s['phi_core_error'] for s in hist),
                   profile=radial(rho, xx, yy).tolist(), history=hist)
        rows.append(row)
        print(f"{name:34s} max rho err {row['max_density_error']:.2e}  central ratio [{row['central_ratio_min']:.4f},{row['central_ratio_max']:.4f}]"
              f"  COM {row['max_com_drift']:.2e}  phi_core err {row['max_phi_core_error']:.2e}  mass {row['max_mass_change']:.1e}  {row['seconds']:.0f}s", flush=True)
        (OUT / f'run_n{n}.json').write_text(json.dumps(dict(n=n, mass=args.mass, rms=rms, T=args.T, dt=args.dt,
                                                           reference_profile=radial(rho0, xx, yy).tolist(), rows=rows)) + '\n')


if __name__ == '__main__':
    main()
