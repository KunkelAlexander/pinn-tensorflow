"""Stationary 2D fuzzy-dark-matter halo: self-consistent eigenstate superposition, verified with spectral gravity.

A single wave function has no stationary |psi|^2 except the soliton, so a 'stationary halo' means a random-phase
superposition of eigenstates of its own mean-field potential, psi = sum_nm sqrt(f_nm) e^{i alpha_nm} R_nm(r) e^{i m theta},
with occupations f(E) chosen so that sum f |psi_nm|^2 reproduces the density that sources the potential (Lin et al. 2018,
Yavetz, Li & Hui 2021). The time-averaged density is then stationary while the granules interfere and fluctuate.

Occupation (lowered isothermal / King): f(E) = c [exp(-(E-E0)/T) - exp(-(Ec-E0)/T)] for E < Ec, Ec = Phi(r_c).
Potential: the same periodic spectral Poisson solve the wave runs use (G=1, lap phi = rho), so the box images are included.
Radial eigenproblem per angular momentum m: flux-form FD on cell-centred r_j, natural regularity at r=0.
"""
import argparse
import json
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal
from soliton_gravity import Gravity, grid, radial, L, OUT

RMAX, NR = 16., 320


def radial_grid():
    dr = RMAX / NR
    return (np.arange(NR) + .5) * dr, dr


def radial_states(Phi, m, emax):
    """Eigenvalues/eigenvectors (R normalised: int R^2 r dr = 1) of -1/2 (R''+R'/r) + (m^2/2r^2 + Phi) R = E R, E < emax."""
    r, dr = radial_grid()
    rp, rm = r + dr / 2, r - dr / 2
    diag = .5 * (rp + rm) / dr ** 2 + r * (m * m / (2 * r * r) + Phi)
    off = -.5 * rp[:-1] / dr ** 2
    d = diag / r
    e = off / np.sqrt(r[:-1] * r[1:])
    E, v = eigh_tridiagonal(d, e, select='v', select_range=(Phi.min() - 1., emax))
    return E, v / np.sqrt(r * dr)[:, None]


def poisson_profile(rho_r, n, spec, nb=256):
    """Azimuthal-mean periodic spectral potential (radial profile on r_j) of a radial density profile."""
    # isolated radial potential analytically (r Phi' = M(<r)/2pi), plus the smooth periodic-image/background correction
    # measured as the coarsely binned azimuthal mean of (periodic spectral solve - isolated potential)
    x = grid(n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    rr = np.hypot(xx, yy)
    r, dr = radial_grid()
    mcum = np.cumsum(2 * np.pi * r * rho_r * dr) - np.pi * r * rho_r * dr
    iso = np.concatenate([[0.], np.cumsum(.5 * (mcum[1:] / r[1:] + mcum[:-1] / r[:-1]) * dr / (2 * np.pi))]) + .5 * dr * mcum[0] / r[0] / (2 * np.pi)
    rho2 = np.interp(rr, r, rho_r, right=0.)
    phi = spec(rho2)
    # checked numerically: the azimuthal mean of (periodic - isolated) is the neutralising background -rho_bar r^2/4 to <0.01
    # out to r=16 (the lattice images only add r^4 cos(4 theta), which averages out), so no fit is needed
    b = -(rho_r * 2 * np.pi * r * dr).sum() / L ** 2 / 4
    return iso + b * r ** 2, rho2, phi


def solve(n, mass, T, rc, mmax=48, iters=80, mix=.5, verbose=True):
    spec = Gravity(n, 'spectral')
    r, dr = radial_grid()
    rho_r = np.exp(-r ** 2 / (2 * 2.5 ** 2))
    rho_r *= mass / (2 * np.pi * np.sum(rho_r * r) * dr)
    Phi, _, _ = poisson_profile(rho_r, n, spec)
    for it in range(iters):
        Ec = np.interp(rc, r, Phi)
        states = {}
        for m in range(mmax + 1):
            E, R = radial_states(Phi, m, Ec)
            if len(E) == 0:
                break
            states[m] = (E, R)
        E0 = min(E[0] for E, _ in states.values())
        w = lambda E: np.maximum(np.exp(-(E - E0) / T) - np.exp(-(Ec - E0) / T), 0.)
        mult = lambda m: 1 if m == 0 else 2
        norm = sum(mult(m) * w(E).sum() for m, (E, _) in states.items())
        c = mass / norm
        new = np.zeros(NR)
        for m, (E, R) in states.items():
            new += mult(m) * (R ** 2 @ (c * w(E))) / (2 * np.pi)
        res = np.linalg.norm(new - rho_r) / np.linalg.norm(new)
        rho_r = (1 - mix) * rho_r + mix * new
        Phi, _, _ = poisson_profile(rho_r, n, spec)
        if verbose and (it % 10 == 0 or res < 1e-7):
            print(f'  iter {it:3d} residual {res:.2e}  peak {rho_r[0]:.3f}  states {sum(len(E) for E, _ in states.values())}  mmax {max(states)}', flush=True)
        if res < 1e-7:
            break
    fm = {m: c * w(E) for m, (E, _) in states.items()}
    # mean-field kinetic energy: sum f (E - <Phi>)
    K = sum(mult(m) * (fm[m] * (E - (R ** 2 * Phi[:, None] * r[:, None] * dr).sum(0))).sum() for m, (E, R) in states.items())
    return dict(states=states, f=fm, rho_r=rho_r, Phi=Phi, Ec=Ec, E0=E0, K=K, res=res, mmax=max(states),
                nstates=sum(len(E) for E, _ in states.values()))


def build_psi(sol, n, seed):
    x = grid(n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    rr, th = np.hypot(xx, yy), np.arctan2(yy, xx)
    r, _ = radial_grid()
    rng = np.random.default_rng(seed)
    psi = np.zeros((n, n), complex)
    for m, (E, R) in sol['states'].items():
        for sgn in ((1,) if m == 0 else (1, -1)):
            g = R @ (np.sqrt(sol['f'][m]) * np.exp(2j * np.pi * rng.random(len(E))))
            gi = (CubicSpline(r, g.real)(np.minimum(rr, r[-1])) + 1j * CubicSpline(r, g.imag)(np.minimum(rr, r[-1]))) * (rr < r[-1])
            psi += gi * np.exp(1j * sgn * m * th) / np.sqrt(2 * np.pi)
    return psi, xx, yy


def centre_and_rest(psi, xx, yy, passes=3):
    """Galilean-boost away the net momentum and shift the centroid to the origin.

    A random-phase superposition carries a random net momentum, so left alone the whole halo translates through the
    periodic box (rms about the origin then grows linearly even though the halo is internally stationary).
    """
    n = psi.shape[0]
    h2 = (L / n) ** 2
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    for _ in range(passes):
        f = np.fft.fft2(psi)
        p = abs(f) ** 2
        vx, vy = (k[:, None] * p).sum() / p.sum(), (k[None, :] * p).sum() / p.sum()
        psi = psi * np.exp(-1j * (vx * xx + vy * yy))
        rho = abs(psi) ** 2
        cx, cy = (rho * xx).sum() / rho.sum(), (rho * yy).sum() / rho.sum()
        psi = np.fft.ifft2(np.fft.fft2(psi) * np.exp(1j * (k[:, None] * cx + k[None, :] * cy)))
    return psi


def shape(rho_r, mass):
    r, dr = radial_grid()
    mr = np.cumsum(2 * np.pi * r * rho_r * dr)
    rms = np.sqrt(np.sum(2 * np.pi * r ** 3 * rho_r * dr) / mass)
    r50 = r[np.searchsorted(mr, .5 * mass)]
    r90 = r[np.searchsorted(mr, .9 * mass)]
    return dict(peak=float(rho_r[0]), rms=float(rms), r50=float(r50), r90=float(r90))


def evolve_diag(w0, spec, T, dt, every, xx, yy):
    """Exact Fourier Strang evolution with spectral gravity, recording density snapshots, K, 2K/(M^2/4pi), E and momentum."""
    n = w0.shape[0]
    h2 = (L / n) ** 2
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    drift = np.exp(-.5j * dt * k2)
    w = w0.copy()
    phi = spec(abs(w) ** 2)
    snaps, extra = [], []
    mass = (abs(w) ** 2).sum() * h2
    for j in range(round(T / dt) + 1):
        if j % every == 0:
            rho = abs(w) ** 2
            f = np.fft.fft2(w)
            K = .5 * np.sum(k2 * abs(f) ** 2) / n ** 2 * h2
            gx = np.fft.ifft2(1j * k[:, None] * np.fft.fft2(phi)).real
            gy = np.fft.ifft2(1j * k[None, :] * np.fft.fft2(phi)).real
            vir = (rho * (xx * gx + yy * gy)).sum() * h2
            px = (k[:, None] * abs(f) ** 2).sum() / n ** 2 * h2
            py = (k[None, :] * abs(f) ** 2).sum() / n ** 2 * h2
            extra.append((K, 2 * K / (mass ** 2 / (4 * np.pi)), vir / (mass ** 2 / (4 * np.pi)), K + .5 * (rho * phi).sum() * h2, px, py))
            snaps.append(rho)
        if j == round(T / dt):
            break
        w = w * np.exp(-.5j * dt * phi)
        w = np.fft.ifft2(np.fft.fft2(w) * drift)
        phi = spec(abs(w) ** 2)
        w = w * np.exp(-.5j * dt * phi)
    return np.array(snaps), np.array(extra)


def diagnostics(snaps, extra, times, xx, yy, mass):
    n = snaps.shape[1]
    h2 = (L / n) ** 2
    r2 = xx ** 2 + yy ** 2
    out = dict(times=list(times))
    out.update({key: extra[:, i].tolist() for i, key in enumerate(['K', 'kinetic_virial_ratio', 'force_virial_ratio', 'energy', 'px', 'py'])})
    out['mass'] = [float(a.sum() * h2) for a in snaps]
    out['rms'] = [float(np.sqrt((a * r2).sum() * h2 / mass)) for a in snaps]
    out['cx'] = [float((a * xx).sum() * h2 / mass) for a in snaps]
    out['cy'] = [float((a * yy).sum() * h2 / mass) for a in snaps]
    out['core'] = [float(a[r2 < 1.].mean()) for a in snaps]
    out['peak'] = [float(a.max()) for a in snaps]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--mass', type=float, default=80.)
    ap.add_argument('--T', type=float, default=2.)
    ap.add_argument('--rc', type=float, default=6.)
    ap.add_argument('--mmax', type=int, default=48)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--tend', type=float, default=0., help='if >0 also run the spectral evolution to this time')
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    n = args.n
    print(f'self-consistent solve: mass {args.mass} T {args.T} r_c {args.rc}', flush=True)
    sol = solve(n, args.mass, args.T, args.rc, args.mmax)
    s = shape(sol['rho_r'], args.mass)
    virial = 2 * sol['K'] / (args.mass ** 2 / (4 * np.pi))
    print(f"converged residual {sol['res']:.1e}; {sol['nstates']} states, m up to {sol['mmax']}; shape {s}; mean-field 2K/(M^2/4pi) = {virial:.3f}", flush=True)
    top = max(sol['f'][m].max() for m in sol['f'])
    edge = max(sol['f'][sol['mmax']].max() / top, 0)
    print(f'largest occupation in highest m channel relative to the maximum: {edge:.1e}', flush=True)
    if args.tend <= 0:
        return
    psi0, xx, yy = build_psi(sol, n, args.seed)
    psi0 = centre_and_rest(psi0, xx, yy)
    psi0 *= np.sqrt(args.mass / (np.sum(abs(psi0) ** 2) * (L / n) ** 2))
    spec = Gravity(n, 'spectral')
    every = round(.1 / args.dt)
    snaps, extra = evolve_diag(psi0, spec, args.tend, args.dt, every, xx, yy)
    times = [j * args.dt * every for j in range(len(snaps))]
    d = diagnostics(snaps, extra, times, xx, yy, args.mass)
    d.update(args=vars(args), target_rho_r=sol['rho_r'].tolist(), target_shape=s, mean_field_virial=float(virial), r_grid=radial_grid()[0].tolist(),
             profile_r_edges=np.linspace(0, 12., 49).tolist(),
             profile_t=np.round([radial(a, xx, yy, 48, 12.) for a in snaps], 6).tolist())
    (OUT / f'stationary_halo_n{n}{args.tag}.json').write_text(json.dumps(d) + '\n')
    np.savez_compressed(OUT / f'stationary_halo_n{n}{args.tag}_psi0.npz', psi0=psi0.astype(np.complex64), final=snaps[-1].astype(np.float32))
    print('rms over time', np.round(d['rms'][::max(1, len(d['rms']) // 10)], 3).tolist())
    print('core density over time', np.round(d['core'][::max(1, len(d['core']) // 10)], 3).tolist())


if __name__ == '__main__':
    main()
