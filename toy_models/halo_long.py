"""Long-time fidelity of gravity treatments on the stationary halo: secular relaxation, granule statistics, spurious forces.

For each gravity treatment the same exact Fourier wave solver runs to T (default 48, ~16 dynamical times of the halo).
Every 0.5 time units we record, measured identically for all methods (spectral functionals on the wave function):
  mass, centroid and momentum (a gravity treatment that is not self-adjoint exerts a spurious net force on the halo),
  K and E = K + 1/2 int rho phi_spectral (energy error), rms about the origin and about the centroid, core density,
  peak, the azimuthal profile about the origin, and the granule contrast std(rho / <rho>_az) in 0.5 < r < 4.
"""
import argparse
import json
from time import perf_counter
import numpy as np
from soliton_gravity import Gravity, grid, radial, L, OUT

CASES = {
    'spectral': dict(kind='spectral'),
    'fd2': dict(kind='fd2'),
    'fd4': dict(kind='fd4'),
    'coarse_r2_fd2': dict(kind='coarse', ratio=2, parent='fd2'),
    'coarse_r4_fd2': dict(kind='coarse', ratio=4, parent='fd2'),
    'coarse_r4_fd4': dict(kind='coarse', ratio=4, parent='fd4'),
    'coarse_r8_fd2': dict(kind='coarse', ratio=8, parent='fd2'),
    'coarse_r8_fd4': dict(kind='coarse', ratio=8, parent='fd4'),
    'coarse_r2_cic_fd2': dict(kind='coarse', ratio=2, parent='fd2', adjoint=1),
    'coarse_r4_cic_fd2': dict(kind='coarse', ratio=4, parent='fd2', adjoint=1),
    'coarse_r4_tsc_fd2': dict(kind='coarse', ratio=4, parent='fd2', adjoint=2),
    'coarse_r4_tsc_fd4': dict(kind='coarse', ratio=4, parent='fd4', adjoint=2),
    'coarse_r8_tsc_fd2': dict(kind='coarse', ratio=8, parent='fd2', adjoint=2),
    'coarse_r8_tsc_fd4': dict(kind='coarse', ratio=8, parent='fd4', adjoint=2),
    'galerkin_r4_tsc': dict(kind='galerkin', ratio=4, adjoint=2),
    'galerkin_r8_tsc': dict(kind='galerkin', ratio=8, adjoint=2),
    'galerkin_r8_cic_half48': dict(kind='galerkin', ratio=8, half=48, adjoint=1),
    'galerkin_r8_tsc_half32': dict(kind='galerkin', ratio=8, half=32, adjoint=2),
    'galerkin_r8_tsc_half48': dict(kind='galerkin', ratio=8, half=48, adjoint=2),
    'galerkin_r4_tsc_half32': dict(kind='galerkin', ratio=4, half=32, adjoint=2),
    'patch_r4_half32': dict(kind='patch', ratio=4, parent='fd2', half=32),
    'patch_r4_half48': dict(kind='patch', ratio=4, parent='fd2', half=48),
    'patch_r8_half32': dict(kind='patch', ratio=8, parent='fd2', half=32),
    'patch_r8_half48': dict(kind='patch', ratio=8, parent='fd2', half=48),
    'mlc_r8_half32': dict(kind='mlc', ratio=8, parent='fd2', half=32),
    'mlc_r8_half48': dict(kind='mlc', ratio=8, parent='fd2', half=48),
}


def run(w0, grav, spec, T, dt, every, xx, yy):
    n = w0.shape[0]
    h2 = (L / n) ** 2
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    drift = np.exp(-.5j * dt * k2)
    r = np.hypot(xx, yy)
    ids = np.minimum((r / 12. * 48).astype(int), 48)
    inner = (r > .5) & (r < 4.)
    w = w0.copy()
    mass = (abs(w) ** 2).sum() * h2
    phi = grav(abs(w) ** 2)
    rec = {key: [] for key in ('mass', 'cx', 'cy', 'px', 'py', 'K', 'E', 'rms', 'rms_c', 'core', 'peak', 'contrast', 'profile', 'profile_c')}
    for j in range(round(T / dt) + 1):
        if j % every == 0:
            rho = abs(w) ** 2
            f = np.fft.fft2(w)
            a2 = abs(f) ** 2
            K = .5 * np.sum(k2 * a2) / n ** 2 * h2
            ps = spec(rho)
            cx, cy = (rho * xx).sum() * h2 / mass, (rho * yy).sum() * h2 / mass
            prof = np.bincount(ids.ravel(), weights=rho.ravel(), minlength=49) / np.maximum(np.bincount(ids.ravel(), minlength=49), 1)
            smooth = prof[ids]
            rec['mass'].append(float((rho).sum() * h2))
            rec['cx'].append(float(cx)); rec['cy'].append(float(cy))
            rec['px'].append(float((k[:, None] * a2).sum() / n ** 2 * h2)); rec['py'].append(float((k[None, :] * a2).sum() / n ** 2 * h2))
            rec['K'].append(float(K)); rec['E'].append(float(K + .5 * (rho * ps).sum() * h2))
            rec['rms'].append(float(np.sqrt((rho * r ** 2).sum() * h2 / mass)))
            rec['rms_c'].append(float(np.sqrt((rho * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() * h2 / mass)))
            rec['core'].append(float(rho[r < 1.].mean())); rec['peak'].append(float(rho.max()))
            rec['contrast'].append(float(np.std(rho[inner] / smooth[inner])))
            rec['profile'].append(np.round(prof[:48], 6).tolist())
            idc = np.minimum((np.hypot(xx - cx, yy - cy) / 12. * 48).astype(int), 48).ravel()  # profile about the density centroid
            profc = np.bincount(idc, weights=rho.ravel(), minlength=49) / np.maximum(np.bincount(idc, minlength=49), 1)
            rec['profile_c'].append(np.round(profc[:48], 6).tolist())
        if j == round(T / dt):
            break
        w = w * np.exp(-.5j * dt * phi)
        w = np.fft.ifft2(np.fft.fft2(w) * drift)
        phi = grav(abs(w) ** 2)
        w = w * np.exp(-.5j * dt * phi)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--psi0', default=str(OUT / 'stationary_halo_n256_T5rc8_psi0.npz'))
    ap.add_argument('--T', type=float, default=48.)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--cases', nargs='+', default=list(CASES))
    ap.add_argument('--tag', default='_long')
    args = ap.parse_args()
    w0 = np.load(args.psi0)['psi0'].astype(complex)
    n = w0.shape[0]
    x = grid(n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    spec = Gravity(n, 'spectral')
    every = round(.5 / args.dt)
    path = OUT / f'halo_n{n}{args.tag}.json'
    out = dict(n=n, T=args.T, dt=args.dt, args=vars(args), times=[j * .5 for j in range(round(args.T / .5) + 1)], cases={})
    for name in args.cases:
        tic = perf_counter()
        kw = CASES[name]
        grav = spec if kw['kind'] == 'spectral' else Gravity(n, **kw)
        rec = run(w0, grav, spec, args.T, args.dt, every, xx, yy)
        rec['seconds'] = perf_counter() - tic
        out['cases'][name] = rec
        path.write_text(json.dumps(out) + '\n')
        print(f"{name:20s} {rec['seconds']:6.0f}s  rms {rec['rms'][0]:.3f}->{rec['rms'][-1]:.3f}  core {np.mean(rec['core'][:10]):.2f}->{np.mean(rec['core'][-10:]):.2f}  "
              f"contrast {np.mean(rec['contrast']):.3f}  centroid {np.hypot(rec['cx'][-1], rec['cy'][-1]):.3f}  |p| {np.hypot(rec['px'][-1], rec['py'][-1]):.2e}  "
              f"E drift {rec['E'][-1] / rec['E'][0] - 1:+.2e}", flush=True)


if __name__ == '__main__':
    main()
