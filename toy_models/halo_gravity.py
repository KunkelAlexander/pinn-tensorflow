"""Bound granular halo (soliton-like core + interfering envelope): gravity-solver sensitivity.

Exact Fourier wave solver for every run; only the potential routine changes. Reference is the
same wave solver with spectral gravity. Floor = spectral gravity with half timestep and with
a 1e-8 perturbed start (the system is chaotic, so late-time pointwise agreement is not expected).
"""
import argparse
import json
from time import perf_counter
import numpy as np
from soliton_gravity import Gravity, grid, green, radial, L, OUT


def halo_initial(n, mass=30., R=4., ell=1.2, theta_rms=3., seed=1):
    x = grid(n)
    xx, yy = np.meshgrid(x, x, indexing='ij')
    rng = np.random.default_rng(seed)
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    filt = np.exp(-.25 * ell ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    theta = np.fft.ifft2(np.fft.fft2(rng.standard_normal((n, n))) * filt).real
    theta *= theta_rms / theta.std()
    w = np.exp(-(xx ** 2 + yy ** 2) / (4 * R ** 2)) * np.exp(1j * theta)
    w *= np.sqrt(mass / (np.sum(abs(w) ** 2) * (L / n) ** 2))
    return w, xx, yy


def run(w0, phi_fn, T, dt, every, xx, yy, n):
    h2 = (L / n) ** 2
    k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    drift = np.exp(-.5j * dt * (k[:, None] ** 2 + k[None, :] ** 2))
    w = w0.copy()
    phi = phi_fn(abs(w) ** 2)
    snaps = []
    for j in range(round(T / dt) + 1):
        if j % every == 0:
            snaps.append(abs(w) ** 2)
        if j == round(T / dt):
            break
        w = w * np.exp(-.5j * dt * phi)
        w = np.fft.ifft2(np.fft.fft2(w) * drift)
        phi = phi_fn(abs(w) ** 2)
        w = w * np.exp(-.5j * dt * phi)
    return np.array(snaps)


PBINS, PRMAX = 48, 12.  # per-snapshot azimuthal profiles: 48 bins of 0.25 out to r=12


def compare(s, ref, xx, yy):
    err = np.array([np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(s, ref)])
    # time-averaged azimuthal profile, and the radial range containing the bulk of the mass
    p = np.mean([radial(a, xx, yy, 32, 8.) for a in s], axis=0)
    q = np.mean([radial(a, xx, yy, 32, 8.) for a in ref], axis=0)
    core = np.array([a[(xx ** 2 + yy ** 2) < 1.].mean() for a in s])
    coreref = np.array([a[(xx ** 2 + yy ** 2) < 1.].mean() for a in ref])
    return dict(density_error_t=err.tolist(), density_error_max=float(err.max()),
                density_error_mean=float(err.mean()),
                mean_profile_error=float(np.linalg.norm(p - q) / np.linalg.norm(q)),
                core_density_error_mean=float(np.mean(abs(core - coreref) / coreref)),
                core_density_mean_bias=float(core.mean() / coreref.mean() - 1),
                profile=p.tolist(), reference_profile=q.tolist(),
                profile_t=np.round([radial(a, xx, yy, PBINS, PRMAX) for a in s], 6).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--T', type=float, default=4.)
    ap.add_argument('--dt', type=float, default=.005)
    ap.add_argument('--tag', default='')
    ap.add_argument('--mass', type=float, default=30.)
    ap.add_argument('--R', type=float, default=4.)
    ap.add_argument('--theta', type=float, default=3.)
    ap.add_argument('--ell', type=float, default=1.2)
    ap.add_argument('--suite', default='inherited', choices=['inherited', 'mlc', 'stationary'])
    ap.add_argument('--psi0', default='', help='npz with initial wave function (stationary_halo.py output); overrides mass/R/theta/ell')
    args = ap.parse_args()
    n, T, dt = args.n, args.T, args.dt
    every = round(.1 / dt)
    w0, xx, yy = halo_initial(n, args.mass, args.R, args.ell, args.theta)
    if args.psi0:
        w0 = np.load(args.psi0)['psi0'].astype(complex)
        args.mass = float((abs(w0) ** 2).sum() * (L / n) ** 2)
    r2 = xx ** 2 + yy ** 2
    print('halo: mass', float((abs(w0) ** 2).sum() * (L / n) ** 2), 'peak', float((abs(w0) ** 2).max()), flush=True)
    spec = Gravity(n, 'spectral')
    ref = run(w0, spec, T, dt, every, xx, yy, n)
    print('reference done; final rms radius',
          float(np.sqrt((ref[-1] * r2).sum() * (L / n) ** 2 / args.mass)), 'peak', float(ref[-1].max()), flush=True)
    rms_t = [float(np.sqrt((a * r2).sum() * (L / n) ** 2 / args.mass)) for a in ref]
    print('reference rms radius over time', np.round(rms_t[::max(1, len(rms_t) // 10)], 3).tolist(), flush=True)
    rows = []
    finals = {'reference': ref[-1].astype(np.float32), 'initial': ref[0].astype(np.float32)}
    ref_profile_t = np.round([radial(a, xx, yy, PBINS, PRMAX) for a in ref], 6).tolist()
    times = [j * dt * every for j in range(len(ref))]

    def add(name, snaps, meta):
        c = compare(snaps, ref, xx, yy)
        rows.append(dict(name=name, **meta, **c))
        finals[name] = snaps[-1].astype(np.float32)
        np.savez_compressed(OUT / f'halo_n{n}{args.tag}_final_density.npz', **finals)
        print(f"{name:34s} rho err max {c['density_error_max']:.2e} mean {c['density_error_mean']:.2e}  "
              f"mean-profile err {c['mean_profile_error']:.2e}  core err {c['core_density_error_mean']:.2e} bias {c['core_density_mean_bias']:+.2e}", flush=True)
        (OUT / f'halo_n{n}{args.tag}.json').write_text(json.dumps(dict(n=n, T=T, dt=dt, args=vars(args), times=times, profile_r_edges=np.linspace(0, PRMAX, PBINS + 1).tolist(),
                                                       reference_rms_t=rms_t, reference_profile_t=ref_profile_t, rows=rows)) + '\n')

    # noise floors
    add('floor_half_dt', run(w0, spec, T, dt / 2, every * 2, xx, yy, n), {})
    rng = np.random.default_rng(7)
    add('floor_perturbed_1e-8', run(w0 * (1 + 1e-8 * (rng.standard_normal(w0.shape) + 1j * rng.standard_normal(w0.shape))), spec, T, dt, every, xx, yy, n), {})
    cases = [('fd2', dict(kind='fd2')), ('fd4', dict(kind='fd4'))]
    if args.suite == 'inherited':
        for r in (2, 4, 8):
            cases.append((f'coarse_r{r}_fd2', dict(kind='coarse', ratio=r, parent='fd2')))
        for half in (16, 32, 48, 64):
            for r in (2, 4, 8):
                cases.append((f'patch_r{r}_half{half}', dict(kind='patch', ratio=r, parent='fd2', half=half)))
    elif args.suite == 'stationary':  # stationary halo: grid, coarse, patch and mlc over a wide range of patch sizes
        for r in (2, 4, 8):
            cases.append((f'coarse_r{r}_fd2', dict(kind='coarse', ratio=r, parent='fd2')))
        cases.append(('coarse_r8_spectral', dict(kind='coarse', ratio=8, parent='spectral')))
        for half in (8, 16, 32, 48, 64):
            for r in (4, 8):
                cases.append((f'patch_r{r}_half{half}', dict(kind='patch', ratio=r, parent='fd2', half=half)))
                cases.append((f'mlc_r{r}_half{half}', dict(kind='mlc', ratio=r, parent='fd2', half=half)))
    else:  # inherited-boundary patch vs method of local corrections (tapered), at matched patch size and parent ratio
        for r in (4, 8):
            cases.append((f'coarse_r{r}_fd2', dict(kind='coarse', ratio=r, parent='fd2')))
        for half in (16, 32, 48):
            for r in (4, 8):
                cases.append((f'patch_r{r}_half{half}', dict(kind='patch', ratio=r, parent='fd2', half=half)))
                cases.append((f'mlc_r{r}_half{half}', dict(kind='mlc', ratio=r, parent='fd2', half=half)))
    for name, kw in cases:
        tic = perf_counter()
        add(name, run(w0, Gravity(n, **kw), T, dt, every, xx, yy, n), dict(kw, seconds=perf_counter() - tic))


if __name__ == '__main__':
    main()
