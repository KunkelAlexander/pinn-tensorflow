"""2D compact halo C (T=2 rc=6 M=80): (1) static radial-force error of coarse solvers on the spectral halo density;
(2) stage decomposition (restriction/parent/prolongation) and (3) composite fine-core + coarse-far, all +mean force.
usage: halo_2d_stages.py <seed> ; writes halo_2d_stages_s<seed>.json. Summary: summarize_2d_stages.py"""
import json, sys
import numpy as np
import stationary_halo as sh
import halo_2d_b3 as hb
from halo_2d_b3 import n, h, x, xx, yy, kx, ky, spec, with_mf, profile, evolve, make as make_mean
from soliton_gravity import interp_matrix, OUT

def make_stage(R, restr, parent, prol):
    m = n // R; H = L = 32. / m
    xc = -16. + (np.arange(m) + .5) * H
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H)
    s2 = np.sin(kc[:, None] * H / 2) ** 2 + np.sin(kc[None, :] * H / 2) ** 2
    k2 = kc[:, None] ** 2 + kc[None, :] ** 2
    g = np.zeros((m, m))
    if parent == 'fd2': g[s2 > 0] = -H ** 2 / (4 * s2[s2 > 0])
    else: g[k2 > 0] = -1 / k2[k2 > 0]
    T = interp_matrix(xc, x, 4)
    lo = np.abs(2 * np.pi * np.fft.fftfreq(n, d=h)) < np.pi / H - 1e-9
    kf = 2 * np.pi * np.fft.fftfreq(n, d=h)
    idxf = np.nonzero(lo)[0]
    idxc = np.round(kf[lo] / (2 * np.pi / 32.)).astype(int) % m
    Rm = np.zeros((m, n))
    for i in range(m): Rm[i, i * R:(i + 1) * R] = 1 / R
    def grav(rho):
        if restr == 'mean':
            Fc = np.fft.fft2(Rm @ rho @ Rm.T)
        else:
            F = np.fft.fft2(rho); Fc = np.zeros((m, m), complex)
            Fc[np.ix_(idxc, idxc)] = F[np.ix_(idxf, idxf)] * (m / n) ** 2
        Fc = Fc * g
        if prol == 'lag4': return T @ np.fft.ifft2(Fc).real @ T.T
        G = np.zeros((n, n), complex); G[np.ix_(idxf, idxf)] = Fc[np.ix_(idxc, idxc)] * (n / m) ** 2
        return np.fft.ifft2(G).real
    return grav

def composite(R, rw, wid=1.5):
    far = make_mean(R, 0)
    r = np.hypot(xx, yy)
    w = np.clip(.5 * (1 + np.cos(np.pi * (r - rw) / wid)), 0, 1); w[r <= rw] = 1; w[r >= rw + wid] = 0
    return lambda rho: spec(rho * w) + far(rho * (1 - w))

def radial_force(phi, rho):
    F = np.fft.fft2(phi); fx = -np.fft.ifft2(1j * kx * F).real; fy = -np.fft.ifft2(1j * ky * F).real
    r = np.hypot(xx, yy); fr = (fx * xx + fy * yy) / np.maximum(r, 1e-9)
    e = hb.EDGES; idx = np.digitize(r.ravel(), e) - 1; ok = (idx >= 0) & (idx < len(e) - 1)
    return np.bincount(idx[ok], weights=fr.ravel()[ok], minlength=len(e) - 1) / np.maximum(np.bincount(idx[ok], minlength=len(e) - 1), 1)

if __name__ == '__main__':
    seed = int(sys.argv[1]); mass = 80.
    sol = sh.solve(n, mass, 2., 6., verbose=False)
    psi0, _, _ = sh.build_psi(sol, n, seed); psi0 = sh.centre_and_rest(psi0, xx, yy); psi0 *= np.sqrt(mass / (np.sum(abs(psi0) ** 2) * h * h))
    rho0 = abs(psi0) ** 2
    gs = {}
    for R in [4, 8, 16]:
        H = R * h
        gs[f'H={H:.1f} mean/fd2/lag4'] = make_stage(R, 'mean', 'fd2', 'lag4')
        gs[f'H={H:.1f} mean/spec/lag4'] = make_stage(R, 'mean', 'spec', 'lag4')
        gs[f'H={H:.1f} fourier/spec/fourier'] = make_stage(R, 'fourier', 'spec', 'fourier')
    for R in [8, 16]:
        for rw in [1.5, 3.]:
            gs[f'H={R*h:.1f} comp rw={rw:g}'] = composite(R, rw)
    fref = radial_force(spec(rho0), rho0)
    force = {nm: (radial_force(g(rho0), rho0) / fref - 1).tolist() for nm, g in gs.items()}
    print('static radial-force error at r=0.25..:', {nm: np.round(np.array(v)[[0, 1, 2, 4, 8]], 2).tolist() for nm, v in force.items()}, flush=True)
    tend, tavg, dt = 40., 15., .01; every = 50
    res = {'static_force': force}
    jobs = {'spectral': spec}; jobs.update({nm: with_mf(g) for nm, g in gs.items()})
    for nm, g in jobs.items():
        sn = evolve(psi0, g, tend, dt, every); ts = np.arange(len(sn)) * every * dt
        ps = [profile(s) for s in sn]
        prof = np.mean([p for (p, c), t in zip(ps, ts) if t >= tend - tavg], axis=0)
        res[nm] = dict(profile=prof.tolist(), drift=float(max(np.hypot(*c) for p, c in ps)))
        print(f'seed {seed} {nm:28s} drift {res[nm]["drift"]:.3f}', flush=True)
    (OUT / f'halo_2d_stages_s{seed}.json').write_text(json.dumps(res) + '\n')
