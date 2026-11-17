"""Minimal gravity test: ONE off-centre soliton, spectral gravity vs a coarse-grid fd2 Poisson solve (cell-mean restriction + quartic Lagrange).

Exact gravity: the soliton is stationary and keeps its momentum (translation invariance).
Coarse grid: the soliton accelerates by itself (spurious self-force) when its width sigma is comparable to the coarse cell H, and its
core is smeared.  Scan sigma/H (set through the mass, sigma ~ M^-1/2) and the sub-cell offset; report displacement and net momentum at T.
Wave solver identical in all runs (exact Fourier Strang). Usage: python minimal_soliton_drift.py MASS [MASS ...]
"""
import sys
import json
import numpy as np
from soliton_gravity import Gravity, ground_state, evolve, L, OUT

N = 256
h = L / N
T, DT = 4., .002
SOLVERS = {'spectral': dict(kind='spectral'), 'coarse r4': dict(kind='coarse', ratio=4, parent='fd2'),
           'coarse r8': dict(kind='coarse', ratio=8, parent='fd2')}
OFFS = [0, 2, 4, 6]       # extra fine-cell shift in x (0 .. 0.75 coarse cells for r8)


def moments(w, xx, yy):
    rho = abs(w) ** 2
    m = rho.sum() * h * h
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    px = (w.conj() * np.fft.ifft2(1j * k[:, None] * np.fft.fft2(w))).sum().imag * h * h
    py = (w.conj() * np.fft.ifft2(1j * k[None, :] * np.fft.fft2(w))).sum().imag * h * h
    return (xx * rho).sum() * h * h / m, (yy * rho).sum() * h * h / m, px, py, rho.max()


out = {}
for mass in [float(a) for a in sys.argv[1:]]:
    w0, xx, yy = ground_state(N, mass)
    rho = abs(w0) ** 2
    sig = float(np.sqrt((rho * (xx ** 2 + yy ** 2)).sum() / rho.sum() / 2))
    print(f'mass {mass:g}: per-axis sigma = {sig:.3f}  (= {sig / (4 * h):.2f} H4, {sig / (8 * h):.2f} H8)', flush=True)
    rec = {}
    for off in OFFS:
        w1 = np.roll(np.roll(w0, 40 + off, 0), 43, 1)
        c0 = moments(w1, xx, yy)
        for name, kw in SOLVERS.items():
            if name == 'spectral' and off != OFFS[0]:
                continue
            w, hist = evolve(w1, Gravity(N, **kw), T, DT, 10 ** 9, xx, yy)
            c = moments(w, xx, yy)
            rec.setdefault(name, []).append(dict(off=off, dx=float(np.hypot(c[0] - c0[0], c[1] - c0[1])), p=float(np.hypot(c[2], c[3])), peak=float(c[4] / c0[4])))
    for name, rr in rec.items():
        print(f"  {name:10s} displacement {max(q['dx'] for q in rr):.4f} (max over offsets)  |p| {max(q['p'] for q in rr):.3e}  peak ratio min {min(q['peak'] for q in rr):.3f}", flush=True)
    out[str(mass)] = dict(sigma=sig, rec=rec)
    (OUT / ('minimal_soliton_drift_' + '_'.join(sys.argv[1:]) + '.json')).write_text(json.dumps(out) + '\n')
