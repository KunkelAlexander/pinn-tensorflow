"""Minimal gravity test: one Gaussian blob (a 'granule') and two blobs, no wave evolution, seconds per method.

(1) Self-force: a single blob of width sigma at sub-cell offset s (in coarse cells) exerts a net force on itself F = -int rho grad(phi).
    Exact gravity gives 0 (translation invariance). Plot |F| / (M^2 / 2 pi sigma) versus sigma / H and versus offset.
(2) Pair force: two equal blobs a distance d apart; compare the force on blob 1 with the spectral value (softening / lag of the force law).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import Gravity, grid, L, OUT

N = 256
h = L / N
x = grid(N)
XX, YY = np.meshgrid(x, x, indexing='ij')


def grad4(f):
    gx = (-np.roll(f, -2, 0) + 8 * np.roll(f, -1, 0) - 8 * np.roll(f, 1, 0) + np.roll(f, 2, 0)) / (12 * h)
    gy = (-np.roll(f, -2, 1) + 8 * np.roll(f, -1, 1) - 8 * np.roll(f, 1, 1) + np.roll(f, 2, 1)) / (12 * h)
    return gx, gy


def blob(cx, cy, s, mass=1.):
    rho = np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * s * s))
    return rho * mass / (rho.sum() * h * h)


def force_on(rho_src, rho_test, phi):
    gx, gy = grad4(phi)
    return -np.array([(rho_test * gx).sum(), (rho_test * gy).sum()]) * h * h


R = 8
M = {'spectral': dict(kind='spectral'),
     f'coarse r{R} (cell mean + Lagrange)': dict(kind='coarse', ratio=R, parent='fd2'),
     f'coarse r{R} TSC': dict(kind='coarse', ratio=R, parent='fd2', adjoint=2),
     f'inherited patch r{R} half32': dict(kind='patch', ratio=R, parent='fd2', half=32),
     f'Galerkin r{R} half32': dict(kind='galerkin', ratio=R, half=32, adjoint=2),
     'fd2 (fine grid)': dict(kind='fd2')}
grav = {k: Gravity(N, **v) for k, v in M.items()}
H = R * h
sig = np.array([.25, .5, 1., 1.5, 2., 3.]) * H / 1.0   # sigma in units of the coarse cell
rng = np.random.default_rng(0)
offs = rng.uniform(0, H, size=(12, 2)) + np.array([5 * H, 5 * H])    # blob centred away from patch centre too, random sub-cell offset
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
print(f'coarse cell H = {H:.2f}; self-force |F|/(M^2/2 pi sigma), max over 12 random sub-cell offsets')
print(f"{'sigma/H':>8s} " + ' '.join(f'{k[:26]:>26s}' for k in M))
res = {k: [] for k in M}
for s in sig:
    row = []
    for k in M:
        worst = 0.
        for cx, cy in offs:
            rho = blob(cx, cy, s)
            f = force_on(rho, rho, grav[k](rho))
            worst = max(worst, np.hypot(*f))
        res[k].append(worst / (1 / (2 * np.pi * s)))
        row.append(res[k][-1])
    print(f'{s / H:8.2f} ' + ' '.join(f'{v:26.2e}' for v in row))
for k in M:
    ax[0].loglog(sig / H, np.maximum(res[k], 1e-9), 'o-', label=k)
ax[0].set_xlabel('blob width sigma / coarse cell H'); ax[0].set_ylabel('max net self-force / (M^2 / 2 pi sigma)')
ax[0].set_title('(1) self-force of a single Gaussian blob'); ax[0].grid(alpha=.3, which='both'); ax[0].legend(fontsize=8)
# offset dependence at sigma = H/2
s = .5 * H
sh = np.linspace(0, H, 17)
for k in M:
    fx = []
    for o in sh:
        rho = blob(5 * H + o, 5 * H + .37 * o, s)
        fx.append(np.hypot(*force_on(rho, rho, grav[k](rho))) / (1 / (2 * np.pi * s)))
    ax[1].semilogy(sh / H, np.maximum(fx, 1e-9), label=k)
ax[1].set_xlabel('sub-cell offset of the blob / H'); ax[1].set_title('self-force vs offset, sigma = H/2'); ax[1].grid(alpha=.3, which='both')
# pair force
s = .5 * H
ds = np.array([.5, 1, 1.5, 2, 3, 4, 6, 8]) * H
print('\npair force on blob 1 relative to spectral, sigma = H/2 (separation d in coarse cells), mean over 6 offsets')
for k in M:
    rel = []
    for d in ds:
        vals = []
        for o in offs[:6]:
            r1, r2 = blob(5 * H + o[0] - 5 * H, 5 * H + o[1] - 5 * H, s), blob(5 * H + o[0] - 5 * H + d, 5 * H + o[1] - 5 * H, s)
            fe = force_on(None, r1, grav['spectral'](r2))
            f = force_on(None, r1, grav[k](r2))
            vals.append(f[0] / fe[0] - 1)
        rel.append(np.mean(vals))
    ax[2].semilogx(ds / H, rel, 'o-', label=k)
    print(f'{k[:32]:32s} ' + ' '.join(f'{v:+7.3f}' for v in rel))
ax[2].axhline(0, color='k', lw=.6); ax[2].set_ylim(-.6, .6)
ax[2].set_xlabel('separation d / H'); ax[2].set_ylabel('F / F_spectral - 1'); ax[2].set_title('(2) pair force (sigma = H/2)'); ax[2].grid(alpha=.3, which='both')
fig.tight_layout()
fig.savefig(OUT / 'minimal_selfforce.png', dpi=130)
print('wrote', OUT / 'minimal_selfforce.png')
