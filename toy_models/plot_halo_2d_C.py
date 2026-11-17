import json, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stationary_halo as sh
import halo_2d_b3 as hb
from soliton_gravity import OUT
res = json.loads((OUT / 'halo_2d_b3_C.json').read_text())
seeds = list(res); ctr = .5 * (hb.EDGES[1:] + hb.EDGES[:-1])
names = ['pert', 'H=0.5 mean+mf', 'H=1.0 mean+mf', 'H=2.0 mean+mf', 'H=1.0 B3+mf']
col = {'pert': '0.6', 'H=0.5 mean+mf': '#1b9e77', 'H=1.0 mean+mf': '#d95f02', 'H=2.0 mean+mf': '#7570b3', 'H=1.0 B3+mf': '#e7298a'}
P = lambda s, n: np.array(res[s][n]['profile'])
ref = np.mean([P(s, 'spectral') for s in seeds], axis=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].loglog(ctr, ref, 'k', lw=2, label='spectral (reference)')
for n in names[1:]:
    ax[0].loglog(ctr, np.mean([P(s, n) for s in seeds], axis=0), color=col[n], label=n, ls='--' if 'B3' in n else '-')
ax[0].set(xlabel='r', ylabel=r'$\langle\rho\rangle$ (time & seed avg)', title='Radial density, compact 2D halo (T=2, rc=6, M=80)'); ax[0].legend(fontsize=8)
for n in names:
    e = np.mean([P(s, n) / P(s, 'spectral') - 1 for s in seeds], axis=0)
    ax[1].semilogx(ctr, e, color=col[n], label=n, ls='--' if 'B3' in n else '-')
ax[1].axhline(0, color='k', lw=.6); ax[1].set_yscale('symlog', linthresh=1); ax[1].set(xlabel='r', ylabel=r'$\rho/\rho_{\rm ref}-1$ (symlog)', title='Relative error vs spectral'); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'halo_2d_C_profiles.png', dpi=140)
# 2D maps (seed 1, time-averaged last 15 time units)
sol = sh.solve(hb.n, 80., 2., 6., verbose=False)
psi0, _, _ = sh.build_psi(sol, hb.n, 1); psi0 = sh.centre_and_rest(psi0, hb.xx, hb.yy); psi0 *= np.sqrt(80. / (np.sum(abs(psi0) ** 2) * hb.h ** 2))
runs = {'spectral': hb.spec, 'H=0.5 cell mean+mf': hb.with_mf(hb.make(8 // 2, 0)), 'H=1.0 cell mean+mf': hb.with_mf(hb.make(8, 0))}
maps = {}
for nm, g in runs.items():
    sn = hb.evolve(psi0, g, 40., .01, 50); maps[nm] = np.mean(sn[-30:], axis=0); print(nm, flush=True)
ext = [-8, 8, -8, 8]; sl = slice(hb.n // 2 - 64, hb.n // 2 + 64)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
for a, (nm, m) in zip(ax, maps.items()):
    im = a.imshow(np.log10(m[sl, sl].T + 1e-6), origin='lower', extent=ext, vmin=-3, vmax=np.log10(maps['spectral'].max()), cmap='magma'); a.set(title=nm, xlabel='x', ylabel='y'); plt.colorbar(im, ax=a, label=r'$\log_{10}\langle\rho\rangle$')
fig.tight_layout(); fig.savefig(OUT / 'halo_2d_C_maps.png', dpi=140)
