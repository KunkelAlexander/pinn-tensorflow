"""Plot/tabulate halo_1d results: time-averaged centroid-following |x| profile, relative error vs same-seed spectral, noise floor."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT

tag = sys.argv[1] if len(sys.argv) > 1 else '_M64'
d = json.loads((OUT / f'halo_1d{tag}.json').read_text())
E = np.array(d['edges'])
xc = np.sqrt(np.maximum(E[:-1], 1e-3) * E[1:])
seeds = list(d['runs'])
names = [n for n in d['runs'][seeds[0]] if n not in ('spectral',)]
P = {n: np.array([d['runs'][s][n]['profile'] for s in seeds]) for n in d['runs'][seeds[0]]}
ref = P['spectral']
rel = {n: P[n] / ref - 1 for n in names}
cols = {'spectral (perturbed)': 'gray'}
cmap = plt.get_cmap('viridis')
coarse = [n for n in names if n.startswith('coarse')]
for i, n in enumerate(coarse):
    cols[n] = cmap(i / max(len(coarse) - 1, 1) * .85)

fig, ax = plt.subplots(2, 3, figsize=(16, 8), gridspec_kw=dict(height_ratios=[1.6, 1]))
s0 = seeds[0]
ax[0, 0].loglog(xc, ref[0], 'k', lw=2.4, label='spectral (ref)')
for n in names:
    ax[0, 0].loglog(xc, P[n][0], color=cols[n], ls=':' if 'perturbed' in n else '-', label=n)
ax[0, 0].set_title(f'time-averaged density, seed {s0} (t in last {d["args"]["tavg"]:g} of {d["args"]["tend"]:g})')
ax[0, 0].set_ylabel(r'$\langle\rho\rangle(|x-x_c|)$'); ax[0, 0].legend(fontsize=7)
for n in names:
    m = np.median(rel[n], 0)
    lo, hi = np.percentile(rel[n], 16, 0), np.percentile(rel[n], 84, 0)
    ax[1, 0].semilogx(xc, m, color=cols[n], ls=':' if 'perturbed' in n else '-', lw=1.6)
    ax[1, 0].fill_between(xc, lo, hi, color=cols[n], alpha=.12)
ax[1, 0].axhline(0, color='k', lw=.6); ax[1, 0].set_ylim(-1, 1.5); ax[1, 0].set_xlabel('|x - x_c|')
ax[1, 0].set_ylabel(r'$\langle\rho\rangle/\langle\rho\rangle_{spec}-1$  (median, 16-84% over seeds)')
for n in names:
    ax[0, 1].semilogx(xc, np.median(abs(rel[n]), 0), color=cols[n], ls=':' if 'perturbed' in n else '-', lw=1.6, label=n)
ax[0, 1].set_yscale('log'); ax[0, 1].set_title('median |relative error| over seeds'); ax[0, 1].legend(fontsize=7); ax[0, 1].grid(alpha=.3, which='both')
ax[1, 1].axis('off')
t = np.array(d['runs'][s0]['spectral']['times'])
for n in ['spectral'] + names:
    ax[0, 2].plot(t, d['runs'][s0][n]['centroid'], color=cols.get(n, 'k'), ls=':' if 'perturbed' in n else '-', label=n)
    ax[1, 2].semilogy(t, d['runs'][s0][n]['peak'], color=cols.get(n, 'k'), ls=':' if 'perturbed' in n else '-')
ax[0, 2].set_title(f'centroid x_c(t), seed {s0}'); ax[1, 2].set_title('peak density'); ax[1, 2].set_xlabel('t')
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.grid(alpha=.3, which='both')
Lbox = d['x'][-1] - d['x'][0] + (d['x'][1] - d['x'][0])
fig.suptitle(f'1D halo (M={d["args"]["mass"]:g}, T={d["args"]["T"]:g}, r_c={d["args"]["rc"]:g}, L={Lbox:g}, N={len(d["x"])}): spectral vs coarse fd2 gravity, {len(seeds)} seeds')
fig.tight_layout()
out = OUT / f'halo_1d{tag}.png'
fig.savefig(out, dpi=130)
print('wrote', out)

bands = {'core |x|<1': (0, 1), 'mid 1-4': (1, 4), 'outer 4-10': (4, 10), 'far 10-40': (10, 40)}
print(f"{'method':24s}" + ''.join(f'{b:>16s}' for b in bands) + '    max|centroid| per seed')
for n in names:
    row = f'{n:24s}'
    for lo, hi in bands.values():
        sel = (xc >= lo) & (xc < hi)
        row += f'{np.median(abs(rel[n][:, sel])):16.3f}'
    row += '    ' + ' '.join(f"{np.abs(d['runs'][s][n]['centroid']).max():.2f}" for s in seeds)
    print(row)
