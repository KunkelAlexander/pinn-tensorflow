"""Late-time relative profile error about the grid origin vs about each run's own density centroid (separates drift from profile change)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT
RANGES = [('core<1', 0., 1.), ('mid1-3', 1., 3.), ('out3-6', 3., 6.), ('far6-10', 6., 10.)]
d = json.loads((OUT / 'halo_n256_longC.json').read_text())
t = np.array(d['times']); late = t >= 24
r = (np.arange(48) + .5) * .25
c = d['cases']; ref = c['spectral']
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
print(f"{'method':24s} {'centroid':>8s} | about ORIGIN: " + ' '.join(f'{n:>8s}' for n, _, _ in RANGES) + ' | about CENTROID: ' + ' '.join(f'{n:>8s}' for n, _, _ in RANGES))
for i, (name, rec) in enumerate(c.items()):
    row = [name, f"{np.hypot(rec['cx'][-1], rec['cy'][-1]):8.3f}"]
    for k, key in enumerate(('profile', 'profile_c')):
        e = np.array(rec[key])[late].mean(0) / np.array(ref[key])[late].mean(0) - 1
        row.append(' '.join(f'{np.mean(e[(r >= lo) & (r < hi)]):+8.3f}' for _, lo, hi in RANGES))
        if name != 'spectral':
            ax[k].semilogx(r, e, color=plt.get_cmap('tab10')(i % 10), label=name, lw=1.5)
    print(f'{row[0]:24s} {row[1]} |               {row[2]} |                  {row[3]}')
for k, ttl in enumerate(('about grid origin', 'about own centroid')):
    ax[k].axhline(0, color='k', lw=.6); ax[k].set_ylim(-.6, .6); ax[k].grid(alpha=.3, which='both'); ax[k].set_title(f'<rho>/<rho>_spec - 1, t>=24, profile {ttl}'); ax[k].set_xlabel('r')
ax[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / 'halo_n256_longC_centred.png', dpi=130)
