import json, os, numpy as np
from soliton_gravity import OUT
Z = {'core': (0, 2), 'mid': (4, 10), 'outer': (10, 24)}
ref = {s: np.array(json.load(open(f'{OUT}/halo_2d_ppw_s{s}.json'))['256']['profile']) for s in (1, 2)}
def err(W, G, n, dt=.01):
    o = {z: [] for z in Z}; d = []
    for s in (1, 2):
        f = f'{OUT}/sens/s{s}_W{W}_G{G}_n{n}_dt{dt}.json'
        if not os.path.exists(f): return None
        r = json.load(open(f)); p = np.array(r['profile']); d.append(r['drift'])
        for z, (a, b) in Z.items(): o[z].append(p[a:b].sum() / ref[s][a:b].sum() - 1)
    return {z: 100 * np.mean(v) for z, v in o.items()}, np.mean(d)
wn = {0: 'exact', 2: 'FD2', 4: 'FD4', 6: 'FD6'}; gn = {0: 'spec', 2: 'FD2', 4: 'FD4', 6: 'FD6'}
for z in ('mid', 'outer'):
    print(f'\n{z} error (%) vs reference; rows = wave scheme / gravity scheme, columns = n (h, points per wavelength at lambda=1.1)')
    print(f'{"wave/grav":14s}' + ''.join(f'n={n} ({1.1*n/32:.1f}ppw)'.rjust(16) for n in (256, 192, 128, 96)))
    for W in (0, 2, 4, 6):
        for G in (0, 2, 4, 6):
            row = [err(W, G, n) if not (W == G == 0 and n == 256) else ({'core': 0, 'mid': 0, 'outer': 0}, 0) for n in (256, 192, 128, 96)]
            print(f'{wn[W]+"/"+gn[G]:14s}' + ''.join((f'{r[0][z]:+.0f}' if r else 'n/a').rjust(16) for r in row))
print('\ntime-step sensitivity, exact/spec, n=128 (outer, mid):')
for dt in (.005, .01, .02, .04):
    r = err(0, 0, 128, dt); print(f' dt={dt}: ' + (f"mid {r[0]['mid']:+.1f}%  outer {r[0]['outer']:+.1f}%  drift {r[1]:.3f}" if r else 'n/a'))
