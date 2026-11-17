import json, numpy as np
from soliton_gravity import OUT
R = [json.loads((OUT / f'halo_2d_stages_s{s}.json').read_text()) for s in (1, 2)]
zones = {'core': (0, 2), 'mid': (4, 10), 'outer': (10, 24)}
names = [n for n in R[0] if n not in ('spectral', 'static_force')]
print(f"{'variant (+mf)':28s} core    mid     outer   drift")
for nm in names:
    row = []
    for a, b in zones.values():
        e = np.mean([np.sum(r[nm]['profile'][a:b]) / np.sum(r['spectral']['profile'][a:b]) - 1 for r in R]); row.append(f'{e:+8.1%}')
    print(f'{nm:28s}', ' '.join(row), f"{np.mean([r[nm]['drift'] for r in R]):.3f}")
print('\nstatic radial-force error (bins r<0.5, 0.5-1, 1-1.5, 2-2.5):')
for nm in names:
    f = np.mean([np.array(r['static_force'][nm]) for r in R], axis=0); print(f'{nm:28s}', np.round(f[[0, 1, 2, 4]], 2))
