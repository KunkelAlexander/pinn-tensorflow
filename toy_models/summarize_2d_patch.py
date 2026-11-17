import json, os, numpy as np
from soliton_gravity import OUT
Z = {'core': (0, 2), 'mid': (4, 10), 'outer': (10, 24)}
ref = {s: np.array(json.load(open(f'{OUT}/halo_2d_ppw_s{s}.json'))['256']['profile']) for s in (1, 2)}
def err(f):
    o = {z: [] for z in Z}
    for s in (1, 2):
        p = np.array(json.load(open(f % s))['profile'])
        for z, (a, b) in Z.items(): o[z].append(p[a:b].sum() / ref[s][a:b].sum() - 1)
    return {z: 100 * np.mean(v) for z, v in o.items()}
print('error (%) vs n=256 exact/spectral reference, mean of 2 seeds; patch = fine direct solve with parent Dirichlet BC')
print(f'{"parent/patch stencil":20s}{"patch half-width":>18s}{"n":>6s}{"ppw":>6s}   core    mid  outer')
for n in (192, 128):
    for hw in (4, 8):
        for pa, pt in ((2, 2), (2, 4), (4, 4)):
            e = err(f'{OUT}/patch/s%d_Pa{pa}_Pt{pt}_hw{hw}_n{n}.json')
            print(f'FD{pa} / FD{pt}'.ljust(20) + f'{hw:18d}{n:6d}{1.1*n/32:6.1f}  {e["core"]:+5.0f}% {e["mid"]:+5.0f}% {e["outer"]:+5.0f}%')
