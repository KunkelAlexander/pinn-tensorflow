"""Relative L2 difference of the density at one snapshot between run s4_rep and other runs (smoothed covering grid at a given level).
usage: python3 cmp_snap.py LEVEL SNAPSHOT_ID TAG [TAG ...]    e.g. python3 cmp_snap.py 1 000001 res3 res5 third_qp4   (runs in /workspace/runs)"""
import sys, numpy as np, yt
yt.set_log_level(50)
lv = int(sys.argv[1]); snap = sys.argv[2]
def load(tag):
    ds = yt.load('/workspace/runs/%s/Data_%s' % (tag, snap))
    N = 64 * 2**lv
    cg = ds.smoothed_covering_grid(level=lv, left_edge=[0, 0, 0], dims=[N, N, 16 * 2**lv])
    return ds.current_time, np.asarray(cg['gas', 'density'])[:, :, 0]
t0, a = load('s4_rep')
print('snapshot', snap, 'level', lv, 'time', t0, 'mean dens', a.mean())
for tag in sys.argv[3:]:
    t, b = load(tag)
    d = b - a
    print('%-10s t=%.4f mean=%.6f  rel L2 diff %.3e   max |diff|/mean %.3e   min %.3e' % (tag, t, b.mean(), np.sqrt((d**2).mean()) / np.sqrt((a**2).mean()), np.abs(d).max() / a.mean(), b.min()))
