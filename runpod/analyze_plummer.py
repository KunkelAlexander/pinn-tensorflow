"""Analyse the Hydro/Gravity Plummer runs (POT_STENCIL_ORDER 2 vs 4).

Run on the pod:  python3 analyze_plummer.py /workspace
Reads  <root>/plum_r<r0>_o<order>/PotError.txt  (diagonal output of the test problem; columns
i j k x y z Dens AbsErr RelErr ... where AbsErr = numerical - analytic potential).

The box truncates the Plummer tail, which adds an almost constant offset to the numerical potential
(the analytic solution assumes an infinite domain). We estimate it from the level-1 cells, where the
discretisation error is negligible, and subtract it, so what remains is discretisation / inter-level error.
"""
import glob
import os
import re
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace"
BOX_CENTER = 50.0

runs = {}
for path in sorted(glob.glob(os.path.join(root, "plum_r*_o*", "PotError.txt"))):
    m = re.search(r"plum_r([0-9.]+)_o(\d)", path)
    r0, order = float(m.group(1)), int(m.group(2))
    d = np.loadtxt(path)
    # the diagonal file lists the levels one after the other; a new level starts when the index i drops
    lev = np.cumsum(np.r_[0, np.diff(d[:, 0]) < 0])
    r = np.sqrt(3.0) * np.abs(d[:, 3] - BOX_CENTER)
    runs[(r0, order)] = dict(lev=lev, r=r, err=d[:, 7], dens=d[:, 6])

print("Potential error after removing the box-truncation offset (offset from level-1 cells with r in [25,40])")
print("Error is numerical - analytic; Phi(0) = -G*M/r0 for scale.\n")
for r0 in sorted({k[0] for k in runs}):
    if (r0, 2) not in runs or (r0, 4) not in runs:
        continue
    a, b = runs[(r0, 2)], runs[(r0, 4)]
    print(f"r0 = {r0}   (finest level-6 cell = 0.0244, so r0 = {r0/0.0244:.0f} cells at level 6)")
    print("  level   r range          max|err| FD2    max|err| FD4    ratio FD2/FD4")
    for L in range(a["lev"].max() + 1):
        out = []
        for run in (a, b):
            sel = (run["lev"] == 1) & (run["r"] > 25) & (run["r"] < 40)
            off = np.median(run["err"][sel]) if sel.any() else 0.0
            m = run["lev"] == L
            out.append((np.max(np.abs(run["err"][m] - off)), run["r"][m].min(), run["r"][m].max()))
        (e2, rmin, rmax), (e4, _, _) = out
        print(f"  L{L}     [{rmin:7.3f},{rmax:7.3f}]   {e2:12.3e}    {e4:12.3e}    {e2/e4 if e4 > 0 else float('inf'):8.2f}")
    print()
