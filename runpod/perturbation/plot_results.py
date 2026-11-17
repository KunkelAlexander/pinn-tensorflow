"""Make figures/band table from data/spectra_t2.csv and the digitized wiki curves. Run from this folder:
   C:/Users/TE/anaconda3/envs/pinn_env/python.exe plot_results.py"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.genfromtxt('data/spectra_t2.csv', delimiter=',', names=True)
k, ref, hyb, fd4 = d['k'], d['ref512_spectral'], d['hybrid_default'], d['hybrid_fd4']
wb = np.loadtxt('wiki_reference/blue_base1024.txt')    # digitized wiki: base-level spectral 1024^2
wg = np.loadtxt('wiki_reference/green_hybrid.txt')     # digitized wiki: hybrid AMR default

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].loglog(k, ref, 'k-', lw=2.5, label='ours: base-level spectral 512$^2$')
ax[0].loglog(k, hyb, 'C3-', lw=1, label='ours: hybrid AMR, default gravity')
ax[0].loglog(k, fd4, 'C0-', lw=1, label='ours: hybrid AMR, FD4 gravity')
ax[0].loglog(wb[:, 0], wb[:, 1], 'k:', lw=1, label='wiki (digitized): spectral 1024$^2$')
ax[0].loglog(wg[:, 0], wg[:, 1], 'C2:', lw=1, label='wiki (digitized): hybrid default')
ax[0].set(xlim=(5, 3500), ylim=(1e-8, 1), xlabel='k [h/Mpc] (index*2pi)', ylabel='P(k)', title='Perturbation, t = 2.0, density power spectrum')
ax[0].grid(True, which='major'); ax[0].legend(fontsize=8, loc='lower left')

ax[1].semilogx(k, hyb / ref, 'C3-', lw=1, label='default / reference')
ax[1].semilogx(k, fd4 / ref, 'C0-', lw=1, label='FD4 / reference')
r = np.interp(k, wb[:, 0], wb[:, 1]); g = np.interp(k, wg[:, 0], wg[:, 1]); m = (k < 1600) & (r > 1e-6)
ax[1].semilogx(k[m], (g / r)[m], 'C2:', lw=1.2, label='wiki hybrid / wiki reference (digitized)')
ax[1].axhline(1, color='k', lw=0.8); ax[1].set(ylim=(0, 8), xlabel='k [h/Mpc]', ylabel='P / P_ref', title='ratio to reference')
ax[1].grid(True, which='major'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('perturbation_spectra_t2.png', dpi=130)

E = [6, 12, 25, 50, 100, 200, 400, 800, 1200, 1609]
gm = lambda p, lo, hi: np.exp(np.log(p[(k >= lo) & (k < hi)]).mean())
print('band        default/ref  FD4/ref')
rows = []
for lo, hi in zip(E[:-1], E[1:]):
    a, b = gm(hyb, lo, hi) / gm(ref, lo, hi), gm(fd4, lo, hi) / gm(ref, lo, hi); rows.append((a, b))
    print('%4d-%4d    %6.2f      %6.2f' % (lo, hi, a, b))
rows = np.log(np.array(rows)); print('RMS log-ratio factor: default %.2f  FD4 %.2f' % tuple(np.exp(np.sqrt((rows**2).mean(axis=0)))))
