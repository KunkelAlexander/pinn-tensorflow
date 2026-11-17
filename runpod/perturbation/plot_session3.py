"""Session 3 figure: default vs best configurations vs the 512^2 spectral reference. Run from this folder:
   C:/Users/TE/anaconda3/envs/pinn_env/python.exe plot_session3.py"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

t = np.genfromtxt('data/spectra_t2.csv', delimiter=',', names=True)
k, ref = t['k'], t['ref512_spectral']
load = lambda n: np.loadtxt('data/%s_t2.txt' % n, delimiter=',')
runs = [('default gravity + spectral interp. (2300 s)', t['hybrid_default'], 'C3'),
        ('FD4 everywhere (session 2)', t['hybrid_fd4'], 'C1'),
        ('best accuracy: FD4 SOR + base symbol at finest dh + CQuad ghosts (1263 s)', load('s4f2cq'), 'C0'),
        ('fast: wave solver from level 2, FD2 SOR, base symbol at finest dh, CQuad ghosts (380 s)', load('w2fd2'), 'C2')]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].loglog(k, ref, 'k-', lw=2.5, label='base-level spectral 512$^2$ (reference)')
for n, p, c in runs: ax[0].loglog(k, p, c, lw=1, label=n)
ax[0].set(xlim=(5, 2000), ylim=(1e-6, 1), xlabel='k [h/Mpc] (index*2pi)', ylabel='P(k)', title='Perturbation, t = 2.0')
ax[0].grid(True, which='major'); ax[0].legend(fontsize=7, loc='lower left')
for n, p, c in runs: ax[1].semilogx(k, p / ref, c, lw=1, label=n)
ax[1].axhline(1, color='k', lw=0.8); ax[1].set(ylim=(0, 8), xlabel='k [h/Mpc]', ylabel='P / P_ref', title='ratio to reference')
ax[1].grid(True, which='major')
plt.tight_layout(); plt.savefig('perturbation_spectra_t2_session3.png', dpi=130)
