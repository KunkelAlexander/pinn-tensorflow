"""Session 4 figure: spectra ratio to the 512^2 spectral reference and total mass error vs time. Run from this folder:
   C:/Users/TE/anaconda3/envs/pinn_env/python.exe plot_session4.py"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.genfromtxt('data/spectra_t2.csv', delimiter=',', names=True)
k, ref = d['k'], d['ref512_spectral']
runs = [('uniform 512$^2$ GramFE wave solver (no hybrid)', 'wave_gramfe', 'k', 2.2),
        ('hybrid, best configuration (s4f2cq)', 's4f2cq', 'C0', 1.4),
        ('hybrid, 3rd-order flux + 4th-order QP', 'third_qp4', 'C2', 1.0),
        ('hybrid, shipped defaults', 'default_ship', 'C3', 1.0)]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for lab, f, c, lw in runs:
    p = np.loadtxt('data/%s_t2.txt' % f, delimiter=',')
    ax[0].semilogx(k, p / ref, color=c, lw=lw, label=lab)
ax[0].axhline(1, color='gray', lw=0.8)
ax[0].set(ylim=(0, 8), xlim=(6, 1700), xlabel='k [h/Mpc]', ylabel='P / P_ref', title='Perturbation, t = 2.0: density power spectrum vs spectral reference')
ax[0].grid(True, which='major'); ax[0].legend(fontsize=8)

t = [1.5, 1.7, 1.8, 1.9, 2.0]
mass = {'uniform GramFE (no hybrid)': ([-0.00, -0.18, -1.23, -3.77, -5.10], 'k'),
        'hybrid, best configuration': ([0.11, 0.06, -0.08, -0.73, -3.72], 'C0'),
        'hybrid, shipped defaults': ([0.11, 0.06, -0.14, -0.97, -3.54], 'C3'),
        'hybrid, 3rd-order flux + 4th-order QP': ([0.05, -0.02, -0.19, -1.22, -4.50], 'C2'),
        '4th-order restriction, phase only (res3)': ([0.10, 0.07, -0.02, -0.95, -10.55], 'C1'),
        '4th-order restriction, density only (res5)': ([0.10, -0.42, -6.18, -23.50, -40.88], 'C4')}
for lab, (m, c) in mass.items():
    ax[1].plot(t, m, 'o-', color=c, label=lab, lw=1.3)
ax[1].set(xlabel='time', ylabel='total mass error [%]', title='Total mass error (Mass_Psi)')
ax[1].set_yscale('symlog', linthresh=1.0)
ax[1].axhline(0, color='gray', lw=0.8); ax[1].grid(True); ax[1].legend(fontsize=7, loc='lower left')
plt.tight_layout(); plt.savefig('perturbation_spectra_t2_session4.png', dpi=130)
print('saved')
