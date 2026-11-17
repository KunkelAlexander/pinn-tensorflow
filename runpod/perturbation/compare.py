import numpy as np, sys, glob, os
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
t = np.genfromtxt(os.path.join(d,'spectra_t2.csv'), delimiter=',', names=True)
k = t['k']; ref = t['ref512_spectral']
runs = {'default': t['hybrid_default'], 'fd4': t['hybrid_fd4']}
for f in sorted(glob.glob(os.path.join(d,'*_t2.txt'))):
    runs[os.path.basename(f)[:-7]] = np.loadtxt(f, delimiter=',')
bands = [(6,25),(25,50),(50,100),(100,200),(200,400),(400,800),(800,1700)]
print('%-10s' % 'band' + ''.join('%9s' % n for n in runs))
for lo, hi in bands:
    m = (k >= lo) & (k < hi)
    print('%-10s' % ('%d-%d' % (lo, hi)) + ''.join('%9.2f' % np.exp(np.mean(np.log(r[m]/ref[m]))) for r in runs.values()))
m = (k > 6)
print('%-10s' % 'RMSlog' + ''.join('%9.2f' % np.exp(np.sqrt(np.mean(np.log(r[m]/ref[m])**2))) for r in runs.values()))
print('%-10s' % 'RMSlogk<400' + ''.join('%9.2f' % np.exp(np.sqrt(np.mean(np.log(r[m&(k<400)]/ref[m&(k<400)])**2))) for r in runs.values()))
