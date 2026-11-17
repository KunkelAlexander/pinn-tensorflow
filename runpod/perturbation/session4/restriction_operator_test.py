import numpy as np
PS1, PS2 = 8, 16
W = np.zeros((PS1, PS2))
for i in range(1, PS1-1):
    W[i, 2*i-1:2*i+3] = np.array([-1, 9, 9, -1]) / 16
W[0, 0:4] = np.array([5, 15, -5, 1]) / 16
W[PS1-1, PS2-4:PS2] = np.array([1, -5, 15, 5]) / 16
def res4(B):   # B[k][j][i], separable, same order as the C++ (x, y, z)
    T = np.einsum('is,kjs->kji', W, B)
    T = np.einsum('js,ksi->kji', W, T)
    return np.einsum('ks,sji->kji', W, T)
def box(B):
    return B.reshape(PS1, 2, PS1, 2, PS1, 2).mean(axis=(1, 3, 5))
def gather(sons):   # sons[LocalID] = PS1^3 arrays; offsets as TABLE_02: bit0->x, bit1->y, bit2->z
    B = np.zeros((PS2, PS2, PS2))
    for lid in range(8):
        ox, oy, oz = (lid & 1) * PS1, ((lid >> 1) & 1) * PS1, ((lid >> 2) & 1) * PS1
        B[oz:oz+PS1, oy:oy+PS1, ox:ox+PS1] = sons[lid]
    return B

print('convergence on a smooth field (error at coarse cell centres, max norm; box average vs exact point value | exact cell average):')
for n in [1, 2, 4]:
    hf = 1.0 / (PS2 * n)      # fine spacing, block of 16 fine cells spans 16*hf
    x = (np.arange(PS2) + 0.5) * hf
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')   # [k][j][i] -> use symmetric f
    f = lambda X, Y, Z: np.sin(2*np.pi*3*X) * np.cos(2*np.pi*2*Y) * np.sin(2*np.pi*Z + 0.3) + 2.0
    B = f(X, Y, Z)
    xc = (2*np.arange(PS1) + 1.0) * hf
    XC, YC, ZC = np.meshgrid(xc, xc, xc, indexing='ij')
    exact_pt = f(XC, YC, ZC)
    e_res4 = np.abs(res4(B) - exact_pt)
    e_box = np.abs(box(B) - exact_pt)
    inner = (slice(1, 7),) * 3
    print(' n=%d  res4 max %.2e (interior %.2e) | box vs point value max %.2e' % (n, e_res4.max(), e_res4[inner].max(), e_box.max()))

print('\nmass preservation of the block: sum(res4)*8 - sum(fine): relative difference on a smooth field and on white noise')
rng = np.random.default_rng(1)
for name, B in [('smooth', f(*np.meshgrid(*(3*[(np.arange(PS2)+0.5)/PS2]), indexing='ij'))), ('noise ', rng.random((PS2, PS2, PS2)))]:
    print(' %s  res4 mean/fine mean - 1 = %+.3e   box = %+.1e' % (name, res4(B).mean()/B.mean() - 1, box(B).mean()/B.mean() - 1))

print('\ngain on a grid-scale mode (fine Nyquist and half Nyquist along x), interior coarse cells:')
for kk, lab in [(PS2/2, 'fine Nyquist'), (PS2/4, 'half fine Nyquist'), (PS2/8, 'quarter')]:
    xi = np.arange(PS2)
    B = np.cos(2*np.pi*kk*xi/PS2)[None, None, :] * np.ones((PS2, PS2, 1))
    xc = 2*np.arange(PS1) + 0.5   # coarse centre in fine index units
    ex = np.cos(2*np.pi*kk*(xc + 0.0)/PS2)     # point sample at the coarse centre (fine index coordinates, cell i centred at i)
    r4 = res4(B)[PS1//2, PS1//2, :]; bx = box(B)[PS1//2, PS1//2, :]
    print(' %-18s |res4|max %.3f   |box|max %.3f   (amplitude of the input = 1)' % (lab, np.abs(r4[1:7]).max(), np.abs(bx[1:7]).max()))
