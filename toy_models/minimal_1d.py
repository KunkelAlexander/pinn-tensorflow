"""1D minimal test: one soliton, spectral gravity vs coarse-grid fd2 Poisson (cell-mean restriction + quartic Lagrange).
1D SP (hbar=m=G=1): i psi_t = -psi''/2 + phi psi, phi'' = rho - mean(rho), periodic box L. Wave solver = exact Fourier Strang.
Reports (a) static net self-force of the soliton density, (b) dynamic displacement and peak change after time T, versus sigma/H.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import interp_matrix, OUT

L, N = 32., 2048
h = L / N
x = -L / 2 + (np.arange(N) + .5) * h
k = 2 * np.pi * np.fft.fftfreq(N, d=h)
gsp = np.zeros(N); gsp[1:] = -1 / k[1:] ** 2


def make_grav(kind, r=1):
    if kind == 'spectral' or r == 1:
        return lambda rho: np.fft.ifft(np.fft.fft(rho) * gsp).real
    m = N // r
    H = L / m
    xc = -L / 2 + (np.arange(m) + .5) * H
    kc = 2 * np.pi * np.fft.fftfreq(m, d=H)
    gc = np.zeros(m); gc[1:] = -H ** 2 / (4 * np.sin(kc[1:] * H / 2) ** 2)
    T = interp_matrix(xc, x, 4)
    return lambda rho: T @ np.fft.ifft(np.fft.fft(rho.reshape(m, r).mean(1)) * gc).real


def ground_state(mass):
    w = np.exp(-x ** 2 / (2 * (2. / mass) ** 2)) + 0j
    w *= np.sqrt(mass / (np.sum(abs(w) ** 2) * h))
    for dt, it in ((.01, 2500), (.002, 2500), (.0005, 3000)):
        dr = np.exp(-.5 * dt * k ** 2)
        for _ in range(it):
            phi = np.fft.ifft(np.fft.fft(abs(w) ** 2) * gsp).real
            w = w * np.exp(-.5 * dt * phi)
            w = np.fft.ifft(np.fft.fft(w) * dr)
            phi = np.fft.ifft(np.fft.fft(abs(w) ** 2) * gsp).real
            w = w * np.exp(-.5 * dt * phi)
            w *= np.sqrt(mass / (np.sum(abs(w) ** 2) * h))
    return w


def evolve(w, grav, T, dt):
    dr = np.exp(-.5j * dt * k ** 2)
    for _ in range(round(T / dt)):
        w = w * np.exp(-.5j * dt * grav(abs(w) ** 2))
        w = np.fft.ifft(np.fft.fft(w) * dr)
        w = w * np.exp(-.5j * dt * grav(abs(w) ** 2))
    return w


def mom(w):
    return float((w.conj() * np.fft.ifft(1j * k * np.fft.fft(w))).sum().imag * h)


def cen(w):
    rho = abs(w) ** 2
    return float((x * rho).sum() / rho.sum())


if __name__ == '__main__':
    R = 32
    H = R * h
    masses = [2, 4, 8, 16, 32, 64]
    res = []
    grs = {'spectral': make_grav('spectral'), f'coarse r{R}': make_grav('coarse', R), 'coarse r16': make_grav('coarse', 16)}
    print(f'N={N} h={h:.4f}, coarse r{R}: H={H:.3f}')
    print(f"{'mass':>5s} {'sigma':>7s} {'sig/H':>6s}   static net self-force/(M^2/2) max over offsets          dynamic T=4: displacement (max offsets) | peak ratio min")
    for M in masses:
        w0 = ground_state(M)
        rho = abs(w0) ** 2
        sig = float(np.sqrt((rho * x ** 2).sum() / rho.sum()))
        row = dict(mass=M, sigma=sig, static={}, dx={}, peak={})
        for name, g in grs.items():
            st, dxs, pk = [], [], []
            for s in np.linspace(0, 1, 4, endpoint=False):
                sh = int(round(48 * 8 + s * R)) if True else 0   # integer fine-cell shifts spanning one coarse cell
                w1 = np.roll(w0, 3 * R + int(s * R))
                r1 = abs(w1) ** 2
                F = -(r1 * np.gradient(g(r1), h)).sum() * h
                st.append(abs(F) / (M ** 2 / 2))
                T0 = 4.
                w2 = evolve(w1, g, T0, .002 if M < 64 else .001)
                dxs.append(abs(cen(w2) - cen(w1)))
                pk.append(float((abs(w2) ** 2).max() / r1.max()))
            row['static'][name], row['dx'][name], row['peak'][name] = max(st), max(dxs), min(pk)
        res.append(row)
        print(f"{M:5d} {sig:7.3f} {sig / H:6.2f}   " + '  '.join(f"{n}: {row['static'][n]:.1e}" for n in grs) + '   |   ' +
              '  '.join(f"{n}: {row['dx'][n]:.3f}/{row['peak'][n]:.2f}" for n in grs), flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    sg = np.array([r['sigma'] for r in res])
    for name in grs:
        ax[0].loglog(sg / H, [max(r['static'][name], 1e-12) for r in res], 'o-', label=name)
        ax[1].loglog(sg / H, [max(r['dx'][name], 1e-9) for r in res], 'o-', label=name)
    ax[0].set_ylabel('static |net self-force| / (M^2/2)'); ax[1].set_ylabel('soliton displacement after T=4')
    for a in ax:
        a.set_xlabel(f'sigma / H (H = {R} fine cells)'); a.grid(alpha=.3, which='both'); a.legend()
    fig.suptitle('1D: lone soliton, spectral vs coarse fd2 gravity')
    fig.tight_layout()
    fig.savefig(OUT / 'minimal_1d.png', dpi=130)
    print('wrote', OUT / 'minimal_1d.png')
