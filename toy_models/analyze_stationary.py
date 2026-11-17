"""Is the stationary-halo candidate stationary under spectral gravity?  Tables + figure from stationary_halo_n256<tag>.json.

Criteria reported: (i) mass/energy/momentum conserved; (ii) virial ratios ~1 and not drifting; (iii) time-window-averaged
azimuthal profiles agree with the self-consistent target and with each other; (iv) rms radius and core density statistics
per window (granule fluctuation vs systematic drift); (v) centroid stays near the origin.
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soliton_gravity import OUT


def annulus_mean(rho_r, r, edges):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        out.append(np.sum(rho_r[m] * r[m]) / np.sum(r[m]) if m.any() else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='_T4rc6')
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--windows', type=int, default=4)
    a = ap.parse_args()
    d = json.loads((OUT / f'stationary_halo_n{a.n}{a.tag}.json').read_text())
    t = np.array(d['times'])
    prof = np.array(d['profile_t'])
    edges = np.array(d['profile_r_edges'])
    rc = .5 * (edges[1:] + edges[:-1])
    target = annulus_mean(np.array(d['target_rho_r']), np.array(d['r_grid']), edges)
    mass = np.array(d['mass'])
    print(f"== {a.tag}: mass {d['args']['mass']}, T {d['args']['T']}, r_c {d['args']['rc']}, target {d['target_shape']}")
    print(f"mass drift {abs(mass / mass[0] - 1).max():.1e}; energy drift {abs(np.array(d['energy']) / d['energy'][0] - 1).max():.1e}; "
          f"max |p| {max(abs(np.array(d['px'])).max(), abs(np.array(d['py'])).max()):.2e}; max centroid {np.hypot(d['cx'], d['cy']).max():.3f}")
    kv, fv = np.array(d['kinetic_virial_ratio']), np.array(d['force_virial_ratio'])
    print(f'2K/(M^2/4pi): mean {kv.mean():.3f} std {kv.std():.3f}   force virial int rho x.grad phi/(M^2/4pi): mean {fv.mean():.3f} std {fv.std():.3f}')
    edges_t = np.linspace(t[0], t[-1] + 1e-9, a.windows + 1)
    wins = [(t >= lo) & (t < hi) for lo, hi in zip(edges_t[:-1], edges_t[1:])]
    wprof = [prof[m].mean(0) for m in wins]
    core, rms = np.array(d['core']), np.array(d['rms'])
    nrm = lambda x, y: np.linalg.norm(x - y) / np.linalg.norm(y)
    print('window        rms (mean+-std)   core density (mean+-std)   peak    profile err vs target   vs window 0   vs previous')
    for i, m in enumerate(wins):
        print(f'[{edges_t[i]:4.1f},{edges_t[i+1]:4.1f})   {rms[m].mean():.3f}+-{rms[m].std():.3f}     {core[m].mean():6.2f}+-{core[m].std():5.2f}'
              f'        {np.mean(np.array(d["peak"])[m]):6.2f}   {nrm(wprof[i], target):.3f}                  {nrm(wprof[i], wprof[0]):.3f}         {nrm(wprof[i], wprof[i-1]) if i else 0:.3f}')
    # least-squares drift of rms and core density per unit time, in units of their fluctuation
    for name, y in (('rms', rms), ('core', core)):
        c = np.polyfit(t, y, 1)
        print(f'linear trend of {name}: {c[0] * (t[-1] - t[0]) / y.mean() * 100:+.1f}% over the run (fluctuation std {y.std() / y.mean() * 100:.1f}%)')
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(t, rms); ax[0, 0].set_title('rms radius'); ax[0, 0].set_xlabel('t')
    ax[0, 1].plot(t, core, label='mean rho, r<1'); ax[0, 1].plot(t, d['peak'], label='peak'); ax[0, 1].set_yscale('log'); ax[0, 1].legend(); ax[0, 1].set_title('core / peak density')
    ax[0, 2].plot(t, kv, label='2K/(M^2/4pi)'); ax[0, 2].plot(t, fv, label='force virial'); ax[0, 2].axhline(1, color='k', lw=.5); ax[0, 2].legend(); ax[0, 2].set_title('virial ratios')
    cm = plt.get_cmap('viridis')
    ax[1, 0].plot(rc, target, 'k', lw=2.5, label='self-consistent target')
    for i, w in enumerate(wprof):
        ax[1, 0].plot(rc, w, color=cm(i / max(len(wprof) - 1, 1)), label=f't in [{edges_t[i]:.0f},{edges_t[i+1]:.0f})')
    ax[1, 0].set_yscale('log'); ax[1, 0].set_ylim(1e-4, 20); ax[1, 0].legend(fontsize=8); ax[1, 0].set_title('window-averaged azimuthal profiles'); ax[1, 0].set_xlabel('r')
    for i, w in enumerate(wprof):
        ax[1, 1].plot(rc, w / target - 1, color=cm(i / max(len(wprof) - 1, 1)))
    ax[1, 1].set_ylim(-.5, .5); ax[1, 1].axhline(0, color='k', lw=.5); ax[1, 1].set_title('window profile / target - 1'); ax[1, 1].set_xlabel('r'); ax[1, 1].set_xlim(0, 8)
    for k in range(0, len(t), max(1, len(t) // 8)):
        ax[1, 2].plot(rc, prof[k], lw=.8, color=cm(k / len(t)))
    ax[1, 2].plot(rc, target, 'k', lw=2); ax[1, 2].set_yscale('log'); ax[1, 2].set_ylim(1e-4, 40); ax[1, 2].set_title('instantaneous profiles (granules)'); ax[1, 2].set_xlabel('r')
    fig.suptitle(f"Stationary halo candidate {a.tag}: spectral gravity, mass {d['args']['mass']}, T {d['args']['T']}, r_c {d['args']['rc']}")
    fig.tight_layout()
    path = OUT / f'stationary_halo_n{a.n}{a.tag}_check.png'
    fig.savefig(path, dpi=130)
    print('wrote', path)


if __name__ == '__main__':
    main()
