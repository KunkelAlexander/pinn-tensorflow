"""Long-run norm stability with a prescribed real potential, no AMR.

Dense maps diagnose a small linear problem; not a proposed production solver.
Matrix powers advance exactly the fixed discrete map up to arithmetic error.
"""
import json
import argparse
from experiments import np,plt
from gramfe_probe import operator,LAMBDA,fft_update
from frozen_interference_gravity import OUT


def kinetic_matrix(n,lam,kind):
    if kind=='periodic_fft':
        q=2*np.pi*np.fft.fftfreq(n)
        return np.fft.ifft(np.exp(-.5j*lam*q*q)[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0)
    op=operator(lam,kind);matrix=np.zeros((n,n),complex)
    for start in range(0,n,16):
        matrix[np.ix_(np.arange(start,start+16),(start+np.arange(-8,24))%n)]=op
    return matrix


def main():
    rows=[];controls=[];mode=16;omega=mode*mode/2
    for n in (80,160):
        h=2*np.pi/n;x=np.arange(n)*h
        initial=np.cos(mode*x)+.3j*np.sin(mode*x+.4)
        initial=initial/np.linalg.norm(initial)
        for factor in (1.,.5):
            lam=LAMBDA*factor;dt=lam*h*h
            for kind in ('periodic_fft','stored_matmul','fft_fused'):
                kinetic=kinetic_matrix(n,lam,kind)
                for ratio in (0.,.05,.2):
                    potential=ratio*omega*np.cos(2*mode*x+.37)
                    kick=np.exp(-.5j*dt*potential)
                    update=kick[:,None]*kinetic*kick[None,:]
                    radius=float(np.max(abs(np.linalg.eigvals(update))))
                    growth=float(np.log(radius)/dt)
                    samples=[]
                    for target in (1.,10.,100.):
                        steps=round(target/dt);time=steps*dt
                        if growth*time>50:
                            samples.append(dict(time=time,steps=steps,skipped='worst-mode growth exceeds exp(50)'));continue
                        evolved=np.linalg.matrix_power(update,steps)@initial
                        samples.append(dict(time=time,steps=steps,mass_ratio=float(np.linalg.norm(evolved)**2)))
                    rows.append(dict(n=n,wave_ppw=n/mode,dt_factor=factor,dt=dt,kind=kind,
                        potential_over_carrier_energy=ratio,spectral_radius=radius,
                        amplitude_growth_rate_per_physical_time=growth,
                        efold_time=None if growth<=0 else 1/growth,samples=samples))
                    if kind=='fft_fused' and factor==1 and ratio==.05:
                        ids=(16*np.arange(n//16)[:,None]+np.arange(-8,24)[None,:])%n
                        runtime=initial.copy()
                        for _ in range(1000):
                            runtime=kick*fft_update((kick*runtime)[ids],lam).reshape(n)
                        fused=np.linalg.matrix_power(update,1000)@initial
                        controls.append(dict(n=n,steps=1000,runtime_vs_fused=float(np.linalg.norm(runtime-fused))))
            print(f'frozen potential n={n}, dt factor={factor} complete',flush=True)
    assert max(c['runtime_vs_fused'] for c in controls)<1e-5,controls
    fft=[r for r in rows if r['kind']=='periodic_fft']
    assert max(abs(s['mass_ratio']-1) for r in fft for s in r['samples'])<1e-8
    result=dict(rows=rows,controls=controls,
        scope='Periodic 1D linear Schrodinger equation, prescribed V=A*cos(2*k*x+0.37), k=16. Same-level patch maps, no AMR or self-gravity. Smooth initial standing wave with complex modulation. Dense matrix powers evaluate long times without loop-roundoff accumulation; actual factored FFT validates 1000 steps. Frozen potential amplitude ratios are chosen controls, not calibrated halo fields. Periodic FFT baseline is unitary on this same grid, not a spatially converged continuum oracle.')
    (OUT/'frozen_potential.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for n,ax in zip((80,160),axs):
        for kind in ('stored_matmul','fft_fused'):
            for factor in (1.,.5):
                rr=[r for r in rows if r['n']==n and r['kind']==kind and r['dt_factor']==factor]
                ax.plot([r['potential_over_carrier_energy'] for r in rr],[r['amplitude_growth_rate_per_physical_time'] for r in rr],
                    'o-' if factor==1 else 's--',label=f'{kind}, dt x {factor}')
        ax.set(xlabel='Potential amplitude / carrier kinetic energy',ylabel='Largest amplitude growth rate',title=f'{n/mode:g} cells per carrier wavelength')
        ax.set_yscale('symlog',linthresh=1e-5);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'frozen_potential.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(controls=controls,nonzero=[r for r in rows if r['kind']!='periodic_fft' and r['potential_over_carrier_energy']==.05]),indent=2))


def runtime_long():
    rows=[];n=160;mode=16;h=2*np.pi/n;dt=LAMBDA*h*h;x=np.arange(n)*h
    initial=np.cos(mode*x)+.3j*np.sin(mode*x+.4);initial/=np.linalg.norm(initial)
    kick=np.exp(-.5j*dt*(.2*mode*mode/2)*np.cos(2*mode*x+.37))
    ids=(16*np.arange(n//16)[:,None]+np.arange(-8,24)[None,:])%n
    targets={round(t/dt) for t in (1.,10.,25.,50.,100.)}
    for kind in ('stored_matmul','fft_runtime'):
        u=initial.copy();op=operator();map_kind='fft_fused' if kind=='fft_runtime' else kind
        update=kick[:,None]*kinetic_matrix(n,LAMBDA,map_kind)*kick[None,:]
        values,vectors=np.linalg.eig(update);j=np.argmax(abs(values));v=vectors[:,j];v/=np.linalg.norm(v)
        worst=v.copy()
        for _ in range(3000):
            padded=(kick*worst)[ids]
            worst=kick*(fft_update(padded) if kind=='fft_runtime' else padded@op.T).reshape(n)
        eigcheck=dict(predicted_norm=float(abs(values[j])**3000),runtime_norm=float(np.linalg.norm(worst)))
        for step in range(1,max(targets)+1):
            padded=(kick*u)[ids]
            u=kick*(fft_update(padded) if kind=='fft_runtime' else padded@op.T).reshape(n)
            if step in targets:
                predicted=np.linalg.matrix_power(update,step)@initial
                row=dict(kind=kind,n=n,steps=step,time=step*dt,mass_ratio=float(np.linalg.norm(u)**2),
                    matrix_power_mass=float(np.linalg.norm(predicted)**2),wave_difference=float(np.linalg.norm(u-predicted)),
                    eigenmode_3000_steps=eigcheck)
                rows.append(row);print(json.dumps(row),flush=True)
    assert all(abs(r['eigenmode_3000_steps']['runtime_norm']/r['eigenmode_3000_steps']['predicted_norm']-1)<1e-3 for r in rows)
    (OUT/'runtime_long.json').write_text(json.dumps(dict(rows=rows,
        scope='Independent repeated runtime updates at n=160, default dt, A/omega=0.2. Late-time amplitudes are seed/roundoff sensitive; agreement of exact amplitude is not required to establish growth.'),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-long',action='store_true');args=parser.parse_args()
    runtime_long() if args.runtime_long else main()
