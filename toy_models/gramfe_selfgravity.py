"""Actual-table Gram-FE kinetic update versus gravity symbol in periodic 1D SP."""
import json
from gramfe_probe import np,plt,OUT,operator,periodic_update,fft_update,LAMBDA
from experiments import modes,inverse_laplacian
from frozen_hamiltonian import setup
from scale_communication import spectral_values


def main():
    _,_,_,ground,mu,residual=setup(257)
    length=20.;final=2.;rows=[]
    for n in (64,128,256):
        h=length/n;x=(np.arange(n)+.5)*h-length/2;k=modes(n,length)
        for boost in (0,12):
            velocity=2*np.pi*boost/length
            initial=spectral_values(ground,x)*np.exp(1j*velocity*x)
            reference=spectral_values(ground,x,shift=velocity*final)*np.exp(1j*velocity*x-1j*(mu+.5*velocity**2)*final)
            for dt_factor in (1.,.5):
                steps=int(np.ceil(final/(dt_factor*LAMBDA*h*h)));dt=final/steps
                for kinetic in ('global_fft','stored_matmul','fft_fused'):
                    op=operator(dt/h**2,kinetic) if kinetic!='global_fft' else None
                    drift=np.exp(-.5j*k*k*dt)
                    for gravity in ('spectral','fd2'):
                        green=inverse_laplacian(n,length,gravity)
                        psi=initial.copy()
                        def potential(p):return np.fft.ifft(green*np.fft.fft(abs(p)**2)).real
                        for _ in range(steps):
                            psi*=np.exp(-.5j*dt*potential(psi))
                            if kinetic=='global_fft':psi=np.fft.ifft(np.fft.fft(psi)*drift)
                            else:psi=periodic_update(psi,op)
                            psi*=np.exp(-.5j*dt*potential(psi))
                        distance=(x-velocity*final+length/2)%length-length/2
                        density=abs(psi)**2;ref_density=abs(reference)**2
                        radius=np.sqrt(np.sum(distance**2*density)/np.sum(density))
                        ref_radius=np.sqrt(np.sum(distance**2*ref_density)/np.sum(ref_density))
                        rows.append(dict(n=n,boost=boost,carrier_ppw=None if boost==0 else n/boost,
                                         dt_factor=dt_factor,steps=steps,dt=dt,kinetic=kinetic,gravity=gravity,
                                         density_error=float(np.linalg.norm(abs(psi)**2-abs(reference)**2)/np.linalg.norm(abs(reference)**2)),
                                         wave_error=float(np.linalg.norm(psi-reference)/np.linalg.norm(reference)),
                                         relative_peak_density_error=float(np.max(density)/np.max(ref_density)-1),
                                         relative_rms_radius_error=float(radius/ref_radius-1),
                                         mass_change=float(np.sum(abs(psi)**2)/np.sum(abs(initial)**2)-1)))
            print('Gram-FE self-gravity',n,boost,flush=True)
    # The fine, global Fourier run validates reference and time discretization.
    accurate=next(r for r in rows if r['n']==256 and r['boost']==0 and r['dt_factor']==.5 and r['kinetic']=='global_fft' and r['gravity']=='spectral')
    assert accurate['density_error']<1e-6,accurate
    result=dict(rows=rows,ground_residual=residual,final_time=final,
                scope='Periodic single-level 1D SP, actual patch-size-8 Gram-FE tables, analytic translated stationary-core reference, exact spectral versus FD2 Poisson symbols. All methods use identical Strang coupling. No AMR interfaces, cosmology, or 3D halo.')
    (OUT/'selfgravity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in rows if r['n']==128 and r['boost']==12 and r['dt_factor']==.5],indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,boost in zip(axs,(0,12)):
        for kinetic in ('global_fft','stored_matmul','fft_fused'):
            for gravity in ('spectral','fd2'):
                rr=[r for r in rows if r['boost']==boost and r['kinetic']==kinetic and r['gravity']==gravity and r['dt_factor']==.5]
                ax.loglog([r['n'] for r in rr],[r['density_error'] for r in rr],'o--' if gravity=='fd2' else 'o-',label=kinetic+' / '+gravity)
        ax.set(xlabel='Grid points',ylabel='Relative density error',title=f'Core boost index {boost}');ax.legend(fontsize=6);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'selfgravity.png',dpi=160);plt.close(fig)


def runtime_check():
    _,_,_,ground,mu,_=setup(257)
    n=128;length=20.;final=2.;h=length/n;x=(np.arange(n)+.5)*h-length/2
    velocity=2*np.pi*12/length
    initial=spectral_values(ground,x)*np.exp(1j*velocity*x)
    reference=spectral_values(ground,x,shift=velocity*final)*np.exp(1j*velocity*x-1j*(mu+.5*velocity**2)*final)
    steps=int(np.ceil(final/(.5*LAMBDA*h*h)));dt=final/steps
    ids=(16*np.arange(n//16)[:,None]+np.arange(-8,24)[None,:])%n
    rows=[]
    for gravity in ('spectral','fd2'):
        green=inverse_laplacian(n,length,gravity);answers=[]
        for runtime in (False,True):
            psi=initial.copy();op=operator(dt/h**2,'fft_fused')
            for _ in range(steps):
                phi=np.fft.ifft(green*np.fft.fft(abs(psi)**2)).real;psi*=np.exp(-.5j*dt*phi)
                psi=fft_update(psi[ids],dt/h**2).ravel() if runtime else periodic_update(psi,op)
                phi=np.fft.ifft(green*np.fft.fft(abs(psi)**2)).real;psi*=np.exp(-.5j*dt*phi)
            answers.append(psi)
        rows.append(dict(gravity=gravity,runtime_vs_fused_wave=float(np.linalg.norm(answers[0]-answers[1])/np.linalg.norm(answers[0])),
                         runtime_density_error=float(np.linalg.norm(abs(answers[1])**2-abs(reference)**2)/np.linalg.norm(abs(reference)**2))))
    assert max(r['runtime_vs_fused_wave'] for r in rows)<1e-7,rows
    (OUT/'runtime_fft_check.json').write_text(json.dumps(dict(n=n,boost=12,rows=rows),indent=2)+'\n')
    print('Runtime FFT check',rows)


if __name__=='__main__':
    import sys
    if '--runtime-check' in sys.argv:runtime_check()
    else:main()
