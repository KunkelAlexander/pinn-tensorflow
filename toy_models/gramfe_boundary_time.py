"""Isolate time interpolation of AMR ghost waves with exact spatial data.

Parent endpoints are analytic, so there is no coarse propagation or spatial
interpolation error. Fine patch groups use reconstructed GAMER Gram-FE updates.
This isolates one coupling mechanism; it is not a full AMR implementation.
"""
import json
from gramfe_probe import np,plt,OUT,operator,LAMBDA


def main():
    n=64;steps=2048;dt=LAMBDA;parent_dt=4*dt
    x=np.arange(n)+.5;ghost_x=np.r_[np.arange(-8,0)+.5,np.arange(n,n+8)+.5]
    ids=16*np.arange(n//16)[:,None]+np.arange(32)[None,:]
    rows=[]
    for ppw in (6.,10.,16.):
        k=np.pi/ppw  # ppw refers to the COARSE spacing 2, fine has twice as many.
        # Stay below the public ELBDM gravity phase cap 2*pi*0.125 = pi/4.
        for potential_rotation in (0.,.25,.5):
            potential=potential_rotation/parent_dt;omega=.5*k*k+potential
            for kind in ('stored_matmul','fft_fused'):
                op=operator(kind=kind)*np.exp(-1j*potential*dt)
                for boundary in ('exact','complex_linear','density_phase'):
                    psi=np.exp(1j*k*x)
                    for step in range(steps):
                        t=step*dt;beta=(step%4)/4;t0=(step//4)*parent_dt
                        if boundary=='exact':ghost=np.exp(1j*k*ghost_x-1j*omega*t)
                        else:
                            left=np.exp(1j*k*ghost_x-1j*omega*t0)
                            right=np.exp(1j*k*ghost_x-1j*omega*(t0+parent_dt))
                            if boundary=='complex_linear':ghost=(1-beta)*left+beta*right
                            else:
                                phase=np.angle(left)+beta*np.angle(right*left.conj())
                                density=(1-beta)*abs(left)**2+beta*abs(right)**2
                                ghost=np.sqrt(density)*np.exp(1j*phase)
                        padded=np.r_[ghost[:8],psi,ghost[8:]]
                        psi=(padded[ids]@op.T).ravel()
                    ref=np.exp(1j*k*x-1j*omega*dt*steps)
                    rows.append(dict(coarse_ppw=ppw,fine_ppw=2*ppw,kind=kind,boundary=boundary,
                                     potential_phase_per_parent_step=potential_rotation,
                                     total_phase_per_parent_step=omega*parent_dt,
                                     wave_error=float(np.linalg.norm(psi-ref)/np.sqrt(n)),
                                     density_error=float(np.linalg.norm(abs(psi)**2-1)/np.sqrt(n)),
                                     mass_ratio=float(np.mean(abs(psi)**2))))
            print('boundary control',ppw,potential_rotation,flush=True)
    assert max(r['density_error'] for r in rows if r['boundary']=='exact')<1e-4
    assert max(r['density_error'] for r in rows if r['boundary']=='density_phase')<1e-4
    result=dict(steps=steps,dt=dt,parent_subcycles=4,rows=rows,
                scope='Prescribed exact coarse endpoints and exact spatial evaluation. Constant potential changes only global phase in the continuum. A phase-density temporal control is exact for this single wave, not for general interference. FFT fused is the high-precision linear-map reconstruction of its actual table/filter, not repeated runtime FFT arithmetic.')
    (OUT/'boundary_time.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in rows if r['coarse_ppw']==10],indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for kind in ('stored_matmul','fft_fused'):
        for boundary in ('exact','complex_linear','density_phase'):
            rr=[r for r in rows if r['kind']==kind and r['boundary']==boundary and r['coarse_ppw']==10]
            axs[0].semilogy([r['potential_phase_per_parent_step'] for r in rr],[max(r['density_error'],1e-15) for r in rr],'o-',label=kind+' '+boundary)
        rr=[r for r in rows if r['kind']==kind and r['boundary']=='complex_linear' and r['potential_phase_per_parent_step']==0]
        axs[1].semilogy([r['fine_ppw'] for r in rr],[r['density_error'] for r in rr],'o-',label=kind)
    axs[0].set(xlabel='Constant-potential phase per parent step',ylabel='Relative density error',title='Same density, different phase rotation');axs[0].legend(fontsize=6,loc='center right');axs[0].grid(alpha=.25)
    axs[1].set(xlabel='Fine-grid points per wavelength',ylabel='Density error: complex time interpolation',title='Zero potential');axs[1].legend();axs[1].grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'boundary_time.png',dpi=160);plt.close(fig)


def refinement_controls():
    # Fixed physical plane wave and final time; vary child spatial resolution
    # separately from parent temporal resolution.
    k=np.pi/10;parent_base=4*LAMBDA;potential=.5/parent_base;omega=.5*k*k+potential
    final=2048*LAMBDA;rows=[]
    for h,time_factor in ((1.,1.),(.5,1.),(1.,.5),(1.,.25)):
        n=round(64/h);dt=LAMBDA*h*h*time_factor;parent_dt=parent_base*time_factor
        steps=round(final/dt);subcycles=round(parent_dt/dt)
        x=(np.arange(n)+.5)*h;gx=np.r_[np.arange(-8,0)+.5,np.arange(n,n+8)+.5]*h
        ids=16*np.arange(n//16)[:,None]+np.arange(32)[None,:]
        op=operator(dt/h**2,'fft_fused')*np.exp(-1j*potential*dt)
        psi=np.exp(1j*k*x)
        for step in range(steps):
            beta=(step%subcycles)/subcycles;t0=(step//subcycles)*parent_dt
            ghost=(1-beta)*np.exp(1j*k*gx-1j*omega*t0)+beta*np.exp(1j*k*gx-1j*omega*(t0+parent_dt))
            padded=np.r_[ghost[:8],psi,ghost[8:]];psi=(padded[ids]@op.T).ravel()
        rows.append(dict(fine_h=h,fine_ppw=2*np.pi/k/h,parent_dt=parent_dt,fine_dt=dt,subcycles=subcycles,
                         density_error=float(np.linalg.norm(abs(psi)**2-1)/np.sqrt(n))))
    assert rows[2]['density_error']<rows[0]['density_error']/3
    assert rows[3]['density_error']<rows[2]['density_error']/3
    (OUT/'boundary_refinement.json').write_text(json.dumps(dict(rows=rows,scope='Same physical wave, potential, box and final time; spatial refinement keeps parent endpoint times fixed, temporal controls reduce both level timesteps.'),indent=2)+'\n')
    print('Boundary refinement controls',rows,flush=True)


if __name__=='__main__':
    import sys
    if '--refinement' in sys.argv:refinement_controls()
    else:main()
