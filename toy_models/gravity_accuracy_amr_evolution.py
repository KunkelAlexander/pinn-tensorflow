"""Small evolving 2D SP control with a hierarchy only in the gravity solve.

The wave grid stays uniform to isolate gravity coupling. Point-centered toy,
not GAMER; coarse sources are point samples, not cell-average restriction.
"""
import json
from experiments import np
from frozen_interference_gravity import coefficients,evaluate,L
from gravity_hierarchy_probe import interpolation
from gravity_accuracy_interface import solve
from gravity_accuracy_evolution import OUT


def green(n,order):
    h=L/n;k=2*np.pi*np.fft.fftfreq(n,d=h);s=np.sin(k*h/2)**2
    axis=k*k if order==0 else 4*s/h**2*(1+s/3 if order==4 else 1)
    lam=axis[:,None]+axis[None,:];g=np.zeros_like(lam);g[lam>0]=-20/lam[lam>0];return g


def run(n,label,shift,dt=.0005,final=.1):
    steps=round(final/dt);dt=final/steps;h=L/n;x=np.arange(n)*h-L/2
    p,_,_=coefficients(7);u=evaluate(p,x,x);k=2*np.pi*np.fft.fftfreq(n,d=h)
    drift=np.exp(-.5j*dt*(k[:,None]**2+k[None,:]**2));finegreen=green(n,0)
    if label.startswith('uniform'):finegreen=green(n,int(label[-1]))
    hierarchy=label.startswith('hierarchy');oracle=label=='oracle4'
    if hierarchy or oracle:
        base_label=label.replace('_projected','')
        order=int(base_label[9]) if hierarchy else 4;degree=int(base_label[-1]) if hierarchy else 4
        coarse=n//2;xp=x[::2];rootgreen=green(coarse,order);transfer=interpolation(xp,x,degree,periodic=True)
        start=n//4+shift;ids=np.arange(start-1,start+coarse+2)%n
        ys=np.arange(n//4-1,n//4+coarse+2)%n
        xx=(np.arange(start-1,start+coarse+2))*h-L/2;yy=(np.arange(n//4-1,n//4+coarse+2))*h-L/2
        interior=ids[2:-2];iy=ys[2:-2]
    max_residual=0.;max_parent_mean_error=0.
    def potential(wave):
        nonlocal max_residual,max_parent_mean_error
        rho=abs(wave)**2
        if not hierarchy:phi=np.fft.ifft2(np.fft.fft2(rho)*finegreen).real
        else:
            coarse_rho=rho[::2,::2]
            if label.endswith('_projected'):
                # Preserve resolved root density coefficients without aliasing
                # unresolved modes into them. Nyquist lines omitted symmetrically.
                rf=np.fft.fft2(rho)/n**2;freq=np.rint(np.fft.fftfreq(n)*n).astype(int)
                ids_keep=np.where(abs(freq)<coarse/2)[0];rootf=np.zeros((coarse,coarse),complex)
                rootf[np.ix_(freq[ids_keep]%coarse,freq[ids_keep]%coarse)]=rf[np.ix_(ids_keep,ids_keep)]
                coarse_rho=np.fft.ifft2(rootf*coarse**2).real
            max_parent_mean_error=max(max_parent_mean_error,abs(float(coarse_rho.mean()-rho.mean())))
            parent=np.fft.ifft2(np.fft.fft2(coarse_rho)*rootgreen).real
            phi=transfer@parent@transfer.T
        if hierarchy or oracle:
            bc=phi[np.ix_(ids,ys)];source=20*(rho[np.ix_(interior,iy)]-rho.mean())
            child,residual=solve(xx,yy,bc,source,order);max_residual=max(max_residual,residual)
            phi[np.ix_(interior,iy)]=child[2:-2,2:-2]
        return phi
    initialmass=float(np.sum(abs(u)**2))
    for _ in range(steps):
        u*=np.exp(-.5j*dt*potential(u));u=np.fft.ifft2(np.fft.fft2(u)*drift);u*=np.exp(-.5j*dt*potential(u))
    return u,dict(steps=steps,dt=dt,mass_change=float(np.sum(abs(u)**2)/initialmass-1),
                  max_poisson_residual=max_residual,max_parent_density_mean_error=max_parent_mean_error)


def main():
    n=96;ref,_=run(192,'uniform0',0,dt=.00025);fine,_=run(384,'uniform0',0,dt=.00025)
    half,_=run(192,'uniform0',0,dt=.000125)
    reference=abs(ref[::2,::2])**2;checks=dict(space=float(np.linalg.norm(abs(ref)**2-abs(fine[::2,::2])**2)/np.linalg.norm(abs(ref)**2)),
        time=float(np.linalg.norm(abs(ref)**2-abs(half)**2)/np.linalg.norm(abs(ref)**2)))
    rows=[];x=np.arange(n)*L/n-L/2;mask=x[:,None]**2+x[None,:]**2<.3**2
    for shift in (0,16):
        for label in ('uniform0','uniform2','uniform4','hierarchy2_2','hierarchy4_2','hierarchy4_4','oracle4'):
            p,stats=run(n,label,shift);rho=abs(p)**2
            rows.append(dict(label=label,shift_cells=shift,shift=shift*L/n,
                density_error=float(np.linalg.norm(rho-reference)/np.linalg.norm(reference)),
                central_density_error=float(np.linalg.norm((rho-reference)[mask])/np.linalg.norm(reference[mask])),**stats))
            print('evolving gravity hierarchy',shift,label,rows[-1]['density_error'],flush=True)
    assert checks['space']<1e-5,checks
    assert checks['time']<1e-5,checks
    assert max(abs(r['mass_change']) for r in rows)<1e-10
    result=dict(rows=rows,checks=checks,
        scope='2D evolving SP, L=2*pi, coupling=20, T=0.1, fine wave grid 96^2, parent gravity grid 48^2, child spans half box per axis. Initial shortest wavelength has 9.6 fine cells, 4.8 parent cells. All wave propagation uses the same global FFT. Hierarchy uses exact available fine density, sampled parent density, and FD2/FD4 direct child solves. Uniform branches apply the chosen gravity operator throughout. Oracle4 supplies spectral potential outside the child and on its boundaries; it changes both the exterior and boundary treatment. No wave AMR, subcycling, relaxation error, cell-average restriction, or cosmology.')
    (OUT/'amr_evolution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
