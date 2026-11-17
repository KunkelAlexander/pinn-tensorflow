"""Moving 2D core, gravity boundaries, and genuinely adaptive wave operators.

Reference FFT grids are diagnostic only. Candidate adaptive runs never allocate
a finest-level full-domain wave grid during evolution. Output evaluation does.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from experiments import np,plt
from adaptive_sipg import Space,L
from gramfe_probe import operator
from gravity_accuracy_interface import solve as child_solve
from gravity_hierarchy_probe import interpolation
from scipy.sparse.linalg import eigsh

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'zoom2d_results';T=.8;G=1.


def packet(x,y,time=0.,center=(-1.5,.3),sigma=.8,velocity=(np.pi,np.pi/8),mass=10.):
    z=sigma*sigma+1j*time;px=0j;py=0j
    for offset in (-L,0.,L):
        q=x-center[0]+offset
        px=px+np.exp(-(q-velocity[0]*time)**2/(2*z)+1j*velocity[0]*q-.5j*velocity[0]**2*time)
        q=y-center[1]+offset
        py=py+np.exp(-(q-velocity[1]*time)**2/(2*z)+1j*velocity[1]*q-.5j*velocity[1]**2*time)
    return np.sqrt(mass/(np.pi*sigma*sigma))*sigma*sigma/z*px*py


def initial(x,y,case='coupled'):
    p=packet(x,y)
    if case=='coupled':
        p=p+.15*packet(x,y,center=(2.,1.),sigma=1.,velocity=(-np.pi/2,-np.pi/4),mass=10.)+.003
    return p


def grid(n):return -L/2+(np.arange(n)+.5)*L/n


def green(n,order=0):
    h=L/n;k=2*np.pi*np.fft.fftfreq(n,d=h);s=np.sin(k*h/2)**2
    axis=k*k if order==0 else 4*s/h**2*(1+s/3 if order==4 else 1)
    denom=axis[:,None]+axis[None,:];out=np.zeros_like(denom);out[denom>0]=-G/denom[denom>0]
    return out


class FourierSampler:
    def __init__(self,n,x,y):
        self.shape=np.shape(x);self.ux,self.ix=np.unique(np.asarray(x).ravel(),return_inverse=True)
        self.uy,self.iy=np.unique(np.asarray(y).ravel(),return_inverse=True)
        self.k=2*np.pi*np.fft.fftfreq(n,d=L/n);origin=grid(n)[0]
        self.ex=np.exp(1j*np.outer(self.ux-origin,self.k));self.ey=np.exp(1j*np.outer(self.uy-origin,self.k));self.n=n
    def __call__(self,field,derivative=(0,0)):
        c=np.fft.fft2(field)/self.n**2
        if derivative!=(0,0):c=c*(1j*self.k[:,None])**derivative[0]*(1j*self.k[None,:])**derivative[1]
        out=self.ex@c@self.ey.T
        return out[self.ix,self.iy].reshape(self.shape)


def reference(n=256,dt=.002,store=True):
    start=perf_counter();x=grid(n);xx,yy=np.meshgrid(x,x,indexing='ij');p=initial(xx,yy)
    k=2*np.pi*np.fft.fftfreq(n,d=L/n);sq=k[:,None]**2+k[None,:]**2;gr=green(n)
    dt=.01/round(.01/dt);sub=round(.01/dt);a=1/(2-2**(1/3));stages=[(dt*f,np.exp(-.5j*sq*dt*f)) for f in (a,1-2*a,a)]
    potentials=[];waves=[];times=[];band=[]
    def phi(w):return np.fft.ifft2(gr*np.fft.fft2(abs(w)**2)).real
    for j in range(round(T/.01)+1):
        if store:
            potentials.append(phi(p))
            if j%10==0:waves.append(p.copy());times.append(j*.01)
            power=abs(np.fft.fft2(p))**2
            band.append(dict(time=j*.01,**{f'power_below_6cells_N{m}':float(power[(abs(k[:,None])>2*np.pi*m/(6*L))|(abs(k[None,:])>2*np.pi*m/(6*L))].sum()/power.sum()) for m in (64,128,256)}))
        if j==round(T/.01):break
        for _ in range(sub):
            for h,drift in stages:
                p*=np.exp(-.5j*h*phi(p));p=np.fft.ifft2(np.fft.fft2(p)*drift);p*=np.exp(-.5j*h*phi(p))
    return dict(final=p,potentials=np.array(potentials),waves=np.array(waves),times=np.array(times),bandwidth=band,
                n=n,dt=dt,seconds=perf_counter()-start)


def norm(v,w=1.):return float(np.sqrt(np.sum(w*abs(v)**2)))


def metrics(p,ref,w=1.,x=None,y=None):
    overlap=np.sum(w*ref.conj()*p);aligned=p*np.exp(-1j*np.angle(overlap));rho=abs(p)**2;rr=abs(ref)**2
    out=dict(wave_error=norm(p-ref,w)/norm(ref,w),aligned_wave_error=norm(aligned-ref,w)/norm(ref,w),
             density_error=norm(rho-rr,w)/norm(rr,w),mass_bias=float(np.sum(w*rho)/np.sum(w*rr)-1))
    if x is not None:
        cmx=np.sum(w*x*rho)/np.sum(w*rho);cmy=np.sum(w*y*rho)/np.sum(w*rho)
        rx=np.sum(w*x*rr)/np.sum(w*rr);ry=np.sum(w*y*rr)/np.sum(w*rr)
        out['center_error']=float(np.hypot(cmx-rx,cmy-ry))
        radius=np.sqrt((x-rx)**2+(y-ry)**2);mask=radius<3
        out['core_density_error']=norm((rho-rr)[mask],np.broadcast_to(w,np.shape(rho))[mask])/norm(rr[mask],np.broadcast_to(w,np.shape(rho))[mask])
        # Fixed physical bins centered on the reference COM; evaluate identically.
        ids=np.minimum((radius/.25).astype(int),15).ravel();wt=np.broadcast_to(w,np.shape(rho)).ravel()
        counts=np.bincount(ids,weights=wt,minlength=16)
        prof=np.bincount(ids,weights=wt*rho.ravel(),minlength=16)/np.maximum(counts,1e-100)
        pref=np.bincount(ids,weights=wt*rr.ravel(),minlength=16)/np.maximum(counts,1e-100)
        out['profile_error']=float(np.linalg.norm(prof-pref)/np.linalg.norm(pref));out['profile']=prof.tolist();out['reference_profile']=pref.tolist()
    return out


def save_reference():
    r=reference();print('Reference 256 complete',r['seconds'],flush=True)
    fine=reference(512,store=False);print('Reference 512 complete',fine['seconds'],flush=True)
    half=reference(256,.001,store=False);print('Reference half time step complete',half['seconds'],flush=True)
    x=grid(256);xx,yy=np.meshgrid(x,x,indexing='ij');sampler=FourierSampler(512,xx,yy)
    checks=dict(space=metrics(r['final'],sampler(fine['final'])),time=metrics(r['final'],half['final']))
    assert checks['space']['wave_error']<1e-6,checks
    assert checks['time']['wave_error']<1e-6,checks
    np.savez_compressed(OUT/'reference.npz',**{k:r[k] for k in ('final','potentials','waves','times')})
    (OUT/'reference.json').write_text(json.dumps(dict(n=256,dt=r['dt'],time=T,coupling=G,checks=checks,
        bandwidth=r['bandwidth'],seconds=r['seconds'],scope='Periodic 2D SP, L=32. Moving Gaussian core mass10, weak counter-moving packet and uniform amplitude .003; not a stationary soliton or 3D halo. Fourier Yoshida4 reference, space/time refinements. Potential stored every .01; wave every .1.'),indent=2)+'\n')


def loadref():return np.load(OUT/'reference.npz')


def sipg(p,kind='adaptive',case='coupled',dt=.01,replay=False):
    start=perf_counter();sp=Space(p,kind);x,y,w,b=sp.samples();values=initial(x,y,case)
    c=sp.project(values,w,b);u=sp.sqrtm*c;initialmass=float(np.vdot(u,u).real)
    initerror=norm(sp.evaluate(c,b)-values,w)/norm(values,w)
    diagnostics=sp.diagnostics();hist=[];steps=round(T/dt);dt=T/steps;advance,degree=sp.exponential(dt)
    ref=None if case=='free' else loadref();sampler=None if case=='free' else FourierSampler(256,x,y)
    nodal_sampler=FourierSampler(256,sp.x,sp.y) if replay else None
    replay_phi=None
    if replay:
        assert abs(dt/.01-round(dt/.01))<1e-12
        replay_phi=[nodal_sampler(ref['potentials'][round(j*dt/.01)]).real for j in range(steps+1)]
    cgstats=sp.cg_probe(abs(c)**2)
    def energy(v):
        cc=v/sp.sqrtm;phi=sp.poisson(abs(cc)**2)
        return float(np.vdot(v,sp.h@v).real+.5*np.dot(sp.m*abs(cc)**2,phi))
    e0=energy(u) if case=='coupled' else float(np.vdot(u,sp.h@u).real);evolution_seconds=0.
    for j in range(steps+1):
        if j%max(1,round(.1/dt))==0 or j==steps:
            truth=packet(x,y,j*dt) if case=='free' else sampler(ref['waves'][round(j*dt/.1)])
            prediction=sp.evaluate(u/sp.sqrtm,b);proj=sp.project(truth,w,b)
            hist.append(dict(time=j*dt,**metrics(prediction,truth,w,x,y),
                reference_projection_error=norm(sp.evaluate(proj,b)-truth,w)/norm(truth,w)))
        if j==steps:break
        tic=perf_counter()
        if case=='coupled':
            phi=replay_phi[j] if replay else sp.poisson(abs(u/sp.sqrtm)**2)
            u*=np.exp(-.5j*dt*phi)
        u=advance(u)
        if case=='coupled':
            phi=replay_phi[j+1] if replay else sp.poisson(abs(u/sp.sqrtm)**2)
            u*=np.exp(-.5j*dt*phi)
        evolution_seconds+=perf_counter()-tic
    e1=energy(u) if case=='coupled' else float(np.vdot(u,sp.h@u).real)
    final=sp.on_uniform(u/sp.sqrtm,256)
    reverse=None
    if case=='free':
        q=grid(256);xx,yy=np.meshgrid(q,q,indexing='ij');true=packet(xx,yy,T)
        diff=np.fft.fft2(final-true);k=np.fft.fftfreq(256)*256
        reverse=float(np.sum(abs(diff[k<0,:])**2)/np.sum(abs(np.fft.fft2(true))**2))
    stats=dict(method='SIPG',case=case,mesh=kind,dt=dt,steps=steps,replay=replay,
        **diagnostics,initial_projection_error=initerror,**{k:v for k,v in hist[-1].items() if k!='time'},
        mass_change=float(np.vdot(u,u).real/initialmass-1),energy_change=(e1-e0)/max(abs(e0),1.),
        chebyshev_degree=degree,kinetic_matvecs=sp.kinetic_calls,poisson_calls=sp.poisson_calls,
        poisson_factor_nnz=sp._poisson.L.nnz+sp._poisson.U.nnz if sp._poisson is not None else 0,
        reverse_error_mass=reverse,cg_probe=cgstats,seconds=perf_counter()-start,evolution_seconds=evolution_seconds,
        factor_setup_seconds=sp.factor_setup_seconds,history=hist)
    return final,stats


def gravity_hierarchy(n,label,placement):
    h=L/n;x=grid(n);rootx=-L/2+(np.arange(n//2)+.5)*2*h;order=4 if '4' in label else 2
    degree=4 if 'quartic' in label else 2;rootgreen=green(n//2,order)
    transfer=interpolation(rootx,x,degree,periodic=True)
    start=0 if placement=='crossing' else n//4;child=n//2
    ids=np.arange(start-1,start+child+2)%n;ys=np.arange(n//4-1,3*n//4+2)%n
    xx=-L/2+(np.arange(start-1,start+child+2)+.5)*h;yy=-L/2+(np.arange(n//4-1,3*n//4+2)+.5)*h
    innerx=ids[2:-2];innery=ys[2:-2];stats=dict(max_residual=0.,max_source_mean_difference=0.)
    def phi(wave):
        rho=abs(wave)**2;coarse=rho.reshape(n//2,2,n//2,2).mean(axis=(1,3))
        stats['max_source_mean_difference']=max(stats['max_source_mean_difference'],abs(float(coarse.mean()-rho.mean())))
        parent=np.fft.ifft2(np.fft.fft2(coarse)*rootgreen).real
        out=transfer@parent@transfer.T;bc=out[np.ix_(ids,ys)]
        childphi,res=child_solve(xx,yy,bc,G*(rho[np.ix_(innerx,innery)]-rho.mean()),order)
        stats['max_residual']=max(stats['max_residual'],res);out[np.ix_(innerx,innery)]=childphi[2:-2,2:-2]
        return out-out.mean()
    return phi,stats


def uniform(n=128,method='gram_fft',gravity='spectral',placement='crossing',dt=.002,case='coupled'):
    start=perf_counter();steps=round(T/dt);dt=T/steps;x=grid(n);xx,yy=np.meshgrid(x,x,indexing='ij');u=initial(xx,yy,case)
    initialmass=float(np.sum(abs(u)**2));h=L/n;k=2*np.pi*np.fft.fftfreq(n,d=h)
    drift=np.exp(-.5j*dt*(k[:,None]**2+k[None,:]**2));extra={}
    if gravity.startswith('hierarchy'):phi,extra=gravity_hierarchy(n,gravity,placement)
    else:
        gr=green(n,0 if gravity=='spectral' else 2)
        def phi(p):return np.fft.ifft2(np.fft.fft2(abs(p)**2)*gr).real
    if method=='gram_fft':
        op=operator(dt/h**2,'fft_fused');ids=(16*np.arange(n//16)[:,None]+np.arange(-8,24))%n
        def column(p):return (p[ids,:].transpose(0,2,1)@op.T).transpose(0,2,1).reshape(n,n)
        def step(p):return column(column(p).T).T
    else:
        def step(p):return np.fft.ifft2(np.fft.fft2(p)*drift)
    tic=perf_counter()
    for _ in range(steps):
        if case=='coupled':u*=np.exp(-.5j*dt*phi(u))
        u=step(u)
        if case=='coupled':u*=np.exp(-.5j*dt*phi(u))
    evolution_seconds=perf_counter()-tic
    ref=packet(xx,yy,T) if case=='free' else FourierSampler(256,xx,yy)(loadref()['final'])
    result=dict(method=method,n=n,dofs=n*n,case=case,gravity=gravity,placement=placement,dt=dt,steps=steps,
                **metrics(u,ref,np.full_like(xx,h*h),xx,yy),mass_change=float(np.sum(abs(u)**2)/initialmass-1),
                seconds=perf_counter()-start,evolution_seconds=evolution_seconds,**extra)
    return u,result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('part',choices=['reference','adaptive','uniform','controls']);args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.part=='reference':save_reference();return
    rows=[];fields={}
    if args.part=='adaptive':
        for case in ('free','coupled'):
            for kind in ('adaptive','covered'):
                for p in (3,5,7):
                    final,r=sipg(p,kind,case);rows.append(r);fields[f'{case}_{kind}_p{p}']=final
                    print(case,kind,p,'rho',r['density_error'],'wave',r['wave_error'],'seconds',r['seconds'],flush=True)
                    (OUT/'adaptive.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
                    np.savez_compressed(OUT/'adaptive_fields.npz',**fields)
    elif args.part=='uniform':
        for case in ('free','coupled'):
            for n in (48,64,96,128,256):
                final,r=uniform(n,'global_fft',case=case);rows.append(r)
                if n in (64,128,256):fields[f'{case}_fft_{n}']=final
                print(case,'FFT',n,r['density_error'],flush=True)
        for n in (128,256):
            for gravity in ('spectral','fd2','hierarchy2_quadratic','hierarchy4_quartic'):
                for placement in (('crossing','covered') if gravity.startswith('hierarchy') else ('crossing',)):
                    final,r=uniform(n,'gram_fft',gravity,placement);rows.append(r);fields[f'coupled_gram_{n}_{gravity}_{placement}']=final
                    print('Gram',n,gravity,placement,r['density_error'],flush=True)
                    (OUT/'uniform.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
                    np.savez_compressed(OUT/'uniform_fields.npz',**fields)
    elif args.part=='controls':
        for p in (5,7):
            for dt in (.005,.02):
                final,r=sipg(p,'adaptive','coupled',dt);rows.append(r);print('time',p,dt,r['density_error'],flush=True)
            final,r=sipg(p,'adaptive','coupled',.01,replay=True);rows.append(r);print('replay',p,r['density_error'],flush=True)
        for n in (128,256):
            for gravity in ('spectral','hierarchy4_quartic'):
                final,r=uniform(n,'gram_fft',gravity,dt=.001);rows.append(r);print('Gram half time',n,gravity,r['density_error'],flush=True)
        (OUT/'controls.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
