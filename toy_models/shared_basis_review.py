"""One conforming high-order space and stiffness matrix for wave and gravity.

Periodic 1D graded mesh, exact polynomial mass/stiffness quadrature, global
Poisson solve, and exact-in-time free-wave propagation. This isolates spatial
accuracy; it is not yet a nonlinear SP solver or a scalable exponential method.
"""
import json
from conditioning_review import OUT, savefig
from experiments import np, plt, eigh, modes
from numpy.polynomial.legendre import Legendre, leggauss, legvander
from scipy.linalg import solve

EDGES=np.array([-10.,-6.,-3.,-1.5,-.75,0.,.75,1.5,3.,6.,10.])


def basis(p,z):
    nodes=np.r_[-1.,Legendre.basis(p).deriv().roots(),1.]
    coefficient=np.linalg.solve(legvander(nodes,p),np.eye(p+1))
    b=legvander(z,p)@coefficient
    derivative=np.column_stack([Legendre.basis(j).deriv()(z) for j in range(p+1)])@coefficient
    return b,derivative


def assemble(p,edges=EDGES,nquad=None):
    n=(len(edges)-1)*p
    z,w=leggauss(nquad or (2*p+6)); b,db=basis(p,z)
    m=np.zeros((n,n));k=m.copy(); xq=[];wq=[];evaluation=[]
    for e,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        ids=(np.arange(p+1)+e*p)%n;h=hi-lo
        m[np.ix_(ids,ids)]+=b.T@(w[:,None]*b)*h/2
        k[np.ix_(ids,ids)]+=db.T@(w[:,None]*db)*2/h
        block=np.zeros((len(z),n));block[:,ids]=b
        evaluation.append(block);xq.extend((lo+hi)/2+h*z/2);wq.extend(w*h/2)
    return m,k,np.array(xq),np.array(wq),np.vstack(evaluation)


def potential(x):
    g=np.exp(-x*x/(2*.4**2))
    return np.cos(2*np.pi*x/20)+.2*g


def source(x):
    g=np.exp(-x*x/(2*.4**2))
    return -(2*np.pi/20)**2*np.cos(2*np.pi*x/20)+.2*g*(x*x/.4**4-1/.4**2)


def initial(x):
    return np.exp(-x*x/(2*.6**2))*np.exp(3j*x)


def free_reference(x,time=.5,n=2048):
    xf=np.arange(n)*20/n-10
    c=np.fft.fft(initial(xf))/n
    k=modes(n,20)
    # Only occupied coefficients are evaluated; omitted absolute tail is bounded.
    occupied=abs(c)>1e-15
    omitted=float(np.sum(abs(c[~occupied])))
    result=np.exp(1j*np.outer(x+10,k[occupied]))@(c[occupied]*np.exp(-.5j*k[occupied]**2*time))
    return result,omitted


def norm(v,w): return float(np.sqrt(np.sum(w*abs(v)**2)))


def main():
    rows=[];checks=[]
    for p in (2,4,6,8,10,12,16,20,24):
        m,k,x,w,b=assemble(p)
        # Independent, much denser quadrature to measure continuum errors.
        _,_,xe,we,be=assemble(p,nquad=100)
        one=np.ones(len(m));mass=m@one
        load=b.T@(w*source(x))
        # Enforce source compatibility and physical zero-mean gauge.
        load-=mass*np.sum(load)/20
        saddle=np.block([[k,mass[:,None]],[mass[None,:],np.zeros((1,1))]])
        phi=solve(saddle,np.r_[-load,0.],assume_a='sym')[:-1]
        exact=potential(xe);exact-=np.sum(we*exact)/20
        gravity_error=norm(be@phi-exact,we)/norm(exact,we)
        e,v=eigh(.5*k,m)
        c0=solve(m,b.T@(w*initial(x)),assume_a='pos')
        c=v@(np.exp(-.5j*e)*(v.T@(m@c0)))
        ref,tail=free_reference(xe)
        ref2,_=free_reference(xe,n=4096)
        pred=be@c
        wave_error=norm(pred-ref,we)/norm(ref,we)
        density_error=norm(abs(pred)**2-abs(ref)**2,we)/norm(abs(ref)**2,we)
        mass_error=float(abs(np.vdot(c,m@c)-np.vdot(c0,m@c0))/np.vdot(c0,m@c0).real)
        d=np.sqrt(np.diag(m)); scaled=m/d[:,None]/d[None,:]
        # Equal-DOF uniform Fourier baseline: adaptivity is not assumed to win.
        nf=len(m); xu=np.arange(nf)*20/nf-10;ku=modes(nf,20)
        fu=np.fft.fft(initial(xu))/nf
        fourier_pred=np.exp(1j*np.outer(xe+10,ku))@(fu*np.exp(-.25j*ku*ku))
        rows.append(dict(degree=p,dofs=len(m),gravity_error=gravity_error,wave_error=wave_error,
                         density_error=density_error,mass_error=mass_error,
                         uniform_fourier_wave_error=norm(fourier_pred-ref,we)/norm(ref,we),
                         initial_projection_error=norm(be@c0-initial(xe),we)/norm(initial(xe),we),
                         scaled_mass_condition=float(np.linalg.cond(scaled)),
                         kinetic_max_eigenvalue=float(e[-1]),reference_tail_bound=tail,
                         reference_refinement=norm(ref-ref2,we)/norm(ref2,we)))
        checks.append(dict(p=p,mass_quadrature=float(abs(np.sum(m)-20)),
                           stiffness_constant_null=float(np.linalg.norm(k@one)),
                           mass_hermiticity=float(np.linalg.norm(m-m.T)),
                           stiffness_hermiticity=float(np.linalg.norm(k-k.T)),
                           poisson_residual=float(np.linalg.norm(k@phi+load))))
    assert max(r['mass_error'] for r in rows)<1e-12
    assert max(r['reference_refinement'] for r in rows)<1e-12
    assert max(c['mass_quadrature'] for c in checks)<1e-12
    assert max(c['stiffness_constant_null'] for c in checks)<1e-10
    assert max(c['poisson_residual'] for c in checks)<1e-10
    assert rows[-1]['gravity_error']<1e-7
    assert rows[-1]['wave_error']<1e-7, rows[-1]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key in ('gravity_error','wave_error','density_error'):
        axes[0].semilogy([r['dofs'] for r in rows],[r[key] for r in rows],'o-',label=key.replace('_',' '))
    axes[0].semilogy([r['dofs'] for r in rows],[r['uniform_fourier_wave_error'] for r in rows],'k--',label='Uniform Fourier wave')
    axes[0].set(xlabel='Degrees of freedom on the same graded mesh',ylabel='Physical L2 relative error')
    axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot(xe,abs(ref)**2,label='Exact free packet, t=0.5')
    axes[1].plot(xe,abs(pred)**2,'--',label='Degree 24')
    for edge in EDGES: axes[1].axvline(edge,color='gray',alpha=.3,lw=.7)
    axes[1].set(xlim=(-6,6),xlabel='x',ylabel='Density');axes[1].legend()
    savefig('shared_basis.png')
    result=dict(mesh=EDGES.tolist(),length=20,final_time=.5,rows=rows,checks=checks,
                scope='Independent manufactured Poisson and moving free-packet problems using exactly the same M and K; no self-gravity coupling or time-stepping cost measured')
    (OUT/'shared_basis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
