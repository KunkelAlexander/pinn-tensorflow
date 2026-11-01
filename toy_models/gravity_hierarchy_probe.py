"""Cheap dimensionality and inherited-boundary controls for the GAMER audit.

Not GAMER: a 2D nested Dirichlet Poisson model isolates effects absent from 1D.
Root symbols match periodic spectral / standard second-order finite differences.
Child sources are exact. Direct solves remove relaxation-tolerance ambiguity.
"""
import json
from functools import lru_cache
from experiments import np,plt
from gramfe_probe import OUT
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import factorized

L=20.
CLUMPS=[(1.,.7,.13,-.07),(2.,1.,3.5,1.7)]


def density(x,y):
    xx,yy=np.meshgrid(x,y,indexing='ij');rho=np.zeros_like(xx)
    for mass,sigma,cx,cy in CLUMPS:
        for ix in (-1,0,1):
            for iy in (-1,0,1):
                rho+=mass/(2*np.pi*sigma**2)*np.exp(-((xx-cx+ix*L)**2+(yy-cy+iy*L)**2)/(2*sigma**2))
    return rho-sum(c[0] for c in CLUMPS)/L**2


def exact(x,y,derivative=(0,0),band=64):
    k=2*np.pi*np.arange(-band,band+1)/L
    kx,ky=np.meshgrid(k,k,indexing='ij');square=kx*kx+ky*ky
    c=np.zeros_like(square,complex)
    for mass,sigma,cx,cy in CLUMPS:
        c+=mass/L**2*np.exp(-.5*sigma**2*square)*np.exp(-1j*(kx*cx+ky*cy))
    mask=square>0;c[mask]/=-square[mask];c[~mask]=0
    c*=(1j*kx)**derivative[0]*(1j*ky)**derivative[1]
    return (np.exp(1j*np.outer(x,k))@c@np.exp(1j*np.outer(k,y))).real


@lru_cache(None)
def solver(n):
    d=diags([np.ones(n-2),-2*np.ones(n-1),np.ones(n-2)],[-1,0,1],format='csc')
    a=kron(d,eye(n-1))+kron(eye(n-1),d)
    return a.tocsc(),factorized(a.tocsc())


def solve_patch(x,boundary):
    n=len(x)-1;h=x[1]-x[0];a,solve=solver(n)
    rhs=h*h*density(x[1:-1],x[1:-1])
    rhs[0,:]-=boundary[0,1:-1];rhs[-1,:]-=boundary[-1,1:-1]
    rhs[:,0]-=boundary[1:-1,0];rhs[:,-1]-=boundary[1:-1,-1]
    phi=boundary.copy();phi[1:-1,1:-1]=solve(rhs.ravel()).reshape(n-1,n-1)
    residual=np.linalg.norm(a@phi[1:-1,1:-1].ravel()-rhs.ravel())/max(np.linalg.norm(rhs),1e-30)
    return phi,float(residual)


def interpolation(parent_x,child_x,degree,periodic=False):
    h=parent_x[1]-parent_x[0];n=len(parent_x);out=np.zeros((len(child_x),n))
    for row,x in enumerate(child_x):
        j=int(np.floor((x-parent_x[0])/h+.5));ids=np.arange(j-degree//2,j+degree//2+1)
        if not periodic:ids=np.clip(ids,0,n-1)
        nodes=parent_x[0]+ids*h
        weights=np.ones(len(ids))
        for r in range(len(ids)):
            for s in range(len(ids)):
                if s!=r:weights[r]*=(x-nodes[s])/(nodes[r]-nodes[s])
        out[row,ids%n]=weights
    return out


def hessian(phi,h):
    j=len(phi)//2
    xx=(phi[j+1,j]-2*phi[j,j]+phi[j-1,j])/h**2
    yy=(phi[j,j+1]-2*phi[j,j]+phi[j,j-1])/h**2
    xy=(phi[j+1,j+1]-phi[j+1,j-1]-phi[j-1,j+1]+phi[j-1,j-1])/(4*h*h)
    return np.array([[xx,xy],[xy,yy]])


def main():
    rows=[];profiles=[]
    href=np.array([[exact([0],[0],(2,0))[0,0],exact([0],[0],(1,1))[0,0]],
                   [exact([0],[0],(1,1))[0,0],exact([0],[0],(0,2))[0,0]]])
    check=float(np.max(abs(exact(np.linspace(-5,5,45),np.linspace(-5,5,45),band=48)-exact(np.linspace(-5,5,45),np.linspace(-5,5,45),band=64))))
    for n in (32,64):
        for symbol in ('fd2','spectral'):
            for degree in (2,4):
                xp=np.arange(n)*L/n-L/2;h=L/n
                k=2*np.pi*np.fft.fftfreq(n,d=h)
                lam=(2*np.sin(k*h/2)/h)**2 if symbol=='fd2' else k*k
                denom=lam[:,None]+lam[None,:];green=np.zeros_like(denom);green[denom>0]=-1/denom[denom>0]
                parent=np.fft.ifft2(np.fft.fft2(density(xp,xp))*green).real
                for level in range(1,7):
                    width=L/2**level;x=np.linspace(-width/2,width/2,n+1);h=x[1]-x[0]
                    transfer=interpolation(xp,x,degree,periodic=(level==1))
                    boundary=transfer@parent@transfer.T
                    numerical,residual=solve_patch(x,boundary)
                    true=exact(x,x);oracle,oracle_residual=solve_patch(x,true)
                    delta=numerical-oracle
                    a,_=solver(n)
                    harmonic=(delta[2:,1:-1]+delta[:-2,1:-1]+delta[1:-1,2:]+delta[1:-1,:-2]-4*delta[1:-1,1:-1])/h**2
                    hh=hessian(numerical,h);ho=hessian(oracle,h);dh=hh-ho
                    tracefree=dh-np.trace(dh)/2*np.eye(2)
                    rows.append(dict(root_n=n,root_symbol=symbol,interpolation_degree=degree,level=level,h=h,
                                     center_hessian_error=float(np.linalg.norm(hh-href)/np.linalg.norm(href)),
                                     oracle_hessian_error=float(np.linalg.norm(ho-href)/np.linalg.norm(href)),
                                     inherited_tidal_error=float(np.linalg.norm(tracefree)/np.linalg.norm(href)),
                                     inherited_hessian_trace=float(np.trace(dh)),
                                     interior_harmonic_residual=float(np.max(abs(harmonic))),poisson_residual=residual))
                    if n==32 and symbol=='fd2' and degree==2 and level in (1,6):
                        xx,yy=np.meshgrid(x,x,indexing='ij')
                        fit=np.linalg.lstsq(np.column_stack([np.ones(xx.size),xx.ravel(),yy.ravel()]),delta.ravel(),rcond=None)[0]
                        detrended=delta-fit[0]-fit[1]*xx-fit[2]*yy
                        profiles.append(dict(level=level,x=x.tolist(),inherited_error_after_affine_removal=detrended.tolist()))
                    xp=x;parent=numerical
                print('gravity hierarchy',n,symbol,degree,flush=True)
    # 1D boundary errors have zero discrete curvature, including after nesting.
    x=np.linspace(-1,1,65);affine=.03+.07*x
    one_d_curvature=float(np.max(abs(np.diff(affine,n=2)/(x[1]-x[0])**2)))
    assert check<1e-12
    assert one_d_curvature<1e-10
    assert max(r['poisson_residual'] for r in rows)<1e-11
    assert max(r['interior_harmonic_residual'] for r in rows)<1e-8
    # Exact 3D trapped Gaussian response to a harmonic potential error:
    # delta Phi=eps*(x^2+y^2-2z^2), whose 3D Laplacian vanishes identically.
    # This is a linear core surrogate, not a self-consistent soliton calculation.
    gaussian=[]
    times=np.linspace(0,20,2001)
    for eps in (.01,.05,.1):
        omega=np.sqrt(np.array([1+2*eps,1+2*eps,1-4*eps]))
        widths=np.cos(times[:,None]*omega)**2+np.sin(times[:,None]*omega)**2/omega**2
        central=1/np.sqrt(np.prod(widths,axis=1))
        gaussian.append(dict(epsilon=eps,max_central_density_change=float(np.max(abs(central-1))),
                             max_rms_radius_change=float(np.max(abs(np.sqrt(np.mean(widths,axis=1))-1))),
                             max_axis_ratio=float(np.max(np.sqrt(np.max(widths,axis=1)/np.min(widths,axis=1))))))
    result=dict(rows=rows,profiles=profiles,reference_hessian=href.tolist(),reference_check=check,
                one_d_boundary_error_curvature=one_d_curvature,gaussian_tidal_response=gaussian,
                scope='2D positive Gaussian clumps with periodic mean subtraction, nested point-centered Dirichlet patches, exact densities, direct FD2 fine solves; quadratic/quartic Lagrange boundary transfer is a controlled model, not the full GAMER interpolation. 3D Gaussian response is analytic linear trapping, not SP or a halo prediction.')
    (OUT/'gravity_hierarchy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(finest=[r for r in rows if r['level']==6],gaussian=gaussian),indent=2))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for n in (32,64):
        for symbol in ('fd2','spectral'):
            rr=[r for r in rows if r['root_n']==n and r['root_symbol']==symbol and r['interpolation_degree']==2]
            axs[0].semilogy([r['level'] for r in rr],[r['center_hessian_error'] for r in rr],'o-',label=f'{symbol}, root {n}')
    rr=[r for r in rows if r['root_n']==32 and r['root_symbol']=='fd2' and r['interpolation_degree']==2]
    axs[0].semilogy([r['level'] for r in rr],[r['oracle_hessian_error'] for r in rr],'k--',label='Exact boundaries, root 32')
    axs[0].set(xlabel='Refinement level',ylabel='Relative Hessian error at center');axs[0].legend(fontsize=8);axs[0].grid(alpha=.25)
    d=profiles[-1];im=axs[1].imshow(np.array(d['inherited_error_after_affine_removal']).T,origin='lower',extent=[d['x'][0],d['x'][-1]]*2,cmap='RdBu_r');fig.colorbar(im,ax=axs[1]);axs[1].set(title='Level 6: inherited tidal error',xlabel='x',ylabel='y')
    fig.tight_layout();fig.savefig(OUT/'gravity_hierarchy.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
