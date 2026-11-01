"""One coarse-fine gravity interface: operator, density and boundary ablations.

Frozen finite-Fourier source, not a full AMR evolution. Quadratic coarse-wave
prolongation is an explicit source-error control, not a model of Gram-FE interpolation.
"""
import json
from functools import lru_cache
from experiments import np,plt
from scipy.sparse import diags,kron,eye
from scipy.sparse.linalg import factorized
from frozen_interference_gravity import coefficients,evaluate,discrete_hessian,L
from gravity_hierarchy_probe import interpolation
from gravity_accuracy_evolution import OUT


def stencil(order):return {2:{-1:1.,0:-2.,1:1.},4:{-2:-1/12,-1:4/3,0:-2.5,1:4/3,2:-1/12}}[order]


@lru_cache(None)
def system(n,order):
    weights=stencil(order);d=diags([np.full(n-1-abs(j),v) for j,v in weights.items()],list(weights),shape=(n-1,n-1),format='csc')
    a=(kron(d,eye(n-1))+kron(eye(n-1),d)).tocsc();return a,factorized(a)


def solve(x,y,boundary,source,order):
    # x/y include two outer layers: physical boundaries at indices 1 and -2.
    n=len(x)-3;h=x[1]-x[0];a,action=system(n,order);rhs=h*h*source.copy()
    for j in range(1,n):
        for offset,w in stencil(order).items():
            nb=j+offset
            if nb<1 or nb>n-1:
                rhs[j-1,:]-=w*boundary[nb+1,2:-2]
                rhs[:,j-1]-=w*boundary[2:-2,nb+1]
    phi=boundary.copy();phi[2:-2,2:-2]=action(rhs.ravel()).reshape(n-1,n-1)
    residual=float(np.linalg.norm(a@phi[2:-2,2:-2].ravel()-rhs.ravel())/max(np.linalg.norm(rhs),1e-30))
    return phi,residual


def main():
    p,rho,truephi=coefficients(7);rows=[];checks={}
    for n in (50,100):
        parent_x=np.arange(n)*L/n-L/2;k=np.arange(-20,21);h=L/(2*n)
        parent_wave=evaluate(p,parent_x,parent_x)
        for order in (2,4):
            s=np.sin(k*L/n/2)**2;axis=4*s/(L/n)**2*(1+s/3 if order==4 else 1)
            lam=axis[:,None]+axis[None,:];parent_coeff=np.zeros_like(rho);mask=lam>0;parent_coeff[mask]=-rho[mask]/lam[mask]
            parent_phi=evaluate(parent_coeff,parent_x,parent_x).real
            for shift in (0.,.7,1.3):
                x=(np.arange(-1,n+2)/n-.5)*(L/2)+shift;y=(np.arange(-1,n+2)/n-.5)*(L/2)+shift*.3
                truth=evaluate(truephi,x,y).real;source=evaluate(rho,x[2:-2],y[2:-2]).real
                px=interpolation(parent_x,x,2,periodic=True);py=interpolation(parent_x,y,2,periodic=True)
                coarsewave=px@parent_wave@py.T;approxsource=abs(coarsewave[2:-2,2:-2])**2-1
                for source_name,used_source in [('exact',source),('coarse_wave_quadratic',approxsource)]:
                    for boundary_name,degree in [('exact',None),('parent_quadratic',2),('parent_quartic',4)]:
                        if degree is None:bc=truth
                        else:
                            tx=interpolation(parent_x,x,degree,periodic=True);ty=interpolation(parent_x,y,degree,periodic=True)
                            bc=tx@parent_phi@ty.T
                        phi,residual=solve(x,y,bc,used_source,order)
                        ht=discrete_hessian(truth[1:-1,1:-1],h);hn=discrete_hessian(phi[1:-1,1:-1],h)
                        ix=int(np.argmin(abs(x[2:-2])));iy=int(np.argmin(abs(y[2:-2])))
                        rows.append(dict(root_n=n,root_wave_ppw=n/10,order=order,shift=shift,source=source_name,boundary=boundary_name,
                            density_source_error=float(np.linalg.norm(used_source-source)/np.linalg.norm(source)),
                            hessian_error=float(np.linalg.norm(hn-ht)/np.linalg.norm(ht)),
                            near_origin_hessian_error=float(np.linalg.norm((hn-ht)[:,ix,iy])/np.linalg.norm(ht[:,ix,iy])),residual=residual))
        print('interface accuracy',n,flush=True)
    # FD4 exactness on a quartic polynomial independently tests boundary handling.
    n=20;x=np.linspace(-1.1,1.1,n+3);xx,yy=np.meshgrid(x,x,indexing='ij')
    truth=xx**4+yy**4+xx**2*yy**2
    phi,res=solve(x,x,truth,14*(xx[2:-2,2:-2]**2+yy[2:-2,2:-2]**2),4)
    checks['fd4_quartic_boundary_exactness']=float(np.max(abs(phi-truth)))
    assert checks['fd4_quartic_boundary_exactness']<1e-11
    assert max(r['residual'] for r in rows)<1e-10
    result=dict(rows=rows,checks=checks,scope='Frozen 2D coherent field with a single shifted coarse-fine interface; exact finite-Fourier reference. Both wave and density initially resolved on parent. FD2/FD4 tensor Laplacians, direct solves; FD4 uses two prescribed boundary/ghost layers. Exact density/exact boundary ablations isolate limitations. Coarse-wave quadratic interpolation is a deliberately low-accuracy source control, not GAMER Gram-FE transfer. This is not evolving AMR.')
    (OUT/'interface.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,n in zip(axs,(50,100)):
        for order in (2,4):
            for source_name,bc in [('exact','exact'),('exact','parent_quadratic'),('exact','parent_quartic'),('coarse_wave_quadratic','exact')]:
                rr=[r for r in rows if r['root_n']==n and r['order']==order and r['source']==source_name and r['boundary']==bc]
                ax.semilogy([r['shift'] for r in rr],[r['near_origin_hessian_error'] for r in rr],'o-' if order==2 else 's--',label=f'FD{order}: '+('approx source' if source_name!='exact' else bc))
        ax.set(xlabel='Patch displacement',ylabel='Hessian error near fixed origin',title=f'Root {n/10:g} cells/wave');ax.grid(alpha=.25);ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'interface.png',dpi=160);plt.close(fig)
    print(json.dumps([r for r in rows if r['root_n']==50 and r['shift']==0],indent=2))


if __name__=='__main__':main()
