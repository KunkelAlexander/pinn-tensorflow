"""Periodic 2D quadtree SIPG operator for scale-communication experiments.

Gauss nodal elements, exact polynomial mass/stiffness and mortar face integrals.
Sparse assembly is a toy implementation; tensor kernels would replace explicit
element matrices at scale. No independent coarse wave or inherited Dirichlet data.
"""
from functools import lru_cache
from time import perf_counter
from experiments import np
from numpy.polynomial.legendre import leggauss,legvander,Legendre
from scipy.sparse import coo_matrix,diags,bmat,csc_matrix
from scipy.sparse.linalg import splu,eigsh,LinearOperator,cg
from scipy.special import jv

L=32.


def mesh(kind='adaptive',shift=0.):
    edges=np.linspace(-L/2,L/2,5)
    cells=[(edges[i],edges[i+1],edges[j],edges[j+1],0) for i in range(4) for j in range(4)]
    if kind=='uniform_coarse':return cells
    if kind=='uniform_fine':
        e=np.arange(-16,17,1.)
        return [(e[i],e[i+1],e[j],e[j+1],3) for i in range(32) for j in range(32)]
    inner=(-4,4,-2,2) if kind=='covered' else (-4,0,-2,2)
    for level,box in enumerate([(-8,8,-8,8),(-4,4,-4,4),inner]):
        out=[]
        for c in cells:
            x0,x1,y0,y1,d=c;cx=(x0+x1)/2;cy=(y0+y1)/2
            if d==level and box[0]+shift<=cx<box[1]+shift and box[2]<=cy<box[3]:
                out.extend([(a,b,s,t,d+1) for a,b in [(x0,cx),(cx,x1)] for s,t in [(y0,cy),(cy,y1)]])
            else:out.append(c)
        cells=out
    # Mesh has dyadic aligned interfaces. Enforce 2:1 balance, including seam.
    while True:
        split=set()
        for i,j,axis,face,lo,hi,periodic in faces(cells):
            if abs(cells[i][4]-cells[j][4])>1:split.add(i if cells[i][4]<cells[j][4] else j)
        if not split:break
        out=[]
        for i,c in enumerate(cells):
            if i not in split:out.append(c);continue
            x0,x1,y0,y1,d=c;cx=(x0+x1)/2;cy=(y0+y1)/2
            out.extend([(a,b,s,t,d+1) for a,b in [(x0,cx),(cx,x1)] for s,t in [(y0,cy),(cy,y1)]])
        cells=out
    return cells


def faces(cells):
    out=[]
    for i,a in enumerate(cells):
        for j,b in enumerate(cells):
            for axis in (0,1):
                start=axis*2;other=(1-axis)*2
                ordinary=abs(a[start+1]-b[start])<1e-12
                periodic=abs(a[start+1]-16)<1e-12 and abs(b[start]+16)<1e-12
                lo=max(a[other],b[other]);hi=min(a[other+1],b[other+1])
                if (ordinary or periodic) and hi-lo>1e-12:out.append((i,j,axis,a[start+1],lo,hi,periodic))
    return out


@lru_cache(None)
def nodal(p):
    z,w=leggauss(p+1);coeff=np.linalg.solve(legvander(z,p),np.eye(p+1))
    return z,w,coeff


def basis(p,z):
    coeff=nodal(p)[2]
    b=legvander(np.asarray(z),p)@coeff
    d=np.column_stack([Legendre.basis(j).deriv()(np.asarray(z)) for j in range(p+1)])@coeff
    return b,d


class Space:
    def __init__(self,p=5,kind='adaptive',shift=0.,penalty=4.):
        start=perf_counter();self.p=p;self.cells=mesh(kind,shift);self.ne=len(self.cells);self.q=p+1;self.block=self.q**2
        self.n=self.ne*self.block;z,w,_=nodal(p);b,d=basis(p,z)
        one=np.eye(self.q);mass=np.diag(w);stiff=d.T@(w[:,None]*d)
        rows=[];cols=[];vals=[];self.x=[];self.y=[];self.m=[]
        def block(ids,local):
            ii,jj=np.meshgrid(ids,ids,indexing='ij');rows.extend(ii.ravel());cols.extend(jj.ravel());vals.extend(local.ravel())
        for e,(x0,x1,y0,y1,level) in enumerate(self.cells):
            hx=x1-x0;hy=y1-y0;xx,yy=np.meshgrid((x0+x1)/2+hx*z/2,(y0+y1)/2+hy*z/2,indexing='ij')
            self.x.extend(xx.ravel());self.y.extend(yy.ravel());self.m.extend((hx*hy/4*np.outer(w,w)).ravel())
            local=hy/hx*np.kron(stiff,mass)+hx/hy*np.kron(mass,stiff)
            block(np.arange(self.block)+e*self.block,local)
        self.face_list=faces(self.cells);zf,wf=leggauss(p+2)
        for i,j,axis,coord,lo,hi,periodic in self.face_list:
            tangent=(lo+hi)/2+(hi-lo)*zf/2;weight=wf*(hi-lo)/2
            trace=[];normal=[];sizes=[]
            for e,sign in [(i,1),(j,-1)]:
                x0,x1,y0,y1,_=self.cells[e];hx=x1-x0;hy=y1-y0
                if axis==0:
                    bz,dz=basis(p,np.array([sign]));bt,dt=basis(p,2*(tangent-y0)/hy-1)
                    v=np.einsum('i,rj->rij',bz[0],bt).reshape(len(zf),-1)
                    dn=np.einsum('i,rj->rij',dz[0]*2/hx,bt).reshape(len(zf),-1);sizes.append(hx)
                else:
                    bz,dz=basis(p,np.array([sign]));bt,dt=basis(p,2*(tangent-x0)/hx-1)
                    v=np.einsum('ri,j->rij',bt,bz[0]).reshape(len(zf),-1)
                    dn=np.einsum('ri,j->rij',bt,dz[0]*2/hy).reshape(len(zf),-1);sizes.append(hy)
                trace.append(v);normal.append(dn)
            jump=np.hstack([trace[0],-trace[1]]);average=.5*np.hstack(normal)
            sigma=penalty*(p+1)**2/min(sizes)
            local=-jump.T@(weight[:,None]*average)-average.T@(weight[:,None]*jump)+sigma*jump.T@(weight[:,None]*jump)
            ids=np.r_[np.arange(self.block)+i*self.block,np.arange(self.block)+j*self.block]
            block(ids,local)
        self.a=coo_matrix((vals,(rows,cols)),shape=(self.n,self.n)).tocsr();self.a.eliminate_zeros()
        self.x=np.array(self.x);self.y=np.array(self.y);self.m=np.array(self.m);self.sqrtm=np.sqrt(self.m)
        self.h=(diags(1/self.sqrtm)@self.a@diags(1/self.sqrtm)).tocsr()*.5
        self.null=self.sqrtm/np.sqrt(self.m.sum())
        self.radius=float(np.max(np.asarray(abs(self.h).sum(axis=1)).ravel()))
        self._poisson=None;self.kinetic_calls=0;self.poisson_calls=0;self.factor_setup_seconds=0.;self.build_seconds=perf_counter()-start

    def diagnostics(self):
        symmetry=float(np.max(abs((self.a-self.a.T).data))) if (self.a-self.a.T).nnz else 0.
        null=float(np.linalg.norm(self.a@np.ones(self.n))/np.linalg.norm(self.a.data))
        return dict(degree=self.p,elements=self.ne,dofs=self.n,nnz=self.a.nnz,
                    assembled_bytes=int(self.a.data.nbytes+self.a.indices.nbytes+self.a.indptr.nbytes),
                    min_cell=min(c[1]-c[0] for c in self.cells),max_cell=max(c[1]-c[0] for c in self.cells),
                    symmetry_error=symmetry,relative_null_error=null,spectral_upper_bound=self.radius,
                    build_seconds=self.build_seconds,leaf_levels={d:sum(c[4]==d for c in self.cells) for d in range(4)})

    def exponential(self,dt,tol=2e-13):
        # Spectrum(H) is nonnegative; Gershgorin bounds it above by radius.
        r=self.radius/2;z=dt*r
        count=max(16,int(abs(z)+12*np.sqrt(abs(z)+1)+25))
        coefficients=np.array([jv(0,z)]+[2*(-1j)**j*jv(j,z) for j in range(1,count)],complex)
        while len(coefficients)>2 and sum(abs(coefficients[-min(8,len(coefficients)):]))<tol/10:
            coefficients=coefficients[:-1]
        # Keep a few extra terms beyond the tolerance truncation.
        count=len(coefficients)+8;coefficients=np.array([jv(0,z)]+[2*(-1j)**j*jv(j,z) for j in range(1,count)],complex)
        phase=np.exp(-1j*z)
        def action(u):
            t0=u;t1=(self.h@u)/r-u;answer=coefficients[0]*t0+coefficients[1]*t1
            for c in coefficients[2:]:
                t2=2*((self.h@t1)/r-t1)-t0;answer+=c*t2;t0,t1=t1,t2
            self.kinetic_calls+=len(coefficients)-1
            return phase*answer
        return action,len(coefficients)-1

    def poisson(self,rho,coupling=1.):
        if self._poisson is None:
            start=perf_counter()
            m=csc_matrix(self.m[:,None]);matrix=bmat([[self.a,m],[m.T,None]],format='csc')
            self._poisson=splu(matrix)
            self.factor_setup_seconds=perf_counter()-start
        rhs=-coupling*self.m*(rho-np.dot(self.m,rho)/self.m.sum())
        self.poisson_calls+=1
        return self._poisson.solve(np.r_[rhs,0.])[:-1]

    def samples(self,nquad=None):
        q=nquad or 2*self.p+3;z,w=leggauss(q);b,_=basis(self.p,z)
        xs=[];ys=[];weights=[]
        for x0,x1,y0,y1,_ in self.cells:
            x,y=np.meshgrid((x0+x1)/2+(x1-x0)*z/2,(y0+y1)/2+(y1-y0)*z/2,indexing='ij')
            xs.append(x);ys.append(y);weights.append((x1-x0)*(y1-y0)/4*np.outer(w,w))
        return np.array(xs),np.array(ys),np.array(weights),b

    def evaluate(self,c,b):return np.einsum('qi,eij,rj->eqr',b,c.reshape(self.ne,self.q,self.q),b,optimize=True)

    def project(self,values,weights,b):
        load=np.einsum('qi,eqr,rj->eij',b,weights*values,b,optimize=True)
        return load.ravel()/self.m

    def on_uniform(self,c,n):
        x=np.arange(n)*L/n-L/2;out=np.empty((n,n),complex)
        for e,(x0,x1,y0,y1,_) in enumerate(self.cells):
            ix=np.where((x>=x0-1e-12)&(x<x1-1e-12))[0];iy=np.where((x>=y0-1e-12)&(x<y1-1e-12))[0]
            if not len(ix) or not len(iy):continue
            bx,_=basis(self.p,2*(x[ix]-x0)/(x1-x0)-1);by,_=basis(self.p,2*(x[iy]-y0)/(y1-y0)-1)
            out[np.ix_(ix,iy)]=bx@c.reshape(self.ne,self.q,self.q)[e]@by.T
        return out

    def cg_probe(self,rho):
        # Two-level p-coarse additive preconditioner; coarse sparse saddle solve.
        # This is an iteration probe, not a claim of scalable mesh multigrid.
        z=nodal(self.p)[0];linear=np.column_stack([(1-z)/2,(1+z)/2]);interp=np.kron(linear,linear)
        rr=[];cc=[];vv=[]
        for e in range(self.ne):
            block=self.sqrtm[e*self.block:(e+1)*self.block,None]*interp
            i,j=np.meshgrid(np.arange(self.block)+e*self.block,np.arange(4)+e*4,indexing='ij')
            rr.extend(i.ravel());cc.extend(j.ravel());vv.extend(block.ravel())
        p=coo_matrix((vv,(rr,cc)),shape=(self.n,4*self.ne)).tocsr()
        a=2*self.h;coarse=(p.T@a@p).tocsc();cm=np.asarray(p.T@self.null).ravel();cmat=csc_matrix(cm[:,None])
        factor=splu(bmat([[coarse,cmat],[cmat.T,None]],format='csc'))
        diagonal=a.diagonal()
        def mat(v):return a@v+self.null*np.dot(self.null,v)
        def pre(v):
            c=factor.solve(np.r_[p.T@v,0.])[:-1]
            return v/diagonal+p@c+self.null*np.dot(self.null,v)
        rhs=-self.sqrtm*(rho-np.dot(self.m,rho)/self.m.sum());counter=[0]
        def callback(x):counter[0]+=1
        start=perf_counter();sol,info=cg(LinearOperator(a.shape,matvec=mat),rhs,
            M=LinearOperator(a.shape,matvec=pre),rtol=1e-10,atol=0,maxiter=1000,callback=callback)
        return dict(iterations=counter[0],info=int(info),relative_residual=float(np.linalg.norm(mat(sol)-rhs)/np.linalg.norm(rhs)),
                    seconds=perf_counter()-start,coarse_dofs=4*self.ne,coarse_factor_nnz=factor.L.nnz+factor.U.nnz)
