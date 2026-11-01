"""Sampling, representation and prior controls for the nonuniform-grid claim.

Exact dense matrices isolate information loss, with no NUFFT implementation or
neural optimizer in the loop. The Gaussian conditional mean is the optimal
mean-square predictor over ALL estimators for the explicitly stated prior.
It is not a trained neural network or an SP solver.
"""
from __future__ import annotations
import json
from pathlib import Path
from experiments import np, plt
from scipy.linalg import svd
from numpy.polynomial.legendre import legvander, leggauss

OUT = Path(__file__).resolve().parent / 'review_results'
OUT.mkdir(exist_ok=True)
K = 25


def fourier(x):
    k = np.arange(1, K+1)
    return np.column_stack((np.ones(len(x)), np.sqrt(2)*np.cos(np.outer(x,k)),
                            np.sqrt(2)*np.sin(np.outer(x,k))))


def weights(x):
    gaps = np.diff(np.r_[x[-1]-2*np.pi, x, x[0]+2*np.pi])
    return (gaps[:-1]+gaps[1:])/(4*np.pi)


def grids():
    n=128
    uniform = -np.pi+2*np.pi*(np.arange(n)+.37)/n
    q=2*np.pi*(np.arange(n)+.37)/n
    clustered = 2*np.arctan((1-.9)/(1+.9)*np.tan((q-np.pi)/2))
    def backbone(nb):
        coarse=-np.pi+2*np.pi*(np.arange(nb)+.37)/nb
        fine=np.linspace(-.59,.59,n-nb)
        return np.sort(np.r_[coarse,fine])
    return dict(uniform=uniform,clustered=np.sort(clustered),
                halo_base20=backbone(20),halo_base52=backbone(52))


EDGES = np.r_[-np.pi,np.linspace(-.6,.6,9),np.pi]


def local_polynomials(x):
    """50 elementwise orthonormal functions, versus 51 global Fourier modes.

    This is a representation diagnostic, not a discontinuous Galerkin PDE
    solver: no claim about interface fluxes follows from its condition number.
    Each basis has unit norm in integral dx/(2*pi).
    """
    b=np.zeros((len(x),50))
    for j,(lo,hi) in enumerate(zip(EDGES[:-1],EDGES[1:])):
        mask=(x>=lo)&(x<hi)
        z=(2*x[mask]-lo-hi)/(hi-lo)
        b[mask,j*5:(j+1)*5]=legvander(z,4)*np.sqrt((2*np.arange(5)+1)*2*np.pi/(hi-lo))
    return b


def savefig(name):
    plt.tight_layout(); plt.savefig(OUT/name,dpi=150); plt.close()


def main():
    xf=-np.pi+2*np.pi*np.arange(8192)/8192
    bf=fourier(xf); lf=local_polynomials(xf)
    k=np.r_[0,np.arange(1,K+1),np.arange(1,K+1)]
    inv=np.zeros(len(k)); inv[k!=0]=-1/k[k!=0]**2
    rows=[]; singular={}; matrices={}; prior_rows=[]
    # Population-optimal Gaussian estimators; no finite training set is fitted.
    # y = B c + eta, eta ~ N(0, sigma^2 I) in volume-weighted sensor coordinates.
    sigma=1e-6
    priors={'bandlimited8':(k<=8).astype(float),
            'low_frequency':np.exp(-(k/5.)**2), 'broadband':np.ones(len(k))}
    priors={name:v/np.sum(v) for name,v in priors.items()}
    for name,x in grids().items():
        w=weights(x); a=fourier(x); b=np.sqrt(w)[:,None]*a
        u,s,vh=svd(b,full_matrices=False)
        singular[name]=s.tolist(); matrices[name]=(b,u,s,vh)
        l=np.sqrt(w)[:,None]*local_polynomials(x)
        sl=svd(l,compute_uv=False)
        rows.append(dict(grid=name,sensors=len(x),fourier_dofs=len(k),local_dofs=l.shape[1],
                         fourier_unweighted_condition=float(np.linalg.cond(a)),
                         fourier_weighted_condition=float(s[0]/s[-1]),
                         local_weighted_condition=float(sl[0]/sl[-1]) if sl[-1]>0 else None,
                         max_gap=float(np.max(np.diff(np.r_[x,x[0]+2*np.pi]))),
                         fourier_rank=int(np.sum(s>1e-12*s[0])),local_rank=int(np.sum(sl>1e-12*sl[0]))))
        for training_name,cov in priors.items():
            # SVD form avoids the squared conditioning of normal equations.
            uc,sc,vhc=svd(b*np.sqrt(cov)[None,:],full_matrices=False)
            predictor=(np.sqrt(cov)[:,None]*vhc.T*(sc/(sc*sc+sigma*sigma))[None,:])@uc.T
            defect=predictor@b-np.eye(len(k))
            for test_name,test_cov in priors.items():
                for target,multiplier in [('wave',np.ones(len(k))),('potential',inv)]:
                    bias=np.linalg.norm(multiplier[:,None]*defect*np.sqrt(test_cov)[None,:])**2
                    variance=sigma*sigma*np.linalg.norm(multiplier[:,None]*predictor)**2
                    target_power=np.sum(multiplier**2*test_cov)
                    prior_rows.append(dict(grid=name,training_prior=training_name,test_prior=test_name,
                                           target=target,expected_relative_rms=float(np.sqrt((bias+variance)/target_power))))

    b,u,s,vh=matrices['halo_base20']
    # Select a small but numerically reliable singular value, not roundoff nullity.
    j=int(np.argmin(abs(np.log10(np.maximum(s,1e-300))+7)))
    v=vh[j]; f=bf@v
    dt=.2; phase=np.exp(-.5j*k*k*dt)
    # A common nonzero background prevents the ambiguity from being merely an
    # irrelevant global phase. The pair 1 +/- i*f has exactly the same initial
    # density and mass, nearly identical sampled waves, but different currents.
    evolved=bf@(phase*v)
    plus=1+1j*evolved;minus=1-1j*evolved
    halo=abs(xf)<.6
    density_difference=abs(plus)**2-abs(minus)**2
    # Observing the background alone is consistent with either wave when the
    # weighted noise is +/- i Bv. Any predictor must miss at least one target.
    ambiguity=dict(grid='halo_base20',singular_index=j,singular_value=float(s[j]),
                   weighted_observation_norm=float(np.linalg.norm(b@v)),
                   wave_norm=float(np.linalg.norm(v)),
                   evolved_wave_norm=float(np.linalg.norm(phase*v)),
                   potential_norm=float(np.linalg.norm(inv*v)),
                   initial_halo_wave_norm=float(np.sqrt(np.mean(abs(f[abs(xf)<.6])**2))),
                   evolved_halo_wave_norm=float(np.sqrt(np.mean(abs(evolved[halo])**2))),
                   equal_initial_mass=float(1+np.dot(v,v)),
                   evolved_halo_density_difference_rms=float(np.sqrt(np.mean(density_difference[halo]**2))),
                   evolved_global_density_difference_rms=float(np.sqrt(np.mean(density_difference**2))),
                   noise_sigma_per_weighted_sensor=sigma)

    # Projection accuracy on independently integrated physical space: conditioning
    # is not accuracy, and changing the space changes which fields are admissible.
    z,wq=leggauss(80); xq=[]; wphys=[]
    for lo,hi in zip(EDGES[:-1],EDGES[1:]):
        xq.extend((lo+hi)/2+(hi-lo)*z/2);wphys.extend(wq*(hi-lo)/(4*np.pi))
    xq=np.array(xq); wphys=np.array(wphys)
    bq=fourier(xq);lq=local_polynomials(xq)
    projection=[]
    for case,fun in [('localized_carrier',lambda x:np.exp(-x*x/(2*.17**2))*np.exp(40j*x)),
                     ('global_carrier',lambda x:np.exp(20j*x))]:
        target=fun(xq); norm=np.sqrt(np.sum(wphys*abs(target)**2))
        for label,mat in [('fourier51',bq),('local50',lq)]:
            coeff=mat.T@(wphys*target)
            projection.append(dict(case=case,basis=label,
                              best_projection_error=float(np.sqrt(np.sum(wphys*abs(mat@coeff-target)**2))/norm)))

    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name,spec in singular.items(): ax[0].semilogy(spec,label=name)
    ax[0].set(xlabel='Singular-value index',ylabel='Singular value of weighted sampling')
    ax[0].legend(fontsize=8);ax[0].grid(alpha=.25)
    ax[1].plot(xf,f,label='Poorly observed wave at t=0')
    ax[1].plot(xf,abs(bf@(phase*v)),label='Magnitude at t=0.2')
    x=grids()['halo_base20'];ax[1].plot(x,np.zeros(len(x)),'.',label='Sensors')
    ax[1].axvspan(-.6,.6,color='gray',alpha=.12);ax[1].set_xlabel('x')
    ax[1].legend(fontsize=8);ax[1].grid(alpha=.25)
    savefig('conditioning_controls.png')

    # Independent checks of norm conventions and the population Bayes calculation.
    checks={
        'local_basis_exact_mass':float(np.linalg.norm(lq.T@(wphys[:,None]*lq)-np.eye(50))),
        'fourier_exact_mass':float(np.linalg.norm(bq.T@(wphys[:,None]*bq)-np.eye(51))),
        'singular_pair_identity':float(abs(np.linalg.norm(b@v)-s[j])),
        'free_unitarity':float(abs(np.linalg.norm(phase*v)-np.linalg.norm(v))),
    }
    assert checks['local_basis_exact_mass']<1e-11
    assert checks['fourier_exact_mass']<1e-11
    assert checks['singular_pair_identity']<1e-13
    assert checks['free_unitarity']<1e-14
    assert rows[-1]['fourier_weighted_condition']<10
    assert next(r for r in rows if r['grid']=='halo_base20')['local_weighted_condition']<10
    # Monte Carlo independent validation of one analytic expected risk.
    rng=np.random.default_rng(20261101); cov=priors['low_frequency']
    uc,sc,vhc=svd(b*np.sqrt(cov)[None,:],full_matrices=False)
    predictor=(np.sqrt(cov)[:,None]*vhc.T*(sc/(sc*sc+sigma*sigma))[None,:])@uc.T
    samples=np.sqrt(priors['broadband'])[:,None]*rng.normal(size=(51,4000))
    noise=sigma*rng.normal(size=(128,4000))
    empirical=float(np.sqrt(np.mean(np.sum(abs(predictor@(b@samples+noise)-samples)**2,axis=0))))
    analytic=next(r['expected_relative_rms'] for r in prior_rows if r['grid']=='halo_base20'
                  and r['training_prior']=='low_frequency' and r['test_prior']=='broadband' and r['target']=='wave')
    checks['bayes_risk_monte_carlo_relative_difference']=abs(empirical/analytic-1)
    assert checks['bayes_risk_monte_carlo_relative_difference']<.04
    result=dict(normalization='L2(dx/(2*pi)), Fourier/local bases orthonormal in physical volume',
                noise_model='Independent sigma=1e-6 Gaussian noise per volume-weighted sensor; not equal raw point noise',
                grids=rows,ambiguity=ambiguity,prior_predictions=prior_rows,
                projection=projection,checks=checks,
                limitations='Dense exact sampling and analytic Gaussian priors; no neural network trained; local discontinuous basis is not a PDE solver')
    (OUT/'conditioning.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(grids=rows,ambiguity=ambiguity,projection=projection,checks=checks),indent=2))


if __name__=='__main__': main()
