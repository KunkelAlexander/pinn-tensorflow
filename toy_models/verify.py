"""Numerical cross-checks of identities, convergence, and saved experiments."""
import json
from pathlib import Path
from experiments import inverse_laplacian,mapped_operators,modes,np,OUT,relative
from representations import band_rhs


def check(condition,message):
    if not condition: raise AssertionError(message)
    print('PASS',message)


def main():
    rng=np.random.default_rng(908)
    n=65; length=20.
    _,_,jac,h,a=mapped_operators(n,0.,length)
    k=modes(n,length)
    spectral=np.fft.ifft((k*k/2)[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
    check(relative(h,spectral)<2e-14,'uniform limit of mapped kinetic equals Fourier kinetic')
    _,_,jac,h,a=mapped_operators(n,.8,length)
    check(relative(h,h.T)<1e-14,'mapped kinetic is Hermitian')
    check(np.linalg.norm(h@np.sqrt(jac))<1e-10,'physical constant has zero mapped kinetic energy')
    check(np.linalg.norm(a@np.ones(n))<1e-10,'mapped Poisson has the correct constant nullspace')

    # Independent FFT evaluation of the nonlinear Galerkin operator.
    ks=np.r_[np.arange(-28,-19),np.arange(20,29)]
    c=rng.normal(size=len(ks))+1j*rng.normal(size=len(ks));c/=np.linalg.norm(c)
    rhs,energy=band_rhs(ks,coupling=20.,carrier=24)
    full=np.zeros(512,dtype=complex);full[ks%512]=c
    psi=np.fft.ifft(full*512)
    phi=np.fft.ifft(np.fft.fft(abs(psi)**2)*inverse_laplacian(512,2*np.pi)).real*20
    nonlinear=(np.fft.fft(phi*psi)/512)[ks%512]
    expected=-1j*((ks*ks-24*24)/2*c+nonlinear)
    check(relative(rhs(0,c),expected)<2e-13,'sparse-band convolution agrees with independent overintegrated FFT')
    direction=rng.normal(size=len(ks))+1j*rng.normal(size=len(ks))
    eps=1e-6
    numerical=(energy(c+eps*direction)-energy(c-eps*direction))/(2*eps)
    physical_h=1j*rhs(0,c)+24*24/2*c
    analytic=2*np.vdot(direction,physical_h).real
    check(abs(numerical-analytic)/max(1.,abs(analytic))<2e-8,'band energy gradient agrees with Hamiltonian action')

    def read(name):
        value=json.loads((OUT/f'{name}.json').read_text())
        return value.get('results',value)
    g=read('gravity')
    find=lambda op,n: next(r for r in g if r['operator']==op and r['n']==n)
    ratio=1/np.sinc(1/5)**2-1
    check(abs(find('fd2',80)['potential_error']-ratio)<1e-12,'FD2 potential error agrees with analytic symbol')
    check(find('spectral_padded_source',24)['potential_error']<1e-11,'padded density source removes two-wave aliasing')
    sp=read('sp')
    check(sp['reference_density_disagreement']<1e-9,'SP reference converges under grid and step refinement')
    mass=max(r['mass_error'] for r in sp['resolution'])
    check(mass<2e-10,'uniform SP experiments conserve mass')
    for method,target in [('strang',4.),('yoshida4',16.)]:
        d={r['dt']:r['density_error'] for r in sp['time'] if r['method']==method}
        ratio=d[.0125]/d[.00625]
        check(.8*target<ratio<1.2*target,f'{method} reaches its expected temporal convergence order')
    mapped=read('mapped_sp')
    check(mapped['reference_disagreement']<1e-9,'mapped SP comparison reference converges')
    check(max(r['mass_error'] for r in mapped['resolution'])<2e-10,'mapped SP conserves physical weighted mass')
    frozen=read('frozen')
    check(frozen['stationary_residual']<1e-10,'nonlinear stationary state solves the SP eigenproblem')
    check(frozen['perturbed_reference_disagreement']<1e-9,'perturbed core time reference converges')
    d=[r for r in frozen['rows'] if r['case']=='stationary' and r['method']=='reference_strang']
    check(max(r['density_error'] for r in d)<1e-10,'reference Hamiltonian splitting preserves the nonlinear equilibrium')
    print('All numerical cross-checks passed.')


if __name__=='__main__': main()
