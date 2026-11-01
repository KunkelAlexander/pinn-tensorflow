"""Scale communication controls, independent of the previous output directories.

Dimensionless periodic i psi_t = -psi_xx/2 + Phi psi, Phi_xx = |psi|^2-mean.
Dense Galerkin operators are accuracy oracles, not a proposed 3D implementation.
No physical coarse wave is used as the gravity source in the coupled experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from experiments import np, plt, modes, inverse_laplacian
from shared_basis_review import assemble
from frozen_hamiltonian import setup
from scipy.linalg import eigh
from scipy.integrate import solve_ivp

OUT = Path(__file__).resolve().parent / 'communication_results'
OUT.mkdir(exist_ok=True)
L = 20.
# 0.5-wide elements on [-5,0]; 1-wide elements elsewhere, including periodic seam.
EDGES = np.r_[np.arange(-10., -5., 1.), np.arange(-5., 0., .5), np.arange(0., 11., 1.)]
T = 2.


def write(name, value):
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def norm(v, w):
    return float(np.sqrt(np.sum(w * abs(v)**2)))


def relative(v, ref, w):
    return norm(v-ref, w)/norm(ref, w)


def spectral_values(values, x, shift=0., cutoff=1e-14):
    """Evaluate a periodic Fourier interpolant on independent physical points."""
    coeff = np.fft.fft(values)/len(values)
    occupied = abs(coeff) > cutoff
    k = modes(len(values), L)[occupied]
    return np.exp(1j*np.outer(x+L/2-shift, k)) @ coeff[occupied]


class Space:
    def __init__(self, degree, edges=EDGES, quadrature=None):
        self.degree = degree
        self.edges = edges
        self.m, self.k, self.x, self.w, self.b = assemble(degree, edges, quadrature or 2*degree+4)
        self.e, self.v = eigh(.5*self.k, self.m)
        self.e[0] = 0.
        self.q = self.b @ self.v
        self.green = np.zeros(len(self.e))
        self.green[1:] = -1/(2*self.e[1:])

    def project(self, values):
        return self.q.T @ (self.w*values)

    def evaluation(self, x):
        # Local barycentric interpolation avoids dense global evaluation storage.
        from shared_basis_review import basis
        b = np.zeros((len(x), len(self.e)))
        xp = (np.asarray(x)+L/2) % L-L/2
        el = np.searchsorted(self.edges, xp, side='right')-1
        for j, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            rows = np.where(el == j)[0]
            if len(rows):
                ids = (np.arange(self.degree+1)+j*self.degree) % len(self.e)
                b[np.ix_(rows, ids)] = basis(self.degree, 2*(xp[rows]-lo)/(hi-lo)-1)[0]
        return b @ self.v

    def gravity(self, a):
        psi = self.q @ a
        load = self.project(abs(psi)**2)
        return self.green*load

    def nonlinear(self, a):
        psi = self.q @ a
        phi = self.q @ self.gravity(a)
        return self.project(phi*psi)

    def energy(self, a):
        load = self.project(abs(self.q @ a)**2)
        return float(np.sum(self.e*abs(a)**2)+.5*np.dot(load, self.green*load).real)

    def evolve(self, a, time=T, rtol=2e-9):
        # Integrate the full projected nonlinear equation, not a pointwise FE kick.
        # Interaction picture treats the kinetic subproblem exactly.
        def rhs(t, y):
            phase = np.exp(-1j*self.e*t)
            return -1j*phase.conj()*self.nonlinear(phase*y)
        start = perf_counter()
        sol = solve_ivp(rhs, (0., time), a.astype(complex), method='DOP853',
                        rtol=rtol, atol=rtol*.02)
        if not sol.success:
            raise RuntimeError(sol.message)
        result = np.exp(-1j*self.e*time)*sol.y[:, -1]
        return result, dict(rhs_calls=sol.nfev, steps=len(sol.t)-1,
                           seconds=perf_counter()-start,
                           mass_drift=float(abs(np.vdot(result,result)-np.vdot(a,a))/np.vdot(a,a).real),
                           energy_drift=float(abs(self.energy(result)-self.energy(a))/max(1.,abs(self.energy(a)))))


def packet(x):
    # Periodic sum; a positive carrier with negligible negative-frequency power.
    return sum(np.exp(-(x+2.+j*L)**2/(2*.7**2))*np.exp(4j*(x+2.+j*L)) for j in (-1,0,1))


def wave_metrics(pred, ref, initial_ref, x, w, shift, final=T):
    overlap = np.sum(w*pred.conj()*ref)
    aligned = pred*np.exp(1j*np.angle(overlap))
    # Right-going packet center is -2+4T+shift. Compare a transmitted window.
    center = -2+4*final+shift
    distance = (x-center+L/2)%L-L/2
    mask = abs(distance)<2.
    local_overlap = np.sum(w[mask]*pred[mask]*ref[mask].conj())
    return dict(wave_error=relative(pred,ref,w), phase_aligned_error=relative(aligned,ref,w),
                density_error=relative(abs(pred)**2,abs(ref)**2,w),
                transmitted_phase=float(np.angle(local_overlap)),
                transmitted_complex_error=relative(pred[mask],ref[mask],w[mask]))


def linear():
    nref=2048; xu=np.arange(nref)*L/nref-L/2; ku=modes(nref,L)
    initial=packet(xu)
    # Short enough to cross the interface, without reaching the periodic seam.
    final=1.25
    reference=np.fft.ifft(np.fft.fft(initial)*np.exp(-.5j*ku**2*final))
    xfine=np.arange(4096)*L/4096-L/2
    ref_fine=np.fft.ifft(np.fft.fft(packet(xfine))*np.exp(-.5j*modes(4096,L)**2*final))
    reference_check=relative(spectral_values(reference,xfine),ref_fine,np.ones(4096))
    rows=[]
    for p in (3,4,6,8,10):
        sp=Space(p)
        _,_,xe,we,be=assemble(p,EDGES,80)
        qe=be@sp.v
        qu=sp.evaluation(xu)
        for shift in (-2.,0.,2.):
            a=sp.project(packet(sp.x-shift))
            evolved=np.exp(-1j*sp.e*final)*a
            pred=qe@evolved; ref=spectral_values(reference,xe,shift)
            pu=qu@evolved; ru=spectral_values(reference,xu,shift)
            # Error projected onto negative physical Fourier modes: a reflection
            # diagnostic, not a unique reflected/transmitted scattering split.
            error_coeff=np.fft.fft(pu-ru)/nref
            reverse_error_mass=float(L*np.sum(abs(error_coeff[ku<0])**2)/(L/nref*np.sum(abs(initial)**2)))
            same_n=len(a); xs=np.arange(same_n)*L/same_n-L/2; ks=modes(same_n,L)
            baseline=np.fft.ifft(np.fft.fft(packet(xs-shift))*np.exp(-.5j*ks**2*final))
            rows.append(dict(degree=p,dofs=len(a),shift=shift,
                             initial_projection_error=relative(qe@a,packet(xe-shift),we),
                             mass_drift=float(abs(np.vdot(evolved,evolved)-np.vdot(a,a))/np.vdot(a,a).real),
                             reverse_error_mass=reverse_error_mass,
                             uniform_fourier_error=relative(spectral_values(baseline,xe),ref,we),
                             **wave_metrics(pred,ref,initial,xe,we,shift,final)))
        print('linear degree',p,flush=True)
    assert reference_check<1e-11
    assert max(r['mass_drift'] for r in rows)<1e-12
    assert max(r['wave_error'] for r in rows if r['degree']==10)<2e-5
    result=dict(final_time=final,mesh=EDGES.tolist(),reference_refinement=reference_check,rows=rows,
                reverse_metric='L2 mass of negative-Fourier projection of numerical minus reference wave; not pure reflection probability')
    write('linear',result)
    return result


def source_communication():
    # High carrier, smooth envelope, and a long beat. Density does not inherit
    # the carrier frequency. All runs use quadrature that resolves the wave.
    def wave(x):
        envelope=sum(np.exp(-(x+j*L)**2/(2*.9**2)) for j in (-1,0,1))
        return envelope*(np.exp(1j*(2*np.pi*30/L)*x)+.6*np.exp(1j*(2*np.pi*32/L)*x))
    xu=np.arange(4096)*L/4096-L/2;rho=abs(wave(xu))**2
    phi=np.fft.ifft(np.fft.fft(rho)*inverse_laplacian(len(xu),L)).real
    rows=[]; edges=np.linspace(-10,10,21)
    for p in (1,2,3,4,6,8):
        sp=Space(p,edges,64)
        a=sp.project(wave(sp.x)); represented=sp.q@a
        exact_load=sp.project(abs(wave(sp.x))**2)
        coarse_load=sp.project(abs(represented)**2)
        good=sp.q@(sp.green*exact_load); bad=sp.q@(sp.green*coarse_load)
        ref=spectral_values(phi,sp.x).real
        rows.append(dict(degree=p,dofs=len(a),wave_projection_error=relative(represented,wave(sp.x),sp.w),
                         density_source_potential_error=relative(good,ref,sp.w),
                         coarse_wave_source_potential_error=relative(bad,ref,sp.w),
                         represented_mass_fraction=float(np.sum(sp.w*abs(represented)**2)/np.sum(sp.w*abs(wave(sp.x))**2))))
    assert rows[-1]['density_source_potential_error']<1e-7
    assert rows[1]['coarse_wave_source_potential_error']>10*rows[1]['density_source_potential_error']
    result=dict(rows=rows,scope='Static source diagnostic. The globally coarse wave projections deliberately discard details; the full wave is retained when forming the correct load. No claim that such a coarse wave resolves the input.')
    write('source',result)
    return result


def interface_controls():
    """Compare like local resolutions and separate initial from generated error."""
    n=2048; x=np.arange(n)*L/n-L/2; k=modes(n,L); final=1.25
    ref=np.fft.ifft(np.fft.fft(packet(x))*np.exp(-.5j*k*k*final))
    mass=L/n*np.sum(abs(packet(x))**2)
    rows=[]
    meshes={'graded':EDGES,'uniform_coarse':np.linspace(-10,10,21),
            'uniform_fine':np.linspace(-10,10,41)}
    for p in (4,6,8,10):
        for name,edges in meshes.items():
            sp=Space(p,edges); q=sp.evaluation(x)
            a=sp.project(packet(sp.x)); initial=q@a
            pred=q@(np.exp(-1j*sp.e*final)*a)
            # Exact continuum propagation of this same projected initial state.
            # Subtract it to isolate the evolution-operator error (on 2048 samples).
            projected_ref=np.fft.ifft(np.fft.fft(initial)*np.exp(-.5j*k*k*final))
            delta=np.fft.fft(pred-projected_ref)/n
            rows.append(dict(degree=p,mesh=name,dofs=len(a),
                             initial_projection_error=relative(initial,packet(x),np.ones(n)),
                             wave_error=relative(pred,ref,np.ones(n)),
                             propagation_error=relative(pred,projected_ref,np.ones(n)),
                             generated_reverse_error_mass=float(L*np.sum(abs(delta[k<0])**2)/mass)))
        print('interface controls degree',p,flush=True)
    # A second sampling resolution checks the negative-mode diagnostic itself.
    sp=Space(6);a=sp.project(packet(sp.x)); checks=[]
    for n in (2048,4096):
        x=np.arange(n)*L/n-L/2;k=modes(n,L);q=sp.evaluation(x)
        initial=q@a;pred=q@(np.exp(-1j*sp.e*final)*a)
        exact=np.fft.ifft(np.fft.fft(initial)*np.exp(-.5j*k*k*final))
        delta=np.fft.fft(pred-exact)/n
        checks.append(float(L*np.sum(abs(delta[k<0])**2)/mass))
    metric_difference=abs(checks[0]-checks[1])/checks[1]
    assert metric_difference<.01,checks
    result=dict(rows=rows,negative_mode_sampling_relative_difference=metric_difference,
                scope='Uniform coarse/fine have the same element sizes as the corresponding graded regions. DOF counts differ. Generated reverse error subtracts exact FFT propagation of the projected initial state; it is not a unique reflection coefficient.')
    write('interface_controls',result)
    return result


def analytic_core_control():
    """Independent nonlinear reference: an exactly translating stationary core."""
    x,_,_,ground,mu,residual=setup(257)
    velocity=2*np.pi*4/L; shift=-2.; final=1.
    sp=Space(10)
    initial=spectral_values(ground,sp.x,shift)*np.exp(1j*velocity*sp.x)
    a=sp.project(initial);answer,stats=sp.evolve(a,time=final,rtol=2e-10)
    _,_,xe,we,be=assemble(10,EDGES,64)
    ref=spectral_values(ground,xe,shift+velocity*final)*np.exp(1j*velocity*xe-1j*(mu+.5*velocity**2)*final)
    pred=be@sp.v@answer
    result=dict(degree=10,time=final,velocity=velocity,ground_residual=residual,
                wave_error=relative(pred,ref,we),density_error=relative(abs(pred)**2,abs(ref)**2,we),**stats)
    assert result['wave_error']<1e-5,result
    write('analytic_core',result)
    print('analytic core control',result,flush=True)
    return result


def monochromatic_interface():
    """Exact semidiscrete scattering at two semi-infinite uniform FE chains.

Condense element interiors at frequency E=k^2/2. The remaining two-face
dynamic stiffness determines both bulk dispersion and interface reflection.
Use kh<pi, on the first propagation band, so branch choice is unambiguous.
This control has neither packet projection nor finite-time integration error.
"""
    from shared_basis_review import basis
    from numpy.polynomial.legendre import leggauss
    def response(p,h,k):
        z,w=leggauss(2*p+4);b,db=basis(p,z)
        m=b.T@(w[:,None]*b)*h/2
        kinetic=db.T@(w[:,None]*db)/h
        d=kinetic-.5*k*k*m
        outer=[0,p];inner=list(range(1,p))
        s=d[np.ix_(outer,outer)]
        if inner:
            s=s-d[np.ix_(outer,inner)]@np.linalg.solve(d[np.ix_(inner,inner)],d[np.ix_(inner,outer)])
        a=(s[0,0]+s[1,1])/2;b=(s[0,1]+s[1,0])/2
        cosine=-a/b
        assert -1<cosine<1,(p,h,k,cosine)
        theta=np.arccos(cosine)
        return theta/h,-b*np.sin(theta)
    rows=[]
    for p in (2,3,4,6,8):
        for k in (1.,2.,2.8):
            for ratio in (2.,4.,8.):
                kl,zl=response(p,1/ratio,k);kr,zr=response(p,1.,k)
                r=(zl-zr)/(zl+zr);t=1+r
                rows.append(dict(degree=p,physical_k=k,refinement_ratio=ratio,
                                 reflection_probability=float(r*r),
                                 flux_transmission=float(zr/zl*t*t),
                                 coarse_relative_wavenumber_error=float(abs(kr/k-1)),
                                 fine_relative_wavenumber_error=float(abs(kl/k-1))))
    flux_error=max(abs(r['reflection_probability']+r['flux_transmission']-1) for r in rows)
    assert flux_error<1e-12
    # Independent closed-form dispersion of consistent linear finite elements.
    kcheck=1.7;hcheck=.6
    qcheck,_=response(1,hcheck,kcheck)
    qexact=np.arccos((1-(kcheck*hcheck)**2/3)/(1+(kcheck*hcheck)**2/6))/hcheck
    linear_dispersion_check=float(abs(qcheck-qexact))
    assert linear_dispersion_check<1e-12
    _,zcheck=response(8,1.,2.)
    continuum_flux_check=float(abs(zcheck-1.))  # Z -> k/2 = 1.
    assert continuum_flux_check<1e-10
    # Matched continuum dynamic stiffness would have zero reflection: verify
    # that numerical scattering disappears with polynomial refinement.
    assert max(r['reflection_probability'] for r in rows if r['degree']==8)<1e-14
    result=dict(rows=rows,flux_balance_error=flux_error,
                linear_element_dispersion_check=linear_dispersion_check,
                continuum_flux_check=continuum_flux_check,
                scope='Stationary semi-infinite 1D interface, constant kinetic coefficient, exact consistent FE mass; first propagation band only. Reflection probability here is a genuine scattering coefficient, unlike the packet negative-mode diagnostic.')
    write('monochromatic_interface',result)
    print('monochromatic interface control saved',flush=True)
    return result


def fft_sp(initial,dt,time=T):
    n=len(initial);k=modes(n,L);green=inverse_laplacian(n,L)
    steps=round(time/dt);dt=time/steps
    gamma=1/(2-2**(1/3)); factors=(gamma,1-2*gamma,gamma)
    stages=[(dt*f,np.exp(-.5j*k*k*dt*f)) for f in factors]
    psi=initial.copy()
    for _ in range(steps):
        for h,drift in stages:
            phi=np.fft.ifft(green*np.fft.fft(abs(psi)**2)).real
            psi*=np.exp(-.5j*h*phi)
            psi=np.fft.ifft(np.fft.fft(psi)*drift)
            phi=np.fft.ifft(green*np.fft.fft(abs(psi)**2)).real
            psi*=np.exp(-.5j*h*phi)
    return psi


def core_bath_initial():
    xg,_,_,ground,mu,residual=setup(257)
    # Moving core meets an incoming coherent packet. Normalize the full sum,
    # including its interference term, once for all translated experiments.
    def raw(x):
        core=spectral_values(ground,x,shift=-2.)*np.exp(1j*(2*np.pi*4/L)*x)
        bath=sum(np.exp(-(x-2.+j*L)**2/(2*.7**2))*np.exp(-3j*(x-2.+j*L)+.4j) for j in (-1,0,1))
        return core+.3*bath
    xr=np.arange(1024)*L/1024-L/2
    factor=np.sqrt(4/(L/len(xr)*np.sum(abs(raw(xr))**2)))
    def initial(x):return factor*raw(x)
    return initial,residual


def coupled():
    initial,residual=core_bath_initial()
    xr=np.arange(1024)*L/1024-L/2
    r0=initial(xr); ref=fft_sp(r0,.002)
    ref_time=fft_sp(r0,.001)
    xf=np.arange(1536)*L/1536-L/2
    ref_space=fft_sp(initial(xf),.002)
    controls=dict(time_wave=relative(ref,ref_time,np.ones(len(xr))),
                  space_wave=relative(spectral_values(ref,xf),ref_space,np.ones(len(xf))),ground_residual=residual)
    assert controls['time_wave']<1e-7,controls
    assert controls['space_wave']<1e-8,controls
    print('SP reference checks',controls,flush=True)
    phi_ref=np.fft.ifft(np.fft.fft(abs(ref_time)**2)*inverse_laplacian(len(xr),L)).real
    rows=[]; profiles={}; invariants=[]
    for p in (4,6,8,10):
        sp=Space(p)
        _,_,xe,we,be=assemble(p,EDGES,64);qe=be@sp.v
        for shift in (-2.,0.,2.):
            a=sp.project(initial(sp.x-shift))
            answer,stats=sp.evolve(a)
            pred=qe@answer; reference=spectral_values(ref_time,xe,shift)
            potential=qe@sp.gravity(answer); potential_ref=spectral_values(phi_ref,xe,shift).real
            density=abs(pred)**2; density_ref=abs(reference)**2
            overlap=np.sum(we*pred.conj()*reference)
            # Core-centered aperture in the reference, tracked by translated density peak.
            center=xe[np.argmax(density_ref)];mask=abs((xe-center+L/2)%L-L/2)<1.
            row=dict(degree=p,dofs=len(a),shift=shift,
                     initial_projection_error=relative(qe@a,initial(xe-shift),we),
                     wave_error=relative(pred,reference,we),
                     phase_aligned_error=relative(pred*np.exp(1j*np.angle(overlap)),reference,we),
                     density_error=relative(density,density_ref,we),
                     potential_error=relative(potential,potential_ref,we),
                     aperture_mass_error=float(abs(np.sum(we[mask]*(density[mask]-density_ref[mask])))/np.sum(we[mask]*density_ref[mask])),
                     **stats)
            if shift==0. and p in (4,10):
                # A tighter integration on the identical spatial operator isolates time error.
                tight,tstats=sp.evolve(a,rtol=2e-11)
                row['time_refinement_wave_difference']=float(np.linalg.norm(answer-tight)/np.linalg.norm(tight))
                profiles[str(p)]=dict(x=xe.tolist(),density=density.tolist(),reference=density_ref.tolist())
            rows.append(row)
            print('SP',p,shift,'wave',row['wave_error'],'density',row['density_error'],'nfev',stats['rhs_calls'],flush=True)
        # Verify the nonlinear map is the derivative of the SAME discrete energy.
        rng=np.random.default_rng(183+p)
        direction=rng.normal(size=len(a))+1j*rng.normal(size=len(a));direction/=np.linalg.norm(direction)
        eps=1e-6
        finite=(sp.energy(a+eps*direction)-sp.energy(a-eps*direction))/(2*eps)
        analytic=2*np.vdot(direction,sp.e*a+sp.nonlinear(a)).real
        invariants.append(dict(degree=p,energy_derivative_error=float(abs(finite-analytic)/max(1.,abs(analytic))),
                               orthonormal_error=float(np.linalg.norm(sp.q.T@(sp.w[:,None]*sp.q)-np.eye(len(a)),ord=2))))
    assert max(r['energy_derivative_error'] for r in invariants)<1e-7,invariants
    assert max(r['orthonormal_error'] for r in invariants)<1e-10
    assert max(r['mass_drift'] for r in rows)<1e-7
    assert max(r['energy_drift'] for r in rows)<1e-7
    assert max(r.get('time_refinement_wave_difference',0.) for r in rows)<1e-6
    assert max(r['density_error'] for r in rows if r['degree']==10)<1e-4
    result=dict(final_time=T,mesh=EDGES.tolist(),reference=controls,rows=rows,invariants=invariants,profiles=profiles,
                scope='Coupled full-density Galerkin SP with consistent quadrature and energy derivative. Dense interaction-picture DOP853; no performance claim. One-dimensional periodic gravity, not a spherical 3D halo.')
    write('coupled',result)
    return result


def time_audit():
    """Resolve the mixed spatial/temporal floor exposed by the degree-10 run."""
    initial,residual=core_bath_initial()
    xr=np.arange(1024)*L/1024-L/2;reference=fft_sp(initial(xr),.001)
    sp=Space(10);a=sp.project(initial(sp.x))
    _,_,xe,we,be=assemble(10,EDGES,64);qe=be@sp.v
    ref=spectral_values(reference,xe)
    answer,stats=sp.evolve(a,rtol=2e-11)
    pred=qe@answer
    result=dict(degree=10,shift=0.,rtol=2e-11,
                wave_error=relative(pred,ref,we),density_error=relative(abs(pred)**2,abs(ref)**2,we),
                initial_projection_error=relative(qe@a,initial(xe),we),**stats)
    assert result['wave_error']<1e-7,result
    write('time_audit',result)
    print('Tight time audit',result,flush=True)
    return result


def plots():
    lin=json.loads((OUT/'linear.json').read_text()); src=json.loads((OUT/'source.json').read_text()); sp=json.loads((OUT/'coupled.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for shift in (-2.,0.,2.):
        rows=[r for r in lin['rows'] if r['shift']==shift]
        axs[0].semilogy([r['degree'] for r in rows],[r['wave_error'] for r in rows],'o-',label=f'shift {shift:g}')
        axs[1].semilogy([r['degree'] for r in rows],[r['reverse_error_mass'] for r in rows],'o-',label=f'shift {shift:g}')
        rows=[r for r in sp['rows'] if r['shift']==shift]
        axs[2].semilogy([r['degree'] for r in rows],[r['density_error'] for r in rows],'o-',label=f'shift {shift:g}')
    for ax,title,ylabel in zip(axs,['Exact-time linear wave','Counterpropagating error','Coupled self-gravity'],['Relative wave error','Negative-mode error mass / initial mass','Relative density error']):
        ax.set(xlabel='Polynomial degree',ylabel=ylabel,title=title);ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'translation.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for key,label in [('density_source_potential_error','Source from full density'),('coarse_wave_source_potential_error','Source from projected coarse wave')]:
        axs[0].semilogy([r['degree'] for r in src['rows']],[r[key] for r in src['rows']],'o-',label=label)
    axs[0].set(xlabel='Degree of gravity / coarse wave space',ylabel='Relative potential error');axs[0].legend();axs[0].grid(alpha=.25)
    data=sp['profiles']['10'];axs[1].plot(data['x'],data['reference'],'k-',label='Fourier reference')
    for p in ('4','10'):
        data=sp['profiles'][p];axs[1].plot(data['x'],data['density'],'--',label='Degree '+p)
        error=abs(np.array(data['density'])-np.array(data['reference']))/max(data['reference'])
        axs[2].semilogy(data['x'],np.maximum(error,1e-16),label='Degree '+p)
    for edge in EDGES:axs[1].axvline(edge,color='grey',alpha=.2,lw=.5)
    axs[1].set(xlabel='x',ylabel='Density at T=2',xlim=(-5,5));axs[1].legend()
    axs[2].set(xlabel='x',ylabel='Absolute density error / reference peak',xlim=(-5,5));axs[2].legend();axs[2].grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'source_and_profile.png',dpi=160);plt.close(fig)
    scattering=json.loads((OUT/'monochromatic_interface.json').read_text())
    controls=json.loads((OUT/'interface_controls.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ratio in (2.,4.,8.):
        rows=[r for r in scattering['rows'] if r['refinement_ratio']==ratio and r['physical_k']==2.8]
        axs[0].semilogy([r['degree'] for r in rows],[max(r['reflection_probability'],1e-16) for r in rows],'o-',label=f'{ratio:g}:1 interface')
    axs[0].set(xlabel='Polynomial degree',ylabel='Reflection probability (display floor 1e-16)',title='Monochromatic k=2.8');axs[0].legend();axs[0].grid(alpha=.25)
    for mesh in ('graded','uniform_coarse','uniform_fine'):
        rows=[r for r in controls['rows'] if r['mesh']==mesh]
        axs[1].semilogy([r['degree'] for r in rows],[r['propagation_error'] for r in rows],'o-',label=mesh.replace('_',' '))
    axs[1].set(xlabel='Polynomial degree',ylabel='Wave error from same projected initial state',title='Bulk-resolution controls');axs[1].legend();axs[1].grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'scattering_and_controls.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['linear','source','coupled','controls','analytic','scattering','time_audit','plots','all'],default='all');args=parser.parse_args()
    if args.only in ('linear','all'):linear()
    if args.only in ('source','all'):source_communication()
    if args.only in ('coupled','all'):coupled()
    if args.only in ('controls','all'):interface_controls()
    if args.only in ('analytic','all'):analytic_core_control()
    if args.only in ('scattering','all'):monochromatic_interface()
    if args.only in ('time_audit','all'):time_audit()
    if args.only in ('plots','all'):plots()
