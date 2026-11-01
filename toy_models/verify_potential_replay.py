"""Independent endpoint-time-potential check and saved experiment validation."""
import json
from potential_replay import OUT, np, errors, rollout, gravity_map


def main():
    # Spatially constant Phi(t)=sin(t) commutes with kinetic evolution. This
    # independently checks temporal indexing of prescribed kicks and phase norms.
    n=32;x=np.arange(n)*2*np.pi/n;trials=[]
    for steps in (20,40):
        dt=.7/steps;t=np.arange(steps+1)*dt
        phi=np.sin(t)[:,None]*np.ones((1,n))
        wave=np.exp(2j*x)[None,:]*np.exp(-1j*(2*t+1-np.cos(t)))[:,None]
        ref=dict(waves={n:wave},pots={n:phi})
        p,row=rollout(ref,2*np.pi,1,n,dt,'global_fft','prescribed','spectral')
        integral=np.trapezoid(np.sin(t),t)
        analytic_discrete=np.exp(2j*x)*np.exp(-1j*(2*.7+integral))
        assert np.max(abs(p-analytic_discrete))<2e-13
        assert row['density_l2']<2e-13
        trials.append(row['wave_l2'])
    assert 3.99<trials[0]/trials[1]<4.01
    # The 1D child solve must satisfy its own difference equation exactly;
    # subtracting a potential gauge constant leaves this equation unchanged.
    n=64;length=20.;h=length/n;x=np.arange(n)*h
    rho=1+.2*np.cos(2*np.pi*x/length)+.3*np.cos(8*np.pi*x/length)
    phi=gravity_map(n,length,1,'hierarchy2')(rho);lo,hi=n//4,3*n//4
    residual=(phi[lo:hi-1]-2*phi[lo+1:hi]+phi[lo+2:hi+1])/h**2-(rho[lo+1:hi]-rho.mean())
    assert max(abs(residual))<2e-13
    result=json.loads((OUT/'results.json').read_text());control=json.loads((OUT/'results_controls.json').read_text())
    rows=result['rows']+control['rows']
    assert len(result['rows'])==180,len(result['rows'])
    assert len(control['rows'])==44,len(control['rows'])
    assert max(abs(r['work_closure']) for r in rows)<1e-8
    mass=max(abs(r['mass_change']) for r in rows if r['kinetic']=='global_fft')
    assert mass<1e-10
    space=max(c['space']['wave_l2'] for c in result['checks'].values())
    time=max(c['time']['wave_l2'] for c in result['checks'].values())
    assert space<1e-7 and time<1e-7
    assert max(r['wave_l2'] for r in result['plane_calibration'] if r['kinetic']=='global_fft')<1e-10
    comparison=[]
    for c in control['rows']:
        b=next(r for r in result['rows'] if all(r[key]==c[key] for key in ('case','n','kinetic','branch','gravity')))
        comparison.append(dict(case=c['case'],n=c['n'],kinetic=c['kinetic'],branch=c['branch'],gravity=c['gravity'],
            baseline_density=b['density_l2'],half_dt_density=c['density_l2'],
            baseline_wave=b['wave_l2'],half_dt_wave=c['wave_l2']))
    validation=dict(endpoint_quadrature_wave_errors=trials,endpoint_quadrature_ratio=trials[0]/trials[1],
                    manufactured_child_residual=float(max(abs(residual))),global_fft_max_mass_error=mass,
                    max_reference_space_wave_error=space,max_reference_time_wave_error=time,
                    timestep_comparison=comparison)
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='timestep_comparison'},indent=2))


if __name__=='__main__':main()
