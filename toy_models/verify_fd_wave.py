"""Source port, continuum-flux order, reference reuse and timestep checks."""
import json
from fd_wave_crosscheck import OUT,ROOT,np,independent_checks,make_step


def flux_order():
    rows=[]
    for order in (2,4):
        for n in (32,64,128,256,512):
            x=np.arange(n)*2*np.pi/n;h=2*np.pi/n;k=np.fft.fftfreq(n)*n
            p=(1+.25*np.cos(x))*np.exp(3j*x)+.3*np.exp(-2j*x)
            derivative=np.fft.ifft(1j*k*np.fft.fft(p));current=np.imag(p.conj()*derivative)
            exact=-np.fft.ifft(1j*k*np.fft.fft(current)).real
            plus=np.roll(p,-1)
            if order==2:face=(p+plus)/2;d=plus-p
            else:
                minus=np.roll(p,1);plus2=np.roll(p,-2)
                face=(-minus+15*p+15*plus-plus2)/28;d=(minus-15*p+15*plus-plus2)/12
            flux=2*np.imag(face.conj()*d);computed=-(flux-np.roll(flux,1))/(2*h*h)
            error=float(np.linalg.norm(computed-exact)/np.linalg.norm(exact))
            rows.append(dict(laplacian_flag_order=order,n=n,continuity_rhs_relative_error=error))
    return rows


def main():
    a=json.loads((OUT/'results.json').read_text());fine=json.loads((OUT/'results_fine.json').read_text())
    control=json.loads((OUT/'results_controls.json').read_text());old=json.loads((ROOT/'potential_replay_results/results.json').read_text())
    assert len(a['rows'])==192,len(a['rows'])
    assert len(fine['rows'])==45,len(fine['rows'])
    assert len(control['rows'])==40,len(control['rows'])
    reuse=[]
    for r in a['rows']:
        mapping={'gram_fft':'fft_fused','gram_matmul':'stored_matmul','global_fft':'global_fft'}
        if r['method'] not in mapping:continue
        o=next(s for s in old['rows'] if s['kinetic']==mapping[r['method']] and all(s[k]==r[k] for k in ('case','n','branch','gravity')))
        reuse.append(abs(r['density_l2']-o['density_l2']))
    assert max(reuse)<1e-9,max(reuse)
    # Constant-amplitude Fourier mode: correction preserves amplitude, while
    # leaving the FD Taylor phase unchanged; no exact-phase substitution allowed.
    plane=[]
    n=64;p=np.exp(5j*np.arange(n)*2*np.pi/n)
    for order in (2,4):
        raw,_,_=make_step(n,2*np.pi,.0002,f'fd_lap{order}')(p)
        corrected,bad,_=make_step(n,2*np.pi,.0002,f'fd_lap{order}_mass')(p)
        amp=float(max(abs(abs(corrected)-1)));phase=float(max(abs(np.angle(corrected/raw))))
        assert amp<1e-13 and phase<1e-13 and bad==0
        plane.append(dict(order=order,amplitude_error=amp,phase_change=phase))
    rows=a['rows']+fine['rows']+control['rows']
    mass=[abs(r['mass_change']) for r in rows if ('_mass' in r['method'] or 'fd4flux' in r['method']) and r['skipped_correction_cellsteps']==0]
    assert max(mass)<1e-10,max(mass)
    pairs=[];times=[]
    for collection in (a,fine):
        for r in collection['rows']:
            if r['method'] not in ('fd_lap4','fd_lap4_mass','gram_fft_fd4flux'):continue
            gram=next(s for s in collection['rows'] if s['method']=='gram_fft' and all(s[k]==r[k] for k in ('case','n','branch','gravity')))
            q=np.array(r['late_folded_profile']);p=np.array(gram['late_folded_profile']);ref=np.array(r['reference_late_folded_profile'])
            pairs.append(dict(case=r['case'],n=r['n'],method=r['method'],branch=r['branch'],gravity=r['gravity'],
                pair_profile_difference=float(np.linalg.norm(q-p)/np.linalg.norm(ref)),
                method_profile_error=r['late_folded_profile_error'],gram_profile_error=gram['late_folded_profile_error']))
    for c in control['rows']:
        b=next(r for r in a['rows'] if all(r[k]==c[k] for k in ('case','n','method','branch','gravity')))
        times.append(dict(case=c['case'],n=c['n'],method=c['method'],gravity=c['gravity'],
            density_base=b['density_l2'],density_half=c['density_l2'],profile_base=b['late_folded_profile_error'],profile_half=c['late_folded_profile_error']))
    result=dict(stencil_port=independent_checks(),plane_correction=plane,flux_spatial_order=flux_order(),
                max_previous_result_difference=max(reuse),max_mass_error_when_no_guard=max(mass),
                guarded_runs=[{k:r[k] for k in ('case','n','method','branch','gravity','skipped_correction_cellsteps','mass_change')} for r in rows if r['skipped_correction_cellsteps']],
                profile_pairs=pairs,time_controls=times)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(max_previous_result_difference=max(reuse),max_mass_error_when_no_guard=max(mass),guarded_runs=len(result['guarded_runs'])),indent=2))


if __name__=='__main__':main()
