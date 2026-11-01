"""Measure kinetic-energy transfer separately in gravity and kinetic substeps."""
import json
from experiments import np
from gravity_accuracy_evolution import OUT,initial_wave,evolve


def main():
    _,ref=evolve(initial_wave(768,'enveloped'),2*np.pi,100,'spectral','global_fft',.5,.0002,order=4,track_kick_work=True)
    reference=ref['discrete_gravity_kick_work'];rows=[]
    for n in (96,192):
        for kinetic in ('global_fft','fft_fused'):
            for gravity in ('fd2','fd4','spectral'):
                _,stats=evolve(initial_wave(n,'enveloped'),2*np.pi,100,gravity,kinetic,.5,.0002,track_kick_work=True)
                w=stats['discrete_gravity_kick_work'];d=stats['numerical_kinetic_drift_work']
                rows.append(dict(n=n,kinetic=kinetic,gravity=gravity,gravity_kick_work=w,numerical_kinetic_work=d,
                    relative_gravity_work_bias=w/reference-1,balance_error=w+d-stats['kinetic_change']))
        print('discrete gravity work',n,flush=True)
    assert max(abs(r['balance_error']) for r in rows)<1e-9
    assert max(abs(r['numerical_kinetic_work']) for r in rows if r['kinetic']=='global_fft')<1e-7
    result=dict(rows=rows,reference_gravity_work=reference,
        scope='Changes in spectral kinetic energy measured before/after every gravity kick and kinetic step, summed separately. This is a discrete work budget, not time quadrature of j dot force. The older sampled integrated_gravity_work field is under-resolved for rapid fringes and is not used for conclusions.')
    (OUT/'work.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
