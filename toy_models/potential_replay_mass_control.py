"""Diagnostic only: how much does Gram-FE mass drift drive gravity feedback?

Global normalization is not GAMER's local continuity/flux correction and is not
proposed as a production remedy. It preserves only total mass, not local flux.
"""
import json
from potential_replay import OUT,reference,initial,rollout,np


def main():
    ref=reference(initial('strong_envelope',768),2*np.pi,100.,.0002,2500,[96,192])
    rows=[]
    for n in (96,192):
        for kinetic in ('fft_fused','stored_matmul'):
            for branch in ('prescribed','self_consistent'):
                p,r=rollout(ref,2*np.pi,100.,n,.0002,kinetic,branch,'spectral',conserve_mass=True)
                r['case']='strong_envelope';rows.append(r)
                print('mass control',n,kinetic,branch,r['density_l2'],flush=True)
    assert max(abs(r['mass_change']) for r in rows)<1e-12
    assert max(abs(r['work_closure']) for r in rows)<1e-8
    (OUT/'mass_control.json').write_text(json.dumps(dict(rows=rows,
        scope='Strong envelope only, same reference and kinetics as main test, spectral gravity. Globally normalize wave after each kinetic step to preserve initial mass. No local conservation or flux correction: NOT the GAMER conservation algorithm and NOT a production recommendation. Work of normalization is tracked separately.'),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
