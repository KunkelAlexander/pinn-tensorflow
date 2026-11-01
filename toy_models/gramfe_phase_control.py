"""Does replacing the Taylor kinetic phase remove frozen-potential growth?"""
import json
from experiments import np
from gramfe_probe import fft_tables,fft_update,operator,LAMBDA
from frozen_interference_gravity import OUT


def patch_exact(lam):
    # Basis-vector application of actual double-precision FFT extension.
    u=np.eye(32);pl,pr,fe,fo=fft_tables()
    al=u[:,:14]@pl.T;ar=u[:,-14:]@pr.T
    extension=.5*(ar+al)@fe+.5*(ar-al)@fo
    q=2*np.pi*np.fft.fftfreq(64)
    multiplier=np.exp(-.5j*lam*q*q)*np.exp(-32*np.log(10)*(abs(q)/np.pi)**200)
    return np.fft.ifft(np.fft.fft(np.concatenate([u,extension],axis=1),axis=1)*multiplier,axis=1)[:,8:24].T


def main():
    rows=[]
    for factor in (1.,.5):
        lam=LAMBDA*factor
        operators={'fused_taylor':operator(lam,'fft_fused'),
            'runtime_basis_taylor':fft_update(np.eye(32),lam).T,'runtime_basis_exact_phase':patch_exact(lam)}
        discrepancy=float(np.linalg.norm(operators['fused_taylor']-operators['runtime_basis_taylor'],2))
        for n in (80,160):
            h=2*np.pi/n;dt=lam*h*h;x=np.arange(n)*h
            for ratio in (0.,.2):
                kick=np.exp(-.5j*dt*ratio*128*np.cos(32*x+.37))
                for name,op in operators.items():
                    kinetic=np.zeros((n,n),complex)
                    for start in range(0,n,16):
                        kinetic[np.ix_(np.arange(start,start+16),(start+np.arange(-8,24))%n)]=op
                    radius=float(max(abs(np.linalg.eigvals(kick[:,None]*kinetic*kick[None,:]))))
                    rows.append(dict(n=n,dt_factor=factor,potential_ratio=ratio,phase=name,radius=radius,
                        growth_rate=float(np.log(radius)/dt),basis_vs_fused_operator_difference=discrepancy))
    (OUT/'phase_control.json').write_text(json.dumps(dict(rows=rows,
        scope='Double-precision basis reconstruction controls the effect of replacing the Taylor phase by an exact exponential while retaining FFT extension and filter. Basis/Taylor versus high-precision fused/Taylor reports numerical reconstruction sensitivity.'),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
