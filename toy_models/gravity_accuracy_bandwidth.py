"""Check evolved bandwidth and a weaker-gravity counterpart of the envelope test."""
import json
from experiments import np,plt
from gravity_accuracy_evolution import OUT,evolve,compare,initial_wave,sample_periodic


def main():
    fields=np.load(OUT/'evolution_fields.npz');rows=[];weak=[]
    for case in ('standing','enveloped'):
        ref=fields['reference_'+case];wave=np.fft.fft(ref)/len(ref);density=np.fft.fft(abs(ref)**2)/len(ref)
        k=abs(np.fft.fftfreq(len(ref))*len(ref));density[0]=0
        for n in (80,96,160,192):
            rows.append(dict(case=case,n=n,
                wave_power_beyond_nyquist=float(np.sum(abs(wave[k>=n/2])**2)/np.sum(abs(wave)**2)),
                wave_power_below_five_cells=float(np.sum(abs(wave[k>n/5])**2)/np.sum(abs(wave)**2)),
                density_unresolved_relative_rms=float(np.sqrt(np.sum(abs(density[k>=n/2])**2)/np.sum(abs(density)**2)))))
    # Make sure the comparison oracle samples, rather than filters, fine density.
    x=np.arange(768)*2*np.pi/768;target=np.arange(80)*2*np.pi/80
    sample_check=float(max(abs(sample_periodic(np.cos(63*x),80).real-np.cos(63*target))))
    assert sample_check<1e-12
    ref,stats=evolve(initial_wave(768,'enveloped'),2*np.pi,10,'spectral','global_fft',.5,.0002,order=4)
    half,_=evolve(initial_wave(768,'enveloped'),2*np.pi,10,'spectral','global_fft',.5,.0001,order=4)
    space,_=evolve(initial_wave(1536,'enveloped'),2*np.pi,10,'spectral','global_fft',.5,.0002,order=4)
    for n in (96,192):
        for kinetic in ('global_fft','fft_fused'):
            for gravity in ('fd2','fd4','spectral'):
                p,stats=evolve(initial_wave(n,'enveloped'),2*np.pi,10,gravity,kinetic,.5,.0002)
                weak.append(dict(n=n,kinetic=kinetic,gravity=gravity,**compare(p,ref),steps=stats['steps']))
        print('weak gravity bandwidth control',n,flush=True)
    result=dict(rows=rows,weak_gravity=weak,checks=dict(reference_point_sample=sample_check,
        weak_time=compare(ref,half)['density_error'],weak_space=compare(ref,space)['density_error']),
        scope='Reference wave power and density fluctuation RMS beyond each target grid limit at final time. Five cells refers to wave wavelengths. Weak envelope case changes only coupling from 100 to 10; it is an additional control, not replacement of the difficult case.')
    (OUT/'bandwidth.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,4))
    for case in ('standing','enveloped'):
        ref=fields['reference_'+case];rho=np.fft.fft(abs(ref)**2)/len(ref);k=np.fft.fftfreq(len(ref))*len(ref)
        mask=(k>0)&(k<120);ax.semilogy(k[mask],abs(rho[mask]),label=case)
    for edge in (40,48,80,96):ax.axvline(edge,ls='--',alpha=.3,color='black')
    ax.set(xlabel='Density wave number',ylabel='Density Fourier amplitude',title='Evolved density spectrum; dashed lines are grid Nyquist limits');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'bandwidth.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
