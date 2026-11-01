"""Cross-check follow-up artifacts and inspect the growing-mode bandwidth."""
import json
import re
from pathlib import Path
from experiments import np
from gramfe_probe import LAMBDA
from gramfe_frozen_potential import kinetic_matrix
from frozen_interference_gravity import OUT


def main():
    root=Path(__file__).resolve().parent
    files=['frozen_interference_gravity.py','gramfe_frozen_potential.py','gramfe_phase_control.py',
           'build_frozen_gravity_report.py','verify_frozen_gravity.py']
    for file in files:compile((root/file).read_text(encoding='utf-8'),str(root/file),'exec')
    data={p.stem:json.loads(p.read_text()) for p in OUT.glob('*.json')}
    assert max(data['frozen_gravity']['checks'].values())<1e-11
    baseline=[r for r in data['frozen_potential']['rows'] if r['kind']=='periodic_fft']
    mass_defect=max(abs(s['mass_ratio']-1) for r in baseline for s in r['samples'])
    assert mass_defect<1e-8
    modes=[];n=160;h=2*np.pi/n;dt=LAMBDA*h*h;x=np.arange(n)*h
    kick=np.exp(-.5j*dt*25.6*np.cos(32*x+.37))
    for kind in ('stored_matmul','fft_fused'):
        update=kick[:,None]*kinetic_matrix(n,LAMBDA,kind)*kick[None,:]
        vals,vecs=np.linalg.eig(update);v=vecs[:,np.argmax(abs(vals))]
        spectrum=abs(np.fft.fft(v))**2;wave=abs(np.fft.fftfreq(n)*n)
        index=np.argmax(spectrum)
        modes.append(dict(kind=kind,dominant_wave_number=float(wave[index]),dominant_ppw=float(n/wave[index]),
            spectral_power_below_five_ppw=float(np.sum(spectrum[wave>n/5])/np.sum(spectrum))))
    text=(root/'frozen_gravity.html').read_text(encoding='utf-8')
    links=re.findall(r'href="([^"]+)"',text)
    local=[link for link in links if not link.startswith(('http','#'))]
    assert all((root/link).exists() for link in local)
    assert text.count('src="data:image/png;base64,')==3
    result=dict(python_files_compiled=files,local_report_links_checked=local,embedded_images=3,
                max_periodic_fft_mass_defect=mass_defect,growing_modes=modes,
                numerical_checks='Analytic density, FFT Poisson, exact cubic alias coefficients, direct residuals, harmonic trace, repeated runtime eigenmode growth and smooth-state evolution.')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
