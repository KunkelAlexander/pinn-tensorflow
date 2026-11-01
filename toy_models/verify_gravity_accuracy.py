"""Independent numerical controls and validation for gravity-accuracy results."""
import json
import re
from pathlib import Path
from experiments import np
from gravity_accuracy_evolution import OUT,sample_periodic
from gravity_accuracy_amr_evolution import run


def extra_controls():
    baseline=json.loads((OUT/'amr_evolution.json').read_text());rows=[]
    for shift in (0,16):
        for label in ('hierarchy4_4','hierarchy4_4_projected'):
            u,stats=run(96,label,shift,dt=.00025)
            ref,_=run(192,'uniform0',0,dt=.000125)
            target=abs(ref[::2,::2])**2
            rows.append(dict(label=label,shift_cells=shift,density_error=float(np.linalg.norm(abs(u)**2-target)/np.linalg.norm(target)),**stats))
            print('refined-time and source control',shift,label,rows[-1]['density_error'],flush=True)
    (OUT/'amr_controls.json').write_text(json.dumps(dict(rows=rows,
        scope='Half the comparison time step and a reference with half the original reference step. Projected-parent source omits unresolved Fourier modes, preserving root-resolved density coefficients and mean; point-source and projected-source branches otherwise identical. This is a diagnostic oracle, not GAMER restriction.'),indent=2)+'\n')


def main():
    root=Path(__file__).resolve().parent
    files=['gravity_accuracy_evolution.py','gravity_accuracy_equilibrium.py','gravity_accuracy_interface.py',
           'gravity_accuracy_amr_evolution.py','gravity_accuracy_bandwidth.py','gravity_accuracy_work.py','build_gravity_accuracy_report.py','verify_gravity_accuracy.py']
    for file in files:compile((root/file).read_text(encoding='utf-8'),str(root/file),'exec')
    data={p.stem:json.loads(p.read_text()) for p in OUT.glob('*.json')}
    assert data['bandwidth']['checks']['reference_point_sample']<1e-12
    assert data['interface']['checks']['fd4_quartic_boundary_exactness']<1e-11
    assert max(r['residual'] for r in data['equilibrium']['equilibria'])<1e-9
    x=np.arange(768)*2*np.pi/768;y=np.arange(96)*2*np.pi/96
    field=np.exp(17j*x)+.4*np.exp(-31j*x)
    expected=abs(np.exp(17j*y)+.4*np.exp(-31j*y))**2
    sampling=float(max(abs(sample_periodic(abs(field)**2,96).real-expected)))
    assert sampling<1e-12
    t=(root/'gravity_accuracy.html').read_text(encoding='utf-8')
    local=[link for link in re.findall(r'href="([^"]+)"',t) if not link.startswith(('http','#'))]
    assert all((root/link).exists() for link in local)
    assert t.count('src="data:image/png;base64,')==5
    result=dict(python_files_compiled=files,independent_density_sample_identity=sampling,
                embedded_plots=5,local_links=local,json_files=list(data),
                scope='All experiment assertions passed; report artifacts checked. No GAMER binary compiled or cosmological halo run.')
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    import sys
    extra_controls() if '--extra-controls' in sys.argv else main()
