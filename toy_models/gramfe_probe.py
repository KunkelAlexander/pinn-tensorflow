"""Reconstruct the pinned GAMER patch-size-8 Gram-FE kinetic column update.

Uses the actual stored tables. High-precision contraction avoids blaming a
float64 port for cancellation the production MATMUL builder handles in float128.
This is not a compiled GAMER run and does not reproduce its complete AMR loop.
"""
import argparse
import json
import math
import re
import hashlib
from functools import lru_cache
from pathlib import Path
from experiments import np, plt
import mpmath as mp

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'gramfe_results';OUT.mkdir(exist_ok=True)
HEADER=ROOT/'sources/gamer/include/GramFE_ExtensionTables.h'
LAMBDA=4/np.pi*.2


def tokens(name,which=0):
    text=HEADER.read_text()
    blocks=re.findall(r'\b'+name+r'\[[^;]+?=\s*\{(.*?)\};',text,re.S)
    return re.findall(r'[-+]?\d+\.\d+(?:e[-+]\d+)?',blocks[which],re.I)


def matrix(name,shape,exact=False):
    vals=tokens(name)
    if exact:
        z=[mp.mpc(vals[i],vals[i+1]) for i in range(0,len(vals),2)]
        return mp.matrix([[z[i*shape[1]+j] for j in range(shape[1])] for i in range(shape[0])])
    z=np.array(vals,float);return (z[::2]+1j*z[1::2]).reshape(shape)


@lru_cache(None)
def fft_tables():
    pl=np.array(tokens('Pl'),float).reshape(14,14)
    pr=np.array(tokens('Pr'),float).reshape(14,14)
    fe=np.array(tokens('Fe',4),float).reshape(14,32)
    fo=np.array(tokens('Fo',4),float).reshape(14,32)
    return pl,pr,fe,fo


def prepare():
    cache=OUT/'operators.npz'
    metadata=OUT/'operator_provenance.json'
    fingerprint=dict(header_sha256=hashlib.sha256(HEADER.read_bytes()).hexdigest(),decimal_precision=55,builder_version=1)
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text())==fingerprint:
        return np.load(cache)
    mp.mp.dps=55
    f=matrix('GramFE_FFT',(64,64),True)
    inv=matrix('GramFE_IFFT',(16,64),True)
    ext=matrix('GramFE_Extend',(64,32),True)
    # Analytic DFT control only changes the transform coefficients, not extension.
    analytic_f=mp.matrix([[mp.exp(-2j*mp.pi*r*j/64) for j in range(64)] for r in range(64)])
    analytic_inv=mp.matrix([[mp.exp(2j*mp.pi*(i+8)*r/64)/64 for r in range(64)] for i in range(16)])
    k=[2*mp.mpf(str(np.pi))/64*(r if r<=32 else r-64) for r in range(64)]
    saved={}
    for label,forward,inverse in [('stored',f,inv),('analytic_dft',analytic_f,analytic_inv)]:
        fe=forward*ext
        for filter_name,decay,power in [('matmul',16*mp.mpf('2.302585092994046'),100),('none',mp.mpf(0),100)]:
            if label=='analytic_dft' and filter_name=='matmul':continue
            coeff=[]
            filt=[mp.exp(-decay*abs(mp.mpf(r if r<=32 else r-64)/32)**power) for r in range(64)]
            for order in range(7):
                weighted=mp.matrix([[filt[r]*(-.5j*k[r]**2)**order/mp.factorial(order)*fe[r,j] for j in range(32)] for r in range(64)])
                result=inverse*weighted
                coeff.append(np.array(result.tolist(),complex))
            saved[label+'_'+filter_name]=np.array(coeff)
            print('prepared',label,filter_name,flush=True)
    # Fused FFT variant: supplied double tables, high-precision contraction.
    pl,pr,even,odd=fft_tables()
    ex=mp.matrix(64,32)
    for i in range(32):ex[i,i]=1
    for r in range(32):
        for j in range(14):
            ex[32+r,j]=sum((mp.mpf(float(even[o,r]))-mp.mpf(float(odd[o,r])))*mp.mpf(float(pl[o,j]))/2 for o in range(14))
            ex[32+r,18+j]=sum((mp.mpf(float(even[o,r]))+mp.mpf(float(odd[o,r])))*mp.mpf(float(pr[o,j]))/2 for o in range(14))
    fe=analytic_f*ex;coeff=[]
    for order in range(7):
        weighted=mp.matrix([[mp.exp(-32*mp.mpf('2.302585092994046')*abs(mp.mpf(r if r<=32 else r-64)/32)**200)*(-.5j*k[r]**2)**order/mp.factorial(order)*fe[r,j] for j in range(32)] for r in range(64)])
        coeff.append(np.array((analytic_inv*weighted).tolist(),complex))
    saved['fft_fused']=np.array(coeff)
    np.savez(cache,**saved)
    metadata.write_text(json.dumps(fingerprint,indent=2)+'\n')
    return saved


def operator(lam=LAMBDA,kind='stored_matmul'):
    coeff=prepare()[kind]
    return sum(c*lam**j for j,c in enumerate(coeff))


def fft_update(u,lam=LAMBDA):
    # Vectorized independent transcription of the FFT runtime operations.
    pl,pr,fe,fo=fft_tables()
    al=u[...,:14]@pl.T;ar=u[...,-14:]@pr.T
    extension=.5*(ar+al)@fe+.5*(ar-al)@fo
    padded=np.concatenate([u,extension],axis=-1)
    k=2*np.pi*np.fft.fftfreq(64);z=-.5j*lam*k*k
    mult=sum(z**n/math.factorial(n) for n in range(7))*np.exp(-32*np.log(10)*(abs(k)/np.pi)**200)
    return np.fft.ifft(np.fft.fft(padded,axis=-1)*mult,axis=-1)[...,8:24]


def periodic_update(psi,op):
    n=len(psi);ids=(16*np.arange(n//16)[:,None]+np.arange(-8,24)[None,:])%n
    return (psi[ids]@op.T).reshape(n)


def diagnostics():
    stored=operator(0,'stored_none'); ideal=operator(0,'analytic_dft_none')
    crop=np.eye(32)[8:24]
    rows=[]
    for ppw in (4.,6.,8.,10.,12.,16.,24.,32.):
        theta=2*np.pi/ppw;u=np.exp(1j*theta*np.arange(-8,24));ref=u[8:24]*np.exp(-.5j*LAMBDA*theta**2)
        for kind in ('stored_matmul','fft_fused'):
            out=operator(kind=kind)@u
            ratio=out/ref
            rows.append(dict(ppw=ppw,kind=kind,relative_error=float(np.linalg.norm(out-ref)/4),
                             mean_phase_error=float(np.angle(np.mean(ratio))),
                             mean_amplitude_error=float(abs(np.mean(ratio))-1),
                             patch_modulation=float(np.linalg.norm(ratio-np.mean(ratio))/4)))
        runtime=fft_update(u)
        rows.append(dict(ppw=ppw,kind='fft_runtime',relative_error=float(np.linalg.norm(runtime-ref)/4),
                         fused_runtime_difference=float(np.linalg.norm(runtime-operator(kind='fft_fused')@u)/4)))
    # Same-level Bloch matrix: one physical wavelength can couple patch harmonics.
    bloch=[]
    for ppw in (4.,6.,8.,10.,12.,16.,24.,32.):
        theta=2*np.pi/ppw
        for kind in ('stored_matmul','fft_fused'):
            op=operator(kind=kind);b=np.zeros((16,16),complex)
            for j in range(32):
                location=j-8;offset=location//16;local=location%16
                b[:,local]+=op[:,j]*np.exp(1j*theta*16*offset)
            vec=np.exp(1j*theta*np.arange(16));out=np.linalg.matrix_power(b,10000)@vec
            exact=vec*np.exp(-.5j*theta*theta*LAMBDA*10000)
            bloch.append(dict(ppw=ppw,kind=kind,steps=10000,
                              wave_error=float(np.linalg.norm(out-exact)/4),
                              mass_ratio=float(np.linalg.norm(out)**2/16),
                              spectral_radius=float(max(abs(np.linalg.eigvals(b))))))
    result=dict(lambda_dt_over_dx2=LAMBDA,
                stored_zero_time_no_filter_matrix_error=float(np.linalg.norm(stored-crop,ord=2)),
                analytic_dft_zero_time_no_filter_matrix_error=float(np.linalg.norm(ideal-crop,ord=2)),
                one_step=rows,bloch=bloch,
                scope='Patch-size 8, ghost 8, 14 Gram boundary samples, 32 extension points; MATMUL stored tables contracted at 55 decimal digits then cast to float64; FFT fused control uses actual FFT tables with analytic DFT. No AMR, gravity, mass rescaling, or density floor.')
    assert result['analytic_dft_zero_time_no_filter_matrix_error']<1e-20
    assert max(r['relative_error'] for r in rows if r['ppw']==32)<1e-6
    (OUT/'dispersion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


def long_run_spectrum():
    rows=[]
    for kind in ('stored_matmul','fft_fused'):
        for factor in (1.,.5,.25):
            op=operator(LAMBDA*factor,kind);worst=(0.,None,None)
            for phase in np.linspace(-np.pi,np.pi,513):
                b=np.zeros((16,16),complex)
                for j in range(32):
                    location=j-8;b[:,location%16]+=op[:,j]*np.exp(1j*phase*(location//16))
                vals=np.linalg.eigvals(b);index=np.argmax(abs(vals));radius=float(abs(vals[index]))
                if radius>worst[0]:worst=(radius,float(phase),b)
            radius,phase,b=worst;growth=np.log(radius)
            values,vectors=np.linalg.eig(b);j=np.argmax(abs(values));v=vectors[:,j];v/=np.linalg.norm(v)
            coefficients=np.fft.fft(v*np.exp(-1j*phase*np.arange(16)/16))
            wave_numbers=phase/16+2*np.pi*np.fft.fftfreq(16)
            dominant=int(np.argmax(abs(coefficients)));dominant_ppw=2*np.pi/abs(wave_numbers[dominant])
            # Independently iterate the actual factored FFT operations with
            # quasiperiodic ghost data, or the MATMUL column action.
            evolved=v.copy();locations=np.arange(-8,24)
            for _ in range(10000):
                padded=evolved[locations%16]*np.exp(1j*phase*(locations//16))
                evolved=fft_update(padded,LAMBDA*factor) if kind=='fft_fused' else op@padded
            predicted=radius**10000;measured=float(np.linalg.norm(evolved))
            rows.append(dict(kind=kind,dt_factor=factor,max_spectral_radius=radius,bloch_phase=phase,
                             amplitude_efold_steps=None if growth<=0 else 1/growth,
                             steps_for_1e7_amplification=None if growth<=0 else np.log(1e7)/growth,
                             steps_for_1e14_amplification=None if growth<=0 else np.log(1e14)/growth,
                             max_single_step_singular_value=float(np.linalg.svd(b,compute_uv=False)[0]),
                             unstable_mode_dominant_ppw=float(dominant_ppw),
                             predicted_10000_step_amplitude=float(predicted),
                             runtime_10000_step_amplitude=measured,
                             runtime_prediction_difference=abs(measured-predicted)))
    assert max(r['runtime_prediction_difference'] for r in rows)<1e-3,rows
    result=dict(rows=rows,scope='Dense Bloch spectrum of the same-level linear map. Positive growth is conditional linear-mode evidence, not measured activation in a cosmological run; roundoff, filtering, subcycling, gravity, and physical perturbations affect seeding. Smaller dt retains a per-step filter and need not monotonically improve this diagnostic.')
    (OUT/'long_run_spectrum.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--long-run',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.long_run:long_run_spectrum()
    else:diagnostics()
