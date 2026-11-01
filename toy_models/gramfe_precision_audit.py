"""Audit stored DFT precision without changing any GAMER source."""
import json
import math
from gramfe_probe import mp,np,OUT,matrix,operator,LAMBDA,fft_tables


def main():
    mp.mp.dps=60
    f=matrix('GramFE_FFT',(64,64),True);inv=matrix('GramFE_IFFT',(16,64),True);ext=matrix('GramFE_Extend',(64,32),True)
    af=mp.matrix([[mp.exp(-2j*mp.pi*r*j/64) for j in range(64)] for r in range(64)])
    ai=mp.matrix([[mp.exp(2j*mp.pi*(i+8)*r/64)/64 for r in range(64)] for i in range(16)])
    dft_error=float(max(abs(f[i,j]-af[i,j]) for i in range(64) for j in range(64)))
    ifft_error=float(max(abs(inv[i,j]-ai[i,j]) for i in range(16) for j in range(64)))
    k=[2*mp.mpf(str(np.pi))/64*(r if r<=32 else r-64) for r in range(64)]
    multiplier=[]
    for r in range(64):
        z=-.5j*mp.mpf(str(LAMBDA))*k[r]**2
        multiplier.append(sum(z**j/mp.factorial(j) for j in range(7))*mp.exp(-16*mp.mpf('2.302585092994046')*abs(mp.mpf(r if r<=32 else r-64)/32)**100))
    direct=inv*mp.diag(multiplier)*f*ext
    fixed=ai*mp.diag(multiplier)*af*ext
    pl,pr,even,odd=fft_tables();factorized_ext=mp.matrix(64,32)
    for i in range(32):factorized_ext[i,i]=1
    for r in range(32):
        for j in range(14):
            factorized_ext[32+r,j]=sum((mp.mpf(float(even[o,r]))-mp.mpf(float(odd[o,r])))*mp.mpf(float(pl[o,j]))/2 for o in range(14))
            factorized_ext[32+r,18+j]=sum((mp.mpf(float(even[o,r]))+mp.mpf(float(odd[o,r])))*mp.mpf(float(pr[o,j]))/2 for o in range(14))
    factorized_control=np.array((ai*mp.diag(multiplier)*af*factorized_ext).tolist(),complex)
    direct=np.array(direct.tolist(),complex);fixed=np.array(fixed.tolist(),complex)
    agreement=float(np.linalg.norm(direct-operator(),ord=2))
    rows=[]
    for ppw in (6,8,10,16,32):
        theta=2*np.pi/ppw;u=np.exp(1j*theta*np.arange(-8,24));ref=u[8:24]*np.exp(-.5j*LAMBDA*theta**2)
        rows.append(dict(ppw=ppw,stored_error=float(np.linalg.norm(direct@u-ref)/4),
                         corrected_dft_error=float(np.linalg.norm(fixed@u-ref)/4),
                         factorized_extension_error=float(np.linalg.norm(factorized_control@u-ref)/4),
                         stored_float32_error=float(np.linalg.norm(direct.astype(np.complex64)@u.astype(np.complex64)-ref)/4)))
    assert agreement<1e-13
    # DFT-only correction does NOT fix the observed smooth-wave accuracy floor.
    assert max(abs(r['corrected_dft_error']-r['stored_error']) for r in rows)<1e-12
    assert rows[-1]['factorized_extension_error']<1e-12,rows
    result=dict(dft_max_entry_error=dft_error,ifft_max_entry_error=ifft_error,
                direct_builder_vs_polynomial_cache=agreement,rows=rows,
                scope='Actual stored constants parsed as decimal at 60 digits. DFT-only control retains extension and MATMUL filter; factorized-extension control builds extension from FFT branch Pl/Pr/Fe/Fo while retaining MATMUL filter. DFT-only correction does not improve the observed smooth-wave floor. Not a compiled binary128 or GPU test. No original source modified.')
    np.save(OUT/'corrected_dft_control.npy',fixed)
    (OUT/'precision_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
