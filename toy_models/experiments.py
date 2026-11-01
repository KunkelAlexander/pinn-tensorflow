"""Small deterministic experiments for possible 3D Schrodinger--Poisson solvers.

Run with the project's pinn_env Python. All generated files stay beside this file.
Dense matrices are diagnostic oracles, never proposed production implementations.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter

# Keep the tiny dense problems from spawning many BLAS threads.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh, lu_factor, lu_solve
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator, expm_multiply

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def phase_error(a, b):
    overlap = np.vdot(a, b)
    return relative(a * np.exp(1j * np.angle(overlap)), b)


def modes(n, length):
    return 2 * np.pi * np.fft.fftfreq(n, d=length / n)


def inverse_laplacian(n, length, kind="spectral"):
    k = modes(n, length)
    dx = length / n
    s = np.sin(k * dx / 2) ** 2
    if kind == "spectral":
        lam = k * k
    elif kind == "fd2":
        lam = 4 * s / dx**2
    elif kind == "fd4":
        lam = 4 * s * (1 + s / 3) / dx**2
    elif kind == "fd6":
        lam = 4 * s * (1 + s / 3 + 8 * s**2 / 45) / dx**2
    else:
        raise ValueError(kind)
    multiplier = np.zeros(n)
    multiplier[1:] = -1 / lam[1:]
    return multiplier


def poisson(rho, length, kind="spectral", coupling=1.0):
    return np.fft.ifft(np.fft.fft(rho) * inverse_laplacian(len(rho), length, kind)).real * coupling


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=155)
    plt.close()


def gravity_transfer():
    """Two exactly represented waves whose density needs twice the bandwidth."""
    m, length, amplitude = 8, 2 * np.pi, 0.8
    xfine = np.arange(4096) * length / 4096
    exact = -2 * amplitude * np.cos(2 * m * xfine) / (2 * m)**2
    rows = []
    for n in (24, 32, 40, 48, 64, 80, 96, 128, 192, 256):
        x = np.arange(n) * length / n
        psi = np.exp(1j * m * x) + amplitude * np.exp(-1j * m * x)
        for kind in ("spectral", "fd2", "fd4", "fd6"):
            phi = poisson(abs(psi)**2, length, kind)
            rows.append(dict(n=n, wave_ppw=n / m, density_ppw=n / (2 * m),
                             operator=kind, potential_error=relative(resample(phi, len(xfine)), exact)))
        # This uses only the represented psi coefficients. It adds quadrature
        # nodes for the nonlinear source, not extra independent wave unknowns.
        padded_psi = resample(psi, 2*n)
        padded_phi = poisson(abs(padded_psi)**2, length)
        rows.append(dict(n=n, wave_ppw=n/m, density_ppw=2*n/(2*m),
                         operator='spectral_padded_source', source_n=2*n,
                         potential_error=relative(resample(padded_phi,len(xfine)),exact)))
    plt.figure(figsize=(7, 4))
    for kind in ("spectral", "spectral_padded_source", "fd2", "fd4", "fd6"):
        data = [r for r in rows if r["operator"] == kind]
        plt.semilogy([r["wave_ppw"] for r in data], [max(r["potential_error"], 1e-15) for r in data], "o-", label=kind)
    plt.xlabel("Points per wavefunction wavelength (density has half as many)")
    plt.ylabel("Relative potential error on independent fine grid")
    plt.legend(); plt.grid(alpha=.25)
    savefig("gravity_transfer.png")
    return rows


def restriction_cancellation():
    """Literal scalar reproduction of Flu_FixUp_Restrict.cpp:453--475.

    Eight fine values, unit density, exactly zero complex average. This does
    not run GAMER, nor does it exercise the optional phase restriction path.
    """
    fine = np.array([1., -1.] * 4, dtype=complex)
    rho_corr = float(np.mean(abs(fine)**2))
    re, im = np.mean(fine).real, np.mean(fine).imag
    rho_wrong = re * re + im * im
    if rho_wrong <= 0 or rho_corr <= 0:
        rho_out, scale = 0., 0.
    else:
        rho_out, scale = rho_corr, np.sqrt(rho_corr / rho_wrong)
    return dict(fine_values=[[float(z.real), float(z.imag)] for z in fine],
                expected_coarse_density=rho_corr, returned_coarse_density=rho_out,
                returned_complex=[float(re * scale), float(im * scale)],
                scope="Reduced reproduction, standard complex restriction only; not a GAMER integration test")


def inherited_boundary():
    """Isolate inherited potential boundary errors from local solve residuals.

    A 1D refined interval has manufactured exact density. Its fine-grid
    discrete problem is solved directly to roundoff for each boundary choice.
    This is an AMR mechanism test, not a replica of GAMER's interpolation.
    """
    length=2*np.pi; nc=32
    xc=np.arange(nc)*length/nc
    exact=lambda x: np.cos(x)+.01*np.cos(12*x)
    source=lambda x: -np.cos(x)-1.44*np.cos(12*x)
    ends=np.array([.43,2.83]); rows=[]
    for kind in ('exact_boundary','coarse_spectral','coarse_fd2'):
        if kind=='exact_boundary': bc=exact(ends)
        else:
            phi=poisson(source(xc),length,'spectral' if kind=='coarse_spectral' else 'fd2')
            spline=CubicSpline(np.r_[xc,length],np.r_[phi,phi[0]],bc_type='periodic')
            bc=spline(ends)
        for n in (32,64,128,256):
            x=np.linspace(*ends,n+2); h=x[1]-x[0]
            rhs=source(x[1:-1])*h*h
            rhs[0]-=bc[0];rhs[-1]-=bc[1]
            matrix=np.zeros((3,n));matrix[0,1:]=1;matrix[1,:]=-2;matrix[2,:-1]=1
            phi=solve_banded((1,1),matrix,rhs)
            residual=np.r_[bc[0],phi,bc[1]]
            residual=(residual[2:]-2*residual[1:-1]+residual[:-2])/h**2-source(x[1:-1])
            rows.append(dict(boundary=kind,fine_interior_n=n,
                             relative_potential_error=relative(phi,exact(x[1:-1])),
                             max_discrete_residual=float(np.max(abs(residual))),
                             boundary_error=[float(v) for v in bc-exact(ends)]))
    plt.figure(figsize=(7,4))
    for kind in ('exact_boundary','coarse_spectral','coarse_fd2'):
        d=[r for r in rows if r['boundary']==kind]
        plt.loglog([r['fine_interior_n'] for r in d],[r['relative_potential_error'] for r in d],'o-',label=kind)
    plt.xlabel('Interior points in refined interval');plt.ylabel('Relative fine potential error')
    plt.legend();plt.grid(alpha=.25);savefig('inherited_boundary.png')
    return rows


def nufft_diagnostics():
    """Exact dense NUDFTs remove any implementation/NUFFT tolerance ambiguity."""
    rng = np.random.default_rng(1931)
    nmode, nsample = 41, 100
    k = np.arange(-(nmode // 2), nmode // 2 + 1)
    coefficients = (rng.normal(size=nmode) + 1j * rng.normal(size=nmode)) / np.sqrt(nmode)
    grid = np.arange(2048) * 2 * np.pi / 2048
    fine_matrix = np.exp(1j * np.outer(grid, k))
    rows = []
    for name, alpha in (("uniform", 0.), ("mild", .5), ("strong", .9), ("extreme", .98)):
        # Inverse CDF, placing most points near zero, leaving a large outer gap.
        q = np.arange(nsample) * 2 * np.pi / nsample
        if alpha == 0:
            x = q.copy()
        else:
            x = 2 * np.arctan((1 - alpha) / (1 + alpha) * np.tan((q - np.pi) / 2)) + np.pi
        order = np.argsort(x); x = x[order]
        gaps = np.diff(np.r_[x[-1] - 2 * np.pi, x, x[0] + 2 * np.pi])
        w = (gaps[:-1] + gaps[1:]) / (4 * np.pi)
        matrix = np.exp(1j * np.outer(x, k))
        values = matrix @ coefficients
        direct = matrix.conj().T @ values / nsample
        weighted = matrix.conj().T @ (w * values)
        fit = np.linalg.lstsq(matrix, values, rcond=1e-14)[0]
        noisy = values + 1e-12 * (rng.normal(size=nsample) + 1j * rng.normal(size=nsample))
        fit_noisy = np.linalg.lstsq(matrix, noisy, rcond=1e-14)[0]
        rows.append(dict(distribution=name, condition=float(np.linalg.cond(matrix)),
                         max_gap_nyquist_wavelengths=float(gaps.max() / (2 * np.pi / max(abs(k)))),
                         adjoint_error=relative(direct, coefficients), weighted_adjoint_error=relative(weighted, coefficients),
                         least_squares_error=relative(fit, coefficients),
                         noisy_fine_grid_error=relative(fine_matrix @ fit_noisy, fine_matrix @ coefficients),
                         rank=int(np.linalg.matrix_rank(matrix))))
    plt.figure(figsize=(7, 4))
    for key in ("adjoint_error", "weighted_adjoint_error", "least_squares_error", "noisy_fine_grid_error"):
        plt.semilogy([r["distribution"] for r in rows], [max(r[key], 1e-16) for r in rows], "o-", label=key.replace("_", " "))
    plt.ylabel("Relative error"); plt.legend(fontsize=8); plt.grid(alpha=.25)
    savefig("nonuniform_fourier.png")
    return rows


def linear_time():
    """Static smooth periodic Hamiltonian, eigen-decomposition as time oracle.

    Add a common energy shift: physically it changes only the global phase.
    Count full-H exponential actions to expose work hidden by large steps.
    """
    n, length, final = 96, 2 * np.pi, 1.
    x = np.arange(n) * length / n
    k = modes(n, length)
    kinetic = np.fft.ifft((k*k/2)[:, None] * np.fft.fft(np.eye(n), axis=0), axis=0).real
    potential = -15 * np.cos(x) + 3 * np.cos(2*x)
    hamiltonian = kinetic + np.diag(potential)
    energies, vectors = eigh(hamiltonian)
    initial = (vectors[:, 0] + .5j * vectors[:, 2] + .2 * vectors[:, 5]) / np.sqrt(1.29)
    reference = vectors @ (np.exp(-1j * final * energies) * (vectors.T @ initial))
    rows = []
    for shift in (0., 500.):
        shifted = hamiltonian + shift * np.eye(n)
        for method in ("strang", "yoshida4", "crank_nicolson", "gauge_cn", "full_exponential"):
            for steps in (1, 2, 4, 8, 16, 32, 64, 128):
                if method == "full_exponential" and steps != 1:
                    continue
                dt = final / steps; psi = initial.astype(complex); matvecs = [0]
                start = perf_counter()
                if method in ("crank_nicolson", "gauge_cn"):
                    hc = hamiltonian if method == "gauge_cn" else shifted
                    lu = lu_factor(np.eye(n) + .5j * dt * hc)
                    right = np.eye(n) - .5j * dt * hc
                for _ in range(steps):
                    if method in ("strang", "yoshida4"):
                        a = 1 / (2 - 2**(1/3))
                        stages = (1.,) if method == "strang" else (a, 1 - 2*a, a)
                        for factor in stages:
                            h = dt * factor
                            psi *= np.exp(-.5j * h * (potential + shift))
                            psi = np.fft.ifft(np.exp(-.5j * h * k*k) * np.fft.fft(psi))
                            psi *= np.exp(-.5j * h * (potential + shift))
                    elif method in ("crank_nicolson", "gauge_cn"):
                        psi = lu_solve(lu, right @ psi)
                        if method == "gauge_cn":
                            psi *= np.exp(-1j * dt * shift)
                    else:
                        # Shift the known trace analytically, as expm_multiply does.
                        def action(v):
                            matvecs[0] += v.shape[1] if v.ndim == 2 else 1
                            return -1j * dt * (shifted @ v)
                        def adjoint(v):
                            matvecs[0] += v.shape[1] if v.ndim == 2 else 1
                            return 1j * dt * (shifted @ v)
                        operator = LinearOperator((n, n), matvec=action, rmatvec=adjoint, matmat=action, rmatmat=adjoint, dtype=complex)
                        psi = expm_multiply(operator, psi, traceA=-1j * dt * np.trace(shifted))
                rows.append(dict(method=method, shift=shift, steps=steps, dt=dt,
                                 density_error=relative(abs(psi)**2, abs(reference)**2),
                                 phase_aligned_error=phase_error(psi, reference),
                                 mass_error=float(abs(np.vdot(psi, psi).real - 1)),
                                 seconds=perf_counter()-start, full_h_matvecs=matvecs[0]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, shift in zip(axes, (0., 500.)):
        for method in ("strang", "yoshida4", "crank_nicolson", "gauge_cn"):
            data = [r for r in rows if r["method"] == method and r["shift"] == shift]
            ax.loglog([r["steps"] for r in data], [r["density_error"] for r in data], "o-", label=method)
        ax.set_title(f"Common energy offset {shift:g}"); ax.set_xlabel("Steps to fixed final time"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Relative density error"); axes[0].legend(fontsize=8)
    savefig("linear_time.png")
    return rows


def sp_initial(x, length):
    """Smooth localized counterstreaming packets; no cosmological expansion."""
    carrier = 2 * np.pi * 8 / length
    psi = np.exp(-(x+2)**2 / (2*.9**2)) * np.exp(1j*carrier*x)
    psi += .8 * np.exp(-(x-2)**2 / (2*.9**2)) * np.exp(-1j*carrier*x + .3j)
    return psi.astype(complex)


def sp_evolve(n, dt, final, gravity="spectral", method="strang", length=20.):
    x = np.arange(n) * length/n - length/2
    psi = sp_initial(x, length)
    # Identical continuum normalization, rather than renormalizing each coarse grid.
    xf = np.arange(4096) * length/4096 - length/2
    norm = np.sqrt(np.sum(abs(sp_initial(xf, length))**2) * length/4096 / 4)
    psi /= norm
    initial_mass = np.sum(abs(psi)**2) * length/n
    k = modes(n, length)
    steps = int(round(final/dt)); dt = final/steps
    a = 1 / (2 - 2**(1/3))
    factors = (1.,) if method == "strang" else (a, 1-2*a, a)
    multipliers = [(dt*f, np.exp(-.5j * dt*f * k*k)) for f in factors]
    invlap = inverse_laplacian(n, length, gravity)
    def phi(state):
        return np.fft.ifft(np.fft.fft(abs(state)**2)*invlap).real
    def energy(state):
        kinetic = np.sum(abs(np.fft.fft(state))**2*k*k/2) / n * length/n
        return float(kinetic + .5*np.sum(abs(state)**2*phi(state))*length/n)
    initial_energy = energy(psi)
    start = perf_counter()
    for _ in range(steps):
        for h, drift in multipliers:
            psi *= np.exp(-.5j*h*phi(psi))
            psi = np.fft.ifft(drift*np.fft.fft(psi))
            psi *= np.exp(-.5j*h*phi(psi))
    return psi, dict(seconds=perf_counter()-start, steps=steps,
                     poisson_calls=2*len(factors)*steps,
                     mass_error=float(abs(np.sum(abs(psi)**2)*length/n-initial_mass)/initial_mass),
                     relative_energy_drift=float(abs(energy(psi)-initial_energy)/max(abs(initial_energy),1e-15)))


def sp_coupling():
    final = 2.
    reference, _ = sp_evolve(1024, .0005, final, method="yoshida4")
    check, _ = sp_evolve(1536, .00025, final, method="yoshida4")
    consistency = relative(resample(abs(reference)**2, len(check)), abs(check)**2)
    rows = []
    for n in (48, 64, 80, 96, 128, 192, 256):
        for kind in ("spectral", "fd2", "fd4", "fd6"):
            psi, stats = sp_evolve(n, .002, final, gravity=kind, method="yoshida4")
            rows.append(dict(n=n, gravity=kind, method="yoshida4", dt=.002,
                             density_error=relative(resample(abs(psi)**2, len(reference)), abs(reference)**2),
                             wave_error=phase_error(resample(psi,len(reference)),reference), **stats))
    times = []
    for method in ("strang", "yoshida4"):
        for dt in (.2, .1, .05, .025, .0125, .00625):
            psi, stats = sp_evolve(256, dt, final, method=method)
            times.append(dict(method=method, dt=dt,
                              density_error=relative(resample(abs(psi)**2,len(reference)),abs(reference)**2), **stats))
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    for kind in ("spectral", "fd2", "fd4", "fd6"):
        data = [r for r in rows if r["gravity"] == kind]
        axes[0].loglog([r["n"] for r in data], [r["density_error"] for r in data], "o-", label=kind)
    for method in ("strang", "yoshida4"):
        data = [r for r in times if r["method"] == method]
        axes[1].loglog([r["poisson_calls"] for r in data], [r["density_error"] for r in data], "o-", label=method)
    axes[0].set_xlabel("Wave grid points"); axes[1].set_xlabel("Poisson solves")
    axes[0].set_ylabel("Relative density error at T=2")
    for ax in axes: ax.grid(alpha=.25); ax.legend()
    savefig("sp_coupling.png")
    np.savez_compressed(OUT/'sp_reference.npz', x=np.arange(1024)*20/1024-10, psi=reference)
    return dict(reference_density_disagreement=consistency, resolution=rows, time=times)


def mapped_operators(n, alpha, length):
    """Unitary coordinate transform u=sqrt(J) psi on an odd periodic q-grid.

    H = -1/(2 sqrt(J)) D 1/J D 1/sqrt(J),
    A_P = -D 1/J D,  A_P phi = -J (rho - physical_mean(rho)).
    Both are Hermitian; odd N avoids a spurious Nyquist nullspace in D D.
    """
    q = np.arange(n)*2*np.pi/n - np.pi
    x = length/(2*np.pi)*(q-alpha*np.sin(q))
    jac = length/(2*np.pi)*(1-alpha*np.cos(q))
    kq = modes(n,2*np.pi)
    derivative = np.fft.ifft((1j*kq)[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0).real
    stiffness = -(derivative * (1/jac)[None,:]) @ derivative
    kinetic = .5 * stiffness / np.sqrt(jac[:,None]*jac[None,:])
    return q, x, jac, kinetic, stiffness


def mapped_spectral():
    """Spatial test: localized linear bound state and manufactured Poisson.

    Dense eigensolves are verification oracles. A 3D implementation must use
    matrix-free tensor contractions and preconditioned elliptic/exponential actions.
    """
    length=20.
    def potential(x): return -25*np.exp(-x*x/(2*.45**2))
    nref=1025
    xr=np.arange(nref)*length/nref-length/2
    kr=modes(nref,length)
    kinetic=np.fft.ifft((kr*kr/2)[:,None]*np.fft.fft(np.eye(nref),axis=0),axis=0).real
    er, vr=eigh(kinetic+np.diag(potential(xr)),subset_by_index=(0,0))
    pr=vr[:,0]/np.sqrt(length/nref)
    rows=[]
    for alpha in (0., .8):
        for n in (33,49,65,97,129,193):
            q,x,jac,h,a=mapped_operators(n,alpha,length)
            values,vectors=eigh(h+np.diag(potential(x)),subset_by_index=(0,0))
            state=vectors[:,0]/np.sqrt(2*np.pi/n)
            # Evaluate u(q) on independent physical x nodes by inverting the map.
            qr=np.linspace(-np.pi,np.pi,nref,endpoint=False)
            for _ in range(16):
                qr -= (length/(2*np.pi)*(qr-alpha*np.sin(qr))-xr)/(length/(2*np.pi)*(1-alpha*np.cos(qr)))
            coeff=np.fft.fft(state)/n
            uf=np.exp(1j*np.outer(qr+np.pi,modes(n,2*np.pi)))@coeff
            pf=uf/np.sqrt(length/(2*np.pi)*(1-alpha*np.cos(qr)))
            # Manufactured phi in physical space: cosine, source phi_xx.
            phi=np.cos(2*np.pi*x/length)
            source=-(2*np.pi/length)**2*phi
            source -= np.sum(jac*source)/np.sum(jac)
            ap_e,ap_v=eigh(a)
            rhs=-jac*source
            pp=ap_v[:,1:]@((ap_v[:,1:].T@rhs)/ap_e[1:])
            pp-=np.sum(jac*pp)/np.sum(jac)
            exact=phi-np.sum(jac*phi)/np.sum(jac)
            rows.append(dict(n=n,alpha=alpha,min_dx=float(np.min(np.diff(x))),max_dx=float(np.max(np.diff(x))),
                             ground_energy_error=float(abs(values[0]-er[0])/abs(er[0])),
                             density_error=relative(abs(pf)**2,abs(pr)**2),
                             mass_error=float(abs(np.sum(jac*abs(state/np.sqrt(jac))**2)*2*np.pi/n-1)),
                             hermitian_defect=relative(h,h.T),
                             poisson_error=relative(np.sqrt(jac)*pp,np.sqrt(jac)*exact),
                             max_kinetic_eigenvalue=float(np.linalg.eigvalsh(h)[-1])))
    plt.figure(figsize=(7,4))
    for alpha in (0.,.8):
        data=[r for r in rows if r['alpha']==alpha]
        plt.semilogy([r['n'] for r in data],[r['density_error'] for r in data],"o-",label=f"mapping alpha={alpha}")
    plt.xlabel("Total grid points"); plt.ylabel("Bound-state density error on independent physical grid")
    plt.legend(); plt.grid(alpha=.25); savefig("mapped_spectral.png")
    return dict(reference_n=nref, reference_energy=float(er[0]), rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',nargs='+',choices=['gravity','restriction','amr','nufft','linear','sp','mapped'])
    args=parser.parse_args()
    jobs={'gravity':gravity_transfer,'restriction':restriction_cancellation,'amr':inherited_boundary,'nufft':nufft_diagnostics,
          'linear':linear_time,'sp':sp_coupling,'mapped':mapped_spectral}
    for name,job in jobs.items():
        if args.only and name not in args.only: continue
        start=perf_counter(); result=job()
        payload=dict(experiment=name, python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                     elapsed_seconds=perf_counter()-start,results=result)
        (OUT/f'{name}.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f'{name}: {payload["elapsed_seconds"]:.2f}s -> {OUT/name}.json',flush=True)


if __name__=='__main__': main()
