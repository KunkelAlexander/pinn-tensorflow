#!/bin/bash
# Build GAMER (ELBDM, GPU) on a Runpod RTX 4090 pod.
# Usage (inside the pod):  bash setup_gamer.sh [wave|hybrid|gravity|gravity_cpu]
#   wave   (default)  ELBDM wave scheme + self-gravity, no comoving.
#                     For Soliton, IsolatedHalo (isolated BC, non-cosmological problems).
#   gravity           HYDRO + gravity + GPU (no MPI, no ELBDM): the Hydro/Gravity test problem, which measures the
#                     Poisson-solver error against analytic NFW/Hernquist potentials (POT_STENCIL_ORDER 2 vs 4).
#   gravity_cpu       the same without GPU (OpenMP only): reference implementation to cross-check the GPU Poisson solver;
#                     run with GAMER_POT_DEBUG=1 to print the L2/L4 residuals of every Poisson call.
#   hybrid            ELBDM hybrid scheme, GramFE wave solver, comoving, GSL, spectral interpolation.
#                     For LSS_Hybrid / LSS_Hybrid_Zoomin (cosmological light/heavy halo).
# The binary is /workspace/gamer/bin/gamer and is copied to /workspace/gamer_<flavor>_bin, because
# building one flavor overwrites bin/gamer (and `make clean` removes the objects of the other).
# Everything persistent lives under /workspace; the apt packages are lost on a pod restart.
set -euo pipefail
FLAVOR="${1:-wave}"

# 1. System libraries (safe to re-run)
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  libfftw3-dev libfftw3-mpi-dev libhdf5-dev libgsl-dev openmpi-bin libopenmpi-dev
pip install yt h5py          # only for the yt analysis scripts of the test problems

# 2. Source: the fork with the 4th-order Poisson solver (POT_STENCIL_ORDER), branch fd4-poisson, based on upstream
#    e36797538. Override with GAMER_REPO / GAMER_BRANCH, e.g. GAMER_REPO=https://github.com/gamer-project/gamer.git
#    GAMER_BRANCH=main for plain upstream. An existing clone is switched to the fork branch and fast-forwarded.
GAMER_REPO="${GAMER_REPO:-https://github.com/KunkelAlexander/gamer.git}"
GAMER_BRANCH="${GAMER_BRANCH:-fd4-poisson}"
[ -d /workspace/gamer/.git ] || git clone "$GAMER_REPO" /workspace/gamer
git -C /workspace/gamer remote set-url origin "$GAMER_REPO"
git -C /workspace/gamer fetch origin "$GAMER_BRANCH"
git -C /workspace/gamer checkout "$GAMER_BRANCH"
git -C /workspace/gamer merge --ff-only "origin/$GAMER_BRANCH"
echo "GAMER source: $(git -C /workspace/gamer log -1 --format='%h %s')"

# 3. Debian puts HDF5/MPI in non-standard places; build prefixes GAMER's configure can use
D=/workspace/deps
mkdir -p $D/hdf5 $D/mpi
ln -sfn /usr/include/hdf5/serial               $D/hdf5/include
ln -sfn /usr/lib/x86_64-linux-gnu/hdf5/serial  $D/hdf5/lib
ln -sfn /usr/lib/x86_64-linux-gnu/openmpi/include $D/mpi/include
ln -sfn /usr/lib/x86_64-linux-gnu/openmpi/lib     $D/mpi/lib
ln -sfn /usr/bin                               $D/mpi/bin

# 4. Machine config: GCC + CUDA 12.8, compute capability 8.9 (RTX 4090).
#    -std=gnu++14 (NOT c++14): the GramFE tables use GNU `q` (__float128) literals, which strict c++14 rejects.
cat > /workspace/gamer/configs/runpod_4090.config <<'EOF'
CUDA_PATH       /usr/local/cuda
FFTW2_PATH      /usr
FFTW3_PATH      /usr
MPI_PATH        /workspace/deps/mpi
HDF5_PATH       /workspace/deps/hdf5
GRACKLE_PATH    /usr
GSL_PATH        /usr
LIBYT_PATH      /usr
CUFFTDX_PATH    /usr
CXX     g++
CXX_MPI mpicxx
CXXFLAG -g
CXXFLAG -O3
CXXFLAG -std=gnu++14
CXXFLAG -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -Wno-maybe-uninitialized -Wno-unused-but-set-variable -Wno-unknown-pragmas -Wno-unused-result
OPENMPFLAG -fopenmp
LIBFLAG
NVCCFLAG_COM -O3
NVCCFLAG_FLU -Xptxas -dlcm=ca -prec-div=false -ftz=true
NVCCFLAG_POT -Xptxas -dlcm=ca
GPU_COMPUTE_CAPABILITY 890
EOF

# 5. Configure and build. Options are the ones in example/test_problem/ELBDM/<problem>/generate_make.sh.
cd /workspace/gamer/src
COMMON="--machine=runpod_4090 --mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --model=ELBDM --gravity=true"
case "$FLAVOR" in
  wave)   python3 configure.py $COMMON ;;
  hybrid) python3 configure.py $COMMON --elbdm_scheme=HYBRID --wave_scheme=GRAMFE --gramfe_scheme=MATMUL \
                                       --comoving=true --gsl=true --spectral_interpolation=true ;;
  gravity) python3 configure.py --machine=runpod_4090 --gpu=true --hdf5=true --fftw=FFTW3 --model=HYDRO --gravity=true ;;
  gravity_cpu) python3 configure.py --machine=runpod_4090 --gpu=false --hdf5=true --fftw=FFTW3 --model=HYDRO --gravity=true --openmp=true ;;
  *) echo "unknown flavor '$FLAVOR' (use wave, hybrid, gravity or gravity_cpu)"; exit 1 ;;
esac
make clean >/dev/null 2>&1 || true
make -j"$(nproc)"
cp /workspace/gamer/bin/gamer "/workspace/gamer_${FLAVOR}_bin"
echo "Built: /workspace/gamer_${FLAVOR}_bin"
