#!/bin/bash
# Build GAMER for the ELBDM/Perturbation test (non-comoving): hybrid (GPU, GRAMFE) and base-level-spectral wave binaries.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libfftw3-dev libfftw3-mpi-dev libhdf5-dev libgsl-dev openmpi-bin libopenmpi-dev
pip install -q yt h5py
[ -d /workspace/gamer/.git ] || git clone https://github.com/KunkelAlexander/gamer.git /workspace/gamer
cd /workspace/gamer
git fetch origin fd4-poisson && git checkout fd4-poisson && git merge --ff-only origin/fd4-poisson
echo "GAMER source: $(git log -1 --format='%h %s')"
D=/workspace/deps; mkdir -p $D/hdf5 $D/mpi
ln -sfn /usr/include/hdf5/serial $D/hdf5/include
ln -sfn /usr/lib/x86_64-linux-gnu/hdf5/serial $D/hdf5/lib
ln -sfn /usr/lib/x86_64-linux-gnu/openmpi/include $D/mpi/include
ln -sfn /usr/lib/x86_64-linux-gnu/openmpi/lib $D/mpi/lib
ln -sfn /usr/bin $D/mpi/bin
cat > configs/runpod_4090.config <<'EOT'
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
EOT
cd src
python3 configure.py --machine=runpod_4090 --mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --model=ELBDM \
  --elbdm_scheme=HYBRID --wave_scheme=GRAMFE --gramfe_scheme=MATMUL --gravity=true --comoving=false --gsl=true --spectral_interpolation=true
make clean >/dev/null 2>&1 || true
make -j"$(nproc)"
cp ../bin/gamer /workspace/gamer_pert_hybrid_bin
python3 configure.py --machine=runpod_4090 --mpi=true --hdf5=true --fftw=FFTW3 --model=ELBDM --elbdm_scheme=WAVE --gravity=true --comoving=false
make clean >/dev/null 2>&1 || true
make -j"$(nproc)"
cp ../bin/gamer /workspace/gamer_pert_wave_bin
echo BUILD_DONE
