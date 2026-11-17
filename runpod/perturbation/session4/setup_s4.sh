#!/bin/bash
# Session 4 one-time pod setup: deps, fork build (two binaries), spectral tables, base input dir, spectrum script.
# Run with: nohup bash setup_s4.sh > /workspace/setup_s4.log 2>&1 &     (about 15 min)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libfftw3-dev libfftw3-mpi-dev libhdf5-dev libgsl-dev openmpi-bin libopenmpi-dev unzip curl
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

# spectrum script (np.asarray fix for the unyt dtype error) and base input directory
sed 's/data\[field\]/np.asarray(data[field])/g' example/yt/extract_power_spectrum.py > /workspace/extract_ps_np.py
mkdir -p /workspace/base_input /workspace/runs && cd /workspace/base_input
cp /workspace/gamer/example/test_problem/ELBDM/Perturbation/Input__* .
mkdir -p tbl && cd tbl && bash /workspace/gamer/example/test_problem/ELBDM/LSS_Hybrid/download_spectral_interpolation_tables.sh > /dev/null 2>&1 || true
cd /workspace/base_input
for t in interpolation_tables boundary2extension_tables; do p=$(find tbl -maxdepth 3 -type d -name $t | head -1); [ -n "$p" ] && cp -r "$p" . ; done
ls /workspace/base_input

# best-accuracy configuration of session 3 (s4f2cq): FD4 SOR, FFT symbol FD2 at the finest spacing, CQuad ghost zones, spectral refinement
setp(){ k=$1; v=$2; if grep -q "^$k[[:space:]]" Input__Parameter; then sed -i -E "s/^$k[[:space:]]+[^[:space:]]+/$k $v/" Input__Parameter; else echo "$k $v" >> Input__Parameter; fi; }
setp END_T 2.0; setp OUTPUT_DT 1.0; setp OPT__OUTPUT_POT 0
setp POT_FFT_SYMBOL 2; setp POT_STENCIL_ORDER 4
setp OPT__FLU_INT_SCHEME 4; setp OPT__REF_FLU_INT_SCHEME 8
grep -E '^(END_T|OUTPUT_DT|OPT__OUTPUT_POT|POT_FFT_SYMBOL|POT_STENCIL_ORDER|OPT__FLU_INT_SCHEME|OPT__REF_FLU_INT_SCHEME|ELBDM_FIRST_WAVE_LEVEL|MAX_LEVEL|NX0_TOT_X) ' Input__Parameter | cut -c1-50

build(){ name=$1; shift
  cd /workspace/gamer/src
  python3 configure.py --machine=runpod_4090 --mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --model=ELBDM --elbdm_scheme=HYBRID \
     --wave_scheme=GRAMFE --gramfe_scheme=MATMUL --gravity=true --comoving=false --gsl=true --spectral_interpolation=true "$@"
  make clean >/dev/null 2>&1 || true
  make -j"$(nproc)"
  cp ../bin/gamer /workspace/bin_$name
  echo "BUILT $name"
}
build base
build third_qp4 --hybrid_scheme=THIRD --hybrid_qp_order=4
echo SETUP_DONE
