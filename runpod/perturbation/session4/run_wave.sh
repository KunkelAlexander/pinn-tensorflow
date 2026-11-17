#!/bin/bash
# usage: run_wave.sh TAG NTHREAD ['KEY VALUE' ...]
# Uniform 512x512x16 GramFE wave run (no AMR, no fluid), same setup as the spectral reference but with the local spectral GramFE solver.
# Needs /workspace/bin_wave_gramfe, built with:
#   python3 configure.py --machine=runpod_4090 --mpi=true --hdf5=true --fftw=FFTW3 --gpu=true --model=ELBDM --elbdm_scheme=WAVE \
#      --wave_scheme=GRAMFE --gramfe_scheme=MATMUL --gravity=true --comoving=false --gsl=true --spectral_interpolation=true
TAG=$1; NT=$2; shift 2
D=/workspace/runs/$TAG; rm -rf $D; mkdir -p $D
cp -r /workspace/base_input/Input__* /workspace/base_input/interpolation_tables /workspace/base_input/boundary2extension_tables $D/
cd $D
cp Input__Parameter_BaseSpectral Input__Parameter
setp(){ k=$1; v=$2; if grep -q "^$k[[:space:]]" Input__Parameter; then sed -i -E "s/^$k[[:space:]]+[^[:space:]]+/$k $v/" Input__Parameter; else echo "$k $v" >> Input__Parameter; fi; }
setp OMP_NTHREAD $NT; setp END_T 2.0; setp OUTPUT_DT 1.0; setp OPT__OUTPUT_POT 0; setp ELBDM_BASE_SPECTRAL 0; setp MAX_LEVEL 0
for kv in "$@"; do setp ${kv% *} ${kv#* }; done
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -np 1 --bind-to none /workspace/bin_wave_gramfe > log 2>&1
python3 /workspace/extract_ps_np.py -s 2 -e 2 -lv 0 -L 1 -ndim 2 > ps.log 2>&1
f=$(ls power_spectrum_*_000002_lv_00_*ndim_2.txt | head -1)
tail -256 $f | paste -sd, > spec.csv
echo "FINISHED $(grep 'Total Simulation Time' Record__Timing | head -1)" >> log
