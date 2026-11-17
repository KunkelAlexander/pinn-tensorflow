#!/bin/bash
# usage: run_variant.sh TAG BINARY NTHREAD ['KEY VALUE' ...]   -> /workspace/runs/TAG/{log,Record__Timing,spec.csv}
TAG=$1; BIN=$2; NT=$3; shift 3
D=/workspace/runs/$TAG; rm -rf $D; mkdir -p $D
cp -r /workspace/base_input/Input__* /workspace/base_input/interpolation_tables /workspace/base_input/boundary2extension_tables $D/
cd $D
setp(){ k=$1; v=$2; if grep -q "^$k[[:space:]]" Input__Parameter; then sed -i -E "s/^$k[[:space:]]+[^[:space:]]+/$k $v/" Input__Parameter; else echo "$k $v" >> Input__Parameter; fi; }
setp OMP_NTHREAD $NT
for kv in "$@"; do setp ${kv% *} ${kv#* }; done
[ -n "$EXTRA_CMD" ] && eval "$EXTRA_CMD"   # optional extra shell command run in the run directory (e.g. sed on Input__Flag_*)
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -np 1 --bind-to none /workspace/bin_$BIN > log 2>&1
python3 /workspace/extract_ps_np.py -s 2 -e 2 -lv 3 -L 1 -ndim 2 > ps.log 2>&1
f=$(ls power_spectrum_*_000002_lv_03_*ndim_2.txt | head -1)
tail -256 $f | paste -sd, > spec.csv
echo "FINISHED $(grep 'Total Simulation Time' Record__Timing | head -1)" >> log
