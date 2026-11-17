#!/bin/bash
# Second batch (reordered). Only res3 and res5 finished before the pod was shut down; flagfine reached t = 1.3 and snap025 t = 1.04.
# Needs /workspace/bin_base2 (= bin_base plus OPT__RESTRICT_ORDER 3/5, fork commit bb90fd39e).
cd /workspace
while [ ! -f /workspace/queue1.done ]; do sleep 20; done
while ! grep -q 'BUILT base2' /workspace/build_base2.log; do sleep 20; done
( ./run_variant.sh res3 base2 8 'OPT__RESTRICT_ORDER 3' ;  EXTRA_CMD="sed -i 's/ 0.03 / 0.01 /' Input__Flag_Interference; sed -i 's/ 1.0 / 0.3 /' Input__Flag_Spectral" ./run_variant.sh flagfine base2 8 ;  ./run_variant.sh resphase base2 8 'OPT__RES_PHASE 1' ) &
( ./run_variant.sh res5 base2 8 'OPT__RESTRICT_ORDER 5' ;  ./run_variant.sh snap025 base2 8 'OUTPUT_DT 0.25' ;  ./run_variant.sh dt05 base2 8 'DT__HYBRID_CFL 0.1' 'DT__HYBRID_VELOCITY 0.5' ) &
wait
echo QUEUE2_DONE > /workspace/queue2.done
