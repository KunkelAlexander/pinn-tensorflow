#!/bin/bash
# First test batch, two runs at a time with 8 threads each. Tags: s4_rep (noise floor), res4 (4th-order restriction),
# third_qp4 (3rd-order flux + 4th-order QP), res4_third_qp4 (both)
cd /workspace
( ./run_variant.sh s4_rep         base       8 ;  ./run_variant.sh third_qp4      third_qp4 8 ) &
( ./run_variant.sh res4           base       8 'OPT__RESTRICT_ORDER 4' ;  ./run_variant.sh res4_third_qp4 third_qp4 8 'OPT__RESTRICT_ORDER 4' ) &
wait
echo QUEUE1_DONE > /workspace/queue1.done
