#!/bin/bash
# Fine-tuning steps 2,3,4,6,7 (plan Section 4) — SCREEN at seed 42 only.
# Candidates that clear the current config on val then get the full 3 seeds (queue_tune_confirm.sh).
# SEQUENTIAL=1 throughout: parallel bank builds were OOM-killed at K=8 (plan Section 6).
cd "$(dirname "$0")/.."
until grep -q NUMNN_DONE ~/DR_12_logs/tune_numnn.log; do sleep 60; done
run() { name=$1; shift; VAL_ONLY=1 SEQUENTIAL=1 bash PatchCore/run_two_bank.sh "$name" "$@" || echo "FAILED $name"; }
run kfield16_s42 --k 8 --k-field 16 --seed 42     # step 2: rebalance the PD bank toward Field
run img128_s42   --k 8 --image-size 128 --seed 42       # step 4: no upsampling, cheap
run l3_s42       --k 8 --layers layer3 --seed 42        # step 7: depth
run l34_s42      --k 8 --layers layer3 layer4 --seed 42
run bb_rn101_s42 --k 8 --backbone resnet101 --seed 42   # step 6: bigger backbone
run k12_s42      --k 12 --seed 42                 # step 3: most expensive, last
echo SCREEN_DONE
