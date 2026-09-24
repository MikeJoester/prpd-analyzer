#!/bin/bash
# E1 retry after the first attempt was OOM-killed (layer1 => 56x56 patch grid, 4x the patches).
# K=1 keeps the patch budget equal to the start config (1 x 3136 ~= 4 x 784), so the feature type
# is compared at equal memory/coreset cost. Waits for the main queue to finish first.
cd "$(dirname "$0")/.."
until grep -q QUEUE_DONE ~/DR_12_logs/queue_e1_e5.log; do sleep 60; done
export VAL_ONLY=1
run() { bash PatchCore/run_two_bank.sh "$@" || echo "FAILED $1"; }
run e1_l12_k1  --layers layer1 layer2 --k 1
run e1_l123_k1 --layers layer1 layer2 layer3 --k 1
run e1_l23_k1  --layers layer2 layer3 --k 1   # control: start config at K=1, isolates the layer effect
echo RETRY_DONE
