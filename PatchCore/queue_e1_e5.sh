#!/bin/bash
# Model-selection queue (Plans/PatchCore_training_plan.md, Section 5). One change at a time from the
# start config (wideresnet50, layer2+layer3, dim 1024, coreset 0.01, K=4, seed 42). Val only.
# E5 (window aggregation) needs no extra run: every run reports all aggregations on val.
cd "$(dirname "$0")/.."
export VAL_ONLY=1
run() { bash PatchCore/run_two_bank.sh "$@" || echo "FAILED $1"; }
run e1_l12      --layers layer1 layer2
run e1_l123     --layers layer1 layer2 layer3
run e2_resnet50 --backbone resnet50
run e3_cs0001   --coreset 0.001
run e4_k2       --k 2
SEQUENTIAL=1 run e4_k8 --k 8
run e3_cs005    --coreset 0.05
echo QUEUE_DONE
