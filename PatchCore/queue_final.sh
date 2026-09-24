#!/bin/bash
# Step 7 (plan Section 8): final config x 3 seeds, scored on val AND test, evaluated once each.
# Final config chosen on val in step 6: wideresnet50, layer2+layer3, dim 1024, coreset 0.01, K=8.
# Starts after the E1 retry queue; on completion releases the SVM/EfficientAD queue.
cd "$(dirname "$0")/.."
until grep -q RETRY_DONE ~/DR_12_logs/queue_e1_retry.log; do sleep 60; done
for seed in 42 43 44; do
  bash PatchCore/run_two_bank.sh final_k8_s$seed --k 8 --seed $seed || echo "FAILED final_k8_s$seed"
done
source ~/anaconda3/etc/profile.d/conda.sh && conda activate danenv
for seed in 42 43 44; do
  python Comparison/evaluate_two_model.py --run Results/patchcore/final_k8_s$seed --title "PatchCore (s$seed)" \
    > Results/patchcore/final_k8_s$seed/logs/evaluate_common.log 2>&1 || echo "FAILED common eval s$seed"
done
echo FINAL_DONE
touch ~/DR_12_logs/PATCHCORE_FINAL_DONE
