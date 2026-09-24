#!/bin/bash
# SVM + EfficientAD on the PatchCore v2 data, after PatchCore is finished.
# Waits for ~/DR_12_logs/PATCHCORE_FINAL_DONE (written when PatchCore step 7 completes).
# Same data/split/K=4 windows as PatchCore, two models per method, seeds 42/43/44.
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate danenv
until [[ -f ~/DR_12_logs/PATCHCORE_FINAL_DONE ]]; do sleep 60; done
for seed in 42 43 44; do
  R=Results/svm/ocsvm_v2_s$seed
  python SVM/one_class/ocsvm_v2.py --out $R --seed $seed > ~/DR_12_logs/ocsvm_v2_s$seed.log 2>&1 \
    && python Comparison/evaluate_two_model.py --run $R --title "One-Class SVM (s$seed)" >> ~/DR_12_logs/ocsvm_v2_s$seed.log 2>&1 \
    || echo "FAILED svm s$seed"
done
for seed in 42 43 44; do
  R=Results/efficientad/effad_v2_s$seed
  CUDA_VISIBLE_DEVICES=0 python EfficientAD/efficientad_v2.py --cls pd    --out $R --seed $seed > ~/DR_12_logs/effad_v2_s${seed}_pd.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python EfficientAD/efficientad_v2.py --cls noise --out $R --seed $seed > ~/DR_12_logs/effad_v2_s${seed}_noise.log 2>&1 &
  wait
  python Comparison/evaluate_two_model.py --run $R --title "EfficientAD (s$seed)" > ~/DR_12_logs/effad_v2_s${seed}_eval.log 2>&1 || echo "FAILED effad s$seed"
done
echo SVM_EFFAD_DONE
