#!/bin/bash
# Fix: SVM + EfficientAD at K=8 train windows/file, matching the final PatchCore config (final_k8_s*).
# The earlier K=4 runs (ocsvm_v2_s*, effad_v2_s*) are kept for reference but are NOT the comparison.
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate danenv

for seed in 42 43 44; do
  R=Results/efficientad/effad_v2_k8_s$seed
  CUDA_VISIBLE_DEVICES=0 python EfficientAD/efficientad_v2.py --cls pd    --out $R --seed $seed --k 8 > ~/DR_12_logs/effad_v2_k8_s${seed}_pd.log 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 python EfficientAD/efficientad_v2.py --cls noise --out $R --seed $seed --k 8 > ~/DR_12_logs/effad_v2_k8_s${seed}_noise.log 2>&1 &
  p2=$!
  wait $p1 && wait $p2 \
    && python Comparison/evaluate_two_model.py --run $R --title "EfficientAD K=8 (s$seed)" > ~/DR_12_logs/effad_v2_k8_s${seed}_eval.log 2>&1 \
    || echo "FAILED effad s$seed"
done

for seed in 42 43 44; do
  R=Results/svm/ocsvm_v2_k8_s$seed
  python SVM/one_class/ocsvm_v2.py --out $R --seed $seed --k 8 > ~/DR_12_logs/ocsvm_v2_k8_s$seed.log 2>&1 \
    && python Comparison/evaluate_two_model.py --run $R --title "One-Class SVM K=8 (s$seed)" >> ~/DR_12_logs/ocsvm_v2_k8_s$seed.log 2>&1 \
    || echo "FAILED svm s$seed"
done
echo K8_DONE
