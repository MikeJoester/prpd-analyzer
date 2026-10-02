#!/bin/bash
# Extra seeds for the final PatchCore config (K=8) plus EfficientAD and One-Class SVM, so the paper can
# report mean +/- sd over 10 seeds instead of a range over 3. Config is UNCHANGED -- this is not tuning:
# val and test are both scored, and each run is evaluated once with the standard protocol.
# SEQUENTIAL=1: parallel bank builds were OOM-killed at K=8 (fine-tuning plan, Section 6).
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate danenv
export PYTHONPATH="$PWD/PatchCore/patchcore-inspection/src:${PYTHONPATH:-}"
SEEDS="${SEEDS:-45 46 47 48 49 50 51}"

for seed in $SEEDS; do
  R=Results/patchcore/final_k8_s$seed
  if [[ -e "$R" ]]; then echo "skip existing $R"; else
    SEQUENTIAL=1 bash PatchCore/run_two_bank.sh final_k8_s$seed --k 8 --seed $seed || echo "FAILED patchcore s$seed"
    python Comparison/evaluate_two_model.py --run "$R" --title "PatchCore (s$seed)" \
      > "$R/logs/evaluate_common.log" 2>&1 || echo "FAILED patchcore eval s$seed"
  fi
  echo "PATCHCORE_SEED_DONE $seed"
done

for seed in $SEEDS; do
  R=Results/efficientad/effad_v2_k8_s$seed
  if [[ ! -e "$R" ]]; then
    CUDA_VISIBLE_DEVICES=0 python EfficientAD/efficientad_v2.py --cls pd    --out $R --seed $seed --k 8 > ~/DR_12_logs/effad_s${seed}_pd.log 2>&1 &
    p1=$!
    CUDA_VISIBLE_DEVICES=1 python EfficientAD/efficientad_v2.py --cls noise --out $R --seed $seed --k 8 > ~/DR_12_logs/effad_s${seed}_noise.log 2>&1 &
    p2=$!
    wait $p1 && wait $p2 && python Comparison/evaluate_two_model.py --run $R --title "EfficientAD (s$seed)" \
      > ~/DR_12_logs/effad_s${seed}_eval.log 2>&1 || echo "FAILED effad s$seed"
  fi
  S=Results/svm/ocsvm_v2_k8_s$seed
  if [[ ! -e "$S" ]]; then
    python SVM/one_class/ocsvm_v2.py --out $S --seed $seed --k 8 > ~/DR_12_logs/ocsvm_s$seed.log 2>&1 \
      && python Comparison/evaluate_two_model.py --run $S --title "One-Class SVM (s$seed)" >> ~/DR_12_logs/ocsvm_s$seed.log 2>&1 \
      || echo "FAILED svm s$seed"
  fi
done
echo MORE_SEEDS_DONE
