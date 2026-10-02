#!/bin/bash
# Fine-tuning step 1: --num-nn 3 / 5 / 9 (plan Section 4). No bank rebuild: the seed-42/43/44 banks
# from final_k8_s* are reused and only the scoring changes. Val only.
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate danenv
export PYTHONPATH="$PWD/PatchCore/patchcore-inspection/src:${PYTHONPATH:-}"
DATA=PatchCore/patchcore_data/prpd_v2
for k in 3 5 9; do
  for seed in 42 43 44; do
    RUN=Results/patchcore/nn${k}_s${seed}
    [[ -e "$RUN" ]] && { echo "skip existing $RUN"; continue; }
    mkdir -p "$RUN/logs"
    echo "num-nn $k seed $seed (banks reused from final_k8_s$seed)" > "$RUN/command.txt"
    CUDA_VISIBLE_DEVICES=0 python PatchCore/score_two_bank.py --run "$RUN" --bank-run Results/patchcore/final_k8_s$seed \
      --cls pd --num-nn $k --data-root $DATA --splits val > "$RUN/logs/score_pd.log" 2>&1 &
    p1=$!
    CUDA_VISIBLE_DEVICES=1 python PatchCore/score_two_bank.py --run "$RUN" --bank-run Results/patchcore/final_k8_s$seed \
      --cls noise --num-nn $k --data-root $DATA --splits val > "$RUN/logs/score_noise.log" 2>&1 &
    p2=$!
    wait $p1 && wait $p2 && python PatchCore/evaluate_two_bank.py --run "$RUN" --val-only \
      > "$RUN/logs/evaluate.log" 2>&1 || echo "FAILED $RUN"
    echo "done $RUN"
  done
done
echo NUMNN_DONE
