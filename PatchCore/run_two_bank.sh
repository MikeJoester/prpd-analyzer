#!/bin/bash
# PatchCore v2 two-bank run on the GPU server (Plans/PatchCore_training_plan.md, Section 4.2).
#
#   bash PatchCore/run_two_bank.sh <run_name> [train_two_bank.py args...] [-- score_two_bank.py args...]
#
# Builds the PD bank on GPU 0 and the Noise bank on GPU 1 in parallel, scores val+test the same
# way, then evaluates. Output: Results/patchcore/<run_name>/ (refuses to reuse an existing folder).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
cd "$REPO_ROOT"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate danenv
export PYTHONPATH="$HERE/patchcore-inspection/src:${PYTHONPATH:-}"

NAME="$1"; shift
TRAIN_ARGS=(); SCORE_ARGS=(); target=TRAIN_ARGS
for a in "$@"; do
  if [[ "$a" == "--" ]]; then target=SCORE_ARGS; continue; fi
  eval "$target+=(\"\$a\")"
done

DATA=PatchCore/patchcore_data/prpd_v2
# VAL_ONLY=1 for model-selection runs (plan Section 5): score and report val only; test stays sealed.
EVAL_FLAGS=()
if [[ "${VAL_ONLY:-0}" == "1" ]]; then SCORE_ARGS+=(--splits val); EVAL_FLAGS+=(--val-only); fi
RUN=Results/patchcore/$NAME
[[ -e "$RUN" ]] && { echo "run folder exists: $RUN" >&2; exit 1; }
mkdir -p "$RUN/logs"
echo "VAL_ONLY=${VAL_ONLY:-0} $0 $NAME ${TRAIN_ARGS[*]:-} -- ${SCORE_ARGS[*]:-}" > "$RUN/command.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN/git_head.txt" 2>/dev/null || echo "no git" > "$RUN/git_head.txt"

# SEQUENTIAL=1 builds the banks one after the other (for settings whose feature pool would not fit
# in host RAM twice, e.g. K=8 at dim 1024). Default: both at once, one per GPU.
if [[ "${SEQUENTIAL:-0}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=1 python "$HERE/train_two_bank.py" --cls noise --data-root $DATA --out "$RUN" "${TRAIN_ARGS[@]}" > "$RUN/logs/train_noise.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 python "$HERE/train_two_bank.py" --cls pd    --data-root $DATA --out "$RUN" "${TRAIN_ARGS[@]}" > "$RUN/logs/train_pd.log" 2>&1
else
  pids=()
  CUDA_VISIBLE_DEVICES=0 python "$HERE/train_two_bank.py" --cls pd    --data-root $DATA --out "$RUN" "${TRAIN_ARGS[@]}" > "$RUN/logs/train_pd.log" 2>&1 & pids+=($!)
  CUDA_VISIBLE_DEVICES=1 python "$HERE/train_two_bank.py" --cls noise --data-root $DATA --out "$RUN" "${TRAIN_ARGS[@]}" > "$RUN/logs/train_noise.log" 2>&1 & pids+=($!)
  for p in "${pids[@]}"; do wait "$p"; done
fi

pids=()
CUDA_VISIBLE_DEVICES=0 python "$HERE/score_two_bank.py" --run "$RUN" --cls pd    --data-root $DATA "${SCORE_ARGS[@]}" > "$RUN/logs/score_pd.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=1 python "$HERE/score_two_bank.py" --run "$RUN" --cls noise --data-root $DATA "${SCORE_ARGS[@]}" > "$RUN/logs/score_noise.log" 2>&1 & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

python "$HERE/evaluate_two_bank.py" --run "$RUN" "${EVAL_FLAGS[@]}" 2>&1 | tee "$RUN/logs/evaluate.log"
echo "RUN_DONE $RUN"
