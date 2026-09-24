#!/bin/bash
# run_prpd_patchcore.sh

# Activate environment
source /home/dannt/miniconda3/etc/profile.d/conda.sh
conda activate base

# Define paths (relative to this script, so it can be run from anywhere)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"
export PYTHONPATH="$HERE/patchcore-inspection/src:$PYTHONPATH"
DATAPATH="$HERE/patchcore_data"
SAVE_PATH="$REPO_ROOT/Results/patchcore/patchcore_results"

# Execute PatchCore
# - We specify WideResNet-50 (-b wideresnet50)
# - We hook ONLY layer1 and layer2 to prevent texture learning (-le layer1 -le layer2)
# - We set imagesize and resize to our 128x256 tensor size
# - We set anomaly_scorer_num_nn to 1 for strict K-NN distance
# - We use a 10% coreset sample for the Memory Bank (-p 0.1)

cd "$REPO_ROOT"  # outputs (e.g. patchcore_raw_scores.npz) are written relative to the repo root
python "$HERE/patchcore-inspection/bin/run_patchcore.py" \
  --gpu 0 \
  --seed 42 \
  --save_patchcore_model \
  --log_group PRPD_PatchCore \
  --log_project PRPD_Results "$SAVE_PATH" \
  patch_core \
    -b resnet50 \
    -le layer1 \
    -le layer2 \
    --pretrain_embed_dimension 256 \
    --target_embed_dimension 256 \
    --anomaly_scorer_num_nn 1 \
    --patchsize 3 \
  sampler \
    -p 0.001 \
    approx_greedy_coreset \
  dataset \
    --batch_size 2 \
    --resize 256 \
    --imagesize 224 \
    -d prpd \
    mvtec "$DATAPATH"
