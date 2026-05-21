#!/bin/bash
# Run complete Feature 3 pipeline
# Usage: bash feature3/scripts/run_all.sh checkpoints/best_model.pt 0 cpu

CHECKPOINT=${1:-"checkpoints/best_model.pt"}
TASK_IDX=${2:-0}
DEVICE=${3:-"cpu"}
RESULTS="results/feature3"

echo "====================================="
echo "Feature 3: Complete Pipeline"
echo "Checkpoint: $CHECKPOINT"
echo "Task Index: $TASK_IDX"
echo "Device:     $DEVICE"
echo "====================================="

echo ""
echo "[Phase 1] Generating Explanations..."
python feature3/scripts/phase1_run_explanations.py \
    --checkpoint "$CHECKPOINT" \
    --task_idx $TASK_IDX \
    --epochs 100 \
    --device "$DEVICE" \
    --output_dir "$RESULTS"

echo ""
echo "[Phase 2] Substructure Analysis..."
python feature3/scripts/phase2_substructure_analysis.py \
    --explanation_dir "$RESULTS" \
    --task_idx $TASK_IDX \
    --output_dir "$RESULTS"

echo ""
echo "[Phase 3] Generating Figures..."
python feature3/scripts/phase3_generate_figures.py \
    --checkpoint "$CHECKPOINT" \
    --task_idx $TASK_IDX \
    --results_dir "$RESULTS" \
    --device "$DEVICE"

echo ""
echo "[Phase 4] Tables & Ablation..."
python feature3/scripts/phase4_generate_tables.py \
    --checkpoint "$CHECKPOINT" \
    --task_idx $TASK_IDX \
    --results_dir "$RESULTS" \
    --device "$DEVICE"

echo ""
echo "====================================="
echo "Feature 3 COMPLETE"
echo "Results: $RESULTS/"
echo "Figures: $RESULTS/figures/"
echo "Tables:  $RESULTS/tables/"
echo "====================================="