#!/bin/bash

# This script runs the entire SliCK-Math workflow sequentially.
# Make sure to configure the paths and parameters in each script before running.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# You can define common variables here and pass them to the python scripts if needed.
# For now, we assume paths are configured inside each Python script.

echo "================================================="
echo "ðŸš€ Starting SliCK-Math Workflow"
echo "================================================="

# Stage 1: SliCK Annotation
echo "
[1/6] Running SliCK Annotation..."
python scripts/1_CoGrad_annotate.py

# Stage 2: Data Augmentation with Contrastive Learning
echo "
[2/6] Augmenting data with RFT and Contrastive CoT..."
python scripts/2_augment_data_contrastive.py

# Stage 3: Build Mixed Datasets
echo "
[3/6] Building mixed datasets for fine-tuning..."
python scripts/3_build_datasets_mixed.py

# Stage 4: Two-Phase Fine-tuning
# Note: This is the most time-consuming step.
echo "
[4/6] Starting Two-Phase Fine-tuning..."
python scripts/4_finetune_twophase.py --dataset_name D_SliCK_Contrastive_Only # Example dataset

# Stage 5: Evaluate Checkpoints
echo "
[5/6] Evaluating all saved checkpoints..."
python scripts/5_evaluate_checkpoints.py D_SliCK_Contrastive_Only # Must match the dataset name from stage 4

# Stage 6: Qualitative Analysis
echo "
[6/6] Analyzing regressions (Right -> Wrong flips)..."
python scripts/6_analyze_regressions.py


echo "================================================="
echo "âœ?Workflow Finished Successfully!"
echo "================================================="


