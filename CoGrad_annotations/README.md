# SliCK-Math: A Workflow for Fine-Tuning LLMs on Mathematical Reasoning

This repository contains a complete workflow for fine-tuning and evaluating Large Language Models (LLMs) on mathematical reasoning tasks, based on the principles of **SliCK (Selective Corpus-Level Knowledge)**.

The core idea is to intelligently classify the model's existing knowledge on a given dataset, and then use this information to construct more effective training curricula. This project provides scripts for data annotation, data augmentation, dataset construction, model fine-tuning, and evaluation.

## Workflow Overview

The workflow is divided into 6 sequential stages, executed by the scripts in the `scripts/` directory:

1.  **`1_CoGrad_annotate.py`**: **SliCK Annotation**
    -   Uses a base LLM to annotate a raw dataset (e.g., PRM800K, MATH).
    -   Classifies each problem into one of four knowledge categories: `HighlyKnown`, `MaybeKnown`, `WeaklyKnown`, or `Unknown`.

2.  **`2_augment_data_contrastive.py`**: **Data Augmentation**
    -   Applies Rejection-Free Tuning (RFT) to generate high-quality solutions for problems the model is unsure about.
    -   Creates "contrastive pairs" (incorrect vs. correct solutions) to teach the model to recognize and correct its own mistakes.

3.  **`3_build_datasets_mixed.py`**: **Dataset Construction**
    -   Builds various training datasets by mixing knowledge categories in different ratios.
    -   This allows for controlled experiments, such as studying the effect of injecting a certain percentage of "unknown" knowledge.

4.  **`4_finetune_twophase.py`**: **Two-Phase Fine-tuning**
    -   Implements a robust curriculum learning strategy.
    -   **Phase 1**: Consolidates the model's base knowledge on "easy" data (Anchors).
    -   **Phase 2**: Teaches the model new knowledge using "harder" data (Frontiers) while replaying some easy data to prevent catastrophic forgetting.
    -   This script is optimized for large models (e.g., 13B) and uses memory-efficient techniques like in-memory handoff and `safetensors`.

5.  **`5_evaluate_checkpoints.py`**: **Checkpoint Evaluation**
    -   Automates the evaluation of all saved model checkpoints.
    -   Provides a detailed breakdown of accuracy across different SliCK knowledge categories.

6.  **`6_analyze_regressions.py`**: **Qualitative Analysis**
    -   Performs a "regression test" by comparing the fine-tuned model against the base model.
    -   Identifies and reports "Right -> Wrong" flips, where the model's performance degraded after fine-tuning.

## Getting Started

### 1. Setup

**Directory Structure:**

```
.
├── data/                 # Raw and processed data (e.g., train.jsonl, test.jsonl)
├── results/              # Model checkpoints and evaluation results
├── scripts/              # All Python scripts for the workflow
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

**Installation:**

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

### 2. Configuration

Before running the workflow, you need to adjust the paths and parameters inside the scripts in the `scripts/` directory. Key variables to modify include:

-   `MODEL_PATH`: Path to the base LLM you want to use.
-   `BASE_PATH`: The root directory for your project's data and results.
-   `INPUT_FILE` / `TEST_FILE`: Paths to your raw dataset files.

### 3. Running the Workflow

You can run the entire workflow sequentially using the provided shell script:

```bash
bash run_workflow.sh
```

Alternatively, you can run each script individually in the order specified by their numbered prefixes.

## Citation

If you use this workflow in your research, please consider citing the original papers and ideas that inspired it.


