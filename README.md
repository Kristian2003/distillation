# Tiny but Mighty: Tuning Alpha and Temperature for Distilled SLMs in Assertion Generation

This repository contains the code and resources for the Bachelor's thesis project: "Tiny but Mighty: Tuning Alpha and Temperature for Improved Performance in Distilled SLMs for Assertion Generation" by Kristian Hristov, supervised by Mitchell Olsthoorn and Annibale Panichella at TU Delft.

## Abstract

Testing of software is crucial to the quality of the final product - manual test assertion creation has become a significant bottleneck in the development process, which delays release. Large language models (LLMs) have shown promise in generating assertions automatically due to their fluency in both natural languages and code, and their speed. However, LLMs face deployment issues like high computation time, latency, or limited functionality in smaller, local counterparts. Knowledge distillation, transferring knowledge from a "teacher" model to a "student," can enable smaller, faster models. This research explores the effectiveness of knowledge distillation for developing a smaller, efficient model for assertion generation. Using CodeT5 as the teacher, a student model learns iteratively. Evaluation metrics include assertion accuracy, similarity to teacher output and ground truth, model size, and inference time. Results indicate the student model achieved about 1/3 the capability of the teacher, suggesting potential for creating efficient, yet reliable assertion generation tools.

## Key Features

*   **Knowledge Distillation:** Implements knowledge distillation from a larger "teacher" model (e.g., `Salesforce/codet5-base`) to a smaller "student" model (e.g., `Salesforce/codet5-small`).
*   **Hyperparameter Tuning:** Focuses on tuning `alpha` (balancing hard and soft losses) and `temperature` (for soft target generation) to optimize student model performance.
*   **Assertion Generation Task:** Tailored for generating test assertions for Java methods.
*   **Comprehensive Evaluation:** Uses metrics such as precision, recall, F1-score, accuracy (exact match), character-level similarity, model size, and inference time.
*   **Dataset Handling:** Custom `IterableDataset` for efficiently loading data from nested ZIP archives containing JSONL files (potentially gzipped), with support for compressed teacher logits.
*   **Modular Scripts:**
    *   `script.py`: Core script for training and evaluating the distilled student model.
    *   `alpha_graphs.py`: Generates plots showing the impact of temperature on various metrics.
    *   `temperature_graphs.py`: Generates plots showing the impact of alpha on various metrics.
*   **Reproducibility:** Aims to provide the necessary tools to replicate the experiments and analyses presented in the thesis.

## Directory Structure
distillation/
├── README.md
├── script.py # Main distillation training and evaluation script
├── compress.py # Utility for (de)compressing teacher logits (as used in the project)
├── temperature_graphs.py # Script to generate temperature graphs
├── alpha_graphs.py # Script to generate alpha graphs
├── results_temp # Directory for temperature results
├── results_alpha # Directory for alpha results
├── graphs_temp/ # Output directory for temperature graphs
├── graphs_alpha/ # Output directory for alpha graphs
└── requirements.txt # Python dependencies


## Setup

1.  **Clone the Repository**

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    *   You will need to obtain the dataset and place it.
    *   The `script.py` expects paths to the main dataset ZIP and internal paths to the specific train/validation JSONL files within the nested ZIP structure.
    *   The `compress.py` file is crucial for decompressing teacher logits if they were stored in a compressed format (e.g., 8-bit representation from original 32-bit floats using LZ4, as hinted in the paper). Ensure this utility is present and works correctly with your data format.

## Usage

### 1. Running Distillation Experiments (`script.py`)

The `script.py` is the main script to train and evaluate the student model.

**Key Arguments:**

*   `--distillation_data_path`: Path to the main outer ZIP file containing your dataset.
*   `--train_outer_zip_internal_path`: Internal path within the main ZIP to the *inner ZIP* containing training data.
*   `--train_inner_zip_jsonl_filename`: Filename of the JSONL file within the *inner training ZIP*.
*   `--train_jsonl_is_gzipped`: Flag if the training JSONL is gzipped.
*   `--val_outer_zip_internal_path`: Internal path within the main ZIP to the *inner ZIP* containing validation data.
*   `--val_inner_zip_jsonl_filename`: Filename of the JSONL file within the *inner validation ZIP*.
*   `--val_jsonl_is_gzipped`: Flag if the validation JSONL is gzipped.
*   `--focal_content_field_name`: Name of the field in JSONL for the focal code (e.g., "focal_file").
*   `--teacher_model_name`: Hugging Face model name for the teacher (e.g., `Salesforce/codet5-base`).
*   `--student_model_name`: Hugging Face model name for the student (e.g., `Salesforce/codet5-small`).
*   `--output_dir`: Directory to save models, logs, and evaluation results (JSON summary).
*   `--epochs`: Number of training epochs.
*   `--batch_size`, `--eval_batch_size`: Batch sizes for training and evaluation.
*   `--learning_rate`, `--weight_decay`: Optimizer parameters.
*   `--alpha_kl_loss_weight`: Weight for the soft KL divergence loss (range 0.0 to 1.0). (1-alpha) is for hard CE loss.
*   `--temperature`: Temperature for scaling teacher logits before KL divergence.
*   `--save_best_model`: Flag to save the best model based on `--best_model_metric`.
*   `--best_model_metric`: Metric to determine the best model (e.g., `avg_f1_s_vs_gt_script1`).

**Example Command (Single Run):**

```bash
python script.py \
    --distillation_data_path ./codet5.zip \
    --train_outer_zip_internal_path "codet5/distillation_data_training.jsonl.zip" \
    --train_inner_zip_jsonl_filename "distillation_data_training.jsonl" \
    --val_outer_zip_internal_path "codet5/distillation_data_validation.jsonl.zip" \
    --val_inner_zip_jsonl_filename "distillation_data_validation.jsonl" \
    --output_dir ./distilled_model_output \
    --student_model_name salesforce/codet5-small \
    --teacher_model_name Salesforce/codet5-base \
    --batch_size 2 \
    --eval_batch_size 4 \
    --alpha 0.5 \
    --temperature 1.0 \
    --save_best_model \
    --seed 42
```
### 2. Generating Graphs
After running the experiments and collecting multiple distillation_summary.json files in dedicated directories (e.g., results_temp/, results_alpha/), you can use the plotting scripts.

Use "alpha_graphs.py" for the graphs for the alpha parameter:

```bash
    python alpha_graphs.py
```

Use "temperature_graphs.py" for the graphs for the temperature parameter:

```bash
    python temperature_graphs.py
```