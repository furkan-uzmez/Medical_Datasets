# Product Context

## Why this project exists
Medical imaging analysis requires rigorous benchmarking and robust handling of real-world data complexities. This project specifically targets the **PadChest-GR Benchmark** dataset to establish baselines for image classification and report grounding. It addresses the need for:
1.  **Standardized Benchmarking**: Using the official test split (`study_is_benchmark=True`) for comparable results.
2.  **Automated Diagnosis**: Developing models to distinguish between Normal and Abnormal chest X-rays.

## Problems it solves
-   **Multi-Label to Binary Simplification**: Converts complex multi-label finding data into a clean Binary (Normal/Abnormal) target for initial modeling.
-   **Class Imbalance**: Addresses the imbalance between Normal and Pathology cases using weighted loss functions during training.
-   **Noisy Labels**: Handles data where validation status is low (`study_is_validation=False`) by using only the "Benchmark" subset which is considered the ground truth for this specific task scope.
-   **Data Ingestion**: Provides tools (`dataset.py`, `torch_dataset.py`) to efficiently load long-format CSV data.

## Dataset Characteristics
### PadChest-GR
-   **Size**: ~4,555 unique images in the Benchmark set.
-   **Format**: Long-format CSV.
-   **Labels**:
    -   **Multi-Label**: Original labels include specific findings (e.g., "Infiltration", "Effusion").
    -   **Binary (Derived)**: "Normal" vs. "Abnormal" (any pathology).
-   **Splits**: Train / Validation / Test (defined in the CSV).

## User Experience Goals
-   **Reproducibility**: One-command training and evaluation.
-   **Visibility**: Clear logging of training metrics (Loss, Accuracy) and Evaluation results (AUC, F1).
-   **Ease of Access**: Simple conversion scripts to prepare data for different tasks (e.g., `convert_to_binary_csv.py`).
