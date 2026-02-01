# System Patterns

## System Architecture
The workspace is organized as a collection of independent dataset analysis projects. The currently active architecture is for **PadChest-GR Binary Classification**.

### Pipeline Flows
1.  **Data Ingestion & Transformation**:
    -   *Input*: `master_table.csv` (Multi-label).
    -   *Process*: `convert_to_binary_csv.py`.
    -   *Output*: `master_table_binary.csv` (Binary).
2.  **Model Training**:
    -   *Script*: `train.py`.
    -   *Input*: Binary CSV + Images.
    -   *Mechanism*: PyTorch DataLoader -> ResNet18 -> Weighted BCE Loss.
    -   *Output*: `.pth` Checkpoints.
3.  **Evaluation**:
    -   *Script*: `evaluate_torch_script.py`.
    -   *Input*: Test Split + Best Checkpoint.
    -   *Output*: Console Metrics (Accuracy, AUC, F1).

## Key Technical Decisions
-   **Derived Datasets**: Instead of complex on-the-fly label processing during training, we pre-process the CSV labels into a derived file (`master_table_binary.csv`). This simplifies the `Dataset` class and ensures consistency.
-   **Modular Scripts**: Training and Evaluation are decoupled into separate scripts to allow for running evaluation on any saved checkpoint without re-running training code.
-   **Standardized Interfaces**: All scripts use `argparse` for flexible configuration (batch size, paths, epochs).

## Component Relationships
-   `utils/torch_dataset.py` is the shared dependency for both training and evaluation scripts.
-   `checkpoints/` directory acts as the bridge between Training and Verification phases.
