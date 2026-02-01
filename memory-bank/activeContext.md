# Active Context

## Current Work Focus
-   **Binary Classification Pipeline**: Finalizing and documenting the pipeline for training ResNet18 on the PadChest-GR binary task.
-   **Documentation Improvements**: Establishing a project-wide `README` and updating the Memory Bank to reflect the current codebase state.
-   **Codebase structure**: Ensuring `PadChest-GR` directory is self-contained with its own requirements and scripts.

## Recent Changes
-   **General**:
    -   Created project-root `README.md` for workspace navigation.
    -   Updated `PadChest-GR/README.md` with detailed usage instructions.
-   **PadChest-GR Implementation**:
    -   **`convert_to_binary_csv.py`**: Implemented script to flatten multi-label data into binary targets.
    -   **`train.py`**: Created robust training script with checkpointing, logging, and weighted BCE loss.
    -   **`evaluate_torch_script.py`**: Implemented standalone evaluation script calculating AUC, F1, and Classification Reports.
    -   **`utils/torch_dataset.py`**: Refined dataset class to handle image loading and transformations.
-   **Analysis**:
    -   `analysis.ipynb`: Completed initial EDA.
    -   `analysis2.ipynb`: Added patient-level statistics.

## Next Steps
-   **Run Benchmarks**: Execute `evaluate_torch_script.py` on the full test set and record results in `PadChest-GR/README.md` or a dedicated results file.
-   **Refine Models**: Experiment with different architectures (e.g., DenseNet121) or hyperparameters if baseline performance is insufficient.
-   **COVID-CXNet**: revisit analysis if priorities shift.

## Active Decisions
-   **Architecture**: Chose ResNet18 as the initial baseline for speed and efficiency.
-   **Metric Selection**: Prioritizing AUC and Weighted F1 due to class imbalance in medical datasets.
-   **File Structure**: Keeping datasets separate (`PadChest-GR`, `COVID-CXNet`) to maintain modularity.
