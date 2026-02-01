# Technical Context

## Technologies Used
-   **Core Language**: Python 3.x
-   **Deep Learning Framework**: PyTorch (`torch`, `torchvision`)
    -   Models: ResNet18 (and potentially others).
    -   Data Loading: `torch.utils.data.Dataset`, `DataLoader`.
-   **Data Manipulation**: Pandas, NumPy.
-   **Image Processing**: Pillow (`PIL`).
-   **Evaluation**: Scikit-learn (for metrics like AUC, F1, Precision, Recall).
-   **Visualization**: Matplotlib, Seaborn (in notebooks).
-   **Progress Tracking**: `tqdm`.

## Development Setup
-   **Environment**: `.venv` (Virtual Environment).
-   **Dependencies**: Defined in `PadChest-GR/requirements.txt`.
-   **Hardware**: GPU acceleration (CUDA) is supported and recommended for training.

## Technical Constraints
-   **Dataset Size**: PadChest is large; data loaders must be efficient (lazy loading images).
-   **Image Resolution**: Standardized to 224x224 for ResNet training (via `transforms`).
-   **Data Consistency**: Handling missing files or corrupt images gracefully in the `Dataset` class (`try-except` blocks).
