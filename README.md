# Chest X-Ray Datasets Analysis Workspace

## Overview
This workspace is dedicated to the analysis, processing, and modeling of various Chest X-Ray datasets. The current focus is on **PadChest-GR** (Grounded Reports) and **COVID-CXNet**.

## Workspace Structure

### 1. [PadChest-GR](PadChest-GR/README.md)
**Focus**: Grounded Reports Analysis & Binary Classification.
-   **Dataset**: PadChest Mark-GR (Benchmark subset).
-   **Key Activities**:
    -   Exploratory Data Analysis (EDA) of grounded reports and findings.
    -   Binary Classification (Normal vs. Abnormal) using ResNet18.
    -   Benchmarking model performance on the official test set.
-   **Key Files**: `analysis.ipynb`, `train.py`, `evaluate_torch_script.py`.

### 2. [COVID-CXNet](COVID-CXNet/)
**Focus**: COVID-19 Chest X-Ray Analysis.
-   **Dataset**: COVID-CXNet dataset.
-   **Key Files**: `analysis.ipynb`, `covidx_merged.csv`.

## Documentation
The project metadata and context are maintained in the **[memory-bank](memory-bank/)** directory:
-   **[Active Context](memory-bank/activeContext.md)**: Current work focus and active decisions.
-   **[Product Context](memory-bank/productContext.md)**: Project goals and dataset characteristics.
-   **[Tech Context](memory-bank/techContext.md)**: Technologies and constraints.
-   **[Project Brief](memory-bank/projectbrief.md)**: High-level overview.

## Getting Started
Navigate to the specific dataset directory for detailed instructions.

```bash
# Example: Using PadChest-GR
cd PadChest-GR
pip install -r requirements.txt
python train.py --help
```
