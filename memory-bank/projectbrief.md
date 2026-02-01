# Project Brief: Chest X-Ray Datasets Analysis

## Overview
This workspace serves as a hub for analyzing and modeling multiple Chest X-Ray datasets. The primary focus currently is on the **PadChest-GR (Grounded Reports)** dataset, specifically the "Benchmark" subset. The project aims to build robust pipelines for data analysis, binary classification (Normal vs. Abnormal), and detailed benchmarking.

## Core Goals
### PadChest-GR
-   **Benchmarking**: Establish a reliable benchmark for the PadChest-GR dataset using the official test set.
-   **Binary Classification**: Train and evaluate models (e.g., ResNet18) to classify images as "Normal" or "Abnormal".
-   **Data Analysis**: Perform comprehensive Exploratory Data Analysis (EDA) to understand class distributions, patient demographics, and label integrity.
-   **Infrastructure**: Create reusable scripts for dataset conversion, training, and evaluation.

### COVID-CXNet
-   **Analysis**: Explore the COVID-CXNet dataset structure and contents.

## Project Scope
-   **Inputs**:
    -   PadChest-GR: `master_table.csv`, `grounded_reports.json`, Images.
    -   COVID-CXNet: CSVs and Images.
-   **Outputs**:
    -   Trained PyTorch Models (`checkpoints/`).
    -   Evaluation Reports and Metrics (Accuracy, AUC, F1).
    -   Derived Datasets (`master_table_binary.csv`).
    -   Analytical Notebooks (`analysis.ipynb`, `analysis2.ipynb`).
