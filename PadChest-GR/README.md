# PadChest-GR: Grounded Reports Analysis

## Overview
This project focuses on the analysis and utilization of the **PadChest-GR (Grounded Reports)** dataset. The goal is to explore the "Benchmark" subset of the PadChest dataset, which connects chest X-ray images with specific radiological findings and their locations.

## Dataset Structure
The project uses the following key files located in the `dataset/` directory:
- **`master_table.csv`**: The primary metadata file containing patient info, study details, and labels.
- **`grounded_reports_20240819.json`**: Contains the "grounded" relationships between report text and image regions.
- **`Padchest_GR_files/`**: Directory containing the chest X-ray images.

## Key Analysis Findings
Based on the exploratory data analysis performed in `analysis.ipynb`:

### 1. Dataset Integrity
- **Total Images**: 4,555 unique images.
- **Benchmark Status**: 100% of this subset is marked as `study_is_benchmark=True`, indicating it is the official test set for comparisons.
- **Validation Status**: 100% of this subset is marked as `study_is_validation=False`, meaning labels are automatically generated and not manually validated by radiologists.

### 2. Data Composition
- **Unique vs Duplicate**: The CSV is in **long-format**. There are 8,787 rows for 4,555 images, meaning many images have multiple findings (multi-label).
- **Missing Values**: 
    - `progression_status`: ~92% missing (expected, as most studies are single visits).
    - `prior_imageID`: ~64% missing (no prior study available).
    - `sentence_en`: ~16% missing (likely "Normal" cases with no specific finding sentence).

### 3. Demographics
- **Age**: Distribution is visualized in the notebook.
- **Sex**: Distribution is visualized in the notebook.

### 4. Normal vs Pathology
- Images are classified as "Normal" if they only contain the 'Normal' label.
- Images with any other label are classified as "Pathology".
- Some ambiguous cases exist where both labels might appear (rare data quality issue).

## Getting Started
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Analysis**:
    Open `analysis.ipynb` in Jupyter Lab or VS Code to interact with the data and view visualizations.

## Project Structure
- `dataset/`: Contains raw data (CSV, JSON, Images).
- `analysis.ipynb`: Main notebook for EDA.
- `memory-bank/`: Project documentation and context.
