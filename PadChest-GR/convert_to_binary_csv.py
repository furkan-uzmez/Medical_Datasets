#!/usr/bin/env python3
"""
Convert multi-label CSV to binary classification CSV.
Each unique ImageID will have exactly one row with a binary label:
  - Normal (0): if ALL labels for that image are "Normal"
  - Abnormal (1): if ANY label for that image is NOT "Normal"
"""

import pandas as pd
import argparse
import os

def convert_to_binary(input_csv, output_csv):
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Original rows: {len(df)}")
    print(f"Unique ImageIDs: {df['ImageID'].nunique()}")
    
    # Group by ImageID and determine binary label
    def get_binary_label(group):
        # If any row has a label other than 'Normal', the image is Abnormal
        labels = group['label_group'].unique()
        if len(labels) == 1 and labels[0] == 'Normal':
            return 'Normal'
        else:
            return 'Abnormal'
    
    # Get first row for each ImageID (to preserve other columns)
    # and compute binary label
    print("Processing...")
    
    # Group by ImageID
    grouped = df.groupby('ImageID')
    
    # Get binary labels
    binary_labels = grouped.apply(get_binary_label).reset_index()
    binary_labels.columns = ['ImageID', 'binary_label']
    
    # Get first row of each group for other metadata
    first_rows = grouped.first().reset_index()
    
    # Replace label_group with binary label
    result = first_rows.merge(binary_labels, on='ImageID')
    result['label_group'] = result['binary_label']
    result = result.drop(columns=['binary_label'])
    
    # Also update 'label' column to match
    result['label'] = result['label_group']
    
    print(f"\nResult rows: {len(result)}")
    print(f"Label distribution:")
    print(result['label_group'].value_counts())
    
    # Split distribution
    if 'split' in result.columns:
        print(f"\nSplit distribution:")
        for split in ['train', 'validation', 'test']:
            split_df = result[result['split'] == split]
            normal = len(split_df[split_df['label_group'] == 'Normal'])
            abnormal = len(split_df[split_df['label_group'] == 'Abnormal'])
            print(f"  {split}: {len(split_df)} (Normal: {normal}, Abnormal: {abnormal})")
    
    # Save
    print(f"\nSaving to {output_csv}...")
    result.to_csv(output_csv, index=False)
    print("Done!")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert multi-label CSV to binary classification CSV")
    parser.add_argument("--input", type=str, default="dataset/master_table.csv", help="Input CSV path")
    parser.add_argument("--output", type=str, default="dataset/master_table_binary.csv", help="Output CSV path")
    args = parser.parse_args()
    
    convert_to_binary(args.input, args.output)
