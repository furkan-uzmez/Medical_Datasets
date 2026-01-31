import torch
import torch.nn as nn
from torchvision import models
from utils.torch_dataset import PadChestDataset
import os
import argparse

def create_dummy(csv_path, img_dir, save_path="dummy_model.pth"):
    print(f"Inspecting dataset at {csv_path} for num_classes...")
    try:
        # We don't need transforms just to get class count
        ds = PadChestDataset(csv_file=csv_path, img_dir=img_dir, split='test')
        if len(ds) == 0:
             print("Test split empty, trying validation...")
             ds = PadChestDataset(csv_file=csv_path, img_dir=img_dir, split='validation')
        if len(ds) == 0:
             print("Validation split empty, trying train...")
             ds = PadChestDataset(csv_file=csv_path, img_dir=img_dir, split='train')
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    num_classes = len(ds.classes)
    print(f"Num classes found: {num_classes}")
    print(f"Classes: {ds.classes}")
    
    # Create model
    print("Creating ResNet18 model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Save
    print(f"Saving state_dict to {save_path}...")
    torch.save(model.state_dict(), save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="dataset/master_table.csv")
    parser.add_argument("--img_dir", type=str, default="dataset/Padchest_GR_files/PadChest_GR")
    args = parser.parse_args()
    
    create_dummy(args.csv_path, args.img_dir)
