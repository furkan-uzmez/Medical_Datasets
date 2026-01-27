import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import argparse
import os
import time
import pandas as pd
from PIL import Image
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 on PadChest-GR (Binary Classification)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data_dir", type=str, default="dataset/Padchest_GR_files/PadChest_GR", help="Path to images")
    parser.add_argument("--csv_file", type=str, default="dataset/master_table_binary.csv", help="Path to binary CSV")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model_name", type=str, default="resnet18_padchest_binary", help="Name for saved model")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


class PadChestBinaryDataset(Dataset):
    """Binary classification dataset: Normal (0) vs Abnormal (1)"""
    
    def __init__(self, csv_file, img_dir, transform=None, split='train', label_col='label_group'):
        self.img_dir = img_dir
        self.transform = transform
        
        try:
            self.data = pd.read_csv(csv_file)
        except Exception:
            self.data = pd.read_csv(csv_file + ".zip")

        if split and 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        self.label_col = label_col
        self.data[self.label_col] = self.data[self.label_col].fillna("Unknown")
        
        # Binary: Normal=0, Abnormal=1
        self.classes = ['Normal', 'Abnormal']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx]['ImageID']
        
        img_path = None
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            temp_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(temp_path):
                img_path = temp_path
        else:
            for ext in ['.png', '.jpg', '.jpeg', '']:
                temp_path = os.path.join(self.img_dir, img_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
        
        if img_path is None:
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color='black')

        label_name = self.data.iloc[idx][self.label_col]
        # Binary: 0 if Normal, 1 if Abnormal
        label = 0 if label_name == 'Normal' else 1

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Shape: (batch, 1) for BCE
        
        optimizer.zero_grad()
        outputs = model(images)  # Shape: (batch, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy (sigmoid > 0.5)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100 * correct / total})
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'val_loss': running_loss / (pbar.n + 1), 'val_acc': 100 * correct / total})

    val_loss = running_loss / len(loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

import csv
import random
import numpy as np

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    """Seed function for dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    args = get_args()
    
    # Set seed for reproducibility
    print(f"Setting random seed to {args.seed}...")
    set_seed(args.seed)
    
    # Create generator for DataLoader to ensure reproducibility
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Logging Setup
    log_file_path = os.path.join(args.save_dir, "training_log.log")
    print(f"Logging metrics to {log_file_path}")
    
    # Write header if file doesn't exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w') as f:
            f.write("=" * 60 + "\n")
            f.write("Training Log - PadChest-GR Binary Classification\n")
            f.write("=" * 60 + "\n\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets (Binary: Normal vs Abnormal)
    print("Loading datasets (Binary Classification: Normal vs Abnormal)...")
    try:
        train_dataset = PadChestBinaryDataset(
            csv_file=args.csv_file,
            img_dir=args.data_dir,
            transform=transform,
            split='train'
        )
        val_dataset = PadChestBinaryDataset(
            csv_file=args.csv_file,
            img_dir=args.data_dir,
            transform=transform,
            split='validation'
        )
    except FileNotFoundError:
        print("Error: Dataset files not found. Please check paths.")
        return

    # Count class distribution
    train_labels = [0 if train_dataset.data.iloc[i][train_dataset.label_col] == 'Normal' else 1 
                    for i in range(len(train_dataset))]
    normal_count = train_labels.count(0)
    abnormal_count = train_labels.count(1)
    
    print(f"Train size: {len(train_dataset)} (Normal: {normal_count}, Abnormal: {abnormal_count})")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    if len(train_dataset) == 0:
        print("Train dataset is empty. Check split name or file paths.")
        return

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # Model (single output for binary classification)
    print("Initializing model...")
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 1)  # Single output for binary
    model = model.to(device)
    
    # Loss and Optimizer
    # Calculate pos_weight for imbalanced data
    pos_weight = torch.tensor([normal_count / abnormal_count]).to(device) if abnormal_count > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("Starting training...")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to file
        with open(log_file_path, mode='a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs}\n")
            f.write(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%\n")
            f.write(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%\n")
            f.write(f"  Epoch Time: {epoch_time:.2f}s\n")
            f.write("-" * 40 + "\n")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
            
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
             save_path = os.path.join(args.save_dir, f"{args.model_name}_epoch_{epoch+1}.pth")
             torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, save_path)
             print(f"Checkpoint saved to {save_path}")

    print("Training complete.")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()

