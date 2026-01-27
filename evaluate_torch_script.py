import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import argparse
from utils.torch_dataset import PadChestDataset
from utils.evaluation_utils import evaluate_model

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate model on PadChest-GR test set")
    parser.add_argument("--csv_path", type=str, default="dataset/master_table.csv", help="Path to csv labels")
    parser.add_argument("--img_dir", type=str, default="dataset/Padchest_GR_files/PadChest_GR", help="Path to images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model weights (optional)")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate on (test, train, validation)")
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms (standard ImageNet normalization typically used for Transfer Learning)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load Dataset
    print(f"Loading dataset from {args.csv_path} (images: {args.img_dir})...")
    try:
        test_dataset = PadChestDataset(
            csv_file=args.csv_path,
            img_dir=args.img_dir,
            transform=transform,
            split=args.split
        )
    except FileNotFoundError:
        print("Error: Dataset files not found. Please check paths.")
        return

    print(f"Dataset size: {len(test_dataset)}")
    if len(test_dataset) == 0:
        print("Dataset is empty. Check split name or file paths.")
        return

    num_classes = len(test_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {test_dataset.classes}")

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # Initialize Model
    # Using ResNet18 as a placeholder baseline. 
    # Replace this with your specific model architecture.
    model = models.resnet18(weights=None) # Start from scratch or use weights='DEFAULT'
    
    # Adjust final layer for number of classes in dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}...")
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading weights: {e}")
    else:
        print("No model path provided or file doesn't exist. Using initialized random weights for demonstration.")

    model = model.to(device)

    # Evaluate
    print("\nStarting evaluation...")
    metrics = evaluate_model(model, test_loader, device, num_classes=num_classes)

    # Print results
    print("\n" + "="*30)
    print("Evaluation Results")
    print("="*30)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if metrics.get('auc') is not None:
        print(f"AUC: {metrics['auc']:.4f}")
    
    print(f"\nWeighted F1 Score: {metrics['f1_weighted']:.4f}")
    print(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    print(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])

if __name__ == "__main__":
    main()
