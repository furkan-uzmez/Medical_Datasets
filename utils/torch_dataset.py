import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class PadChestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, split='test', label_col='label_group'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            split (string): 'train', 'test', or 'validation' to filter data.
            label_col (string): Column to use as label. 
                                'label' is fine-grained, 'label_group' is coarser.
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # Load CSV
        try:
            self.data = pd.read_csv(csv_file)
        except Exception:
            # Try zip if csv not found directly (common in this dataset)
            self.data = pd.read_csv(csv_file + ".zip")

        # Filter by split if provided and column exists
        if split and 'split' in self.data.columns:
            self.data = self.data[self.data['split'] == split].reset_index(drop=True)
        
        self.label_col = label_col
        
        # Handle labels
        # Create a mapping from label to integer
        # We fill NaNs with "Unknown" or similar
        self.data[self.label_col] = self.data[self.label_col].fillna("Unknown")
        self.classes = sorted(self.data[self.label_col].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx]['ImageID']
        
        # Try finding the image with common extensions
        img_path = None
        # Check if img_name already has extension
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
            # Return black image if not found (or raise error based on preference)
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image = Image.new('RGB', (224, 224), color='black')

        label_name = self.data.iloc[idx][self.label_col]
        label = self.class_to_idx[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label
