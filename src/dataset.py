import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WaterbirdsDataset(Dataset):
    def __init__(self, metadata_path, data_dir, split='train', transform=None):
        """
        Waterbirds PyTorch Dataset based on Sagawa et al. (2020)
        Official splits in metadata.csv: 0 -> train, 1 -> val, 2 -> test
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Split mapping: 'train' -> 0, 'val' -> 1, 'test' -> 2
        split_map = {'train': 0, 'val': 1, 'test': 2}
        if isinstance(split, str):
            split_code = split_map[split]
        else:
            split_code = split
            
        self.df = df[df['split'] == split_code].reset_index(drop=True)
        
        # Extract ground truth y (0: landbird, 1: waterbird) and place (0: land, 1: water)
        self.y_array = self.df['y'].values
        self.place_array = self.df['place'].values  # 0: Land background, 1: Water background
        self.filename_array = self.df['img_filename'].values
        
        # Calculate subgroup ID (0 to 3)
        # Group = y * 2 + place
        self.group_array = self.y_array * 2 + self.place_array

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.filename_array[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        y = torch.tensor(self.y_array[idx], dtype=torch.long)
        group = torch.tensor(self.group_array[idx], dtype=torch.long)
        
        return image, y, group

def get_waterbirds_transforms():
    """Standard ImageNet preprocessing transforms for ResNet-50"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, eval_transform