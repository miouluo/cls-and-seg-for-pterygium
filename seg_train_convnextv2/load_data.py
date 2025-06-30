import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import pandas as pd
import cv2
import random

class PterygiumSegmentationDataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None, img_size=(512, 512), is_train=True):
        """
        初始化数据集
        Args:
            root_dir: 图像根目录
            label_file: 分类标签文件
            transform: 数据增强变换
            img_size: 图像大小
            is_train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.is_train = is_train
        
        self.df = pd.read_excel(label_file)
    
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
        # 标准化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 获取图像ID和类别
        row = self.df.iloc[idx]
        img_id = str(row['Image']).zfill(4)
        pterygium_class = row['Pterygium']
        
        img_dir = os.path.join(self.root_dir, img_id)
        img_path = os.path.join(img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        
        if pterygium_class == 0:  
            mask = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        else:  
            if self.is_train:
                mask_path = os.path.join(img_dir, f"{img_id}_label.png")
                if os.path.exists(mask_path):
                    label = Image.open(mask_path).convert('L')
                    label = label.resize(self.img_size, Image.NEAREST)
                    mask = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
                    mask[:, :, 0] = (np.array(label) > 0).astype(np.uint8) * 128
                else:
                    mask = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
            else:
                mask = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        
        mask = Image.fromarray(mask)
        
        image = self.base_transform(image)
        mask = self.base_transform(mask)
        
        # 数据增强
        if self.transform:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            
            torch.manual_seed(seed)
            mask = self.transform(mask)
        
        # 标准化
        image = self.normalize(image)
        
        if self.is_train:
            mask = mask[0:1]  # 只保留R通道
            mask = (mask > 0.5).float()  # 二值化
        else:
            mask = mask.float()
            mask = (mask > 0.5).float() * 128  # R通道为128
        
        return {
            'image': image,
            'mask': mask,
            'id': img_id,
            'class': pterygium_class
        }

def get_transforms(img_size=(512, 512)):
    """
    获取数据增强变换
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # transforms.ColorJitter(
        #     brightness=0.2,
        #     contrast=0.2,
        #     saturation=0.2,
        #     hue=0.1
        # ), 加了掉分，loss震荡
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.8, 1.0),
            ratio=(0.8, 1.2)
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=10
        ),
        transforms.RandomPerspective(
            distortion_scale=0.2,
            p=0.5
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 2.0)
        ),
    ])