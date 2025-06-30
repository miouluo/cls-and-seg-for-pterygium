import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms

class Cutout(object):
    def __init__(self, length=8, n_holes=2):
        self.length = length
        self.n_holes = n_holes

    def __call__(self, img):
        if isinstance(img, Image.Image): 
            w, h = img.size
            img_np = np.array(img)
            
            for n in range(self.n_holes):
                y = random.randint(0, h - self.length)
                x = random.randint(0, w - self.length)
                
                y1 = max(0, y)
                y2 = min(h, y + self.length)
                x1 = max(0, x)
                x2 = min(w, x + self.length)
                
                img_np[y1:y2, x1:x2, :] = 0
                
            return Image.fromarray(img_np)
        else: 
            h = img.size(1)
            w = img.size(2)
            
            mask = torch.ones((h, w), dtype=torch.float32)
            
            for n in range(self.n_holes):
                y = random.randint(0, h - self.length)
                x = random.randint(0, w - self.length)
                
                y1 = max(0, y)
                y2 = min(h, y + self.length)
                x1 = max(0, x)
                x2 = min(w, x + self.length)
                
                mask[y1:y2, x1:x2] = 0
            
            mask = mask.expand_as(img)
            img = img * mask
            
            return img

class PterygiumDataset(Dataset):
    def __init__(self, root_dir, label_file=None, transform=None, mode='train', indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.imgs = []
        
        if mode == 'train' or mode == 'val':
            self.label_df = pd.read_excel(label_file)
            self.img_ids = list(self.label_df['Image'])
            self.labels = list(self.label_df['Pterygium'])
            
            all_samples = []
            self.all_labels = [] 
            for img_id in self.img_ids:
                img_path = os.path.join(root_dir, str(img_id).zfill(4), f"{str(img_id).zfill(4)}.png")
                if os.path.exists(img_path):
                    label = self.labels[self.img_ids.index(img_id)]
                    all_samples.append((img_path, label))
                    self.all_labels.append(label)
            
            if indices is not None:
                self.imgs = [all_samples[i] for i in indices if i < len(all_samples)]
            else:
                self.imgs = all_samples
                
        else: 
            img_list = sorted(os.listdir(root_dir))
            for img_name in img_list:
                if img_name.endswith('.png'):
                    img_path = os.path.join(root_dir, img_name)
                    self.imgs.append((img_path, -1)) 
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_labels(self):
        """返回所有样本的标签，用于分层抽样"""
        return self.all_labels

class PterygiumTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.img_paths = []
        
        img_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
        for img_name in img_list:
            img_path = os.path.join(root_dir, img_name)
            img_id = int(os.path.splitext(img_name)[0])  # 提取图像ID
            self.imgs.append((img_path, img_id))
            self.img_paths.append(img_path)
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path, img_id = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, img_id

# 早停
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = 0 
        self.delta = delta
        self.path = path
        
    def __call__(self, val_score, model):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0
            
    def save_checkpoint(self, val_score, model):
        if self.verbose:
            print(f'验证Score提高 ({self.val_score_max:.3f} --> {val_score:.3f}). 保存模型...')
        torch.save(model, self.path)
        self.val_score_max = val_score

def get_train_transform(image_size=(512, 512)):
    """训练数据增强"""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(
            size=image_size,
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
            kernel_size=(3, 3),
            sigma=(0.1, 2.0)
        ),
        transforms.ColorJitter(
            brightness=0.3, 
            contrast=0.3, 
            saturation=0.3
        ),
        transforms.Resize(image_size),
        Cutout(length=5, n_holes=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_test_transform(image_size=(512, 512)):
    """测试数据增强"""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class TTAWrapper:
    def __init__(self, model, transforms_list):
        self.model = model
        self.transforms_list = transforms_list
        
    def __call__(self, x):
        batch_size = x.shape[0]
        predictions = []
        with torch.no_grad():
            predictions.append(self.model(x))
        for transform in self.transforms_list:
            with torch.no_grad():
                augmented = transform(x)
                predictions.append(self.model(augmented))
        # 平均结果
        return torch.stack(predictions).mean(dim=0)
        
def get_tta_transforms():
    """TTA变换"""
    return [
            # 水平翻转
            lambda x: torch.flip(x, dims=[3]),
            # 垂直翻转
            lambda x: torch.flip(x, dims=[2]),
            # 90度旋转
            lambda x: torch.rot90(x, k=1, dims=[2, 3])
        ]