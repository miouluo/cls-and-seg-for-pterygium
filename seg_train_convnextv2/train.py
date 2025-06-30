import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import datetime
import argparse
from sklearn.model_selection import KFold
import gc
from tqdm import tqdm
import numpy as np
from medpy.metric.binary import hd95
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score

from convnextv2 import ConvNeXtSegmentation
from load_data import PterygiumSegmentationDataset, get_transforms

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.alpha * dice + (1 - self.alpha) * focal

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(pred, target):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Dice系数
    intersection = np.logical_and(pred, target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    
    # Hausdorff距离
    try:
        hd = hd95(pred, target)
        hd_score = 1 / (hd + 1)
    except:
        hd = np.nan
        hd_score = 0
 
    score = 4 * dice + 3 * hd_score
    
    # Sensitivity
    true_positives = np.logical_and(pred, target).sum()
    false_negatives = np.logical_and(np.logical_not(pred), target).sum()
    sensitivity = true_positives / (true_positives + false_negatives + 1e-6)
    
    # Specificity
    true_negatives = np.logical_and(np.logical_not(pred), np.logical_not(target)).sum()
    false_positives = np.logical_and(pred, np.logical_not(target)).sum()
    specificity = true_negatives / (true_negatives + false_positives + 1e-6)
    
    auc = roc_auc_score(target.flatten(), pred.flatten())
    
    return dice, hd, score, sensitivity, specificity, auc

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, fold, save_dir):
    writer = SummaryWriter(f'runs/convnext_fold_{fold}')
    scaler = GradScaler('cuda')
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  
        T_mult=1,  
        eta_min=1e-6  
    )
    
    best_val_score = 0.0 
    patience = 30
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_hd = 0.0
        train_score = 0.0
        train_sensitivity = 0.0
        train_specificity = 0.0
        train_auc = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            dice, hd, score, sensitivity, specificity, auc = calculate_metrics(outputs, masks)
            
            train_loss += loss.item()
            train_dice += dice
            train_hd += hd if not np.isnan(hd) else 0
            train_score += score
            train_sensitivity += sensitivity
            train_specificity += specificity
            train_auc += auc
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'hd95': f'{hd:.4f}' if not np.isnan(hd) else 'nan',
                'score': f'{score:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_hd /= len(train_loader)
        train_score /= len(train_loader)
        train_sensitivity /= len(train_loader)
        train_specificity /= len(train_loader)
        train_auc /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_hd = 0.0
        val_score = 0.0
        val_sensitivity = 0.0
        val_specificity = 0.0
        val_auc = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                # TTA
                outputs = []
                # 原始图像
                outputs.append(model(images))
                # 水平翻转
                outputs.append(torch.flip(model(torch.flip(images, [3])), [3]))
                # 垂直翻转
                outputs.append(torch.flip(model(torch.flip(images, [2])), [2]))
                output = torch.mean(torch.stack(outputs), dim=0)
                loss = criterion(output, masks)
                dice, hd, score, sensitivity, specificity, auc = calculate_metrics(output, masks)
                
                val_loss += loss.item()
                val_dice += dice
                val_hd += hd if not np.isnan(hd) else 0
                val_score += score
                val_sensitivity += sensitivity
                val_specificity += specificity
                val_auc += auc
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}',
                    'hd95': f'{hd:.4f}' if not np.isnan(hd) else 'nan',
                    'score': f'{score:.4f}'
                })
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_hd /= len(val_loader)
        val_score /= len(val_loader)
        val_sensitivity /= len(val_loader)
        val_specificity /= len(val_loader)
        val_auc /= len(val_loader)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('HD95/train', train_hd, epoch)
        writer.add_scalar('HD95/val', val_hd, epoch)
        writer.add_scalar('Score/train', train_score, epoch)
        writer.add_scalar('Score/val', val_score, epoch)
        writer.add_scalar('Sensitivity/train', train_sensitivity, epoch)
        writer.add_scalar('Sensitivity/val', val_sensitivity, epoch)
        writer.add_scalar('Specificity/train', train_specificity, epoch)
        writer.add_scalar('Specificity/val', val_specificity, epoch)
        writer.add_scalar('AUC/train', train_auc, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train HD95: {train_hd:.4f}, Train Score: {train_score:.4f}')
        print(f'Train Sensitivity: {train_sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val HD95: {val_hd:.4f}, Val Score: {val_score:.4f}')
        print(f'Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_fold{fold}.pth'))
            print(f'保存最佳模型，验证得分: {val_score:.4f} (Dice: {val_dice:.4f}, HD95: {val_hd:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered for fold {fold}')
                break
    
    writer.close()
    return model

def main():
    parser = argparse.ArgumentParser(description='训练翼状胬肉分割模型')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    parser.add_argument('--epochs', type=int, default=120, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--folds', type=int, default=5, help='交叉验证折数')
    
    args = parser.parse_args()
    seed_everything(seed=42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'checkpoints/{current_time}'
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = PterygiumSegmentationDataset(
        root_dir='../train',
        label_file='../train_classification_label.xlsx',
        transform=get_transforms((args.img_size, args.img_size)),
        img_size=(args.img_size, args.img_size),
        is_train=True
    )
    
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{args.folds}')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)
        
        model = ConvNeXtSegmentation(
            pretrained=True,
            in_chans=3,
            num_classes=1,
            drop_path_rate=0.1,
            img_size=args.img_size
        ).to(device)
        
        criterion = CombinedLoss(alpha=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02,betas=(0.9, 0.999))
        
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            device, args.epochs, fold, save_dir
        )
        
        del model
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()