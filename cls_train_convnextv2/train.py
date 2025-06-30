import json
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from timm.utils import accuracy, AverageMeter, ModelEma
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from convnextv2 import create_model_convnext
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
from torchvision import datasets
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from datetime import datetime
import random
from load_data import Cutout, PterygiumDataset, EarlyStopping, get_train_transform, get_test_transform, TTAWrapper, get_tta_transforms
from convnextv2 import CustomConvNeXt

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置显卡
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 固定随机种子以便复现结果
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'checkpoints/convnext_pterygium_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    file_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(file_dir, exist_ok=True)

    # 全局参数
    model_lr = 1e-4
    BATCH_SIZE = 6
    EPOCHS = 80
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = True  # 是否使用混合精度
    use_dp = False  # 是否开启dp方式的多卡训练
    classes = 3      # 分类数量：正常、翼状胬肉建议观察、翼状胬肉建议手术
    resume = None  
    CLIP_GRAD = 5.0
    best_score_container = {'value': 0}  # 存储最高分数
    use_ema = False
    model_ema_decay = 0.9998
    start_epoch = 1
    seed = 42
    n_splits = 5
    weight_decay = 0.02
    torch.backends.cudnn.benchmark = True
    
    seed_everything(seed)

    image_size = (512, 512)

    # 数据预处理和增强
    transform_train = get_train_transform(image_size)
    transform_test = get_test_transform(image_size)
    
    mixup_fn = Mixup(
        mixup_alpha=0.9, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=0.1, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=classes)

    full_dataset = PterygiumDataset(
        root_dir='../train',
        label_file='../train_classification_label.xlsx',
        transform=None, 
        mode='train'
    )
    
    all_labels = full_dataset.get_labels()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) 
    fold_accuracies = [] 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
        print(f"\n{'='*50}")
        print(f"开始训练第 {fold+1}/{n_splits} 折")
        print(f"{'='*50}")
        
        fold_dir = os.path.join(file_dir, f'fold_{fold+1}')
        os.makedirs(fold_dir, exist_ok=True)
        best_score_container['value'] = 0
        
        train_dataset = PterygiumDataset(
            root_dir='../train',
            label_file='../train_classification_label.xlsx',
            transform=transform_train,
            mode='train',
            indices=train_idx
        )
        
        val_dataset = PterygiumDataset(
            root_dir='../train',
            label_file='../train_classification_label.xlsx',
            transform=transform_test,
            mode='val',
            indices=val_idx
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            pin_memory=True, 
            shuffle=True,
            drop_last=True, 
            num_workers=8,
            persistent_workers=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            pin_memory=True, 
            shuffle=False, 
            num_workers=8,
            persistent_workers=True
        )
        
        print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
        
        # 类别分布
        train_labels = [label for _, label in train_dataset.imgs]
        val_labels = [label for _, label in val_dataset.imgs]
        
        train_class_counts = {}
        val_class_counts = {}
        
        for label in train_labels:
            train_class_counts[label] = train_class_counts.get(label, 0) + 1
        
        for label in val_labels:
            val_class_counts[label] = val_class_counts.get(label, 0) + 1
        
        print("\n训练集类别分布:")
        for class_id in sorted(train_class_counts.keys()):
            print(f"类别 {class_id}: {train_class_counts[class_id]} 样本 ({train_class_counts[class_id]/len(train_labels)*100:.2f}%)")
        
        print("\n验证集类别分布:")
        for class_id in sorted(val_class_counts.keys()):
            print(f"类别 {class_id}: {val_class_counts[class_id]} 样本 ({val_class_counts[class_id]/len(val_labels)*100:.2f}%)")
        
        criterion_train = SoftTargetCrossEntropy()
        criterion_val = torch.nn.CrossEntropyLoss()

        model_ft = create_model_convnext(
            num_classes=classes,
            pretrained=True,
            drop_path_rate=0.1
        )
        
        if resume:
            model = torch.load(resume)
            print(model['state_dict'].keys())
            model_ft.load_state_dict(model['state_dict'])
            best_score_container['value'] = model['Best_Score']
            start_epoch = model['epoch'] + 1
        
        model_ft.to(DEVICE)
    
        optimizer = optim.AdamW(model_ft.parameters(), lr=model_lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  
            T_mult=1,  
            eta_min=1e-6  
        )
    
        # 设置混合精度，DP多卡，EMA
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        if torch.cuda.device_count() > 1 and use_dp:
            print(f"使用 {torch.cuda.device_count()} 个GPU!")
            model_ft = torch.nn.DataParallel(model_ft)
        
        if use_ema:
            model_ema = ModelEma(
                model_ft,
                decay=model_ema_decay,
                device=DEVICE,
                resume=resume)
        else:
            model_ema = None
            
        # 早停
        early_stopping = EarlyStopping(
            patience=40, 
            verbose=True, 
            path=os.path.join(fold_dir, 'best.pth')
        )
        
        def train(model, device, train_loader, optimizer, epoch, model_ema):
            model.train()
            loss_meter = AverageMeter()
            acc1_meter = AverageMeter()
            acc5_meter = AverageMeter()
            
            # 梯度累积参数
            accumulation_steps = 3  # 累积3次，相当于批次大小为6*3=18
            optimizer.zero_grad()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                samples, targets = mixup_fn(data, target)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        output = model(samples)
                        
                        ce_loss = criterion_train(output, targets)
                        probs = torch.softmax(output, dim=1)
                        pred_probs = torch.zeros((3, 3), dtype=torch.float32, device=device)
                        for i in range(3):
                            for j in range(3):
                                pred_probs[i, j] = probs[target == i, j].sum()
                        total = pred_probs.sum()
                        Acc = (pred_probs[0, 0] + pred_probs[1, 1] + pred_probs[2, 2]) / total
                        
                        # 精确率
                        P1 = pred_probs[0, 0] / pred_probs[:, 0].sum() if pred_probs[:, 0].sum() > 0 else torch.tensor(0.0, device=device)
                        P2 = pred_probs[1, 1] / pred_probs[:, 1].sum() if pred_probs[:, 1].sum() > 0 else torch.tensor(0.0, device=device)
                        P3 = pred_probs[2, 2] / pred_probs[:, 2].sum() if pred_probs[:, 2].sum() > 0 else torch.tensor(0.0, device=device)
                        MacroP = (P1 + P2 + P3) / 3
                        
                        # 召回率
                        R1 = pred_probs[0, 0] / pred_probs[0, :].sum() if pred_probs[0, :].sum() > 0 else torch.tensor(0.0, device=device)
                        R2 = pred_probs[1, 1] / pred_probs[1, :].sum() if pred_probs[1, :].sum() > 0 else torch.tensor(0.0, device=device)
                        R3 = pred_probs[2, 2] / pred_probs[2, :].sum() if pred_probs[2, :].sum() > 0 else torch.tensor(0.0, device=device)
                        
                        # F1分数
                        F1_1 = 2 * P1 * R1 / (P1 + R1) if (P1 + R1) > 0 else torch.tensor(0.0, device=device)
                        F1_2 = 2 * P2 * R2 / (P2 + R2) if (P2 + R2) > 0 else torch.tensor(0.0, device=device)
                        F1_3 = 2 * P3 * R3 / (P3 + R3) if (P3 + R3) > 0 else torch.tensor(0.0, device=device)
                        MacroF1 = (F1_1 + F1_2 + F1_3) / 3
                        
                        Score = Acc + MacroP + MacroF1
                        Avg_score = Score / 3
                        score_loss = 1 - Avg_score
                        alpha = 0.2 
                        loss = alpha * ce_loss + (1 - alpha) * score_loss
                        loss = loss / accumulation_steps
                        
                    scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        
                        if model_ema is not None:
                            model_ema.update(model)
                else:
                    output = model(samples)
                    
                    # 交叉熵
                    ce_loss = criterion_train(output, targets)
                    probs = torch.softmax(output, dim=1)
                    pred_probs = torch.zeros((3, 3), dtype=torch.float32, device=device)
                    for i in range(3):
                        for j in range(3):
                            pred_probs[i, j] = probs[target == i, j].sum()
                    total = pred_probs.sum()
                    Acc = (pred_probs[0, 0] + pred_probs[1, 1] + pred_probs[2, 2]) / total
                    
                    # 精确率
                    P1 = pred_probs[0, 0] / pred_probs[:, 0].sum() if pred_probs[:, 0].sum() > 0 else torch.tensor(0.0, device=device)
                    P2 = pred_probs[1, 1] / pred_probs[:, 1].sum() if pred_probs[:, 1].sum() > 0 else torch.tensor(0.0, device=device)
                    P3 = pred_probs[2, 2] / pred_probs[:, 2].sum() if pred_probs[:, 2].sum() > 0 else torch.tensor(0.0, device=device)
                    MacroP = (P1 + P2 + P3) / 3
                    
                    # 召回率
                    R1 = pred_probs[0, 0] / pred_probs[0, :].sum() if pred_probs[0, :].sum() > 0 else torch.tensor(0.0, device=device)
                    R2 = pred_probs[1, 1] / pred_probs[1, :].sum() if pred_probs[1, :].sum() > 0 else torch.tensor(0.0, device=device)
                    R3 = pred_probs[2, 2] / pred_probs[2, :].sum() if pred_probs[2, :].sum() > 0 else torch.tensor(0.0, device=device)
                    
                    # F1分数
                    F1_1 = 2 * P1 * R1 / (P1 + R1) if (P1 + R1) > 0 else torch.tensor(0.0, device=device)
                    F1_2 = 2 * P2 * R2 / (P2 + R2) if (P2 + R2) > 0 else torch.tensor(0.0, device=device)
                    F1_3 = 2 * P3 * R3 / (P3 + R3) if (P3 + R3) > 0 else torch.tensor(0.0, device=device)
                    MacroF1 = (F1_1 + F1_2 + F1_3) / 3
                    
                    Score = Acc + MacroP + MacroF1
                    Avg_score = Score / 3
                    
                    score_loss = 1 - Avg_score
                    alpha = 0.2
                    loss = alpha * ce_loss + (1 - alpha) * score_loss
                    loss = loss / accumulation_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        if model_ema is not None:
                            model_ema.update(model)
                
                torch.cuda.synchronize()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                
                acc1, acc5 = accuracy(output, target, topk=(1, min(classes, 5)))
                loss_meter.update(loss.item() * accumulation_steps, target.size(0))  # 因为之前除了累积步数，现在要乘回来
                acc1_meter.update(acc1.item(), target.size(0))
                acc5_meter.update(acc5.item(), target.size(0))
                
                # 我们train有360个样本，batch_size=6，则一个epoch有60个批次
                # 我们在一个epoch中输出5次，则每12个批次输出一次
                # 考虑梯度累积为3次，则每20个批次更新一次模型参数，每(60/5)*3=36个批次输出一次
                if (batch_idx + 1) % 12 == 0:
                    effective_batch_size = len(data) * accumulation_steps
                    effective_idx = (batch_idx + 1) // accumulation_steps
                    effective_total = len(train_loader) // accumulation_steps
                    print(f'训练 Epoch: {epoch} [{(batch_idx + 1) * len(data)}/{len(train_loader.dataset)} ({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\t损失: {loss.item() * accumulation_steps:.6f}\t学习率:{lr:.9f}')
                    print(f'CE Loss: {ce_loss.item():.4f}, Score Loss: {score_loss.item():.4f}, Acc: {Acc.item():.4f}, MacroP: {MacroP.item():.4f}, MacroF1: {MacroF1.item():.4f}, Score: {Score.item():.4f}, Avg_score: {Avg_score.item():.4f}')
            
            ave_loss = loss_meter.avg
            acc = acc1_meter.avg
            print(f'epoch:{epoch}\t损失:{ave_loss:.2f}\t准确率:{acc:.2f}')
            return ave_loss, acc


        @torch.no_grad()
        def val(model, device, test_loader, use_tta=True):
            model.eval()
            loss_meter = AverageMeter()
            acc1_meter = AverageMeter()
            acc5_meter = AverageMeter()
            total_num = len(test_loader.dataset)
            print(f"验证数据量: {total_num}, Batches: {len(test_loader)}")
            
            val_list = []
            pred_list = []
            confusion_matrix = np.zeros((3, 3), dtype=int)
            original_model = model
            
            # TTA变换
            if use_tta:
                tta_transforms = get_tta_transforms()
                model = TTAWrapper(model, tta_transforms)
                print("使用测试时增强 (TTA)")
            
            for data, target in test_loader:
                for t in target:
                    val_list.append(t.data.item())
                    
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                loss = criterion_val(output, target)
                
                _, pred = torch.max(output.data, 1)
                for p in pred:
                    pred_list.append(p.data.item())
                
                # 混淆矩阵
                for t, p in zip(target, pred):
                    confusion_matrix[t, p] += 1
                
                acc1, acc5 = accuracy(output, target, topk=(1, min(classes, 5)))
                loss_meter.update(loss.item(), target.size(0))
                acc1_meter.update(acc1.item(), target.size(0))
                acc5_meter.update(acc5.item(), target.size(0))
            
            acc = acc1_meter.avg
            
            # 计算Accuracy, Macro Precision, Macro F1 score
            N11, N12, N13 = confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[0, 2]
            N21, N22, N23 = confusion_matrix[1, 0], confusion_matrix[1, 1], confusion_matrix[1, 2]
            N31, N32, N33 = confusion_matrix[2, 0], confusion_matrix[2, 1], confusion_matrix[2, 2]
            
            Acc = (N11 + N22 + N33) / (N11 + N12 + N13 + N21 + N22 + N23 + N31 + N32 + N33)
            
            P1 = N11 / (N11 + N21 + N31) if (N11 + N21 + N31) != 0 else 0
            P2 = N22 / (N12 + N22 + N32) if (N12 + N22 + N32) != 0 else 0
            P3 = N33 / (N13 + N23 + N33) if (N13 + N23 + N33) != 0 else 0
            MacroP = (P1 + P2 + P3) / 3
            
            R1 = N11 / (N11 + N12 + N13) if (N11 + N12 + N13) != 0 else 0
            R2 = N22 / (N21 + N22 + N23) if (N21 + N22 + N23) != 0 else 0
            R3 = N33 / (N31 + N32 + N33) if (N31 + N32 + N33) != 0 else 0
            
            F1_1 = 2 * P1 * R1 / (P1 + R1) if (P1 + R1) != 0 else 0
            F1_2 = 2 * P2 * R2 / (P2 + R2) if (P2 + R2) != 0 else 0
            F1_3 = 2 * P3 * R3 / (P3 + R3) if (P3 + R3) != 0 else 0
            MacroF1 = (F1_1 + F1_2 + F1_3) / 3
            
            Score = Acc + MacroP + MacroF1 
            Avg_score = Score / 3  # 归一化
            
            print("混淆矩阵:\n")
            print(confusion_matrix)
            print(f'\n验证集: 平均损失: {loss_meter.avg:.4f}\t准确率:{Acc*100:.3f}%\tTop5准确率:{acc5_meter.avg:.3f}%')
            print(f'宏平均精确率: {MacroP*100:.3f}%\t宏平均F1分数: {MacroF1*100:.3f}%')
            print(f'Score: {Score:.4f}\tAvg_score: {Avg_score:.4f}\n')
            
            # 更新Best_Score
            if Score > best_score_container['value']:
                if isinstance(original_model, torch.nn.DataParallel):
                    torch.save(original_model.module, os.path.join(fold_dir, 'best_score.pth'))
                else:
                    torch.save(original_model, os.path.join(fold_dir, 'best_score.pth'))
                best_score_container['value'] = Score
                print(f'新的最佳Score: {Score:.4f}')
                
            if isinstance(original_model, torch.nn.DataParallel):
                state = {
                    'epoch': epoch,
                    'state_dict': original_model.module.state_dict(),
                    'Best_Score': best_score_container['value'] 
                }
                if use_ema:
                    state['state_dict_ema'] = original_model.module.state_dict()
            else:
                state = {
                    'epoch': epoch,
                    'state_dict': original_model.state_dict(),
                    'Best_Score': best_score_container['value']  
                }
                if use_ema:
                    state['state_dict_ema'] = original_model.state_dict()
                
            return val_list, pred_list, loss_meter.avg, Score, Acc, MacroP, MacroF1
    
        # 训练与验证
        log_dir = {}
        train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
        val_macrof1_list, val_macrop_list, val_score_list = [], [], [] 
        
        if resume and os.path.isfile(os.path.join(fold_dir, "result.json")):
            with open(os.path.join(fold_dir, 'result.json'), 'r', encoding='utf-8') as file:
                logs = json.load(file)
                train_acc_list = logs['train_acc']
                train_loss_list = logs['train_loss']
                val_acc_list = logs['val_acc']
                val_loss_list = logs['val_loss']
                epoch_list = logs['epoch_list']
                if 'val_macrof1' in logs:
                    val_macrof1_list = logs['val_macrof1']
                if 'val_macrop' in logs:
                    val_macrop_list = logs['val_macrop']
                if 'val_score' in logs:
                    val_score_list = logs['val_score']
                
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_list.append(epoch)
            log_dir['epoch_list'] = epoch_list
            
            train_loss, train_acc = train(model_ft, DEVICE, train_loader, optimizer, epoch, model_ema)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            log_dir['train_acc'] = train_acc_list
            log_dir['train_loss'] = train_loss_list
            
            if use_ema:
                val_list, pred_list, val_loss, val_score, val_acc, val_macrop, val_macrof1 = val(model_ema.ema, DEVICE, val_loader, use_tta=True)
            else:
                val_list, pred_list, val_loss, val_score, val_acc, val_macrop, val_macrof1 = val(model_ft, DEVICE, val_loader, use_tta=True)
                
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            val_macrop_list.append(val_macrop)
            val_macrof1_list.append(val_macrof1)
            val_score_list.append(val_score)
            
            log_dir['val_acc'] = val_acc_list
            log_dir['val_loss'] = val_loss_list
            log_dir['val_macrop'] = val_macrop_list
            log_dir['val_macrof1'] = val_macrof1_list
            log_dir['val_score'] = val_score_list
            log_dir['best_score'] = best_score_container['value'] 
            
            with open(os.path.join(fold_dir, 'result.json'), 'w', encoding='utf-8') as file:
                file.write(json.dumps(log_dir))
            
            class_mapping = {
                0: "正常",
                1: "翼状胬肉建议观察",
                2: "翼状胬肉建议手术"
            }

            print(classification_report(val_list, pred_list, target_names=list(class_mapping.values())))

            scheduler.step(val_score)
            
            early_stopping(val_score, model_ft)
            if early_stopping.early_stop:
                print("早停触发! 停止训练。")
                break
        
        fold_accuracies.append(best_score_container['value'])
        print(f"第 {fold+1} 折最佳Score: {best_score_container['value']:.4f}")
        
    print("\n" + "="*50)
    print("k折交叉验证结果:")
    for i, score in enumerate(fold_accuracies):
        print(f"第 {i+1} 折Score: {score:.4f}")
    print(f"平均Score: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print("="*50)
    
    with open(os.path.join(base_dir, 'kfold_results.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'fold_scores': fold_accuracies,  
            'mean_score': float(np.mean(fold_accuracies)), 
            'std_score': float(np.std(fold_accuracies)) 
        }, f, indent=4)
        
if __name__ == '__main__':
    main()