import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, accuracy_score
from convnextv2 import LayerNorm, GRN, CustomConvNeXt
from timm.models import create_model
import torch.nn as nn

from load_data import PterygiumTestDataset, TTAWrapper, get_test_transform, get_tta_transforms

def load_model(model_path, device, num_classes=3):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        base_model = create_model(
            'convnextv2_base',
            pretrained=True,
            num_classes=num_classes,
            drop_path_rate=0.1
        )
        model = CustomConvNeXt(base_model, num_classes)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint
    
    model = model.to(device)
    model.eval()
    
    return model

def ensemble_predict(models, images, device, weights=None):
    """加权模型集成"""
    all_outputs = []
    
    for i, model in enumerate(models):
        outputs = model(images.to(device))
        all_outputs.append(outputs)
    if weights is None:
   
        ensemble_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    else:
        weighted_outputs = [out * w for out, w in zip(all_outputs, weights)]
        ensemble_outputs = torch.sum(torch.stack(weighted_outputs), dim=0) / sum(weights)
    
    _, preds = torch.max(ensemble_outputs, 1)
    
    return preds, ensemble_outputs

def predict_single_image(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='眼球翼状胬肉分类模型预测')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='包含模型的目录')
    parser.add_argument('--test_dir', type=str, default='../val_img', help='测试图像目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--ensemble', action='store_true', help='是否使用模型集成')
    parser.add_argument('--no_tta', action='store_true', help='不使用测试时增强')
    parser.add_argument('--weighted', action='store_true', help='是否使用加权集成')
    parser.add_argument('--weights', type=str, default='0.2,0.2,0.2,0.2,0.2', 
                        help='集成权重，用逗号分隔的浮点数"')
    parser.add_argument('--folds', type=str, default='1', 
                        help='要使用的折，用逗号分隔的数字，如"1,2,3"')
    args = parser.parse_args()
    
    if args.weighted:
        try:
            weights = [float(w) for w in args.weights.split(',')]
            print(f"使用加权集成，权重: {weights}")
        except:
            print("警告: 权重格式错误，使用默认等权重")
            weights = None
    else:
        weights = None
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")
    
    if os.path.isdir(args.model_dir):
        model_dirs = [d for d in os.listdir(args.model_dir) if d.startswith('convnext_pterygium_')]
        if model_dirs:
            latest_model_dir = os.path.join(args.model_dir, sorted(model_dirs)[-1])
            print(f"使用最新模型目录: {latest_model_dir}")
        else:
            print(f"错误: 在 {args.model_dir} 中找不到模型目录!")
            return
    else:
        latest_model_dir = args.model_dir
    
    class_names = ["正常", "翼状胬肉建议观察", "翼状胬肉建议手术"]
    print(f"类别映射: {class_names}")
    
    transform = get_test_transform()
    
    test_dataset = PterygiumTestDataset(
        root_dir=args.test_dir,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    models = []
    if args.ensemble:
        folds_dir = os.path.join(latest_model_dir, 'checkpoints')
        fold_dirs = [d for d in os.listdir(folds_dir) if d.startswith('fold_')]
        try:
            user_folds = [int(fold_num) for fold_num in args.folds.split(',')]
            print(f"用户指定使用折: {user_folds}")
            selected_folds = [f'fold_{fold_num}' for fold_num in user_folds]
        except ValueError:
            print(f"警告: 无法解析折号 '{args.folds}'，将使用第1折")
            selected_folds = ['fold_1']
  
        selected_folds = [f for f in selected_folds if f in fold_dirs]
        if not selected_folds:
            print("警告: 指定的折都不存在，将使用所有可用的折")
            selected_folds = fold_dirs

        print(f"将使用以下折进行集成: {selected_folds}")

        for fold_dir in selected_folds:
            model_path = os.path.join(folds_dir, fold_dir, 'best_score.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(folds_dir, fold_dir, 'best.pth')
            
            if os.path.exists(model_path):
                models.append(load_model(model_path, DEVICE))
                print(f"已加载模型: {model_path}")
            else:
                print(f"警告：未找到模型文件 {model_path}，跳过该折。") 
        
        if not models:
            print("错误: 找不到任何模型文件!")
            return
        
        print(f"已加载 {len(models)} 个模型进行集成预测")
    else:
        # 如果不使用集成，使用指定的第一个折或默认第1折
        try:
            fold_num = int(args.folds.split(',')[0])
            fold_dir = os.path.join(latest_model_dir, 'checkpoints', f'fold_{fold_num}')
        except (ValueError, IndexError):
            print(f"警告: 无法解析折号，将使用第1折")
            fold_dir = os.path.join(latest_model_dir, 'checkpoints', 'fold_1')
            
        print(f"非集成模式下使用: {fold_dir}")
        
        model_path = os.path.join(fold_dir, 'best_score.pth')
        
        if not os.path.exists(model_path):
            model_path = os.path.join(fold_dir, 'best.pth')
            if not os.path.exists(model_path):
                print(f"错误: 模型文件 {model_path} 不存在!")
                return
        
        model = load_model(model_path, DEVICE)
        models = [model]
        print(f"已加载模型: {model_path}")
    
    # TTA
    use_tta = not args.no_tta
    if use_tta:
        tta_transforms = get_tta_transforms()
        
        for i in range(len(models)):
            models[i] = TTAWrapper(models[i], tta_transforms)
        
        print("使用测试时增强 (TTA)")
    else:
        print("不使用测试时增强")
    
    predictions = []
    img_ids = []
    confidences = []
    
    print(f"开始对 {len(test_dataset)} 张图像进行预测...")
    for images, ids in tqdm(test_loader, desc="预测中"):
        if args.ensemble:
            preds, outputs = ensemble_predict(models, images, DEVICE, weights if args.weighted else None)
        else:
            with torch.no_grad():
                outputs = models[0](images.to(DEVICE))
                _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        for i, (pred, id) in enumerate(zip(preds, ids)):
            img_ids.append(id.item())
            pred_class = pred.item()
            predictions.append(pred_class)
            conf = probs[i][pred_class].item()
            confidences.append(conf)
    
    results_df = pd.DataFrame({
        'Image': img_ids,
        'Pterygium': predictions,
        'Confidence': confidences
    })

    final_results = pd.DataFrame({
        'Image': img_ids,
        'Pterygium': predictions
    })
    results_df = results_df.sort_values('Image')
    final_results = final_results.sort_values('Image')
    fold_info = args.folds.replace(',', '-')
    final_results.to_excel(f'Classification_Results_fold{fold_info}.xlsx', index=False)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_path = f'pterygium_predictions_fold{fold_info}_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"预测结果已保存至: {results_path}")
    
    detailed_results_path = f'detailed_predictions_fold{fold_info}_{timestamp}.json'
    detailed_results = {
        'predictions': results_df.to_dict(orient='records'),
        'model_info': {
            'model_dir': latest_model_dir,
            'ensemble': args.ensemble,
            'num_models': len(models),
            'use_tta': use_tta,
            'used_folds': args.folds
        },
        'class_names': class_names
    }
    
    with open(detailed_results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    
    print("\n预测类别分布:")
    class_counts = results_df['Pterygium'].value_counts().sort_index()
    for class_id, count in class_counts.items():
        print(f"{class_names[class_id]}: {count} 样本 ({count/len(results_df)*100:.2f}%)")
    
    print("\n置信度统计:")
    print(f"平均置信度: {results_df['Confidence'].mean():.4f}")
    print(f"最小置信度: {results_df['Confidence'].min():.4f}")
    print(f"最大置信度: {results_df['Confidence'].max():.4f}")
    
    print("\n预测完成!")

if __name__ == "__main__":
    main()