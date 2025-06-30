import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pandas as pd
import cv2
from convnextv2 import ConvNeXtSegmentation
from load_data import PterygiumSegmentationDataset
import argparse
import tqdm
from torch.utils.data import Dataset, DataLoader

class PterygiumTestDataset(Dataset):
    def __init__(self, img_paths, img_size=(512, 512)):
        self.img_paths = img_paths
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_id = os.path.basename(img_path).split('.')[0]
        image = Image.open(img_path).convert('RGB')
        orig_size = image.size
        image = self.transform(image)
        
        # 掩码处理
        mask = np.zeros((*self.img_size[::-1], 3), dtype=np.uint8)
        mask = Image.fromarray(mask)
        mask = self.transform(mask)
        mask = mask[0:1]  # 只保留R通道
        mask = (mask > 0.5).float()  # 二值化
        
        return {
            'image': image,
            'mask': mask,
            'id': img_id,
            'orig_size': orig_size
        }

def create_segmentation_image(pred_mask, output_size=None):
    """
    创建符合要求的三通道分割结果图像
    R通道：128表示翼状胬肉区域，0表示其他区域
    G和B通道：全0
    """
    if output_size:
        pred_mask = cv2.resize(pred_mask, (output_size[0], output_size[1]), 
                               interpolation=cv2.INTER_NEAREST)
    
    height, width = pred_mask.shape[:2]
    seg_img = np.zeros((height, width, 3), dtype=np.uint8)
    seg_img[pred_mask > 0, 0] = 128
    
    return seg_img

def ensemble_predictions(predictions_list):
    """
    融合多个模型的预测结果
    """
    if len(predictions_list) == 1:
        return predictions_list[0]
    
    # 对所有预测取平均
    ensemble_preds = {}
    for key in predictions_list[0].keys():
        preds = [pred_dict[key] for pred_dict in predictions_list]
        avg_pred = np.mean(preds, axis=0)
        ensemble_preds[key] = (avg_pred > 0.5).astype(np.uint8)
    
    return ensemble_preds

def main():
    parser = argparse.ArgumentParser(description='翼状胬肉分割预测')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--img_size', type=int, default=512, help='图像大小')
    parser.add_argument('--output_dir', type=str, default='Segmentation_Results', help='输出目录')
    parser.add_argument('--cls_result', type=str, default='./Classification_Results.xlsx', 
                       help='分类结果文件路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    parser.add_argument('--test_dir', type=str, default='../val_img', help='测试图像目录')
    parser.add_argument('--folds', type=str, default='0', help='选择使用哪些折的模型，用逗号分隔，如"0,1,2"')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    selected_folds = [int(fold) for fold in args.folds.split(',')]
    
    base_model_path = args.model_path
    if 'fold' in base_model_path:
        base_model_path = base_model_path[:base_model_path.rfind('fold')]
    
    classification_results = pd.read_excel(args.cls_result)
    
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始批量预测，使用折: {selected_folds}")
    img_paths = []
    class_dict = {}
    
    for _, row in classification_results.iterrows():
        img_id = str(row['Image']).zfill(4)
        img_path = os.path.join(args.test_dir, f"{img_id}.png")
        if os.path.exists(img_path):
            img_paths.append(img_path)
            class_dict[img_id] = row['Pterygium']
    
    test_dataset = PterygiumTestDataset(img_paths, img_size=(args.img_size, args.img_size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    all_fold_predictions = []
    
    for fold in selected_folds:
        if 'fold' in args.model_path:
            model_path = f"{base_model_path}fold{fold}.pth"
        else:
            model_dir = os.path.dirname(args.model_path)
            model_path = os.path.join(model_dir, f"best_model_fold{fold}.pth")

        if not os.path.exists(model_path):
            print(f"警告: 模型文件 {model_path} 不存在，跳过折 {fold}")
            continue
        
        print(f"加载模型: {model_path}")
        model = ConvNeXtSegmentation(
            pretrained=False,
            in_chans=3,
            num_classes=1,
            img_size=args.img_size
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc=f"折 {fold} 预测进度"):
                images = batch['image'].to(device)
                img_ids = batch['id']
                
                # TTA
                outputs = []
                outputs.append(model(images))
                outputs.append(torch.flip(model(torch.flip(images, [3])), [3]))
                outputs.append(torch.flip(model(torch.flip(images, [2])), [2]))
                output = torch.mean(torch.stack(outputs), dim=0)
                
                # 后处理（同验证集）
                preds = torch.sigmoid(output)
                
                for i, img_id in enumerate(img_ids):
                    pred_mask = preds[i].squeeze().cpu().numpy()
                    predictions[img_id] = pred_mask
        
        all_fold_predictions.append(predictions)
    
    if len(all_fold_predictions) > 0:
        final_predictions = ensemble_predictions(all_fold_predictions)
        
        for img_path in img_paths:
            img_id = os.path.basename(img_path).split('.')[0]
            
            if img_id not in final_predictions:
                print(f"警告: 图像 {img_id} 没有找到预测结果，使用全零掩码")
                seg_img = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
            else:
                orig_img = Image.open(img_path)
                original_size = orig_img.size
                
                if class_dict.get(img_id, 0) == 0:
                    seg_img = np.zeros((original_size[1], original_size[0], 3), dtype=np.uint8)
                else:
                    pred_mask = final_predictions[img_id]
                    seg_img = create_segmentation_image(pred_mask, output_size=original_size)
            
            seg_img = Image.fromarray(seg_img)
            seg_img.save(os.path.join(save_dir, f"{img_id}.png"))
        
        print(f"预测完成，使用了 {len(all_fold_predictions)} 个折的模型，结果保存在: {save_dir}")
    else:
        print("没有找到有效的模型，预测失败")

if __name__ == '__main__':
    main()