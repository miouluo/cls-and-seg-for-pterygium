import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import json
import argparse
from convnextv2 import *
from torch.autograd import Variable
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_model_target_layers(model):
    """获取模型的目标层"""
    target_layers = [model.base_model.stages[-1]] 
    return target_layers

def load_model(model_path, device):
    """加载模型并移动到指定设备"""
    print(f"加载模型: {model_path}")
    model = torch.load(model_path, map_location=device) 
    model.eval()
    return model

def ensemble_predict(models, input_tensor, device):
    """使用模型集成进行预测"""
    all_outputs = []
    for model in models:
        with torch.no_grad():
            output = model(input_tensor.to(device))
            all_outputs.append(output)
    
    # 平均输出
    ensemble_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    _, preds = torch.max(ensemble_outputs, 1)
    
    return preds.item(), ensemble_outputs

def get_ensemble_cam(models, input_tensor, predicted_class, cam_method, rgb_img_np, device):
    """获取所有模型的CAM并融合"""
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM
    }
    
    all_grayscale_cams = []
    
    for model in models:
        target_layers = get_model_target_layers(model)
        cam_algorithm = methods[cam_method]
        
        with cam_algorithm(model=model, target_layers=target_layers) as cam:
            targets = [ClassifierOutputTarget(predicted_class)]
            grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)
            all_grayscale_cams.append(grayscale_cam[0, :])
    
    # 平均CAM
    avg_grayscale_cam = np.mean(all_grayscale_cams, axis=0)
    cam_image = show_cam_on_image(rgb_img_np, avg_grayscale_cam, use_rgb=True)
    
    return avg_grayscale_cam, cam_image

def visualization(models, img_path, save_path, cam_method='gradcam++', class_names=None, device=None, ensemble=True):
    """使用CAM方法可视化模型的注意力区域"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rgb_img = Image.open(img_path).convert('RGB')
    rgb_img = rgb_img.resize((224, 224))
    rgb_img_np = np.array(rgb_img) / 255.0
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(rgb_img).unsqueeze(0)
    
    # 预测
    if ensemble and len(models) > 1:
        predicted_class, outputs = ensemble_predict(models, input_tensor, device)
        grayscale_cam, cam_image = get_ensemble_cam(models, input_tensor, predicted_class, cam_method, rgb_img_np, device)
    else:
        model = models[0]
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            predicted_class = pred.item()
        
        # CAM
        methods = {
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM
        }
        
        target_layers = get_model_target_layers(model)
        cam_algorithm = methods[cam_method]
        
        with cam_algorithm(model=model, target_layers=target_layers) as cam:
            targets = [ClassifierOutputTarget(predicted_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img_np, grayscale_cam, use_rgb=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img_np)
    plt.title("原始图像")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cam_image)
    plt.title(f"CAM: {cam_method}" + (" (集成)" if ensemble and len(models) > 1 else ""))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(grayscale_cam, cmap='jet')
    plt.title("热力图")
    plt.axis('off')
    
    if class_names:
        plt.suptitle(f"预测类别: {class_names[predicted_class]} (类别 {predicted_class})", fontsize=14)
    else:
        plt.suptitle(f"预测类别: {predicted_class}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return predicted_class, cam_image

def main():
    parser = argparse.ArgumentParser(description='眼球翼状胬肉模型可视化')
    parser.add_argument('--model_dir', type=str, default='checkpoints', help='包含模型的目录')
    parser.add_argument('--test_dir', type=str, default='../val_img', help='测试图像目录')
    parser.add_argument('--ensemble', action='store_true', default=True, help='是否使用模型集成')
    parser.add_argument('--cam_method', type=str, default='gradcam++', choices=['gradcam', 'scorecam', 'gradcam++', 'xgradcam', 'eigencam'], help='CAM方法')
    parser.add_argument('--num_samples', type=int, default=10, help='要可视化的样本数量')
    args = parser.parse_args()

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
    

    models = []

    if args.ensemble:
        folds_dir = os.path.join(latest_model_dir, 'checkpoints')
        fold_dirs = [d for d in os.listdir(folds_dir) if d.startswith('fold_')]

        selected_folds = ['fold_1', 'fold_2', 'fold_3']
        selected_folds = [f for f in selected_folds if f in fold_dirs]  

        for fold_dir in selected_folds:
            model_path = os.path.join(folds_dir, fold_dir, 'best.pth')
            if os.path.exists(model_path):
                models.append(load_model(model_path, DEVICE))
            else:
                print(f"警告：未找到模型文件 {model_path}，跳过该折。")
        
        if not models:
            print("错误: 找不到任何模型文件!")
            return
        
        print(f"已加载 {len(models)} 个模型进行集成可视化")
    else:
        fold_dir = os.path.join(latest_model_dir, 'checkpoints', 'fold_1')
        model_path = os.path.join(fold_dir, 'best.pth')
        
        if not os.path.exists(model_path):
            model_path = os.path.join(fold_dir, 'best_acc.pth')
            if not os.path.exists(model_path):
                print(f"错误: 模型文件 {model_path} 不存在!")
                return
        
        model = load_model(model_path, DEVICE)
        models = [model]
    
    visualization_dir = os.path.join(latest_model_dir, 'visualization_results')
    os.makedirs(visualization_dir, exist_ok=True)
    
    test_dir = args.test_dir

    sample_images = sorted(os.listdir(test_dir))[:args.num_samples]
    
    for img_name in sample_images:
        img_path = os.path.join(test_dir, img_name)
        save_path = os.path.join(visualization_dir, f"cam_{os.path.splitext(img_name)[0]}.png")
        
        print(f"处理图像: {img_path}")
        predicted_class, _ = visualization(
            models, 
            img_path, 
            save_path, 
            cam_method=args.cam_method,
            class_names=class_names,
            device=DEVICE,
            ensemble=args.ensemble
        )
        print(f"图像 {img_name} 被预测为: {class_names[predicted_class]} (类别 {predicted_class})")
    
    print(f"可视化结果已保存到: {visualization_dir}")

if __name__ == "__main__":
    main()