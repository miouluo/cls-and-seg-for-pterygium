import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from load_data import PterygiumSegmentationDataset, get_transforms

def visualize_data_augmentation(image_path, output_path):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # 原始图
    original_image = Image.open(image_path).convert('RGB')
    train_transform_list = []
    full_transform = get_transforms(512)
    for t in full_transform.transforms:
        if not isinstance(t, transforms.Normalize):
            train_transform_list.append(t)
    
    transform_names = [t.__class__.__name__ for t in train_transform_list]
    transform_for_visualization = transforms.Compose(train_transform_list)
    full_transform_pipeline = get_transforms(512)

    normalized_image_tensor = full_transform_pipeline(original_image)

    # 反归一化
    def unnormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m) # 乘以标准差，加上均值
        return tensor

    # 可视化
    num_augmentations = 7 
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    # 原始图
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 增强
    for i in range(num_augmentations):
        augmented_image_tensor = transform_for_visualization(original_image)
        if not isinstance(augmented_image_tensor, torch.Tensor):
            augmented_image_tensor = transforms.ToTensor()(augmented_image_tensor)

        # Normalize
        augmented_image_tensor = transforms.Normalize(mean, std)(augmented_image_tensor)

        # 反归一化
        display_image_tensor = unnormalize(augmented_image_tensor.clone(), mean, std)
        
        display_image = transforms.ToPILImage()(display_image_tensor)
        
        axes[i+1].imshow(display_image)
        # 变换名为标题
        if i < len(transform_names):
            axes[i+1].set_title(f'{transform_names[i]}')
        else:
            axes[i+1].set_title(f'Augmented: Random Transform {i+1}')
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    plt.close(fig)

if __name__ == '__main__':
    data_root = '..//train'
    label_file_path = '../train_classification_label.xlsx'

    dataset = PterygiumSegmentationDataset(data_root, label_file=label_file_path, transform=None) # 不在这里应用变换
    
    # 可视化第四张图
    if len(dataset) > 3:
        img_id = str(dataset.df.iloc[3]['Image']).zfill(4)
        image_path_to_visualize = os.path.join(data_root, img_id, f"{img_id}.png")
    else:
        print("数据集中的图片数量不足，无法获取第四张图片。")
        exit()

    output_svg_path = './augmented_images.svg'
    visualize_data_augmentation(image_path_to_visualize, output_svg_path)
    print(f"数据增强可视化已保存到: {output_svg_path}")