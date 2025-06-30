# 眼球翼状胬肉分割与诊断挑战赛

本项目使用深度学习方法对眼球翼状胬肉进行分类与分割，运行之前请把目录结构弄成下面的格式，并把官方的数据集放到train/、val_img/、test_img/目录下，注意官方的数据集需要解压，解压之后的目录结构如下：


## 项目结构

```
├── cls_train_convnextv2/          # 分类任务代码
│   ├── train.py                   # 分类模型训练脚本
│   ├── predict.py                 # 分类模型预测脚本
│   ├── convnextv2.py              # 分类模型架构
│   └── load_data.py               # 数据加载
│
├── seg_train_convnextv2/          # 分割任务代码
│   ├── train.py                   # 分割模型训练脚本
│   ├── predict.py                 # 分割模型预测脚本
│   ├── convnextv2.py              # 分割模型验证集使用的模型架构
│   └── convnext_v2.py             # 分割模型测试集使用的模型架构，采样改进，结合反卷积以及双重线性插值
│   └── load_data.py               # 数据加载
│
├── train/                         # 训练图像目录（需要把官方的数据集放到这里）
│   ├── 0001
|   │   ├── 0001.jpg
│   │   ├── 0001_label.png
│   │   ├── 0001_label_viz.png
│   │   ├──...
│   │
│   ├── 0002
│   │   ├── 0002.jpg
│   │   ├── 0002_label.png
│   │   ├── 0002_label_viz.png
│   │   ├──...
│   │
│   ├──...
│
├── val_img/                       # 验证图像目录（需要把官方的数据集放到这里）
│   ├── 0451.jpg
│   ├── 0452.jpg
│   ├── ...
│
├── test_img/                      # 测试图像目录（需要把官方的数据集放到这里）
│   ├── 0601.jpg
│   ├── 0602.jpg
│   ├──...
├── train_classification_label.xlsx # 训练集分类标签
├── requirements.txt               # 项目依赖
├── README.md                      # 项目说明
```

## 环境要求
- Python 3.12.7
- Windows11
- CUDA12.1
- conda 24.7.1
- NVIDIA RTX3060(Driver Version: 556.12, CUDA Version: 12.5)


## 环境安装
```
conda create --name [xx] python=3.12.7
conda activate [xx]
pip install -r requirements.txt
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  --extra-index-url https://download.pytorch.org/whl/cu118
```


## 这是总的思路,看不懂可以去看下面单独复现的过程：

## 训练模型，预测，复现结果

### 分类过程
```
cd cls_train_convnextv2
python train.py
# 验证集预测
python predict.py --img_dir ../val_img --ensemble --folds 1,3 --weighted --weights 0.5,0.5
# 后处理
python ./post_process.py pterygium_predictions_fold1-3_20250503_193149.csv  #请注意这里面生成的文件名，由于时间戳不一样，需要换成实际预测文件，之后运行，文件名带"_submission"为最终结果文件，提交需要重命名为Classification_Results.xlsx
# 测试集预测
python predict.py --img_dir ../test_img --ensemble --folds 1,2,3
# 后处理
python ./post_process.py pterygium_predictions_fold1-2-3_20250522_231310.csv  #同理请注意这里面生成的文件名
```

### 后处理之后的文件需要重命名为Classification_Results.xlsx并放在seg_train_convnextv2目录下面，便于后续分割任务训练，由于分割任务只分割“翼状胬肉建议观察”和“翼状胬肉建议手术”这两列，正常直接通过分类模型预测创建全黑掩码

### 分割过程
```
cd ../
cd seg_train_convnextv2
python train.py
# 把convnext_v2.py里的代码替换成convnextv2.py，再次运行：python train.py
##能够得到两个不同的权重，第一个作为验证集的评估，第二个作为测试集的评估,目前checkpoints保存的两个就是最佳的

# 验证集预测（checkpoints/20250428_020456/是使用seg_train_convnextv2/convnextv2.py为模型架构）
python predict.py --img_dir ../val_img --model_path checkpoints/20250428_020456/ --folds 0,2 --batch_size 64 --img_size 512
# 测试集预测（checkpoints/20250428_020456/是使用seg_train_convnextv2/convnext_v2.py为模型架构）
python predict.py --img_dir ../test_img --model_path checkpoints/20250517_001758/ --folds 0,2,3 --batch_size 64 --img_size 512
```

## 这是验证集和测试集单独复现

### 验证集复现
### 分类过程
```
cd cls_train_convnextv2
python train.py
# 验证集预测
python predict.py --img_dir ../val_img --ensemble --folds 1,3 --weighted --weights 0.5,0.5
# 后处理
python ./post_process.py [pterygium_predictions_fold1-3_20250503_193149.csv]  #请注意这里面生成的文件名，由于时间戳不一样，需要换成实际预测文件，之后运行

### 后处理之后的文件需要重命名为Classification_Results.xlsx并放在seg_train_convnextv2目录下面，便于后续分割任务训练，由于分割任务只分割“翼状胬肉建议观察”和“翼状胬肉建议手术”这两列，正常直接通过分类模型预测创建全黑掩码

cd ../
cd seg_train_convnextv2
python train.py
# 验证集预测（checkpoints/20250428_020456/是使用seg_train_convnextv2/convnextv2.py为模型架构）
python predict.py --img_dir ../val_img --model_path checkpoints/20250428_020456/ --folds 0,2 --batch_size 64 --img_size 512
```

### 测试集复现
```
cd cls_train_convnextv2
# 测试集预测
python predict.py --img_dir ../test_img --ensemble --folds 1,2,3
# 后处理
python ./post_process.py pterygium_predictions_fold1-2-3_20250522_231310.csv  #同理请注意这里面生成的文件名，由于时间戳不一样，需要换成实际预测文件，之后运行

### 后处理之后的文件需要重命名为Classification_Results.xlsx并放在seg_train_convnextv2目录下面，便于后续分割任务训练，由于分割任务只分割“翼状胬肉建议观察”和“翼状胬肉建议手术”这两列，正常直接通过分类模型预测创建全黑掩码

cd ../
cd seg_train_convnextv2
# 把convnext_v2.py里的代码替换成convnextv2.py，之后运行
python train.py
# 测试集预测（checkpoints/20250428_020456/是使用seg_train_convnextv2/convnext_v2.py为模型架构）
python predict.py --img_dir ../test_img --model_path checkpoints/20250517_001758/ --folds 0,2,3 --batch_size 64 --img_size 512
```


## 不明白之处可以联系一下我，wx: Crunk_crunk_crunk
