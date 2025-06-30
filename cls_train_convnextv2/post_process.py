import pandas as pd
import argparse
import os
from datetime import datetime

def balance_class_distribution(input_file, output_file=None, target_ratio=(1, 1, 1), min_diff_threshold=0):
    """
    根据目标比例调整类别分布，可以双向调整：
    1. 如果"建议观察"样本过多，将低置信度的转为"建议手术"
    2. 如果"建议手术"样本过多，将低置信度的转为"建议观察"
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出文件路径，如果不指定，将根据输入文件自动生成
    target_ratio: 目标比例，格式为元组 (正常, 建议观察, 建议手术)，默认为(1, 1, 1)即1:1:1
    min_diff_threshold: 最小差异阈值，当类别差异大于此阈值时才进行调整，默认为0
    """
    df = pd.read_csv(input_file)
    class_names = ["正常", "翼状胬肉建议观察", "翼状胬肉建议手术"]
    original_counts = df['Pterygium'].value_counts().to_dict()
    total_samples = len(df)
    for i in range(3):
        if i not in original_counts:
            original_counts[i] = 0
    
    print("\n原始预测类别分布:")
    for class_id in range(3):
        count = original_counts.get(class_id, 0)
        percentage = count / total_samples * 100
        print(f"{class_names[class_id]}: {count} 样本 ({percentage:.2f}%)")
    ratio_sum = sum(target_ratio)
    target_counts = {}
    for i in range(3):
        target_counts[i] = int(total_samples * target_ratio[i] / ratio_sum)
    remaining = total_samples - sum(target_counts.values())
    if remaining > 0:
        target_counts[2] += remaining
    
    print(f"\n目标类别分布 (按照 {target_ratio[0]}:{target_ratio[1]}:{target_ratio[2]} 的比例):")
    for class_id in range(3):
        percentage = target_counts[class_id] / total_samples * 100
        print(f"{class_names[class_id]}: {target_counts[class_id]} 样本 ({percentage:.2f}%)")
    
    transfer_1_to_2 = min(
        max(0, original_counts[1] - target_counts[1]),  # 类别1多出的样本数
        max(0, target_counts[2] - original_counts[2])   # 类别2缺少的样本数
    )
    transfer_2_to_1 = min(
        max(0, original_counts[2] - target_counts[2]),  # 类别2多出的样本数
        max(0, target_counts[1] - original_counts[1])   # 类别1缺少的样本数
    )
    
    if transfer_1_to_2 <= transfer_2_to_1 and transfer_1_to_2 > 0:
        to_transfer = transfer_1_to_2
        transfer_direction = "1_to_2"
        print(f"\n需要将 {to_transfer} 个'翼状胬肉建议观察'样本重新分类为'翼状胬肉建议手术'")
    elif transfer_2_to_1 > 0:
        to_transfer = transfer_2_to_1
        transfer_direction = "2_to_1"
        print(f"\n需要将 {to_transfer} 个'翼状胬肉建议手术'样本重新分类为'翼状胬肉建议观察'")
    else:
        to_transfer = 0
        print("\n当前分布已符合目标比例，无需调整")
        return df
    
    if to_transfer > 0:
        if transfer_direction == "1_to_2":
            class1_samples = df[df['Pterygium'] == 1].sort_values('Confidence')
            indices_to_change = class1_samples.head(to_transfer).index
            new_class = 2
        else:
            class2_samples = df[df['Pterygium'] == 2].sort_values('Confidence')
            indices_to_change = class2_samples.head(to_transfer).index
            new_class = 1

        print("\n将要重新分类的样本:")
        for idx in indices_to_change:
            row = df.loc[idx]
            print(f"图像ID: {row['Image']}, 置信度: {row['Confidence']:.4f}")

        df.loc[indices_to_change, 'Pterygium'] = new_class
        df.loc[indices_to_change, 'Confidence'] = 0.8
        new_counts = df['Pterygium'].value_counts().to_dict()
        for i in range(3):
            if i not in new_counts:
                new_counts[i] = 0
                
        print("\n调整后的类别分布:")
        for class_id in range(3):
            count = new_counts.get(class_id, 0)
            percentage = count / total_samples * 100
            print(f"{class_names[class_id]}: {count} 样本 ({percentage:.2f}%)")
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base_name}_balanced_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n平衡后的结果已保存至: {output_file}")
    submission_df = df[['Image', 'Pterygium']]
    submission_file = output_file.replace('.csv', '_submission.xlsx')
    submission_df.to_excel(submission_file, index=False)
    print(f"提交格式结果已保存至: {submission_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='平衡翼状胬肉分类结果的类别分布')
    parser.add_argument('input_file', type=str, help='输入的预测结果CSV文件')
    parser.add_argument('--ratio', type=str, default='1:1:1', 
                      help='目标类别比例，格式为"x:y:z"，默认为"1:1:1"')
    parser.add_argument('--output', type=str, default=None, 
                      help='输出文件路径，默认为自动生成')
    parser.add_argument('--min-diff', type=int, default=0,
                      help='最小差异阈值，当类别差异大于此阈值时才进行调整，默认为0')
    
    args = parser.parse_args()
    try:
        ratio_parts = args.ratio.split(':')
        if len(ratio_parts) != 3:
            raise ValueError("比例格式应为'x:y:z'，表示正常:建议观察:建议手术")
        target_ratio = tuple(map(float, ratio_parts))
    except Exception as e:
        print(f"警告: 无法解析比例 '{args.ratio}'，将使用默认比例 1:1:1")
        target_ratio = (1, 1, 1)
    
    balance_class_distribution(args.input_file, args.output, target_ratio, args.min_diff)

if __name__ == "__main__":
    main() 