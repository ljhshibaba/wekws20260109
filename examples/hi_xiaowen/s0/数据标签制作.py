import os
import json
import random
from pathlib import Path

def split_and_generate_labels(
    balanced_folder,
    output_label_folder,
    split_ratio=(1, 1, 8),  # dev:test:train
    positive_prefixes=["base", "aug"],
    audio_formats=(".wav", ".flac", ".mp3", ".m4a"),
    random_seed=42
):
    """
    按比例划分平衡数据集，并生成正/副样本的dev/test/train标签文件
    
    :param balanced_folder: 平衡后的数据集文件夹（正样本+抽取的副样本）
    :param output_label_folder: 标签文件输出文件夹
    :param split_ratio: dev:test:train 比例
    :param positive_prefixes: 正样本前缀
    :param audio_formats: 支持的音频格式
    :param random_seed: 随机种子（保证划分结果可复现）
    """
    # 1. 初始化随机种子
    random.seed(random_seed)
    
    # 2. 遍历文件夹，分离正/副样本（提取文件名，去掉后缀）
    positive_samples = []  # 正样本文件名（无后缀）
    negative_samples = []  # 副样本文件名（无后缀）
    
    print(f"正在遍历平衡数据集文件夹: {balanced_folder}")
    for root, dirs, files in os.walk(balanced_folder):
        for file in files:
            if file.lower().endswith(audio_formats):
                # 提取纯文件名（去掉后缀）
                file_name = os.path.splitext(file)[0]
                # 判断正/副样本
                if any(file.startswith(prefix) for prefix in positive_prefixes):
                    positive_samples.append(file_name)
                else:
                    negative_samples.append(file_name)
    
    # 3. 输出基础统计
    pos_count = len(positive_samples)
    neg_count = len(negative_samples)
    print(f"\n📊 平衡数据集统计:")
    print(f"   正样本数量: {pos_count}")
    print(f"   副样本数量: {neg_count}")
    
    # 校验正副样本数量是否相等（平衡数据集要求）
    if pos_count != neg_count:
        print(f"⚠️  警告：正副样本数量不一致（正{pos_count}/副{neg_count}），仍将按比例划分")
    
    # 4. 计算各数据集划分数量
    total_ratio = sum(split_ratio)
    dev_ratio, test_ratio, train_ratio = split_ratio
    
    # 正样本各集数量
    p_dev_num = int(pos_count * dev_ratio / total_ratio)
    p_test_num = int(pos_count * test_ratio / total_ratio)
    p_train_num = pos_count - p_dev_num - p_test_num
    
    # 副样本各集数量（与正样本同比例）
    n_dev_num = int(neg_count * dev_ratio / total_ratio)
    n_test_num = int(neg_count * test_ratio / total_ratio)
    n_train_num = neg_count - n_dev_num - n_test_num
    
    print(f"\n📝 划分比例 (dev:test:train = {dev_ratio}:{test_ratio}:{train_ratio}):")
    print(f"   正样本 - dev: {p_dev_num} | test: {p_test_num} | train: {p_train_num}")
    print(f"   副样本 - dev: {n_dev_num} | test: {n_test_num} | train: {n_train_num}")
    
    # 5. 随机打乱并划分数据
    # 正样本划分
    random.shuffle(positive_samples)
    p_dev = positive_samples[:p_dev_num]
    p_test = positive_samples[p_dev_num:p_dev_num+p_test_num]
    p_train = positive_samples[p_dev_num+p_test_num:]
    
    # 副样本划分
    random.shuffle(negative_samples)
    n_dev = negative_samples[:n_dev_num]
    n_test = negative_samples[n_dev_num:n_dev_num+n_test_num]
    n_train = negative_samples[n_dev_num+n_test_num:]
    
    # 6. 构建标签数据结构
    def build_label_list(sample_list, keyword_id):
        """生成标签列表（统一格式）"""
        label_list = []
        for utt_id in sample_list:
            label_list.append({
                "utt_id": utt_id,
                "speaker_id": utt_id,
                "keyword_id": keyword_id
            })
        return label_list
    
    # 正样本标签（keyword_id=0）
    p_dev_labels = build_label_list(p_dev, 0)
    p_test_labels = build_label_list(p_test, 0)
    p_train_labels = build_label_list(p_train, 0)
    
    # 副样本标签（keyword_id=-1）
    n_dev_labels = build_label_list(n_dev, -1)
    n_test_labels = build_label_list(n_test, -1)
    n_train_labels = build_label_list(n_train, -1)
    
    # 7. 创建输出文件夹
    os.makedirs(output_label_folder, exist_ok=True)
    
    # 8. 保存JSON文件（格式化输出，便于阅读）
    label_files = {
        "p_dev.json": p_dev_labels,
        "p_test.json": p_test_labels,
        "p_train.json": p_train_labels,
        "n_dev.json": n_dev_labels,
        "n_test.json": n_test_labels,
        "n_train.json": n_train_labels
    }
    
    for file_name, data in label_files.items():
        file_path = os.path.join(output_label_folder, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"✅ 标签文件已保存: {file_path} (共{len(data)}条数据)")
    
    # 9. 输出最终统计
    print("\n" + "="*60)
    print(f"🎉 标签文件生成完成！")
    print(f"   标签输出目录: {output_label_folder}")
    print(f"   总正样本标签数: {len(p_dev_labels)+len(p_test_labels)+len(p_train_labels)}")
    print(f"   总副样本标签数: {len(n_dev_labels)+len(n_test_labels)+len(n_train_labels)}")
    print("="*60)

# ==================== 运行示例 ====================
if __name__ == "__main__":
    # 配置参数（请根据实际路径修改）
    BALANCED_FOLDER = "./balanced_audio_dataset"  # 平衡后的数据集文件夹
    OUTPUT_LABEL_FOLDER = "./balanced_audio_label_files"  # 标签文件输出文件夹
    SPLIT_RATIO = (1, 1, 8)  # dev:test:train = 1:1:8
    RANDOM_SEED = 42  # 固定种子，保证划分结果可复现
    
    # 生成标签文件
    split_and_generate_labels(
        balanced_folder=BALANCED_FOLDER,
        output_label_folder=OUTPUT_LABEL_FOLDER,
        split_ratio=SPLIT_RATIO,
        random_seed=RANDOM_SEED
    )