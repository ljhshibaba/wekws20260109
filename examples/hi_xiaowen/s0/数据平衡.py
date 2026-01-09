import os
import random
import shutil
from pathlib import Path

def balance_samples(
    input_folder,
    output_folder,
    positive_prefixes=["base", "aug"],  # 正样本前缀
    audio_formats=(".wav", ".flac", ".mp3", ".m4a"),  # 支持的音频格式
    random_seed=42  # 固定随机种子，保证每次抽取结果一致
):
    """
    平衡样本：保留所有正样本 + 随机抽取等量副样本，保存到新文件夹
    
    :param input_folder: 原始音频文件夹路径
    :param output_folder: 平衡后样本保存路径
    :param positive_prefixes: 正样本文件名前缀列表
    :param audio_formats: 支持的音频文件格式
    :param random_seed: 随机种子（保证可复现）
    """
    # 1. 初始化随机种子
    random.seed(random_seed)
    
    # 2. 遍历文件夹，分离正样本和副样本
    positive_files = []  # 存储所有正样本路径
    negative_files = []  # 存储所有副样本路径
    
    print(f"正在遍历文件夹: {input_folder}")
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 只处理音频文件
            if file.lower().endswith(audio_formats):
                file_path = os.path.join(root, file)
                # 判断是否为正样本（以指定前缀开头）
                if any(file.startswith(prefix) for prefix in positive_prefixes):
                    positive_files.append(file_path)
                else:
                    negative_files.append(file_path)
    
    # 3. 输出统计信息
    pos_count = len(positive_files)
    neg_count = len(negative_files)
    print(f"\n📊 原始样本统计:")
    print(f"   正样本数量 (base/aug开头): {pos_count}")
    print(f"   副样本数量: {neg_count}")
    
    # 4. 校验副样本数量是否足够
    if neg_count < pos_count:
        raise ValueError(f"副样本数量({neg_count})小于正样本数量({pos_count})，无法满足1:1比例！")
    
    # 5. 随机抽取与正样本等量的副样本
    sampled_negative_files = random.sample(negative_files, pos_count)
    print(f"\n🎯 随机抽取 {len(sampled_negative_files)} 条副样本（与正样本1:1）")
    
    # 6. 合并需要保留的文件（所有正样本 + 抽取的副样本）
    selected_files = positive_files + sampled_negative_files
    print(f"✅ 最终保留样本总数: {len(selected_files)} (正样本{pos_count} + 副样本{len(sampled_negative_files)})")
    
    # 7. 创建输出文件夹（清空原有内容，避免重复）
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    # 8. 复制选中的文件到输出文件夹（保留原目录结构）
    copied_count = 0
    for file_path in selected_files:
        try:
            # 计算相对路径，保持原文件夹结构
            relative_path = os.path.relpath(file_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            
            # 创建输出子目录
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 复制文件
            shutil.copy2(file_path, output_path)  # copy2 保留文件元信息
            copied_count += 1
            
            # 每复制1000个文件输出进度
            if copied_count % 1000 == 0:
                print(f"📤 已复制 {copied_count}/{len(selected_files)} 个文件")
                
        except Exception as e:
            print(f"❌ 复制文件失败 {file_path}: {str(e)}")
    
    # 9. 输出最终结果
    print("\n" + "="*60)
    print(f"🎉 样本平衡完成！")
    print(f"   输出文件夹: {output_folder}")
    print(f"   成功复制文件数: {copied_count}")
    print(f"   正样本数: {pos_count} | 抽取副样本数: {len(sampled_negative_files)}")
    print("="*60)

# ==================== 运行示例 ====================
if __name__ == "__main__":
    # 配置参数（请根据你的实际路径修改）
    INPUT_FOLDER = "/root/wekws/examples/hi_xiaowen/s0/data/mobvoi_hotword_dataset"       # 原始音频文件夹（包含2W副样本+4K正样本）
    OUTPUT_FOLDER = "./balanced_audio_dataset"   # 平衡后的样本保存路径
    RANDOM_SEED = 42                     # 固定种子，保证每次抽取结果一致
    
    # 执行样本平衡
    balance_samples(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        positive_prefixes=["base", "aug"],
        random_seed=RANDOM_SEED
    )
