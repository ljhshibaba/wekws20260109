import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os
from pathlib import Path

def add_noise_to_audio(
    input_path,
    output_path,
    noise_type="impulse",  # white/pink/impulse
    noise_level=0.02,    # 噪音强度，值越大噪音越明显
    seed=42              # 随机种子，保证结果可复现
):
    """
    给音频添加指定类型的噪音
    :param input_path: 输入音频文件路径
    :param output_path: 输出带噪音音频路径
    :param noise_type: 噪音类型
    :param noise_level: 噪音强度系数
    :param seed: 随机种子
    """
    try:
        # 1. 读取音频
        y, sr = librosa.load(input_path, sr=None)  # sr=None 保持原采样率
        np.random.seed(seed)

        # 2. 生成对应类型的噪音
        if noise_type == "white":
            # 白噪音：所有频率分量强度相同
            noise = np.random.normal(0, 1, len(y))
        elif noise_type == "pink":
            # 粉红噪音：低频能量更高，更接近环境噪音
            noise = np.random.normal(0, 1, len(y))
            # 频域处理实现粉红噪音特性
            noise_fft = np.fft.rfft(noise)
            freq = np.fft.rfftfreq(len(y), 1/sr)
            freq[freq == 0] = 1e-8  # 避免除零
            noise_fft /= np.sqrt(freq)
            noise = np.fft.irfft(noise_fft)
        elif noise_type == "impulse":
            # 脉冲噪音：随机出现的尖锐噪声（模拟电流声/爆音）
            noise = np.zeros(len(y))
            num_impulses = int(len(y) * 0.001)  # 脉冲数量
            impulse_pos = np.random.randint(0, len(y), num_impulses)
            noise[impulse_pos] = np.random.normal(0, 5, num_impulses)
        else:
            raise ValueError(f"不支持的噪音类型: {noise_type}")

        # 3. 归一化噪音并叠加到原音频
        noise = noise / np.max(np.abs(noise))  # 噪音归一化到 [-1,1]
        noisy_y = y + noise_level * noise

        # 4. 防止音频过载（幅值超过1会失真）
        noisy_y = np.clip(noisy_y, -1.0, 1.0)

        # 5. 确保输出文件夹存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 6. 保存带噪音的音频
        sf.write(output_path, noisy_y, sr)
        print(f"✅ 带噪音的音频已保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理文件 {input_path} 失败: {str(e)}")
        return False

def batch_process_audio_folder(
    input_folder,
    output_folder,
    noise_type="white",
    noise_level=0.02,
    seed=42
):
    """
    批量处理文件夹中以base/aug开头的音频文件
    :param input_folder: 输入音频文件夹路径
    :param output_folder: 输出音频文件夹路径
    :param noise_type: 噪音类型
    :param noise_level: 噪音强度
    :param seed: 随机种子
    """
    # 支持的音频格式
    supported_formats = (".wav", ".flac", ".mp3", ".m4a")
    
    # 遍历文件夹中的所有文件
    processed_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 筛选条件：1. 是音频文件 2. 文件名以base或aug开头
            if file.lower().endswith(supported_formats) and (file.startswith("base") or file.startswith("aug")):
                # 构建输入文件路径
                input_path = os.path.join(root, file)
                
                # 构建输出文件路径（保持原文件夹结构）
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                # 给输出文件名添加噪音类型后缀（方便区分）
                filename, ext = os.path.splitext(output_path)
                output_path = f"{filename}_{noise_type}_noise{noise_level}{ext}"
                
                # 处理当前文件
                if add_noise_to_audio(input_path, output_path, noise_type, noise_level, seed):
                    processed_count += 1
            else:
                skipped_count += 1
    
    # 输出处理统计
    print("\n" + "="*50)
    print(f"处理完成！总计处理: {processed_count} 个文件，跳过: {skipped_count} 个文件")
    print("="*50)

# ==================== 运行示例 ====================
if __name__ == "__main__":
    # 配置参数（请根据你的实际路径修改）
    INPUT_FOLDER = "/root/wekws/examples/hi_xiaowen/s0/data/mobvoi_hotword_dataset"  # 你的音频文件夹路径
    OUTPUT_FOLDER = "/root/wekws/examples/hi_xiaowen/s0/data/mobvoi_hotword_dataset"  # 增强后的音频保存路径
    NOISE_TYPE = "white"  # 可选：white/pink/impulse
    NOISE_LEVEL = 0.02    # 噪音强度，建议0.01~0.1
    
    # 批量处理
    batch_process_audio_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        noise_type=NOISE_TYPE,
        noise_level=NOISE_LEVEL
    )