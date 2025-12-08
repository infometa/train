"""

import os
import argparse
import numpy as np
import soundfile as sf
import onnxruntime as ort
import time

# 尝试导入高质量重采样库
try:
    import soxr
    SOXR_AVAILABLE = True
except ImportError:
    import librosa
    SOXR_AVAILABLE = False
    print("Warning: soxr not found, falling back to librosa for resampling.")

def load_and_preprocess(path, target_sr=48000):
    """加载音频并预处理：转单声道 -> 重采样 -> 归一化"""
    try:
        audio, sr = sf.read(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

    # 1. 转单声道
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 2. 重采样
    if sr != target_sr:
        if SOXR_AVAILABLE:
            audio = soxr.resample(audio, sr, target_sr, quality="VHQ")
        else:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # 3. 简单归一化 (防止削波，保留动态)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.99
        
    return audio.astype(np.float32), target_sr

def run_inference(model_path, audio_data):
    """运行 ONNX 推理"""
    # 创建推理会话
    # 注意：如果有 GPU，可以把 providers 改为 ['CUDAExecutionProvider']
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    
    input_name = input_meta[0].name
    
    # 检查是否为流式模型（带状态输入）
    if len(input_meta) > 1:
        print("❌ 错误：此脚本仅支持【非流式】模型（一次性处理全长音频）。")
        print("   请使用 export_onnx.py 不带 --streaming 参数导出模型。")
        return None

    # 准备输入 Tensor: [1, 1, T]
    input_tensor = audio_data[np.newaxis, np.newaxis, :]
    
    print(f"Running inference on {model_path}...")
    start_time = time.time()
    
    # 推理
    outputs = session.run(None, {input_name: input_tensor})
    output_tensor = outputs[0]
    
    elapsed = time.time() - start_time
    rtf = elapsed / (len(audio_data) / 48000)
    print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
    
    # 移除 Batch 和 Channel 维度: [1, 1, T] -> [T]
    return output_tensor.squeeze()

def main():
    parser = argparse.ArgumentParser(description="DeepFilterGAN 音色修复离线推理工具")
    parser.add_argument("-m", "--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入音频路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出音频路径 (默认: input_restored.wav)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    # 确定输出路径
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        model_name = Path(args.model).stem
        args.output = f"{base}_{model_name}.wav"

    # 1. 加载
    print(f"Loading {args.input} ...")
    audio, sr = load_and_preprocess(args.input)
    if audio is None: return

    # 2. 推理
    processed_audio = run_inference(args.model, audio)
    if processed_audio is None: return

    # 3. 保存
    sf.write(args.output, processed_audio, 48000)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    from pathlib import Path
    main()