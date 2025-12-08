#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path  # ✅ 修复1：移到顶部

# 尝试导入高质量重采样库
try:
    import soxr
    SOXR_AVAILABLE = True
except ImportError:
    SOXR_AVAILABLE = False
    try:
        import librosa
    except ImportError:
        print("❌ Error: 既没找到 'soxr' 也没找到 'librosa'。请至少安装其中一个：")
        print("   pip install soxr  (推荐)")
        print("   pip install librosa")
        exit(1) # ✅ 修复3：优雅退出
    print("Warning: soxr not found, falling back to librosa for resampling.")

def load_and_preprocess(path, target_sr=48000):
    try:
        audio, sr = sf.read(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        if SOXR_AVAILABLE:
            audio = soxr.resample(audio, sr, target_sr, quality="VHQ")
        else:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # ⚠️ 注意：如果音频本身很小声，这里不会归一化放大，符合预期
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak * 0.99
        
    return audio.astype(np.float32), target_sr

def run_inference(model_path, audio_data):
    # 根据设备选择 Provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_meta = session.get_inputs()
    if len(input_meta) > 1:
        print("❌ 错误：此脚本仅支持【非流式】模型。")
        return None

    input_name = input_meta[0].name
    
    print(f"Running inference on {model_path}...")
    start_time = time.time()

    # ✅ 修复2：简易切片处理（防止爆内存）
    # 如果音频超过 30秒，建议切片。这里为了简单演示，仅做提示，
    # 真正的切片需要处理重叠(Overlap)以避免拼接处有咔哒声。
    if len(audio_data) > 48000 * 30: 
        print("⚠️ 警告：音频过长 (>30s)，一次性推理可能耗尽内存。建议使用流式模型或切片处理。")

    input_tensor = audio_data[np.newaxis, np.newaxis, :]
    
    outputs = session.run(None, {input_name: input_tensor})
    output_tensor = outputs[0]
    
    elapsed = time.time() - start_time
    rtf = elapsed / (len(audio_data) / 48000)
    print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
    
    return output_tensor.squeeze()

def main():
    parser = argparse.ArgumentParser(description="DeepFilterGAN 音色修复离线推理工具")
    parser.add_argument("-m", "--model", type=str, required=True, help="ONNX 模型路径")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入音频路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出路径")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    # 生成默认输出文件名
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        model_name = Path(args.model).stem
        args.output = f"{base}_{model_name}.wav"

    print(f"Loading {args.input} ...")
    audio, sr = load_and_preprocess(args.input)
    if audio is None: return

    processed_audio = run_inference(args.model, audio)
    if processed_audio is None: return

    sf.write(args.output, processed_audio, 48000)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()