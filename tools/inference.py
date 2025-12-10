#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path
import torch
import yaml

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

def load_pt_model(ckpt_path: str, config_path: str):
    """加载 PyTorch checkpoint（仅支持 generator）"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    g_cfg = cfg["model"]["generator"]
    model = __import__("model.generator", fromlist=["CausalUNetGenerator"]).CausalUNetGenerator(
        in_channels=g_cfg["in_channels"],
        out_channels=g_cfg["out_channels"],
        channels=g_cfg["channels"],
        kernel_size=g_cfg["kernel_size"],
        bottleneck_type=g_cfg["bottleneck_type"],
        bottleneck_layers=g_cfg["bottleneck_layers"],
        use_weight_norm=g_cfg["use_weight_norm"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("generator", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.remove_weight_norm()
    model.eval()
    return model


def run_inference(model_path, audio_data, frame_size=480, hidden_size=None, num_layers=None, config_path=None):
    # 根据设备选择 Provider
    if model_path.lower().endswith(".onnx"):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()
        has_state = any(i.name == 'h_in' for i in inputs)

        if has_state:
            if hidden_size is None or num_layers is None:
                print("❌ 流式模型需要指定 --hidden-size 和 --num-layers")
                return None
            input_name = 'input'
            h_name = 'h_in'
        else:
            input_name = inputs[0].name
            h_name = None
        
        print(f"Running inference on {model_path}... (streaming={has_state})")
        start_time = time.time()

        if not has_state:
            if len(audio_data) > 48000 * 30:
                print("⚠️ 警告：音频过长 (>30s)，一次性推理可能耗尽内存。建议使用流式模型或切片处理。")
            input_tensor = audio_data[np.newaxis, np.newaxis, :]
            outputs = session.run(None, {input_name: input_tensor})
            output_tensor = outputs[0].squeeze()
            elapsed = time.time() - start_time
            rtf = elapsed / (len(audio_data) / 48000)
            print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
            return output_tensor

        # 流式推理
        h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
        out = np.zeros_like(audio_data)
        n = len(audio_data)
        idx = 0
        while idx < n:
            end = min(idx + frame_size, n)
            frame = audio_data[idx:end]
            # pad 到固定长度
            if len(frame) < frame_size:
                pad = np.zeros(frame_size, dtype=np.float32)
                pad[:len(frame)] = frame
                frame = pad
            inp = frame[np.newaxis, np.newaxis, :]
            feed = {input_name: inp, h_name: h}
            outputs = session.run(None, feed)
            y = outputs[0][0,0,:len(frame)]
            h = outputs[1]
            out[idx:end] = y[:end-idx]
            idx = end

        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio_data) / 48000)
        print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
        return out

    # ===== PyTorch .pt 推理 =====
    if not model_path.lower().endswith(".pt"):
        print("❌ 不支持的模型格式，仅支持 .onnx 或 .pt")
        return None
    if config_path is None:
        print("❌ 使用 .pt 模型时需要提供 --config 指向训练配置")
        return None
    model = load_pt_model(model_path, config_path)
    device = torch.device("cpu")
    model.to(device)
    print(f"Running PyTorch inference on {model_path}...")
    start_time = time.time()
    with torch.no_grad():
        x = torch.from_numpy(audio_data)[None, None, :].to(device)
        y = model(x).cpu().numpy()[0,0]
    elapsed = time.time() - start_time
    rtf = elapsed / (len(audio_data) / 48000)
    print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
    return y
            return None
        input_name = 'input'
        h_name = 'h_in'
    else:
        input_name = inputs[0].name
        h_name = None
    
    print(f"Running inference on {model_path}... (streaming={has_state})")
    start_time = time.time()

    if not has_state:
        if len(audio_data) > 48000 * 30:
            print("⚠️ 警告：音频过长 (>30s)，一次性推理可能耗尽内存。建议使用流式模型或切片处理。")
        input_tensor = audio_data[np.newaxis, np.newaxis, :]
        outputs = session.run(None, {input_name: input_tensor})
        output_tensor = outputs[0].squeeze()
        elapsed = time.time() - start_time
        rtf = elapsed / (len(audio_data) / 48000)
        print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
        return output_tensor

    # 流式推理
    h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    out = np.zeros_like(audio_data)
    n = len(audio_data)
    idx = 0
    while idx < n:
        end = min(idx + frame_size, n)
        frame = audio_data[idx:end]
        # pad 到固定长度
        if len(frame) < frame_size:
            pad = np.zeros(frame_size, dtype=np.float32)
            pad[:len(frame)] = frame
            frame = pad
        inp = frame[np.newaxis, np.newaxis, :]
        feed = {input_name: inp, h_name: h}
        outputs = session.run(None, feed)
        y = outputs[0][0,0,:len(frame)]
        h = outputs[1]
        out[idx:end] = y[:end-idx]
        idx = end

    elapsed = time.time() - start_time
    rtf = elapsed / (len(audio_data) / 48000)
    print(f"  Time: {elapsed:.3f}s | RTF: {rtf:.3f}x")
    return out

def main():
    parser = argparse.ArgumentParser(description="DeepFilterGAN 音色修复离线推理工具")
    parser.add_argument("-m", "--model", type=str, required=True, help="ONNX 或 .pt 模型路径")
    parser.add_argument("-i", "--input", type=str, required=True, help="输入音频路径")
    parser.add_argument("-o", "--output", type=str, default=None, help="输出路径")
    parser.add_argument("--frame-size", type=int, default=480, help="流式帧长")
    parser.add_argument("--hidden-size", type=int, default=None, help="流式模型隐藏状态维度")
    parser.add_argument("--num-layers", type=int, default=None, help="流式模型层数")
    parser.add_argument("--config", type=str, default="configs/lightweight.yaml", help=".pt 模型对应的训练配置")
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

    processed_audio = run_inference(
        args.model,
        audio,
        frame_size=args.frame_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        config_path=args.config,
    )
    if processed_audio is None: return

    sf.write(args.output, processed_audio, 48000)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
