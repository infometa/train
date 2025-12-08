#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import soundfile as sf
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

class TimbreDataReader(CalibrationDataReader):
    def __init__(self, model_path, data_dir, frame_size=480, hidden_size=384, num_layers=2, num_files=50):
        self.data_dir = data_dir
        self.frame_size = frame_size
        # 你的 Balanced 模型参数
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        
        # 获取输入名
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name # 'input'
        self.state_name = session.get_inputs()[1].name # 'h_in'
        
        # 扫描文件
        self.wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))[:num_files]
        self.file_iter = iter(self.wav_files)
        self.current_frames = []
        self.frame_idx = 0
        
        print(f"Calibration using {len(self.wav_files)} files from {data_dir}...")

    def get_next(self):
        # 如果当前文件处理完了，加载下一个
        if self.frame_idx >= len(self.current_frames):
            try:
                wav_path = next(self.file_iter)
                audio, sr = sf.read(wav_path)
                if audio.ndim > 1: audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                
                # 切分成帧
                self.current_frames = []
                for i in range(0, len(audio) - self.frame_size, self.frame_size):
                    frame = audio[i : i + self.frame_size]
                    # [1, 1, 480]
                    frame_tensor = frame[np.newaxis, np.newaxis, :]
                    self.current_frames.append(frame_tensor)
                
                self.frame_idx = 0
                if not self.current_frames: return self.get_next()
            except StopIteration:
                return None
        
        # 返回一帧数据
        input_data = self.current_frames[self.frame_idx]
        self.frame_idx += 1
        
        # 构造 dummy hidden state (校准时隐状态可以用全0)
        # [num_layers, 1, hidden_size]
        h_in = np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32)
        
        return {self.input_name: input_data, self.state_name: h_in}

def benchmark(model_path, frame_size=480, hidden_size=384, num_layers=2):
    import time
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return
        
    input_name = session.get_inputs()[0].name
    state_name = session.get_inputs()[1].name
    
    # Dummy inputs
    dummy_input = np.random.randn(1, 1, frame_size).astype(np.float32)
    dummy_h = np.zeros((num_layers, 1, hidden_size), dtype=np.float32)
    
    # Warmup
    for _ in range(20):
        session.run(None, {input_name: dummy_input, state_name: dummy_h})
        
    # Bench
    iters = 2000
    start = time.perf_counter()
    for _ in range(iters):
        session.run(None, {input_name: dummy_input, state_name: dummy_h})
    end = time.perf_counter()
    
    lat = (end - start) / iters * 1000
    rtf = lat / (frame_size / 48000 * 1000)
    print(f"[{model_path}] Latency: {lat:.3f} ms | RTF: {rtf:.3f}x")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="FP32 ONNX model")
    parser.add_argument("--output", required=True, help="Output INT8 model")
    parser.add_argument("--data", required=True, help="Path to folder containing wav files for calibration")
    args = parser.parse_args()
    
    # 1. 静态量化
    # 针对 Balanced 模型：hidden_size=384, num_layers=2
    dr = TimbreDataReader(args.input, args.data, hidden_size=384, num_layers=2)
    
    print("Starting Static Quantization (this may take a minute)...")
    quantize_static(
        model_input=args.input,
        model_output=args.output,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ, # 使用 QDQ 格式，通常在 x86 上更快
        weight_type=QuantType.QInt8,  # 静态量化推荐 QInt8
        activation_type=QuantType.QInt8,
        per_channel=True,             # 通道级量化，精度更高
        reduce_range=False,
    )
    print("Quantization Done!")
    
    # 2. 对比测试
    print("\n--- Benchmark ---")
    benchmark(args.input)
    benchmark(args.output)

if __name__ == "__main__":
    main()