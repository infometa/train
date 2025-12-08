#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def benchmark(model_path, frame_size=480):
    """测试模型推理速度"""
    print(f"Benchmarking {model_path} ...")
    try:
        # 使用 CPU 推理
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return float('inf')

    # 获取输入信息
    input_meta = session.get_inputs()
    inputs = {}
    
    # 构造 dummy 输入
    for meta in input_meta:
        shape = meta.shape
        name = meta.name
        # 处理动态维度
        new_shape = []
        for dim in shape:
            if isinstance(dim, str) or dim is None:
                new_shape.append(1) # Batch size
            else:
                new_shape.append(dim)
        
        # 修正时间维度 (通常是最后一个)
        if len(new_shape) >= 1:
            new_shape[-1] = frame_size
            
        inputs[name] = np.random.randn(*new_shape).astype(np.float32)

    # 预热
    for _ in range(10):
        session.run(None, inputs)
        
    # 测试
    start_time = time.perf_counter()
    iters = 1000
    for _ in range(iters):
        session.run(None, inputs)
    end_time = time.perf_counter()
    
    avg_latency_ms = (end_time - start_time) / iters * 1000
    # 10ms 帧长对应的 RTF
    rtf = avg_latency_ms / (frame_size / 48000 * 1000)
    
    print(f"  > Latency: {avg_latency_ms:.3f} ms")
    print(f"  > RTF:     {rtf:.3f}x")
    return rtf

def main():
    parser = argparse.ArgumentParser(description="ONNX 模型动态量化工具 (FP32 -> INT8)")
    parser.add_argument("input", help="输入的 FP32 ONNX 模型路径")
    parser.add_argument("output", help="输出的 INT8 量化模型路径")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        return

    print("=" * 40)
    print(f"Quantizing model...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 40)
    
    # 1. 执行量化
    # 使用 QUInt8 (无符号8位整数) 进行动态量化，这对 CPU 推理最友好
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8, 
    )
    print("Quantization complete!\n")
    
    # 2. 对比测试
    print("--- 原始模型 (FP32) ---")
    rtf_fp32 = benchmark(input_path)
    
    print("\n--- 量化模型 (INT8) ---")
    rtf_int8 = benchmark(output_path)
    
    # 3. 总结
    print("\n" + "=" * 40)
    print(f"Speedup: {rtf_fp32 / rtf_int8:.2f}x")
    if rtf_int8 < 0.8:
        print("✅ Safe for real-time usage!")
    elif rtf_int8 < 1.0:
        print("⚠️ Marginal real-time (risk of underrun).")
    else:
        print("❌ Too slow for real-time.")
    print("=" * 40)

if __name__ == "__main__":
    main()