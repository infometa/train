#!/usr/bin/env python3
"""
导出 ONNX 模型用于 Rust 集成

导出流程：
1. 加载训练好的 Generator
2. 移除 weight norm
3. 导出 ONNX（支持动态 batch 和长度）
4. 简化和优化
5. 验证输出一致性
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import yaml

# ONNX
try:
    import onnx
    from onnxsim import simplify as onnx_simplify
    ONNX_AVAILABLE = True
except ImportError:
    print("Warning: onnx/onnxsim not found. Install with: pip install onnx onnxsim")
    ONNX_AVAILABLE = False

# ONNX Runtime (验证用)
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    print("Warning: onnxruntime not found. Install with: pip install onnxruntime")
    ORT_AVAILABLE = False

from model.generator import CausalUNetGenerator


class ONNXWrapper(nn.Module):
    """ONNX 导出包装器"""
    
    def __init__(self, model: CausalUNetGenerator, return_state: bool = False):
        super().__init__()
        self.model = model
        self.return_state = return_state
    
    def forward(self, x: torch.Tensor, h_in: torch.Tensor = None):
        """
        Args:
            x: [B, 1, T] 或 [B, T] 输入音频
            h_in: 可选隐藏状态 [num_layers, B, hidden]（仅 stateful 导出）
        Returns:
            y: [B, 1, T] 或 [B, T]
            h_out: 可选隐藏状态
        """
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        
        if self.return_state:
            y, h_out = self.model(x, rnn_state=h_in, return_state=True)
        else:
            y = self.model(x)
            h_out = None
        
        if squeeze_output:
            y = y.squeeze(1)
        
        if self.return_state:
            return y, h_out
        return y


def load_model(checkpoint_path: str, config_path: str) -> CausalUNetGenerator:
    """加载训练好的模型"""
    
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    gen_config = config['model']['generator']
    
    # 创建模型
    model = CausalUNetGenerator(
        in_channels=gen_config['in_channels'],
        out_channels=gen_config['out_channels'],
        channels=gen_config['channels'],
        kernel_size=gen_config['kernel_size'],
        bottleneck_type=gen_config['bottleneck_type'],
        bottleneck_layers=gen_config['bottleneck_layers'],
        use_weight_norm=gen_config['use_weight_norm'],
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 处理 DDP 前缀
    state_dict = checkpoint['generator']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    # 移除 weight norm
    model.remove_weight_norm()
    
    return model


def export_onnx(
    model: CausalUNetGenerator,
    output_path: str,
    opset_version: int = 17,
    simplify: bool = True,
    include_state: bool = False,
):
    """导出 ONNX 模型"""
    
    if not ONNX_AVAILABLE:
        raise RuntimeError("ONNX not available. Install with: pip install onnx onnxsim")
    
    model.eval()
    wrapper = ONNXWrapper(model, return_state=include_state)
    
    # 示例输入（1秒音频）
    dummy_input = torch.randn(1, 1, 48000)
    
    input_names = ['input']
    output_names = ['output']
    dynamic_axes = {
        'input': {0: 'batch', 2: 'length'},
        'output': {0: 'batch', 2: 'length'},
    }
    
    example_inputs = (dummy_input,)
    
    # 隐状态（仅支持 GRU/LSTM）
    if include_state:
        if not (hasattr(model.bottleneck, "gru") or hasattr(model.bottleneck, "lstm")):
            raise RuntimeError("include_state=True requires a GRU/LSTM bottleneck")
        num_layers = model.bottleneck.gru.num_layers if hasattr(model.bottleneck, "gru") else getattr(model.bottleneck, "num_layers", 1)
    hidden_size = model.channels[-1]
    # 如果 bottleneck 显式指定了隐藏维度，优先使用
    if hasattr(model, "bottleneck"):
        if hasattr(model.bottleneck, "gru"):
            hidden_size = model.bottleneck.gru.hidden_size
        elif hasattr(model.bottleneck, "lstm"):
            hidden_size = model.bottleneck.lstm.hidden_size
        h0 = torch.zeros(num_layers, 1, hidden_size)
        input_names.append('h_in')
        output_names.append('h_out')
        dynamic_axes['h_in'] = {1: 'batch'}
        dynamic_axes['h_out'] = {1: 'batch'}
        example_inputs = (dummy_input, h0)
    
    print(f"Exporting to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        wrapper,
        example_inputs,
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    
    print(f"Exported: {output_path}")
    
    # 简化
    if simplify:
        print("Simplifying ONNX model...")
        onnx_model = onnx.load(output_path)
        simplified, ok = onnx_simplify(onnx_model)
        if ok:
            onnx.save(simplified, output_path)
            print("Simplified successfully")
        else:
            print("Simplification failed, keeping original")
    
    # 验证
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validated")
    
    # 打印信息
    print(f"\nModel info:")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    
    return output_path


def verify_onnx(
    pytorch_model: CausalUNetGenerator,
    onnx_path: str,
    test_lengths: list = [480, 4800, 48000],
    include_state: bool = False,
):
    """验证 ONNX 输出与 PyTorch 一致"""
    
    if not ORT_AVAILABLE:
        print("Skipping verification (onnxruntime not available)")
        return
    
    print("\nVerifying ONNX model...")
    
    pytorch_model.eval()
    wrapper = ONNXWrapper(pytorch_model, return_state=include_state)
    
    # ONNX Runtime session
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    for length in test_lengths:
        # 随机输入
        x = torch.randn(1, 1, length)
        if include_state:
            num_layers = wrapper.model.bottleneck.gru.num_layers if hasattr(wrapper.model.bottleneck, "gru") else getattr(wrapper.model.bottleneck, "num_layers", 1)
            hidden_size = wrapper.model.channels[-1]
            h0 = torch.zeros(num_layers, 1, hidden_size)
        
        ort_inputs = {'input': x.numpy()}
        if include_state:
            ort_inputs['h_in'] = h0.numpy()
        
        # PyTorch 输出
        with torch.no_grad():
            if include_state:
                pytorch_out, h_out_ref = wrapper(x, h0)
                pytorch_out = pytorch_out.numpy()
                h_out_ref = h_out_ref.detach().numpy()
            else:
                pytorch_out = wrapper(x).numpy()
        
        # ONNX Runtime 输出
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_out = ort_outputs[0]
        if include_state:
            ort_h_out = ort_outputs[1]
        
        # 比较
        max_diff = np.abs(pytorch_out - ort_out).max()
        mean_diff = np.abs(pytorch_out - ort_out).mean()
        
        print(f"  Length {length}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}")
        
        if max_diff > 1e-4:
            print(f"    WARNING: Large difference detected!")
        
        if include_state:
            h_max_diff = np.abs(h_out_ref - ort_h_out).max()
            h_mean_diff = np.abs(h_out_ref - ort_h_out).mean()
            print(f"    Hidden diff: max={h_max_diff:.6f}, mean={h_mean_diff:.8f}")


def export_streaming_onnx(
    model: CausalUNetGenerator,
    output_path: str,
    frame_size: int = 480,
    opset_version: int = 17,
):
    """
    导出流式推理版本（带 RNN 状态输入/输出）
    """
    print("\nExporting streaming (stateful) ONNX...")
    export_onnx(model, output_path, opset_version=opset_version, simplify=True, include_state=True)
    print(f"Streaming model exported: {output_path}")
    print(f"Note: Inference expects input + hidden state and returns updated hidden state.")


def benchmark_onnx(onnx_path: str, frame_size: int = 480, num_iters: int = 1000):
    """性能基准测试"""
    
    if not ORT_AVAILABLE:
        print("Skipping benchmark (onnxruntime not available)")
        return
    
    import time
    
    print(f"\nBenchmarking ONNX model (frame_size={frame_size})...")
    
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    # 预热
    x = np.random.randn(1, 1, frame_size).astype(np.float32)
    for _ in range(10):
        ort_session.run(None, {'input': x})
    
    # 计时
    start = time.perf_counter()
    for _ in range(num_iters):
        ort_session.run(None, {'input': x})
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    rtf = avg_ms / (frame_size / 48000 * 1000)  # Real-Time Factor
    
    print(f"  Average latency: {avg_ms:.3f} ms")
    print(f"  Real-Time Factor: {rtf:.3f}x")
    print(f"  Frames per second: {1000 / avg_ms:.1f}")
    
    if rtf < 1.0:
        print("  ✓ Real-time capable!")
    else:
        print("  ✗ Too slow for real-time")


def main():
    parser = argparse.ArgumentParser(description="Export ONNX model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="timbre_restore.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Export streaming version with hidden state")
    args = parser.parse_args()
    
    # 加载模型
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.config)
    
    # 导出
    if args.streaming:
        export_streaming_onnx(model, args.output, opset_version=args.opset)
        include_state = True
    else:
        export_onnx(
            model,
            args.output,
            opset_version=args.opset,
            simplify=not args.no_simplify,
            include_state=False,
        )
        include_state = False
    
    # 验证
    if args.verify:
        verify_onnx(model, args.output, include_state=include_state)
    
    # 性能测试
    if args.benchmark:
        benchmark_onnx(args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
