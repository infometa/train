#!/usr/bin/env python3
"""
Causal U-Net Generator for Timbre Restoration

设计要点：
1. 所有卷积使用因果 padding（只 pad 左侧）
2. 使用 GRU 作为 bottleneck 捕获时序依赖
3. Skip connections 用于保留细节
4. Weight normalization 稳定训练
5. 残差学习：模型学习 (clean - degraded) 的差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm as weight_norm_param
from typing import List, Optional, Tuple


class CausalConv1d(nn.Module):
    """因果一维卷积"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # 因果 padding：只在左侧 pad
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # 手动 pad
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        
        if use_weight_norm:
            self.conv = weight_norm_param(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 左侧 padding
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    """因果转置卷积（用于上采样）"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        self.stride = stride
        self.kernel_size = kernel_size
        
        # 计算需要裁剪的量
        self.trim = kernel_size - stride
        
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )
        
        if use_weight_norm:
            self.conv = weight_norm_param(self.conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        # 裁剪右侧以保持因果
        if self.trim > 0:
            x = x[:, :, :-self.trim]
        return x


class ResBlock(nn.Module):
    """残差块"""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size,
            dilation=dilation, use_weight_norm=use_weight_norm
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size,
            dilation=1, use_weight_norm=use_weight_norm
        )
        
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x + res


class EncoderBlock(nn.Module):
    """编码器块：下采样"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        self.conv = CausalConv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, use_weight_norm=use_weight_norm
        )
        self.res = ResBlock(out_channels, kernel_size=3, use_weight_norm=use_weight_norm)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        x = self.res(x)
        return x


class DecoderBlock(nn.Module):
    """解码器块：上采样 + skip connection"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        self.upsample = CausalConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, use_weight_norm=use_weight_norm
        )
        # skip connection 后通道数翻倍
        self.conv = CausalConv1d(
            out_channels * 2, out_channels, kernel_size=3,
            use_weight_norm=use_weight_norm
        )
        self.res = ResBlock(out_channels, kernel_size=3, use_weight_norm=use_weight_norm)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.upsample(x))
        
        # 对齐长度
        min_len = min(x.size(-1), skip.size(-1))
        x = x[:, :, :min_len]
        skip = skip[:, :, :min_len]
        
        # 拼接 skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.conv(x))
        x = self.res(x)
        return x


class CausalGRU(nn.Module):
    """因果 GRU bottleneck"""
    
    def __init__(
        self,
        channels: int,
        hidden_size: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            channels,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # 单向 = 因果
        )
        self.proj = nn.Linear(hidden_size, channels)
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None, return_state: bool = False):
        # x: [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x, h_out = self.gru(x, h)
        x = self.proj(x)
        # [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        if return_state:
            return x, h_out
        return x


class CausalLSTM(nn.Module):
    """因果 LSTM bottleneck"""
    
    def __init__(
        self,
        channels: int,
        hidden_size: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            channels,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(hidden_size, channels)
    
    def forward(self, x: torch.Tensor, h: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_state: bool = False):
        x = x.transpose(1, 2)
        x, (h_n, c_n) = self.lstm(x, h)
        x = self.proj(x)
        x = x.transpose(1, 2)
        if return_state:
            return x, (h_n, c_n)
        return x


class CausalUNetGenerator(nn.Module):
    """因果 U-Net Generator
    
    核心改进：残差学习设计
    - 模型学习的是 (clean - degraded) 的差异/残差
    - 输出 = 输入 + 学习到的残差
    - 这样模型任务更简单，收敛更快
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 7,
        bottleneck_type: str = "gru",  # "gru", "lstm", "none"
        bottleneck_layers: int = 2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        self.channels = channels
        
        # 输入投影
        self.input_conv = CausalConv1d(
            in_channels, channels[0], kernel_size=7,
            use_weight_norm=use_weight_norm
        )
        
        # 编码器
        self.encoders = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoders.append(
                EncoderBlock(
                    channels[i], channels[i + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    use_weight_norm=use_weight_norm
                )
            )
        
        # Bottleneck
        if bottleneck_type == "gru":
            self.bottleneck = CausalGRU(
                channels[-1],
                hidden_size=channels[-1],
                num_layers=bottleneck_layers
            )
        elif bottleneck_type == "lstm":
            self.bottleneck = CausalLSTM(
                channels[-1],
                hidden_size=channels[-1],
                num_layers=bottleneck_layers
            )
        else:
            self.bottleneck = nn.Identity()
        
        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(
                DecoderBlock(
                    channels[i], channels[i - 1],
                    kernel_size=kernel_size,
                    stride=2,
                    use_weight_norm=use_weight_norm
                )
            )
        
        # 输出投影
        self.output_conv = CausalConv1d(
            channels[0], out_channels, kernel_size=7,
            use_weight_norm=use_weight_norm
        )
        
        # 残差缩放因子（可学习），初始化为 1.0
        # 模型输出 = 输入 + residual_scale * 网络输出
        self.residual_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor, rnn_state: Optional[torch.Tensor] = None, return_state: bool = False):
        """
        Args:
            x: [B, 1, T] 输入（DF 输出）
            rnn_state: 可选，RNN 隐状态（仅当 bottleneck_type 为 gru/lstm 时使用）
            return_state: 是否返回新的 RNN 隐状态（流式推理需要）
        Returns:
            y: [B, 1, T] 修复后的音频
            h_out (可选): 新的 RNN 隐状态
            
        核心逻辑：
            output = input + residual_scale * network(input)
            模型学习的是残差/差异，而不是完整信号
        """
        input_len = x.size(-1)
        residual = x  # 保存原始输入
        
        # 输入投影
        x = F.leaky_relu(self.input_conv(x), 0.2)
        
        # 编码器（保存 skip connections）
        skips = [x]
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        
        # Bottleneck
        if isinstance(self.bottleneck, (CausalGRU, CausalLSTM)):
            x, h_out = self.bottleneck(x, rnn_state, return_state=True)
        else:
            x = self.bottleneck(x)
            h_out = None
        
        # 解码器
        skips = skips[:-1][::-1]  # 反转，去掉最后一个
        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip)
        
        # 输出投影（这里输出的是"残差/修正量"）
        x = self.output_conv(x)
        
        # 对齐长度
        if x.size(-1) > input_len:
            x = x[:, :, :input_len]
        elif x.size(-1) < input_len:
            x = F.pad(x, (0, input_len - x.size(-1)))
        
        # ========== 核心改进：残差学习 ==========
        # 输出 = 原始输入 + 学习到的残差
        # 模型只需要学习 (clean - degraded)，任务更简单
        x = residual + self.residual_scale * x
        # 使用 tanh 软约束，避免梯度饱和
        x = torch.tanh(x)
        # =========================================
        
        if return_state:
            return x, h_out
        return x
    
    def remove_weight_norm(self):
        """移除 weight norm（用于推理）"""
        def _remove(module):
            try:
                parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
            except Exception:
                pass
        
        self.apply(lambda m: _remove(m) if isinstance(m, nn.Conv1d) else None)


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试
    model = CausalUNetGenerator(
        channels=[64, 128, 256, 512],
        bottleneck_type="gru"
    )
    
    print(f"Parameters: {count_parameters(model) / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(2, 1, 48000)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    
    # 验证因果性：输出的前 N 个样本不应该依赖于输入的后面样本
    x1 = torch.randn(1, 1, 48000)
    x2 = x1.clone()
    x2[:, :, 24000:] = torch.randn(1, 1, 24000)  # 修改后半部分
    
    y1 = model(x1)
    y2 = model(x2)
    
    # 前半部分应该相同（因果性）
    diff = (y1[:, :, :24000] - y2[:, :, :24000]).abs().max()
    print(f"Causality test (should be ~0): {diff.item():.6f}")
    
    # 验证残差学习
    print(f"Residual scale: {model.residual_scale.item():.4f}")
