#!/usr/bin/env python3
"""
Multi-Scale Discriminator for GAN Training

基于 HiFi-GAN 的多尺度判别器设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DiscriminatorBlock(nn.Module):
    """单个判别器"""
    
    def __init__(
        self,
        channels: List[int] = [64, 128, 256, 512, 1024],
        kernel_size: int = 5,
        stride: int = 2,
        groups: List[int] = [1, 4, 16, 64, 256],
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        self.convs = nn.ModuleList()
        
        # 输入层
        self.convs.append(
            norm_fn(nn.Conv1d(1, channels[0], kernel_size=15, stride=1, padding=7))
        )
        
        # 中间层
        in_ch = channels[0]
        # 调整分组，避免大规模回退到 1
        for i, out_ch in enumerate(channels[1:]):
            actual_groups = groups[i] if i < len(groups) else 1
            # 若不整除则缩小分组，直到整除或为 1
            while actual_groups > 1 and (in_ch % actual_groups != 0 or out_ch % actual_groups != 0):
                actual_groups //= 2
            self.convs.append(
                norm_fn(nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=actual_groups
                ))
            )
            in_ch = out_ch
        
        # 输出层
        self.convs.append(
            norm_fn(nn.Conv1d(in_ch, in_ch, kernel_size=5, stride=1, padding=2))
        )
        self.convs.append(
            norm_fn(nn.Conv1d(in_ch, 1, kernel_size=3, stride=1, padding=1))
        )
        
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            output: 判别器输出
            features: 中间层特征（用于 feature matching loss）
        """
        features = []
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.act(x)
            features.append(x)
        
        x = self.convs[-1](x)
        
        return x, features


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器"""
    
    def __init__(
        self,
        scales: int = 3,
        channels: List[int] = [64, 128, 256, 512, 1024],
        kernel_size: int = 5,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        for i in range(scales):
            self.discriminators.append(
                DiscriminatorBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    use_spectral_norm=use_spectral_norm
                )
            )
            
            if i < scales - 1:
                self.downsamplers.append(
                    nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
                )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: [B, 1, T] 输入音频
        Returns:
            outputs: 各尺度判别器输出
            features: 各尺度的中间特征
        """
        outputs = []
        all_features = []
        
        for i, disc in enumerate(self.discriminators):
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)
            
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
        
        return outputs, all_features


class MultiPeriodDiscriminator(nn.Module):
    """多周期判别器（可选，用于增强细节判别）"""
    
    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        channels: List[int] = [32, 128, 512, 1024],
        kernel_size: int = 5,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.periods = periods
        self.discriminators = nn.ModuleList()
        
        for period in periods:
            self.discriminators.append(
                PeriodDiscriminator(
                    period=period,
                    channels=channels,
                    kernel_size=kernel_size,
                    use_spectral_norm=use_spectral_norm
                )
            )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        all_features = []
        
        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            all_features.append(feats)
        
        return outputs, all_features


class PeriodDiscriminator(nn.Module):
    """周期判别器（2D 卷积）"""
    
    def __init__(
        self,
        period: int,
        channels: List[int] = [32, 128, 512, 1024],
        kernel_size: int = 5,
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.period = period
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else lambda x: x
        
        self.convs = nn.ModuleList()
        
        in_ch = 1
        for out_ch in channels:
            self.convs.append(
                norm_fn(nn.Conv2d(
                    in_ch, out_ch,
                    kernel_size=(kernel_size, 1),
                    stride=(3, 1),
                    padding=(kernel_size // 2, 0)
                ))
            )
            in_ch = out_ch
        
        self.convs.append(
            norm_fn(nn.Conv2d(in_ch, in_ch, kernel_size=(5, 1), padding=(2, 0)))
        )
        self.convs.append(
            norm_fn(nn.Conv2d(in_ch, 1, kernel_size=(3, 1), padding=(1, 0)))
        )
        
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Reshape to 2D: [B, 1, T] -> [B, 1, T//period, period]
        b, c, t = x.shape
        
        # Pad if necessary
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), mode='reflect')
            t = x.size(-1)
        
        x = x.view(b, c, t // self.period, self.period)
        
        features = []
        for conv in self.convs[:-1]:
            x = conv(x)
            x = self.act(x)
            features.append(x)
        
        x = self.convs[-1](x)
        x = x.flatten(1, -1)
        
        return x, features


class CombinedDiscriminator(nn.Module):
    """组合判别器：多尺度 + 多周期"""
    
    def __init__(
        self,
        use_msd: bool = True,
        use_mpd: bool = True,
        msd_scales: int = 3,
        mpd_periods: List[int] = [2, 3, 5, 7, 11],
        use_spectral_norm: bool = True,
    ):
        super().__init__()
        
        self.use_msd = use_msd
        self.use_mpd = use_mpd
        
        if use_msd:
            self.msd = MultiScaleDiscriminator(
                scales=msd_scales,
                use_spectral_norm=use_spectral_norm
            )
        
        if use_mpd:
            self.mpd = MultiPeriodDiscriminator(
                periods=mpd_periods,
                use_spectral_norm=use_spectral_norm
            )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        all_features = []
        
        if self.use_msd:
            msd_out, msd_feats = self.msd(x)
            outputs.extend(msd_out)
            all_features.extend(msd_feats)
        
        if self.use_mpd:
            mpd_out, mpd_feats = self.mpd(x)
            outputs.extend(mpd_out)
            all_features.extend(mpd_feats)
        
        return outputs, all_features


if __name__ == "__main__":
    # 测试
    msd = MultiScaleDiscriminator(scales=3)
    mpd = MultiPeriodDiscriminator()
    combined = CombinedDiscriminator()
    
    x = torch.randn(2, 1, 48000)
    
    # MSD
    out, feats = msd(x)
    print(f"MSD outputs: {[o.shape for o in out]}")
    print(f"MSD features per scale: {[len(f) for f in feats]}")
    
    # MPD
    out, feats = mpd(x)
    print(f"MPD outputs: {[o.shape for o in out]}")
    
    # Combined
    out, feats = combined(x)
    print(f"Combined outputs: {len(out)}")
    
    # 参数量
    print(f"\nMSD params: {sum(p.numel() for p in msd.parameters()) / 1e6:.2f}M")
    print(f"MPD params: {sum(p.numel() for p in mpd.parameters()) / 1e6:.2f}M")
