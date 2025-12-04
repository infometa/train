#!/usr/bin/env python3
"""
Loss Functions for Timbre Restoration

包含：
- L1 Loss (时域)
- Multi-Resolution STFT Loss (频域) - 带高频加权
- Adversarial Loss (GAN)
- Feature Matching Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiResolutionSTFTLoss(nn.Module):
    """多分辨率 STFT 损失 - 带高频加权"""
    
    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_lengths: List[int] = [512, 1024, 2048],
        window: str = "hann",
        sample_rate: int = 48000,
        hf_weight: float = 1.5,      # 高频加权倍数
        hf_cutoff: int = 3000,       # 高频起始频率 (Hz)
    ):
        super().__init__()
        
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.sample_rate = sample_rate
        self.hf_weight = hf_weight
        self.hf_cutoff = hf_cutoff
        
        # 预计算窗函数
        self.windows = nn.ParameterList()
        for win_length in win_lengths:
            if window == "hann":
                win = torch.hann_window(win_length)
            elif window == "hamming":
                win = torch.hamming_window(win_length)
            else:
                win = torch.ones(win_length)
            self.windows.append(nn.Parameter(win, requires_grad=False))
    
    def _get_frequency_weight(self, freq_bins: int, fft_size: int, device: torch.device) -> torch.Tensor:
        """生成频率加权向量（高频平滑过渡，sigmoid）"""
        weight = torch.ones(freq_bins, device=device)
        
        hf_start_bin = int(self.hf_cutoff * fft_size / self.sample_rate)
        if hf_start_bin < freq_bins:
            bins = torch.arange(freq_bins, device=device, dtype=torch.float32)
            transition_width = max(1, (freq_bins - hf_start_bin) // 4)
            sigmoid_input = (bins - hf_start_bin) / transition_width
            smooth_ramp = torch.sigmoid(sigmoid_input)
            weight = 1.0 + (self.hf_weight - 1.0) * smooth_ramp
        
        return weight.view(1, -1, 1)  # [1, F, 1] for broadcasting
    
    def stft_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fft_size: int,
        hop_size: int,
        win_length: int,
        window: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算单个分辨率的 STFT 损失"""
        
        # STFT
        pred_stft = torch.stft(
            pred.squeeze(1),
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window.to(pred.device),
            return_complex=True,
        )
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window.to(target.device),
            return_complex=True,
        )
        
        # 幅度谱
        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()
        
        # ========== 高频加权 ==========
        freq_bins = pred_mag.shape[1]
        freq_weight = self._get_frequency_weight(freq_bins, fft_size, pred.device)
        
        # 应用加权
        pred_mag_weighted = pred_mag * freq_weight
        target_mag_weighted = target_mag * freq_weight
        # ===============================
        
        # Spectral Convergence Loss（使用加权后的幅度谱）
        sc_loss = torch.norm(target_mag_weighted - pred_mag_weighted, p="fro") / (torch.norm(target_mag_weighted, p="fro") + 1e-8)
        
        # Log Magnitude Loss（使用加权后的幅度谱）
        pred_log = torch.log(pred_mag_weighted + 1e-8)
        target_log = torch.log(target_mag_weighted + 1e-8)
        mag_loss = F.l1_loss(pred_log, target_log)
        
        return sc_loss, mag_loss
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 1, T] 预测波形
            target: [B, 1, T] 目标波形
        """
        total_loss = 0.0
        
        for fft_size, hop_size, win_length, window in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows
        ):
            sc_loss, mag_loss = self.stft_loss(
                pred, target, fft_size, hop_size, win_length, window
            )
            total_loss = total_loss + sc_loss + mag_loss
        
        return total_loss / len(self.fft_sizes)


class AdversarialLoss(nn.Module):
    """对抗损失"""
    
    def __init__(self, mode: str = "lsgan"):
        """
        Args:
            mode: "lsgan" (least squares) or "vanilla" (BCE)
        """
        super().__init__()
        self.mode = mode
    
    def forward(
        self,
        disc_outputs: List[torch.Tensor],
        is_real: bool,
    ) -> torch.Tensor:
        """
        Args:
            disc_outputs: 判别器输出列表
            is_real: True for real samples, False for fake
        """
        loss = 0.0
        
        for out in disc_outputs:
            if self.mode == "lsgan":
                if is_real:
                    loss = loss + torch.mean((out - 1.0) ** 2)
                else:
                    loss = loss + torch.mean(out ** 2)
            else:  # vanilla
                target = torch.ones_like(out) if is_real else torch.zeros_like(out)
                loss = loss + F.binary_cross_entropy_with_logits(out, target)
        
        return loss / len(disc_outputs)


class FeatureMatchingLoss(nn.Module):
    """特征匹配损失"""
    
    def __init__(self, layer_weights: Optional[List[float]] = None):
        super().__init__()
        self.layer_weights = layer_weights
    
    def forward(
        self,
        fake_features: List[List[torch.Tensor]],
        real_features: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            fake_features: 生成样本的判别器特征
            real_features: 真实样本的判别器特征
        """
        loss = 0.0
        count = 0
        
        for fake_scale, real_scale in zip(fake_features, real_features):
            for idx, (fake_feat, real_feat) in enumerate(zip(fake_scale, real_scale)):
                w = 1.0
                if self.layer_weights and idx < len(self.layer_weights):
                    w = self.layer_weights[idx]
                loss = loss + w * F.l1_loss(fake_feat, real_feat.detach())
                count += 1
        
        return loss / max(count, 1)


class GeneratorLoss(nn.Module):
    """Generator 总损失"""
    
    def __init__(
        self,
        l1_weight: float = 3.0,
        stft_weight: float = 3.0,
        adv_weight: float = 1.0,
        fm_weight: float = 2.0,
        stft_config: Optional[dict] = None,
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.adv_weight = adv_weight
        self.fm_weight = fm_weight
        
        # STFT Loss（带高频加权）
        if stft_config is None:
            stft_config = {
                'fft_sizes': [512, 1024, 2048],
                'hop_sizes': [128, 256, 512],
                'win_lengths': [512, 1024, 2048],
                'sample_rate': 48000,
                'hf_weight': 1.5,      # 高频加权
                'hf_cutoff': 3000,     # 3kHz 以上
            }
        self.stft_loss = MultiResolutionSTFTLoss(**stft_config)
        
        # Adversarial Loss
        self.adv_loss = AdversarialLoss(mode="lsgan")
        
        # Feature Matching Loss
        self.fm_loss = FeatureMatchingLoss(layer_weights=[1.0, 0.5, 0.25, 0.125])
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disc_fake_outputs: Optional[List[torch.Tensor]] = None,
        disc_fake_features: Optional[List[List[torch.Tensor]]] = None,
        disc_real_features: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失（用于日志）
        """
        losses = {}
        
        # L1 Loss
        l1 = F.l1_loss(pred, target)
        losses['l1'] = l1.item()
        
        # STFT Loss（带高频加权）
        stft = self.stft_loss(pred, target)
        losses['stft'] = stft.item()
        
        total = self.l1_weight * l1 + self.stft_weight * stft
        
        # Adversarial Loss (如果有判别器输出)
        if disc_fake_outputs is not None:
            adv = self.adv_loss(disc_fake_outputs, is_real=True)
            losses['adv'] = adv.item()
            total = total + self.adv_weight * adv
        
        # Feature Matching Loss
        if disc_fake_features is not None and disc_real_features is not None:
            fm = self.fm_loss(disc_fake_features, disc_real_features)
            losses['fm'] = fm.item()
            total = total + self.fm_weight * fm
        
        losses['total'] = total.item()
        
        return total, losses


class DiscriminatorLoss(nn.Module):
    """Discriminator 损失"""
    
    def __init__(self):
        super().__init__()
        self.adv_loss = AdversarialLoss(mode="lsgan")
    
    def forward(
        self,
        real_outputs: List[torch.Tensor],
        fake_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            real_outputs: 真实样本的判别器输出
            fake_outputs: 生成样本的判别器输出
        """
        real_loss = self.adv_loss(real_outputs, is_real=True)
        fake_loss = self.adv_loss(fake_outputs, is_real=False)
        
        total = real_loss + fake_loss
        
        losses = {
            'd_real': real_loss.item(),
            'd_fake': fake_loss.item(),
            'd_total': total.item(),
        }
        
        return total, losses


if __name__ == "__main__":
    # 测试
    print("Testing MultiResolutionSTFTLoss with high-frequency weighting...")
    
    stft_loss = MultiResolutionSTFTLoss(
        hf_weight=2.0,
        hf_cutoff=3000,
    )
    
    pred = torch.randn(2, 1, 48000)
    target = torch.randn(2, 1, 48000)
    
    loss = stft_loss(pred, target)
    print(f"STFT Loss: {loss.item():.4f}")
    
    # Generator Loss
    g_loss = GeneratorLoss(
        l1_weight=15.0,
        stft_weight=2.0,
    )
    total, losses = g_loss(pred, target)
    print(f"Generator Loss: {losses}")
    
    # 验证高频加权效果
    print("\nTesting high-frequency weighting effect...")
    
    # 创建只有高频差异的信号
    pred_hf = torch.randn(2, 1, 48000)
    target_hf = pred_hf.clone()
    
    # 在时域添加高频噪声（只影响高频）
    import numpy as np
    t = np.linspace(0, 1, 48000)
    hf_noise = torch.from_numpy(0.1 * np.sin(2 * np.pi * 8000 * t)).float()  # 8kHz
    target_hf[0, 0, :] += hf_noise
    
    loss_hf = stft_loss(pred_hf, target_hf)
    print(f"High-frequency difference loss: {loss_hf.item():.4f}")
    
    # 创建只有低频差异的信号
    pred_lf = torch.randn(2, 1, 48000)
    target_lf = pred_lf.clone()
    lf_noise = torch.from_numpy(0.1 * np.sin(2 * np.pi * 500 * t)).float()  # 500Hz
    target_lf[0, 0, :] += lf_noise
    
    loss_lf = stft_loss(pred_lf, target_lf)
    print(f"Low-frequency difference loss: {loss_lf.item():.4f}")
    
    print(f"\nHF loss should be higher than LF loss due to weighting")
    print(f"Ratio (HF/LF): {loss_hf.item() / loss_lf.item():.2f}x")
