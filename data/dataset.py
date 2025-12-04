#!/usr/bin/env python3
"""
PyTorch Dataset for Timbre Restoration
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
import soxr


class TimbreRestoreDataset(Dataset):
    """音色修复数据集"""
    
    def __init__(
        self,
        file_list: str,
        segment_length: int = 48000,
        sample_rate: int = 48000,
        augment: bool = True,
        align_df_delay: bool = False,
        align_max_shift: int = 1000,
        align_sample_count: int = 32,
    ):
        """
        Args:
            file_list: 文件列表路径，格式为 "degraded_path|clean_path" 每行
            segment_length: 训练片段长度
            sample_rate: 采样率
            augment: 是否启用数据增强
            align_df_delay: 是否估计并补偿 DF 引入的固定延迟
            align_max_shift: 最大对齐偏移（样本）
            align_sample_count: 用于估计偏移的样本对数量
        """
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.augment = augment
        self.align_df_delay = align_df_delay
        self.align_max_shift = align_max_shift
        self.align_sample_count = max(1, align_sample_count)
        
        # 加载文件列表
        self.pairs = []
        with open(file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '|' in line:
                    degraded, clean = line.split('|', 1)
                    if os.path.exists(degraded) and os.path.exists(clean):
                        self.pairs.append((degraded, clean))
        
        print(f"Loaded {len(self.pairs)} audio pairs from {file_list}")

        if self.align_df_delay and len(self.pairs) > 0 and self.align_max_shift > 0:
            print(f"[align] Per-sample alignment enabled, max_shift={self.align_max_shift} samples")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def load_audio(self, path: str) -> np.ndarray:
        """加载音频文件"""
        audio, sr = sf.read(path)
        
        # 转单声道
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # 重采样
        if sr != self.sample_rate:
            audio = soxr.resample(audio, sr, self.sample_rate, quality="HQ")

        # 异常值处理与裁剪
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.99, neginf=-0.99)
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio.astype(np.float32)

    @staticmethod
    def _estimate_offset(clean: np.ndarray, degraded: np.ndarray, max_shift: int, sample_rate: int) -> int:
        """估计单个样本对的相对偏移（degraded 相对于 clean 的延迟，单位：样本）
        使用简化互相关，限制搜索窗口，避免 FFT 引入的索引偏移。
        """
        if max_shift <= 0:
            return 0
        # 下采样以降低计算成本（目标 ~16kHz）
        down_factor = max(1, sample_rate // 16000)
        clean_ds = clean[::down_factor]
        degraded_ds = degraded[::down_factor]
        max_shift_ds = max(1, int(max_shift / down_factor))
        max_shift_ds = min(max_shift_ds, len(clean_ds) - 1, len(degraded_ds) - 1)
        if max_shift_ds <= 0:
            return 0

        # 零均值，提升相关峰显著性
        clean_ds = clean_ds - np.mean(clean_ds)
        degraded_ds = degraded_ds - np.mean(degraded_ds)

        # 直接互相关并限制窗口
        corr = np.correlate(clean_ds, degraded_ds, mode='full')
        lags = np.arange(-len(degraded_ds) + 1, len(clean_ds))
        valid = (lags >= -max_shift_ds) & (lags <= max_shift_ds)
        corr = corr[valid]
        lags = lags[valid]
        if len(lags) == 0:
            return 0

        best_idx = int(np.argmax(corr))
        best_lag = int(lags[best_idx])

        return int(best_lag * down_factor)

    def _apply_offset(self, degraded: np.ndarray, clean: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        """按单条样本估计的延迟对齐 degraded 与 clean"""
        if offset == 0:
            return degraded, clean
        if offset > 0:
            # degraded 滞后：裁掉 degraded 开头
            degraded = degraded[offset:]
            clean = clean[:len(degraded)]
        else:
            # degraded 领先：裁掉 clean 开头
            offset = -offset
            clean = clean[offset:]
            degraded = degraded[:len(clean)]
        return degraded, clean
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        degraded_path, clean_path = self.pairs[idx]
        
        # 加载音频
        degraded = self.load_audio(degraded_path)
        clean = self.load_audio(clean_path)
        
        # 确保长度一致
        min_len = min(len(degraded), len(clean))
        degraded = degraded[:min_len]
        clean = clean[:min_len]

        # 对齐 DF 潜在延迟（逐样本）
        if self.align_df_delay and self.align_max_shift > 0 and min_len > 1000:
            offset = self._estimate_offset(clean, degraded, self.align_max_shift, self.sample_rate)
            if offset != 0:
                degraded, clean = self._apply_offset(degraded, clean, offset)
        
        # 裁剪或填充（对齐后长度可能略短）
        current_len = len(degraded)
        if current_len >= self.segment_length:
            # 为保持对齐，同时引入轻微随机性（最多偏移 500 样本或剩余空间）
            if self.augment:
                max_start = max(0, min(500, current_len - self.segment_length))
                start = random.randint(0, max_start) if max_start > 0 else 0
            else:
                start = 0
            degraded = degraded[start:start + self.segment_length]
            clean = clean[start:start + self.segment_length]
        elif current_len >= int(self.segment_length * 0.9):
            # 略短于目标长度：居中并用反射填充两侧，避免大块零
            deficit = self.segment_length - current_len
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            degraded = np.pad(degraded, (pad_left, pad_right), mode='reflect')
            clean = np.pad(clean, (pad_left, pad_right), mode='reflect')
        else:
            # 远短于目标：零填充（极少出现）
            pad_len = self.segment_length - current_len
            degraded = np.pad(degraded, (0, pad_len), mode='constant')
            clean = np.pad(clean, (0, pad_len), mode='constant')
        
        # 数据增强
        if self.augment:
            # 温和增益（约 ±1dB）
            gain = random.uniform(0.9, 1.1)
            degraded = degraded * gain
            clean = clean * gain
        
        # 转为 Tensor [1, T]
        degraded = torch.from_numpy(degraded.astype(np.float32)).unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
        
        return degraded, clean


class InferenceDataset(Dataset):
    """推理用数据集"""
    
    def __init__(
        self,
        file_list: List[str],
        sample_rate: int = 48000,
    ):
        self.files = file_list
        self.sample_rate = sample_rate
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = soxr.resample(audio, sr, self.sample_rate, quality="HQ")
        
        audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        
        return audio, path


def create_dataloader(
    file_list: str,
    batch_size: int = 16,
    segment_length: int = 48000,
    sample_rate: int = 48000,
    num_workers: int = 8,
    augment: bool = True,
    shuffle: bool = True,
    align_df_delay: bool = False,
    align_max_shift: int = 2000,
    align_sample_count: int = 32,
) -> DataLoader:
    """创建 DataLoader"""
    dataset = TimbreRestoreDataset(
        file_list=file_list,
        segment_length=segment_length,
        sample_rate=sample_rate,
        augment=augment,
        align_df_delay=align_df_delay,
        align_max_shift=align_max_shift,
        align_sample_count=align_sample_count,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <train.txt>")
        sys.exit(1)
    
    dataset = TimbreRestoreDataset(sys.argv[1])
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        degraded, clean = dataset[0]
        print(f"Degraded shape: {degraded.shape}")
        print(f"Clean shape: {clean.shape}")
