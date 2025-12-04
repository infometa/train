#!/usr/bin/env python3
"""
训练脚本（全量内存缓存版）

特点：
- 启动时一次性加载/对齐所有音频到内存，训练时不再触磁盘或 shm
- DataLoader 默认 num_workers=0，避免共享内存问题
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import random
from typing import List, Tuple

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import yaml
from tqdm import tqdm

# 复用原有组件
sys.path.insert(0, str(Path(__file__).parent))
from train import setup_distributed, cleanup_distributed, Trainer  # reuse base logic
from model.generator import CausalUNetGenerator, count_parameters
from model.discriminator import MultiScaleDiscriminator
from model.losses import GeneratorLoss, DiscriminatorLoss


class MemoryTimbreDataset(Dataset):
    """一次性加载并对齐到内存的 Dataset"""

    def __init__(
        self,
        file_list: str,
        segment_length: int = 48000,
        sample_rate: int = 48000,
        augment: bool = True,
        align_df_delay: bool = True,
        align_max_shift: int = 2000,
    ):
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.augment = augment
        self.align_df_delay = align_df_delay
        self.align_max_shift = align_max_shift

        # 读取列表
        pairs: List[Tuple[str, str]] = []
        with open(file_list, "r") as f:
            for line in f:
                line = line.strip()
                if line and "|" in line:
                    d, c = line.split("|", 1)
                    pairs.append((d, c))
        if len(pairs) == 0:
            raise SystemExit(f"空列表: {file_list}")

        # 预加载并对齐
        self.cache: List[Tuple[np.ndarray, np.ndarray]] = []
        for d_path, c_path in tqdm(pairs, desc="Preloading to memory"):
            try:
                d = self.load_audio(d_path)
                c = self.load_audio(c_path)
                min_len = min(len(d), len(c))
                d = d[:min_len]
                c = c[:min_len]
                if self.align_df_delay and min_len > 1000:
                    offset = self._estimate_offset(c, d, self.align_max_shift, self.sample_rate)
                    if offset != 0:
                        d, c = self._apply_offset(d, c, offset)
                self.cache.append((d, c))
            except Exception:
                continue

        print(f"[MemoryDataset] Cached {len(self.cache)} pairs into RAM")

    def __len__(self) -> int:
        return len(self.cache)

    def load_audio(self, path: str) -> np.ndarray:
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            audio = soxr.resample(audio, sr, self.sample_rate, quality="HQ")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.99, neginf=-0.99)
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(np.float32)

    @staticmethod
    def _estimate_offset(clean: np.ndarray, degraded: np.ndarray, max_shift: int, sample_rate: int) -> int:
        """
        与 prepare_dataset.py 保持一致的全精度对齐：
        - 无下采样，样本级精度
        - degraded 在 clean 上滑动，degraded 滞后时返回正 offset
        """
        if np.std(clean) < 1e-4 or np.std(degraded) < 1e-4:
            return 0
        if max_shift <= 0:
            return 0

        clean_ds = clean.flatten() if clean.ndim > 1 else clean
        degraded_ds = degraded.flatten() if degraded.ndim > 1 else degraded

        limit = min(len(clean_ds), len(degraded_ds)) - 1
        real_max_shift = min(max_shift, limit)
        if real_max_shift <= 0:
            return 0

        clean_ds = clean_ds - np.mean(clean_ds)
        degraded_ds = degraded_ds - np.mean(degraded_ds)

        corr = np.correlate(degraded_ds, clean_ds, mode="full")
        zero_idx = len(clean_ds) - 1
        start_idx = max(0, zero_idx - real_max_shift)
        end_idx = min(len(corr), zero_idx + real_max_shift + 1)
        if start_idx >= end_idx:
            return 0

        search_window = corr[start_idx:end_idx]
        best_idx_in_window = int(np.argmax(search_window))
        best_lag_idx = start_idx + best_idx_in_window
        best_lag = best_lag_idx - zero_idx
        return int(best_lag)

    def _apply_offset(self, degraded: np.ndarray, clean: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        if offset == 0:
            return degraded, clean
        if offset > 0:
            degraded = degraded[offset:]
            clean = clean[: len(degraded)]
        else:
            offset = -offset
            clean = clean[offset:]
            degraded = degraded[: len(clean)]
        return degraded, clean

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        degraded, clean = self.cache[idx]
        cur_len = min(len(degraded), len(clean))
        degraded = degraded[:cur_len]
        clean = clean[:cur_len]

        # 裁剪/填充逻辑与原版一致（含反射填充）
        if cur_len >= self.segment_length:
            start = random.randint(0, cur_len - self.segment_length) if self.augment else 0
            degraded = degraded[start : start + self.segment_length]
            clean = clean[start : start + self.segment_length]
        elif cur_len >= int(self.segment_length * 0.9):
            deficit = self.segment_length - cur_len
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            degraded = np.pad(degraded, (pad_left, pad_right), mode="reflect")
            clean = np.pad(clean, (pad_left, pad_right), mode="reflect")
        else:
            pad_len = self.segment_length - cur_len
            degraded = np.pad(degraded, (0, pad_len), mode="constant")
            clean = np.pad(clean, (0, pad_len), mode="constant")

        if self.augment:
            gain = random.uniform(0.9, 1.1)
            degraded = degraded * gain
            clean = clean * gain

        degraded = torch.from_numpy(degraded.astype(np.float32)).unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).unsqueeze(0)
        return degraded, clean


class TrainerMem(Trainer):
    """继承原 Trainer，仅替换 DataLoader 为内存缓存版本"""

    def _build_dataloaders(self):
        data_config = self.config["data"]
        train_config = self.config["training"]

        align_df_delay = data_config.get("align_df_delay", False)
        align_max_shift = data_config.get("align_max_shift", 2000)

        output_dir = Path(data_config["output_dir"])
        train_file = output_dir / "train.txt"
        val_file = output_dir / "val.txt"

        # 训练集（全量预加载）
        train_dataset = MemoryTimbreDataset(
            file_list=str(train_file),
            segment_length=data_config["segment_length"],
            sample_rate=data_config["sample_rate"],
            augment=True,
            align_df_delay=align_df_delay,
            align_max_shift=align_max_shift,
        )
        # 验证集（全量预加载）
        val_dataset = MemoryTimbreDataset(
            file_list=str(val_file),
            segment_length=data_config["segment_length"],
            sample_rate=data_config["sample_rate"],
            augment=False,
            align_df_delay=align_df_delay,
            align_max_shift=align_max_shift,
        )

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise SystemExit("训练/验证集为空，请检查数据生成。")

        self.train_len = len(train_dataset)
        self.val_len = len(val_dataset)

        # 内存版不建议多进程，强制 num_workers=0
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1,
            drop_last=True,
        )

        if self.is_main:
            print(f"[Memory] Train samples: {len(train_dataset)}")
            print(f"[Memory] Val samples: {len(val_dataset)}")


def main():
    parser = argparse.ArgumentParser(description="Train Timbre Restoration Model (In-Memory)")
    parser.add_argument("--config", type=str, default="configs/lightweight.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    trainer = TrainerMem(args.config, args.resume)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
