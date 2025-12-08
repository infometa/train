#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 SNR 过滤已有数据集，快速生成一个去除低 SNR 样本的新数据集。

使用场景：
- 已经生成了 /data/train_full_clean（含 train.txt/val.txt）
- 不想重跑全量数据生成，只想去掉极低 SNR (< 阈值) 的样本后重训

默认行为：
- 从 input/train.txt 和 input/val.txt 读取配对列表
- 计算每条 (degraded, clean) 的 SNR，低于阈值则丢弃
- 将保留样本硬链接（或复制）到新的 output 目录，并写出新的 train.txt/val.txt
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_audio(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def compute_snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    n = min(len(clean), len(degraded))
    if n == 0:
        return -np.inf
    clean = clean[:n]
    degraded = degraded[:n]
    noise = clean - degraded
    p_clean = float(np.mean(clean ** 2) + 1e-12)
    p_noise = float(np.mean(noise ** 2) + 1e-12)
    return 10.0 * np.log10(p_clean / p_noise)


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)  # 硬链接，零拷贝
    except Exception:
        shutil.copy2(src, dst)  # 回退到复制


def filter_split(
    split: str,
    input_root: Path,
    output_root: Path,
    snr_threshold: float,
) -> Tuple[int, int, List[str]]:
    list_path = input_root / f"{split}.txt"
    if not list_path.exists():
        raise SystemExit(f"列表不存在: {list_path}")

    clean_dir_out = output_root / "clean"
    degraded_dir_out = output_root / "degraded"

    kept = 0
    dropped = 0
    out_lines: List[str] = []
    snrs: List[float] = []

    with list_path.open() as f:
        lines = [l.strip() for l in f.readlines() if "|" in l]

    for line in tqdm(lines, desc=f"Filtering {split}", ncols=80):
        deg_path_str, clean_path_str = line.split("|", 1)
        deg_path = Path(deg_path_str)
        clean_path = Path(clean_path_str)

        if not deg_path.exists() or not clean_path.exists():
            dropped += 1
            continue

        clean = load_audio(clean_path)
        degraded = load_audio(deg_path)
        snr = compute_snr(clean, degraded)

        if snr < snr_threshold:
            dropped += 1
            continue

        # 输出文件名沿用原始 basename，避免冲突需保证输入已唯一
        fname = deg_path.name
        copy_or_link(deg_path, degraded_dir_out / fname)
        copy_or_link(clean_path, clean_dir_out / fname)

        out_lines.append(f"{degraded_dir_out / fname}|{clean_dir_out / fname}\n")
        snrs.append(snr)
        kept += 1

    out_list_path = output_root / f"{split}.txt"
    with out_list_path.open("w") as f:
        f.writelines(out_lines)

    mean_snr = np.mean(snrs) if snrs else 0.0
    print(f"[{split}] kept={kept}, dropped={dropped}, mean_snr={mean_snr:.2f} dB")
    return kept, dropped, out_lines


def main():
    parser = argparse.ArgumentParser(description="按 SNR 过滤数据集并生成新列表/文件")
    parser.add_argument("--input", type=str, required=True, help="原始数据集目录（含 train.txt/val.txt）")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--threshold", type=float, default=10.0, help="SNR 阈值（丢弃低于该值的样本）")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"SNR threshold: {args.threshold} dB")

    total_kept = 0
    total_dropped = 0
    for split in ["train", "val"]:
        kept, dropped, _ = filter_split(split, input_root, output_root, args.threshold)
        total_kept += kept
        total_dropped += dropped

    print("\nDone!")
    print(f"Total kept   : {total_kept}")
    print(f"Total dropped: {total_dropped}")
    print(f"New lists    : {output_root/'train.txt'}, {output_root/'val.txt'}")


if __name__ == "__main__":
    main()
