#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本：生成 DeepFilterNet 输出与干净音频的配对数据

流程：
1. 加载干净音频 (VCTK/Aishell3)
2. 随机添加混响 (IR 卷积)
3. 随机添加噪声 (指定 SNR)
4. 过 DeepFilterNet 得到退化音频
5. 保存配对 (degraded, clean)
"""

import os
import sys
import argparse
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from tqdm import tqdm
import yaml

# DeepFilterNet
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from df import config
    DF_AVAILABLE = True
except ImportError:
    print("Warning: DeepFilterNet not found. Install with: pip install deepfilternet")
    DF_AVAILABLE = False


@dataclass
class AudioFile:
    path: Path
    duration: float
    sample_rate: int


def scan_audio_files(root_dir: str, extensions: List[str] = ['.wav', '.flac']) -> List[Path]:
    """递归扫描音频文件"""
    files = []
    root = Path(root_dir)
    for ext in extensions:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def load_audio_file(path: Path, target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """加载音频文件并重采样"""
    audio, sr = sf.read(path)
    
    # 转为单声道
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    # 重采样
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio.astype(np.float32), target_sr


def load_noise_file(path: str, target_sr: int = 48000) -> np.ndarray:
    """加载单个噪声文件"""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def list_audio_paths(root_dir: str, extensions: List[str]) -> List[Path]:
    """列出目录下所有匹配的音频路径"""
    paths = []
    root = Path(root_dir)
    for ext in extensions:
        paths.extend(root.rglob(f"*{ext}"))
    return sorted(paths)


def load_ir_files(ir_dir: str, target_sr: int = 48000) -> List[np.ndarray]:
    """加载所有 IR 文件"""
    ir_files = scan_audio_files(ir_dir, ['.wav'])
    irs = []
    for ir_path in ir_files:
        ir, sr = sf.read(ir_path)
        if ir.ndim > 1:
            ir = ir.mean(axis=1)
        if sr != target_sr:
            ir = librosa.resample(ir, orig_sr=sr, target_sr=target_sr)
        # 归一化 IR
        ir = ir / (np.abs(ir).max() + 1e-8)
        irs.append(ir.astype(np.float32))
    return irs


def add_reverb(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """添加混响（卷积）"""
    reverbed = signal.fftconvolve(audio, ir, mode='full')[:len(audio)]
    # 归一化防止削波
    max_val = np.abs(reverbed).max()
    if max_val > 0.99:
        reverbed = reverbed * 0.99 / max_val
    return reverbed.astype(np.float32)


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按指定 SNR 添加噪声"""
    # 多噪声文件时随机选一个
    if isinstance(noise, list) and len(noise) > 0:
        noise = random.choice(noise)
    # 随机选择噪声片段
    if len(noise) < len(audio):
        # 循环填充
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    
    start = random.randint(0, len(noise) - len(audio))
    noise_segment = noise[start:start + len(audio)]
    
    # 计算增益
    audio_power = np.mean(audio ** 2) + 1e-10
    noise_power = np.mean(noise_segment ** 2) + 1e-10
    
    snr_linear = 10 ** (snr_db / 10)
    noise_gain = np.sqrt(audio_power / (noise_power * snr_linear))
    
    noisy = audio + noise_gain * noise_segment
    
    # 归一化防止削波
    max_val = np.abs(noisy).max()
    if max_val > 0.99:
        noisy = noisy * 0.99 / max_val
    
    return noisy.astype(np.float32)


def process_with_deepfilter(audio: np.ndarray, sr: int, df_model, df_state) -> np.ndarray:
    """使用 DeepFilterNet 处理音频"""
    # DeepFilterNet enhance 函数
    enhanced = enhance(df_model, df_state, audio)
    return enhanced.astype(np.float32)


def process_single_file(
    args: Tuple[int, Path, Path, Path, List[Path], List[Path], dict]
) -> Optional[dict]:
    """处理单个文件（用于多进程）"""
    idx, clean_path, output_clean_dir, output_degraded_dir, irs, noise, config = args
    
    try:
        target_sr = config['sample_rate']
        segment_length = config['segment_length']
        save_segment_length = config.get('save_segment_length', segment_length)
        segment_hop = max(1, config.get('segment_hop', save_segment_length // 2))
        snr_range = config['snr_range']
        ir_prob = config['ir_prob']
        
        # 加载干净音频
        clean, sr = load_audio_file(clean_path, target_sr)
        
        # 跳过太短的音频
        if len(clean) < save_segment_length:
            return None
        
        # 多片段切分（滑窗）
        max_start = len(clean) - save_segment_length
        starts = list(range(0, max_start + 1, segment_hop))
        if starts[-1] != max_start:
            starts.append(max_start)
        
        samples = []
        for seg_idx, start in enumerate(starts):
            clean_seg = clean[start:start + save_segment_length]
            
            # 归一化
            clean_seg = clean_seg / (np.abs(clean_seg).max() + 1e-8) * 0.9
            
            # 创建退化版本
            degraded = clean_seg.copy()
            
            # 随机添加混响（按需加载）
            add_ir = random.random() < ir_prob and len(irs) > 0
            if add_ir:
                ir_path = random.choice(irs)
                ir, _ = load_audio_file(ir_path, target_sr)
                degraded = add_reverb(degraded, ir)
            
            # 随机添加噪声（按需加载）
            snr = random.uniform(snr_range[0], snr_range[1])
            if len(noise) == 0:
                raise RuntimeError("噪声列表为空")
            noise_path = random.choice(noise)
            noise_arr = load_noise_file(str(noise_path), target_sr)
            degraded = add_noise(degraded, noise_arr, snr)
            
            # 文件名
            file_id = f"{idx:08d}_{seg_idx:02d}"
            clean_out = output_clean_dir / f"{file_id}.wav"
            degraded_out = output_degraded_dir / f"{file_id}.wav"
            
            # 保存（这里保存的是加噪版本，后续批量过 DF）
            sf.write(clean_out, clean_seg, target_sr)
            sf.write(degraded_out, degraded, target_sr)
            
            samples.append({
                'id': file_id,
                'clean': str(clean_out),
                'degraded': str(degraded_out),
                'snr': snr,
                'has_reverb': add_ir
            })
        
        return samples
        
    except Exception as e:
        print(f"Error processing {clean_path}: {e}")
        return None


def run_deepfilter_batch(
    input_dir: Path,
    output_dir: Path,
    batch_size: int = 100
) -> None:
    """批量运行 DeepFilterNet"""
    if not DF_AVAILABLE:
        print("DeepFilterNet not available, skipping...")
        return
    
    print("Initializing DeepFilterNet...")
    model, df_state, _ = init_df()
    
    input_files = sorted(input_dir.glob("*.wav"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(input_files)} files with DeepFilterNet...")
    
    for i, input_path in enumerate(tqdm(input_files)):
        output_path = output_dir / input_path.name
        
        audio = None  # keep for fallback save
        sr = df_state.sr()
        try:
            # DeepFilterNet 期望 Torch Tensor，因此使用官方加载函数
            audio, _ = load_audio(str(input_path), sr=sr, verbose=False)
            enhanced = enhance(model, df_state, audio)
            save_audio(str(output_path), enhanced, sr)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            # 复制原文件作为 fallback
            try:
                if audio is None:
                    # 回退读取为 numpy
                    audio_np, _ = load_audio_file(input_path, sr)
                else:
                    # 转 numpy 并调整为 (T, C)
                    audio_np = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                    if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
                        audio_np = audio_np.T
                sf.write(output_path, audio_np, sr)
            except Exception as e2:
                print(f"Fallback save failed for {input_path}: {e2}; copying original.")
                output_path.write_bytes(input_path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description="准备音色修复训练数据")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=None, help="限制处理文件数（调试用）")
    parser.add_argument("--skip_df", action="store_true", help="跳过 DeepFilterNet 处理")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    # 配置校验
    if 'save_segment_length' in data_config:
        if data_config['save_segment_length'] < data_config['segment_length']:
            raise ValueError(
                f"save_segment_length ({data_config['save_segment_length']}) 必须 >= segment_length ({data_config['segment_length']})"
            )
    output_dir = Path(data_config['output_dir'])
    
    # 创建输出目录
    clean_dir = output_dir / "clean"
    noisy_dir = output_dir / "noisy"  # 加噪后（DF 处理前）
    degraded_dir = output_dir / "degraded"  # DF 处理后
    
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    degraded_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描干净音频文件
    print("Scanning audio files...")
    clean_files = []
    vctk_files = []
    aishell_files = []
    
    # VCTK
    if os.path.exists(data_config['vctk_path']):
        vctk_files = scan_audio_files(data_config['vctk_path'], ['.flac'])
        print(f"  VCTK: {len(vctk_files)} files")
    
    # Aishell3
    if os.path.exists(data_config['aishell3_path']):
        aishell_files = scan_audio_files(data_config['aishell3_path'], ['.wav'])
        print(f"  Aishell3: {len(aishell_files)} files")
    
    # 按比例抽取（可选）
    dataset_ratio = data_config.get('dataset_ratio')
    if dataset_ratio:
        r_v = float(dataset_ratio.get('vctk', 0))
        r_a = float(dataset_ratio.get('aishell3', 0))
        ratio_sum = r_v + r_a
        if ratio_sum > 0:
            r_v /= ratio_sum
            r_a /= ratio_sum
        
        total_available = len(vctk_files) + len(aishell_files)
        target_total = args.max_files if args.max_files else total_available
        target_total = min(target_total, total_available)
        
        import random as _random
        _random.shuffle(vctk_files)
        _random.shuffle(aishell_files)
        
        v_target = min(len(vctk_files), int(round(target_total * r_v)))
        a_target = min(len(aishell_files), int(round(target_total * r_a)))
        
        clean_files = vctk_files[:v_target] + aishell_files[:a_target]
        
        # 分配剩余
        leftover = target_total - len(clean_files)
        if leftover > 0:
            extra = vctk_files[v_target:] + aishell_files[a_target:]
            _random.shuffle(extra)
            clean_files.extend(extra[:leftover])
    else:
        clean_files = vctk_files + aishell_files
        if args.max_files:
            clean_files = clean_files[:args.max_files]
    
    print(f"Total: {len(clean_files)} files")
    if len(clean_files) == 0:
        raise SystemExit("未找到任何干净音频，请检查 vctk_path/aishell3_path 配置和路径权限。")
    
    # 加载 IR 和噪声
    print("Loading IR files (paths only)...")
    if not os.path.exists(data_config['ir_path']):
        raise SystemExit(f"IR 路径不存在: {data_config['ir_path']}")
    irs = list_audio_paths(data_config['ir_path'], ['.wav'])
    print(f"  IR paths: {len(irs)}")
    if len(irs) == 0:
        raise SystemExit("IR 路径为空，无法添加混响，请检查 ir_path。")
    
    print("Loading noise file/dir (paths only)...")
    if not os.path.exists(data_config['noise_path']):
        raise SystemExit(f"噪声文件/目录不存在: {data_config['noise_path']}")
    if os.path.isdir(data_config['noise_path']):
        noise = list_audio_paths(data_config['noise_path'], ['.wav'])
        if len(noise) == 0:
            raise SystemExit(f"噪声目录为空: {data_config['noise_path']}")
        print(f"  Noise paths: {len(noise)}")
    else:
        noise = [Path(data_config['noise_path'])]
        print(f"  Noise file: {noise[0]}")
    
    # 准备多进程参数
    process_config = {
        'sample_rate': data_config['sample_rate'],
        'segment_length': data_config['segment_length'],
        'save_segment_length': data_config.get('save_segment_length', data_config['segment_length']),
        'segment_hop': data_config.get('segment_hop', data_config.get('save_segment_length', data_config['segment_length']) // 2),
        'snr_range': data_config['snr_range'],
        'ir_prob': data_config['ir_prob'],
    }
    
    tasks = [
        (i, path, clean_dir, noisy_dir, irs, noise, process_config)
        for i, path in enumerate(clean_files)
    ]
    
    # 随机打乱
    random.shuffle(tasks)
    
    # 多进程处理
    print(f"Processing with {args.num_workers} workers...")
    results = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
    
    print(f"Generated {len(results)} samples")
    
    # 运行 DeepFilterNet
    if not args.skip_df:
        run_deepfilter_batch(noisy_dir, degraded_dir)
    else:
        print("Skipping DeepFilterNet processing (--skip_df)")
        # 复制 noisy 到 degraded（调试用）
        import shutil
        for f in noisy_dir.glob("*.wav"):
            shutil.copy(f, degraded_dir / f.name)
    
    # 更新 results 中的 degraded 路径
    for r in results:
        r['degraded'] = str(degraded_dir / Path(r['degraded']).name)
    
    # 划分训练/验证集
    random.shuffle(results)
    val_size = int(len(results) * data_config['val_ratio'])
    
    train_results = results[val_size:]
    val_results = results[:val_size]
    
    # 保存元数据
    metadata = {
        'train': train_results,
        'val': val_results,
        'config': data_config
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 保存文件列表（简化格式）
    with open(output_dir / "train.txt", 'w') as f:
        for r in train_results:
            f.write(f"{r['degraded']}|{r['clean']}\n")
    
    with open(output_dir / "val.txt", 'w') as f:
        for r in val_results:
            f.write(f"{r['degraded']}|{r['clean']}\n")
    
    print(f"\nDataset prepared:")
    print(f"  Train: {len(train_results)}")
    print(f"  Val: {len(val_results)}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
