# Timbre Restoration Model

基于 **Stochastic Regeneration** 框架的因果音色修复模型，用于补偿 DeepFilterNet 降噪造成的音色损失。

参考论文：[DeepFilterGAN: A Full-band Real-time Speech Enhancement System with GAN-based Stochastic Regeneration](https://arxiv.org/abs/2505.23515)

## 特性

- ✅ **完全因果**：适合实时流式处理，无前瞻延迟
- ✅ **GAN 训练**：生成模型补偿预测模型的 over-suppression
- ✅ **轻量级**：~3.5M 参数，单帧处理 < 1ms
- ✅ **ONNX 导出**：便于 Rust/C++ 集成

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

修改 `configs/default.yaml` 中的数据路径：

```yaml
data:
  vctk_path: "/workspace/vctk/wav48_silence_trimmed"            # 英文
  aishell3_path: "/workspace/Aishell3/train/wav"                # 中文
  ir_path: "/data/impulse/datasets_fullband/impulse_responses"  # IR 递归加载多级子目录
  noise_path: "/data/freesound/datasets_fullband/noise_fullband" # 噪声目录，递归随机抽样
  output_dir: "/workspace/timbre_restore_data"

  # 中/英比例（权重），默认 0.3 : 0.7
  dataset_ratio:
    vctk: 0.3
    aishell3: 0.7
```

运行数据准备（支持分片并行 + 断点续跑）：

```bash
# 小规模调试（随机抽样 200 条，跳过 DF 加速）
python data/prepare_dataset.py \
  --config configs/default.yaml \
  --max_files 200 \
  --skip_df \
  --skip_existing

# 全量单进程（默认 24 worker，在进程内并行切片）
python data/prepare_dataset.py \
  --config configs/default.yaml \
  --skip_existing

# 全量分片并行（4 进程/4 卡示例）
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i \
  python data/prepare_dataset.py \
    --config configs/default.yaml \
    --shard-idx $i --shard-count 4 \
    --skip_existing \
    --num_workers 12 &
done
wait
```

关键参数：
- `--skip_existing` 已存在的 clean/degraded 直接跳过，便于断点续跑/多机多进程并行。
- `--shard-idx/--shard-count` 分片跑，不同进程/机器用不同的 shard-idx。
- `--max_files` 仅用于抽样调试。

### 3. 训练

单卡：
```bash
python train.py --config configs/default.yaml
```

多卡 DDP：
```bash
torchrun --nproc_per_node=4 train.py --config configs/default.yaml
```

或使用脚本：
```bash
chmod +x run.sh
./run.sh all  # 完整流程：准备 → 训练 → 导出
```

### 4. 导出 ONNX

```bash
python export_onnx.py \
    --checkpoint logs/xxx/checkpoints/checkpoint_final.pt \
    --output timbre_restore.onnx \
    --verify \
    --benchmark
```

## 训练时长估算

| 配置 | 数据量 | 50 Epochs |
|------|--------|-----------|
| 单卡 4090 | 10万样本 | ~6-8 小时 |
| 2× 4090 | 10万样本 | ~3.5-4.5 小时 |
| 4× 4090 | 10万样本 | ~2-2.5 小时 |

## 模型架构

```
输入 (DF输出) [B, 1, T]
       │
       ▼
┌─────────────────┐
│ Causal Encoder  │  Conv1d ↓ [64→128→256→512]
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  GRU Bottleneck │  因果时序建模
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Causal Decoder  │  Conv1d ↑ + Skip [512→256→128→64]
└─────────────────┘
       │
       ▼
输出 (修复) [B, 1, T]
```

## 集成到 DeepFilter

参见 `rust_integration/timbre_restore.rs` 示例。

核心集成点（在 `capture.rs`）：

```rust
// DF 处理后
lsnr = df.process(inframe.view(), outframe.view_mut())?;

// 音色修复
if let Some(buf) = outframe.as_slice_mut() {
    timbre_model.process_frame(buf)?;
}

// 后续 EQ/AGC
```

## 目录结构

```
timbre_restore/
├── configs/
│   └── default.yaml        # 训练配置
├── data/
│   ├── prepare_dataset.py  # 数据准备
│   └── dataset.py          # PyTorch Dataset
├── model/
│   ├── generator.py        # 因果 U-Net Generator
│   ├── discriminator.py    # 多尺度判别器
│   └── losses.py           # 损失函数
├── rust_integration/
│   └── timbre_restore.rs   # Rust 集成示例
├── train.py                # 训练入口
├── export_onnx.py          # ONNX 导出
├── run.sh                  # 运行脚本
└── requirements.txt        # Python 依赖
```


## 随机性与过拟合预防

- 数据抽样：干净语音按 `dataset_ratio` 权重随机抽取（含洗牌），IR/噪声目录递归加载，噪声/IR 每次调用都会随机选取，确保样本多样。
- 批次顺序：数据列表与任务列表均打乱，训练 DataLoader 默认 `shuffle=True`。
- 如需复现，可在外部设置随机种子（`PYTHONHASHSEED`、`torch.manual_seed` 等）；默认不固定种子以增加多样性，帮助泛化。

## License

MIT
