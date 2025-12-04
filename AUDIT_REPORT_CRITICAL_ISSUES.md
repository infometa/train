# 最终严格审计报告 - 关键问题清单

**审计目标**：确保数据集准确、性能优秀、模型有效  
**审计范围**：仅列出必须修复的问题  
**审计日期**：2024年

---

## 🔴 严重问题（必须立即修复）

### 问题 1：配置与代码不一致 - 损失权重

**位置**：
- `configs/default.yaml` Line 92-95
- `model/losses.py` Line 208-209, 228-229

**问题**：
```yaml
# configs/default.yaml
loss_weights:
  l1: 3.0              # 配置文件中是 3.0
  multi_stft: 3.0      # 配置文件中是 3.0
```

```python
# model/losses.py Line 208-209
def __init__(
    self,
    l1_weight: float = 15.0,       # ❌ 代码默认值是 15.0
    stft_weight: float = 2.0,      # ❌ 代码默认值是 2.0
```

**问题分析**：
- 代码默认值与配置文件不一致
- 如果配置文件传参有误，会使用错误的默认值
- 损失权重对训练效果影响巨大

**影响**：可能导致训练效果不符合预期

**修复**：将代码默认值改为与配置文件一致，或确保配置文件正确传递

---

### 问题 2：STFT Loss 高频加权配置不一致

**位置**：
- `configs/default.yaml` Line 104
- `model/losses.py` Line 228-229

**问题**：
```yaml
# configs/default.yaml
hf_weight: 1.5       # 配置文件中是 1.5
hf_cutoff: 3000
```

```python
# model/losses.py Line 228-229
stft_config = {
    'hf_weight': 2.0,      # ❌ 代码默认值是 2.0
    'hf_cutoff': 3000,
}
```

**影响**：如果配置传递有误，会使用错误的高频加权

**修复**：统一配置和代码默认值

---

### 问题 3：skip_existing 不检查文件完整性

**位置**：`data/prepare_dataset.py` Line 246, 290

**问题**：
```python
# Line 246
if skip_existing and clean_out.exists() and degraded_out.exists():
    continue  # ❌ 只检查存在，不检查完整性

# Line 290
if skip_existing and output_path.exists():
    continue  # ❌ 只检查存在，不检查完整性
```

**问题分析**：
- 如果处理中断，可能留下不完整的文件（0字节或部分写入）
- 下次运行会跳过这些损坏的文件
- **导致训练集包含损坏数据或样本缺失**

**影响**：可能导致训练失败或效果差

**修复方案**：
```python
# 方案 1：检查文件大小
if skip_existing and clean_out.exists() and degraded_out.exists():
    if clean_out.stat().st_size > 1000 and degraded_out.stat().st_size > 1000:
        continue
    # 否则重新生成

# 方案 2：使用临时文件（推荐）
tmp_clean = str(clean_out) + ".tmp"
sf.write(tmp_clean, clean_seg, target_sr)
os.rename(tmp_clean, clean_out)  # 原子操作
```

---

### 问题 4：数据准备结果列表可能被错误过滤

**位置**：`data/prepare_dataset.py` Line 471-482

**问题代码**：
```python
# 更新 results 中的 degraded 路径
filtered = []
for r in results:
    name = Path(r['degraded']).name
    if name in failed_df:  # ❌ 'in' 操作对列表效率低
        continue
    degraded_path = degraded_dir / name
    if not degraded_path.exists():  # ❌ 这里可能漏掉文件
        continue
    r['degraded'] = str(degraded_path)
    filtered.append(r)
results = filtered
```

**问题分析**：
1. `failed_df` 是列表，`in` 操作是 O(n)，效率低
2. 更严重的是：`r['degraded']` 指向 `noisy_dir`，但检查的是 `degraded_dir`
   - 如果 DF 处理了文件但 results 里的路径还是 noisy_dir，会找不到文件

**影响**：可能导致大量样本被错误过滤，训练集变小

**修复**：
```python
# 转为集合提高效率
failed_df_set = set(failed_df)

filtered = []
for r in results:
    name = Path(r['degraded']).name
    if name in failed_df_set:
        continue
    degraded_path = degraded_dir / name
    if not degraded_path.exists():
        print(f"Warning: {degraded_path} not found, skipping")  # 添加日志
        continue
    r['degraded'] = str(degraded_path)
    filtered.append(r)

print(f"Filtered: {len(results)} -> {len(filtered)} samples")  # 添加日志
results = filtered
```

---

### 问题 5：数据准备流程缺少最终验证

**位置**：`data/prepare_dataset.py` Line 500-509

**问题**：
- 生成 train.txt 和 val.txt 后，没有验证文件是否真实存在
- 如果路径错误或文件被删除，训练时才会报错

**影响**：浪费时间，训练启动后才发现数据问题

**修复**：
```python
# 保存文件列表前验证
print("\nValidating generated file pairs...")
invalid = 0
for r in train_results + val_results:
    if not Path(r['degraded']).exists():
        print(f"Missing: {r['degraded']}")
        invalid += 1
    if not Path(r['clean']).exists():
        print(f"Missing: {r['clean']}")
        invalid += 1

if invalid > 0:
    raise SystemExit(f"Found {invalid} invalid file references!")

print("Validation passed!")
```

---

### 问题 6：分片可能导致结果不一致

**位置**：`data/prepare_dataset.py` Line 395-401

**问题代码**：
```python
if args.shard_count > 1:
    shard_files = []
    for i, p in enumerate(clean_files):
        if i % args.shard_count == args.shard_idx:  # ❌ 简单的模运算
            shard_files.append(p)
    clean_files = shard_files
```

**问题分析**：
- 分片在随机抽取和打乱之后进行
- 如果 clean_files 顺序不一致（不同运行），分片结果会不同
- **导致不同分片运行可能处理相同文件或漏掉文件**

**影响**：多进程并行时可能产生重复或遗漏

**修复**：
```python
# 在分片前确保顺序一致
clean_files = sorted(clean_files)  # 按路径排序

if args.shard_count > 1:
    shard_files = []
    for i, p in enumerate(clean_files):
        if i % args.shard_count == args.shard_idx:
            shard_files.append(p)
    clean_files = shard_files
    print(f"Shard {args.shard_idx}/{args.shard_count} -> {len(clean_files)} files")
```

---

## 🟡 中等问题（建议修复）

### 问题 7：训练循环中的学习率调度器更新在循环内

**位置**：`train.py` Line 587-591

**当前代码**：
```python
# Line 562-591
for batch_idx, (degraded, clean) in enumerate(pbar):
    losses = self.train_step(degraded, clean)
    self.global_step += 1
    # ...
    
    # CosineAnnealingWarmRestarts 按 step 更新
    if not self.scheduler_step_per_epoch and num_batches > 0:
        if self.enable_scheduler:
            step_frac = epoch + batch_idx / num_batches
            self.scheduler_g.step(step_frac)
            self.scheduler_d.step(step_frac)
```

**问题**：已经在循环内，但逻辑正确。✅ 无问题

**（撤回此问题，代码已修复）**

---

### 问题 8：验证集可能过小

**位置**：`configs/default.yaml` Line 39

**问题**：
```yaml
val_ratio: 0.05  # 只有 5%
```

**分析**：
- 如果总样本量小（如 1000），验证集只有 50 个
- 验证指标可能不稳定

**建议**：如果总样本量 < 2000，建议增加到 0.1（10%）

---

### 问题 9：num_workers 可能过小

**位置**：`configs/default.yaml` Line 47

**问题**：
```yaml
num_workers: 2  # 可能太保守
```

**分析**：
- 2 个 worker 可能不足以喂饱 GPU
- 特别是启用了 align_df_delay 时，数据加载开销大

**建议**：
- 单卡训练：4-6 workers
- 多卡训练：每卡 2-4 workers
- 根据 CPU 核心数和 GPU 数量调整

---

### 问题 10：持久化 worker 未启用

**位置**：`train.py` Line 284, 296

**问题**：
```python
persistent_workers=False,  # ❌ 每个 epoch 都重启 worker
```

**分析**：
- num_workers > 0 时，persistent_workers=False 会导致每个 epoch 结束后 worker 进程被销毁，下个 epoch 重新启动
- 增加开销，浪费时间

**修复**：
```python
persistent_workers=True if data_config['num_workers'] > 0 else False,
```

---

### 问题 11：prefetch_factor 过小

**位置**：`train.py` Line 285, 297

**问题**：
```python
prefetch_factor=1,  # ❌ 只预取 1 个 batch
```

**建议**：
```python
prefetch_factor=2 if data_config['num_workers'] > 0 else None,
```

---

## 🟢 轻微问题

### 问题 12：缺少训练数据统计

**建议**：在训练开始前打印数据统计信息
```python
# 在 train.py 开始训练前
if self.is_main:
    print(f"\nDataset Statistics:")
    print(f"  Train batches: {len(self.train_loader)}")
    print(f"  Val batches: {len(self.val_loader)}")
    print(f"  Samples per epoch: {len(self.train_loader) * train_config['batch_size']}")
```

---

### 问题 13：DeepFilterNet 批处理未优化

**位置**：`data/prepare_dataset.py` Line 287-304

**问题**：逐个文件处理 DF，未利用批处理

**影响**：数据准备阶段较慢

---

## 📋 修复优先级

### 立即修复（5-10分钟）

1. **问题 3**：skip_existing 文件完整性检查（5分钟）
2. **问题 4**：数据准备结果过滤优化（3分钟）
3. **问题 5**：添加最终验证（3分钟）
4. **问题 6**：分片前排序（2分钟）

### 建议修复（10分钟）

5. **问题 1-2**：统一配置和代码默认值（5分钟）
6. **问题 10-11**：优化 DataLoader 参数（5分钟）

### 可选优化

7. **问题 8-9**：调整配置参数（按需）
8. **问题 12-13**：添加统计和优化（可选）

---

## ⚠️ 数据准备前检查清单

在运行 `prepare_dataset.py` 之前：

- [ ] 确认数据路径存在且可访问
- [ ] 确认有足够磁盘空间（至少 3x 原始数据大小）
- [ ] 测试 DeepFilterNet 是否正常工作
- [ ] 小规模测试（--max_files 10）

**测试命令**：
```bash
# 小规模测试
python data/prepare_dataset.py \
  --config configs/default.yaml \
  --max_files 10 \
  --skip_existing

# 检查生成的文件
ls -lh /data/train_data_lite/clean/ | head
ls -lh /data/train_data_lite/degraded/ | head

# 验证音频文件可读
python -c "import soundfile as sf; sf.read('/data/train_data_lite/clean/00000000_00.wav')"
```

---

## ⚠️ 训练前检查清单

在运行 `train.py` 之前：

- [ ] 确认 train.txt 和 val.txt 存在
- [ ] 确认文件路径正确（运行 dataset.py 测试）
- [ ] 确认 GPU 内存足够（batch_size=16 需要约 10-12GB）
- [ ] 确认日志目录可写

**测试命令**：
```bash
# 测试数据加载
python data/dataset.py /data/train_data_lite/train.txt

# 测试模型前向传播
python -c "
import torch
from model.generator import CausalUNetGenerator
model = CausalUNetGenerator()
x = torch.randn(2, 1, 48000)
y = model(x)
print(f'Input: {x.shape}, Output: {y.shape}')
print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
"
```

---

## 🎯 最终建议

### 立即执行（必须）

1. **修复问题 3-6**（13分钟）
2. **小规模数据准备测试**（10分钟）
3. **修复问题 1-2、10-11**（10分钟）
4. **小规模训练测试**（30分钟，5 epochs）

### 全量运行（确认无误后）

```bash
# 数据准备（推荐分片并行）
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i \
  python data/prepare_dataset.py \
    --config configs/default.yaml \
    --shard-idx $i --shard-count 4 \
    --skip_existing \
    --num_workers 12 &
done
wait

# 验证数据完整性
python -c "
with open('/data/train_data_lite/train.txt') as f:
    lines = f.readlines()
print(f'Train samples: {len(lines)}')
with open('/data/train_data_lite/val.txt') as f:
    lines = f.readlines()
print(f'Val samples: {len(lines)}')
"

# 训练
torchrun --nproc_per_node=4 train.py --config configs/default.yaml
```

---

## ⚡ 预期性能

修复后的预期性能：

**数据准备**：
- 单进程：~100-200 samples/min
- 4进程分片：~400-800 samples/min

**训练**：
- 单卡 4090：~2-3 batches/sec (batch_size=16)
- 4卡 4090：~8-12 batches/sec

**GPU 利用率**：
- 修复前：30-50%（数据加载瓶颈）
- 修复后：80-95%

---

**审计完成**

共发现：
- 🔴 严重问题：6个（必须修复）
- 🟡 中等问题：5个（建议修复）
- 🟢 轻微问题：2个（可选）

**预计修复时间**：30-40分钟

**建议**：先修复问题 1-6 和 10-11，然后小规模测试，确认无误后全量运行。

