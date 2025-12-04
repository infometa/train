# 数据质量分析报告

**测试日期**：2024年  
**测试结果**：存在问题，需要关注

---

## 📊 测试结果

```
shift_samples: mean=9.7, min=-2, max=902
len_diff: mean=0.0, min=0, max=0
SNR(dB): mean=10.43, min=0.12, max=76.85
```

---

## 🔍 问题分析

### ❌ 问题 1：对齐效果不佳（中等严重）

**现象**：
- `shift mean=9.7`：平均偏移 9.7 个样本（约 0.2ms @ 48kHz）
- `shift max=902`：最差偏移 902 个样本（约 19ms @ 48kHz）
- `shift min=-2`：最好情况接近完美对齐

**严重程度**：🟡 **中等**

**影响**：
1. **平均偏移 9.7 样本**：
   - 约 0.2ms，对于音频处理来说基本可接受
   - 但理想情况应该接近 0（±2 样本以内）
   - 说明存在轻微的系统性偏移

2. **最大偏移 902 样本**：
   - 约 19ms，这是严重的对齐失败
   - 对于 1 秒的训练片段，19ms 的偏移会导致：
     - 模型学习错误的对齐关系
     - 残差学习效果差
     - 训练不稳定

**原因分析**：

1. **DeepFilterNet 改变了波形**：
   ```
   数据流程：
   clean（原始） → add_noise → noisy → DeepFilterNet → degraded
                                                          ↓
   训练时：                                        clean vs degraded
   ```
   - DeepFilterNet 的降噪会改变波形
   - 改变越大，与原始 clean 的相关性越低
   - 互相关对齐算法依赖信号相似性，相关性低时效果差

2. **低 SNR 样本问题**：
   - 噪声很大的样本（SNR < 10dB）
   - DeepFilterNet 处理后可能失真严重
   - 导致与原始 clean 难以对齐

3. **对齐算法局限性**：
   - 当前算法：`dataset.py` Line 85-120
   - 使用 FFT 互相关，在 16kHz 下采样后计算
   - 对于波形差异大的信号，可能找不到正确的峰值

**验证建议**：

检查对齐失败的样本：

```python
import random, numpy as np, pathlib, sys
sys.path.insert(0,'.')
from data.dataset import TimbreRestoreDataset

ds = TimbreRestoreDataset(
    "/data/train_data_lite/train.txt",
    segment_length=48000,
    sample_rate=48000,
    align_df_delay=True,
    align_max_shift=4000,
    augment=False,
)

# 找出对齐最差的样本
worst_shifts = []
for i in range(min(500, len(ds))):
    d, c = ds[i]
    x, y = d.numpy().flatten(), c.numpy().flatten()
    n = min(len(x), len(y), 48000)
    x, y = x[:n], y[:n]
    corr = np.correlate(y, x, mode='full')
    lag = np.argmax(corr) - (n-1)
    if abs(lag) > 100:  # 偏移超过100个样本（2ms）
        worst_shifts.append((i, lag, ds.pairs[i]))

print(f"发现 {len(worst_shifts)} 个对齐不佳的样本")
for i, lag, (deg_path, clean_path) in worst_shifts[:10]:
    print(f"  样本 {i}: shift={lag}, degraded={deg_path}")

# 手动检查这些样本的音频质量
import soundfile as sf
if len(worst_shifts) > 0:
    idx, lag, (deg_path, clean_path) = worst_shifts[0]
    print(f"\n最差样本: shift={lag}")
    print(f"  degraded: {deg_path}")
    print(f"  clean: {clean_path}")
    # 可以听一下这两个文件，看看是否相关性很低
```

---

### 🟡 问题 2：SNR 分布异常（中等）

**现象**：
- `SNR mean=10.43 dB`：平均 SNR 偏低
- `SNR min=0.12 dB`：远低于配置的 5 dB
- `SNR max=76.85 dB`：远高于配置的 30 dB
- 配置的 `snr_range: [5, 30]`

**严重程度**：🟡 **中等**（但可能是正常的）

**原因分析**：

这个 **可能是正常的**，因为：

1. **数据准备时的 SNR**：
   ```python
   # prepare_dataset.py Line 227
   snr = random.uniform(snr_range[0], snr_range[1])  # 5-30 dB
   degraded = add_noise(degraded, noise_arr, snr)
   # 这里计算的是 clean vs noisy 的 SNR
   ```

2. **DeepFilterNet 改变了 SNR**：
   ```
   数据流程：
   clean ─┐
          ├→ SNR=5-30dB ─→ noisy ─→ DeepFilterNet ─→ degraded
   (原始)                                                ↓
                                                  实际 SNR 不同！
   ```
   - DeepFilterNet 的作用是降噪
   - **成功降噪** → SNR 大幅提高（可能到 40-80 dB）
   - **降噪失败** → SNR 可能更低
   - **降噪过度** → 信号失真，SNR 计算异常

3. **测试脚本测量的是最终 SNR**：
   ```python
   # README 自检脚本
   nse = y - x  # clean - degraded
   snr = 10*np.log10(np.mean(y**2)/(np.mean(nse**2)+1e-9))
   ```
   - 这是训练时模型看到的真实 SNR
   - 包含了 DeepFilterNet 的影响

**是否正常**？

- ✅ **SNR max=76.85 dB**：正常，说明 DeepFilterNet 在某些样本上降噪非常成功
- ⚠️ **SNR min=0.12 dB**：可疑，说明有样本几乎没有信号（全是噪声）
- ⚠️ **SNR mean=10.43 dB**：偏低，理论上经过 DeepFilterNet 应该提高

**可能的问题**：

1. **对齐不准导致 SNR 计算错误**：
   - 如果 `degraded` 和 `clean` 没有对齐
   - 计算 `nse = clean - degraded` 时包含了对齐误差
   - 导致 SNR 被低估

2. **DeepFilterNet 处理失败的样本**：
   - 某些样本 DeepFilterNet 可能降噪失败
   - 或者过度处理导致信号失真
   - 这些样本可能不适合训练

3. **极低 SNR 样本（0.12 dB）**：
   - 可能是 DeepFilterNet 几乎完全破坏了信号
   - 或者对齐完全失败
   - 这种样本应该被过滤掉

**验证建议**：

检查低 SNR 样本：

```python
# 找出 SNR < 3 dB 的样本
low_snr_samples = []
for i in range(min(500, len(ds))):
    d, c = ds[i]
    x, y = d.numpy().flatten(), c.numpy().flatten()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    nse = y - x
    snr = 10*np.log10(np.mean(y**2)/(np.mean(nse**2)+1e-9))
    if snr < 3.0:
        low_snr_samples.append((i, snr, ds.pairs[i]))

print(f"发现 {len(low_snr_samples)} 个低 SNR 样本")
for i, snr, (deg_path, clean_path) in low_snr_samples[:10]:
    print(f"  样本 {i}: SNR={snr:.2f}dB")

# 计算低 SNR 样本的占比
low_snr_ratio = len(low_snr_samples) / min(500, len(ds))
print(f"低 SNR 样本占比: {low_snr_ratio*100:.1f}%")

# 如果占比 > 10%，说明有严重问题
```

---

### ✅ 问题 3：长度差异（正常）

**现象**：
- `len_diff: mean=0.0, min=0, max=0`

**评估**：✅ **完全正常**

长度对齐没有问题。

---

## 🎯 综合评估

### 数据质量等级：🟡 **中等（存在问题）**

| 指标 | 状态 | 严重程度 | 影响 |
|------|------|---------|------|
| **对齐偏移** | 🟡 不佳 | 中等 | 个别样本对齐失败 |
| **SNR 分布** | 🟡 异常 | 中等 | 可能影响训练 |
| **长度对齐** | ✅ 正常 | 无 | 无影响 |

---

## 💡 可能的原因总结

### 根本原因：DeepFilterNet 改变了信号

1. **波形改变** → 对齐困难
2. **SNR 改变** → 分布异常
3. **某些样本处理失败** → 极端值

### 具体问题

1. **max shift=902**：
   - 少数样本对齐严重失败
   - 可能是 DeepFilterNet 处理后与原始音频相关性太低

2. **SNR min=0.12**：
   - 极少数样本几乎没有信号
   - 可能是 DeepFilterNet 失败或对齐失败

3. **SNR mean=10.43**：
   - 偏低，但可能是因为对齐误差导致 SNR 被低估
   - 或者 DeepFilterNet 整体效果不够好

---

## 🔧 建议

### 1. 量化问题严重程度（必做，5分钟）

运行上面的验证脚本，统计：
- 对齐失败（shift > 100）的样本占比
- 低 SNR（< 3 dB）的样本占比

**判断标准**：
- 对齐失败 < 5%：可接受，训练时影响不大
- 对齐失败 5-10%：中等，建议改进
- 对齐失败 > 10%：严重，必须修复

- 低 SNR < 5%：可接受
- 低 SNR 5-15%：中等
- 低 SNR > 15%：严重

### 2. 可视化检查（建议，10分钟）

随机抽取几对样本，画波形和频谱：

```python
import matplotlib.pyplot as plt
import soundfile as sf

# 随机选一个样本
idx = random.randint(0, len(ds)-1)
d, c = ds[idx]
deg_path, clean_path = ds.pairs[idx]

# 加载原始文件（看看对齐前的样子）
deg_raw, _ = sf.read(deg_path)
clean_raw, _ = sf.read(clean_path)

# 画图
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# 时域波形
axes[0, 0].plot(clean_raw[:4800])
axes[0, 0].set_title('Clean (raw)')
axes[0, 1].plot(deg_raw[:4800])
axes[0, 1].set_title('Degraded (raw)')

# 对齐后
axes[1, 0].plot(c.numpy().flatten()[:4800])
axes[1, 0].set_title('Clean (aligned)')
axes[1, 1].plot(d.numpy().flatten()[:4800])
axes[1, 1].set_title('Degraded (aligned)')

# 频谱
axes[2, 0].specgram(c.numpy().flatten(), Fs=48000)
axes[2, 0].set_title('Clean spectrum')
axes[2, 1].specgram(d.numpy().flatten(), Fs=48000)
axes[2, 1].set_title('Degraded spectrum')

plt.tight_layout()
plt.savefig('data_quality_check.png')
print(f"样本 {idx}: {clean_path}")
print(f"可视化保存到 data_quality_check.png")
```

看看：
- 波形是否对齐
- 频谱是否相似（DeepFilterNet 是否保留了主要频率成分）

### 3. 短期方案：过滤坏样本（可选）

如果问题样本占比不高（< 10%），可以考虑：
- 在 Dataset 中添加样本质量检查
- 跳过对齐失败或极低 SNR 的样本
- 但需要修改代码

### 4. 中期方案：改进对齐算法（可选）

如果对齐失败率较高（> 10%），可能需要：
- 使用更鲁棒的对齐方法（如 DTW）
- 在频域对齐
- 或者在数据准备时直接保存对齐偏移量

### 5. 长期方案：数据准备流程优化（可选）

考虑改变数据准备流程：
- 选项 A：训练时不用原始 clean，而是用 DeepFilterNet 的输入（noisy）
  - 这样对齐问题会小很多
  - 但模型学习的是 "noisy → clean" 而不是 "degraded → clean"

- 选项 B：在数据准备时预计算对齐并保存
  - 避免每次训练都重新计算
  - 可以使用更准确但更慢的对齐算法

---

## 📋 立即执行清单

### 1. 量化问题（必做，5分钟）

```bash
python - <<'PY'
import random, numpy as np, pathlib, sys
sys.path.insert(0,'.')
from data.dataset import TimbreRestoreDataset

ds = TimbreRestoreDataset(
    "/data/train_data_lite/train.txt",
    segment_length=48000,
    sample_rate=48000,
    align_df_delay=True,
    align_max_shift=4000,
    augment=False,
)

n_samples = min(500, len(ds))
bad_align = 0
low_snr = 0

for i in range(n_samples):
    d, c = ds[i]
    x, y = d.numpy().flatten(), c.numpy().flatten()
    n = min(len(x), len(y), 48000)
    x, y = x[:n], y[:n]
    
    # 检查对齐
    corr = np.correlate(y, x, mode='full')
    lag = np.argmax(corr) - (n-1)
    if abs(lag) > 100:
        bad_align += 1
    
    # 检查 SNR
    nse = y - x
    snr = 10*np.log10(np.mean(y**2)/(np.mean(nse**2)+1e-9))
    if snr < 3.0:
        low_snr += 1

print(f"测试样本数: {n_samples}")
print(f"对齐失败 (shift>100): {bad_align} ({bad_align/n_samples*100:.1f}%)")
print(f"低 SNR (<3dB): {low_snr} ({low_snr/n_samples*100:.1f}%)")

if bad_align/n_samples > 0.10:
    print("\n⚠️ 警告：对齐失败率 > 10%，建议修复")
elif bad_align/n_samples > 0.05:
    print("\n🟡 注意：对齐失败率 5-10%，可接受但建议改进")
else:
    print("\n✅ 对齐质量良好")

if low_snr/n_samples > 0.15:
    print("⚠️ 警告：低 SNR 样本 > 15%，建议检查数据准备流程")
elif low_snr/n_samples > 0.05:
    print("🟡 注意：低 SNR 样本 5-15%，可接受但建议改进")
else:
    print("✅ SNR 分布良好")
PY
```

### 2. 根据结果决定下一步

- **对齐失败 < 5% 且 低SNR < 5%**：✅ 数据质量良好，可以开始训练
- **对齐失败 5-10% 或 低SNR 5-15%**：🟡 可以训练，但建议监控训练效果
- **对齐失败 > 10% 或 低SNR > 15%**：⚠️ 建议先改进数据质量

---

## 🎯 预期结果

### 如果数据质量良好（问题样本 < 5%）

可以直接开始训练，训练时：
- 监控 Loss 曲线
- 如果 Loss 不稳定或不收敛，可能是问题样本导致
- 考虑添加样本过滤逻辑

### 如果数据质量中等（问题样本 5-15%）

可以训练，但：
- 预期训练会略微不稳定
- 少数样本可能产生很大的 Loss（outliers）
- 建议使用 gradient clipping（已配置）
- 考虑添加 Loss 异常检测

### 如果数据质量差（问题样本 > 15%）

建议先改进数据：
- 检查 DeepFilterNet 配置
- 改进对齐算法
- 或过滤坏样本

---

## ✅ 总结

**当前状态**：🟡 **数据质量中等，存在问题但可能可接受**

**关键问题**：
1. 个别样本对齐失败（max shift=902）
2. SNR 分布异常（0.12-76.85 dB）

**下一步**：
1. ✅ 运行量化脚本（5分钟）
2. 📊 根据统计结果决定是否需要改进
3. 🚀 如果问题样本 < 10%，可以开始训练并监控效果

**重要提醒**：
- 上面的异常可能部分是正常的（DeepFilterNet 的影响）
- 需要量化统计才能判断严重程度
- 少数问题样本（< 5%）对训练影响不大

