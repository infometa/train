# audioCompare.py (修复版 V2)
"""
AudioCompare Expert V2 - 自动对齐 + 高清分析版

修复日志：
1. [修复] 移除了 st.audio() 中不支持的 label 参数，改为 st.markdown() 显示。
2. [功能] 包含自动对齐、高清频谱、PSD、LSD/PESQ/STOI 计算。

依赖: pip install streamlit matplotlib numpy scipy librosa soundfile torch pesq pystoi
"""

import streamlit as st
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import librosa
import librosa.display
import io
import os
import urllib.request
from scipy import signal, stats
from scipy.signal import welch

# 尝试导入听感指标库
try:
    from pesq import pesq
    from pystoi import stoi
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# ================= 1. 字体配置 (防乱码) =================
def configure_font():
    font_name = "SimHei.ttf"
    font_url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
    
    if not os.path.exists(font_name):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(font_url, font_name)
        except Exception:
            pass

    if os.path.exists(font_name):
        try:
            fm.fontManager.addfont(font_name)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except: pass
    
    # 回退
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return False

HAS_FONT = configure_font()

# ================= 2. 核心算法：自动对齐 =================

def align_signals(ref, deg, sr, max_shift_ms=200):
    """
    自动对齐两个信号 (Cross-Correlation)
    """
    # 1. 粗略对齐不需要全长，取前 30秒 足够
    max_len = min(len(ref), len(deg), sr * 30)
    ref_slice = ref[:max_len]
    deg_slice = deg[:max_len]
    
    # 2. 归一化去直流
    ref_slice = ref_slice - np.mean(ref_slice)
    deg_slice = deg_slice - np.mean(deg_slice)
    
    # 3. 计算互相关
    corr = signal.correlate(ref_slice, deg_slice, mode='full', method='fft')
    lags = signal.correlation_lags(len(ref_slice), len(deg_slice), mode='full')
    
    # 找到峰值
    best_idx = np.argmax(corr)
    lag = lags[best_idx]
    
    # 4. 应用位移
    if lag > 0:
        deg_aligned = deg[lag:]
        ref_aligned = ref
    elif lag < 0:
        deg_aligned = deg[abs(lag):]
        ref_aligned = ref
    else:
        deg_aligned = deg
        ref_aligned = ref
        
    # 5. 再次强制等长截断
    min_len = min(len(ref_aligned), len(deg_aligned))
    return ref_aligned[:min_len], deg_aligned[:min_len], lag

def load_audio(file, target_sr=48000):
    audio, sr = sf.read(file)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr

# ================= 3. 指标计算 =================

def compute_metrics(ref, deg, sr):
    metrics = {}
    
    # SI-SNR
    eps = 1e-8
    ref_en = np.sum(ref**2) + eps
    scale = np.dot(ref, deg) / ref_en
    proj = scale * ref
    noise = deg - proj
    metrics['SI-SNR'] = 10 * np.log10(np.sum(proj**2) / (np.sum(noise**2) + eps))
    
    # LSD (分频段)
    S_ref = np.abs(librosa.stft(ref))**2
    S_deg = np.abs(librosa.stft(deg))**2
    log_diff = (10 * np.log10(S_ref+eps) - 10 * np.log10(S_deg+eps))**2
    
    freqs = librosa.fft_frequencies(sr=sr)
    mask_high = freqs > 10000
    metrics['LSD High'] = np.mean(np.sqrt(np.mean(log_diff[mask_high], axis=0)))
    metrics['LSD All'] = np.mean(np.sqrt(np.mean(log_diff, axis=0)))
    
    # L1
    metrics['L1'] = np.mean(np.abs(ref - deg))

    # PESQ/STOI
    if METRICS_AVAILABLE:
        try:
            r16 = librosa.resample(ref, orig_sr=sr, target_sr=16000)
            d16 = librosa.resample(deg, orig_sr=sr, target_sr=16000)
            metrics['PESQ'] = pesq(16000, r16, d16, 'wb')
            metrics['STOI'] = stoi(r16, d16, 16000)
        except: pass
        
    return metrics

# ================= 4. 界面逻辑 =================

st.set_page_config(layout="wide", page_title="AudioCompare Expert V2")
st.title("🎛️ 音色修复专家台 V2 (Auto-Align + Hi-Res)")

if not METRICS_AVAILABLE:
    st.warning("提示: 未安装 pesq/pystoi，听感指标将隐藏。")

# 侧边栏配置
st.sidebar.header("🔧 设置")
enable_align = st.sidebar.checkbox("启用自动对齐 (Auto-Align)", value=True, help="自动计算延迟并对齐音频，计算 Diff 和 SNR 必须开启。")
spectrogram_clim = st.sidebar.slider("频谱图动态范围 (dB)", min_value=40, max_value=120, value=80)

col1, col2 = st.columns([1, 2])
with col1:
    ref_file = st.file_uploader("1. 参考音频 (Ref/Clean)", type=["wav", "flac"])
    comp_files = st.file_uploader("2. 待测音频 (Models)", type=["wav", "flac"], accept_multiple_files=True)

if ref_file and comp_files:
    # 1. 加载 Ref
    ref_raw, sr = load_audio(ref_file)
    audio_store = {"Ref": ref_raw}
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 播放 (已对齐)")
    st.sidebar.markdown("**Ref (Original)**") # [修复] 手动显示标签
    st.sidebar.audio(ref_file) # [修复] 移除 label 参数

    # 2. 处理所有待测文件
    results = []
    pbar = st.progress(0)
    
    for i, f in enumerate(comp_files):
        deg_raw, _ = load_audio(f, target_sr=sr)
        
        # 对齐
        if enable_align:
            ref_aligned, deg_aligned, lag = align_signals(ref_raw, deg_raw, sr)
            status_text = f"✅ Shift: {lag}"
        else:
            min_l = min(len(ref_raw), len(deg_raw))
            ref_aligned = ref_raw[:min_l]
            deg_aligned = deg_raw[:min_l]
            status_text = "⚠️ Unaligned"
            
        audio_store[f.name] = deg_aligned
        
        # 计算
        m = compute_metrics(ref_aligned, deg_aligned, sr)
        
        row = {
            "Model": f.name,
            "Align": status_text,
            "SI-SNR": m['SI-SNR'],
            "LSD High": m['LSD High'],
            "PESQ": m.get('PESQ', 0),
            "STOI": m.get('STOI', 0),
            "L1": m['L1']
        }
        results.append(row)
        
        # [修复] 侧边栏播放列表
        st.sidebar.markdown(f"**{f.name}**") # 手动显示标签
        with io.BytesIO() as buf:
            sf.write(buf, deg_aligned, sr, format='WAV')
            st.sidebar.audio(buf) # 移除 label 参数
            
        pbar.progress((i + 1) / len(comp_files))
    
    pbar.empty()

    # === 展示数据 ===
    st.subheader("1. 核心指标对比")
    st.dataframe(
        results,
        column_config={
            "SI-SNR": st.column_config.NumberColumn("SI-SNR (dB) ↑", format="%.2f"),
            "LSD High": st.column_config.NumberColumn("高频失真 (LSD) ↓", format="%.2f"),
            "PESQ": st.column_config.NumberColumn("PESQ (听感) ↑", format="%.2f"),
            "L1": st.column_config.NumberColumn("L1 Error ↓", format="%.5f"),
        },
        use_container_width=True
    )
    
    # === 展示高清图表 ===
    st.subheader("2. 频谱与细节 (High-Res)")
    
    num_files = len(audio_store)
    fig = plt.figure(figsize=(14, 4 * num_files), dpi=150)
    gs = fig.add_gridspec(num_files, 2, width_ratios=[3, 1])

    for idx, (name, y) in enumerate(audio_store.items()):
        # 左图：Spectrogram
        ax_spec = fig.add_subplot(gs[idx, 0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=256)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', 
                               ax=ax_spec, cmap='magma', vmin=-spectrogram_clim, vmax=0)
        ax_spec.set_title(f"{name} - Spectrogram", fontsize=10)
        ax_spec.set_ylim(0, 24000)
        ax_spec.set_xlabel("")
        if idx == num_files - 1: ax_spec.set_xlabel("Time (s)")

        # 右图：PSD (高频特写)
        ax_psd = fig.add_subplot(gs[idx, 1])
        f_p, Pxx = welch(y, sr, nperseg=1024)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        
        if name != "Ref":
            f_ref, P_ref = welch(audio_store["Ref"], sr, nperseg=1024)
            ax_psd.plot(f_ref, 10*np.log10(P_ref+1e-12), color='grey', alpha=0.3, label='Ref')
            
        ax_psd.plot(f_p, Pxx_db, color='tab:orange', linewidth=1.5, label=name)
        ax_psd.set_title("PSD (High Freq)", fontsize=10)
        ax_psd.set_xlim(8000, 24000)
        ax_psd.set_ylim(-100, -20)
        ax_psd.grid(True, alpha=0.3)
        ax_psd.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    
    # === 差分试听 ===
    st.subheader("3. 差分检视 (Residual Check)")
    st.markdown("播放 `Clean - Restored`。**听到了清晰人声 = 修复失败（丢失信息）。**")
    
    diff_cols = st.columns(len(comp_files))
    ref_wav = audio_store["Ref"]
    
    for i, f in enumerate(comp_files):
        name = f.name
        deg_wav = audio_store[name]
        l = min(len(ref_wav), len(deg_wav))
        diff = ref_wav[:l] - deg_wav[:l]
        
        with diff_cols[i]:
            st.markdown(f"**Diff: {name}**")
            with io.BytesIO() as buf:
                sf.write(buf, diff * 2.0, sr, format='WAV')
                st.audio(buf)

else:
    st.info("👋 请先在左侧上传文件。推荐先勾选 'Auto-Align'。")