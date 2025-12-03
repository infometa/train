#!/bin/bash
# 音色修复模型 训练脚本
# 用法: ./run.sh [prepare|train|train_ddp|export|all]

set -e

# 配置
CONFIG="configs/default.yaml"
NUM_GPUS=$(nvidia-smi -L | wc -l)
MASTER_PORT=29500

echo "=================================="
echo "Timbre Restoration Training"
echo "=================================="
echo "GPUs detected: $NUM_GPUS"
echo "Config: $CONFIG"
echo ""

# 1. 数据准备
prepare_data() {
    echo ">>> Step 1: Preparing dataset..."
    python data/prepare_dataset.py --config $CONFIG --num_workers 8
    echo "Dataset prepared!"
}

# 2. 单卡训练
train_single() {
    echo ">>> Training (single GPU)..."
    python train.py --config $CONFIG
}

# 3. 多卡训练 (DDP)
train_ddp() {
    echo ">>> Training (DDP, $NUM_GPUS GPUs)..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train.py --config $CONFIG
}

# 4. 导出 ONNX
export_model() {
    echo ">>> Exporting ONNX model..."
    
    # 找到最新的检查点
    CKPT=$(ls -t logs/*/checkpoints/checkpoint_final.pt 2>/dev/null | head -1)
    
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -t logs/*/checkpoints/checkpoint_latest.pt 2>/dev/null | head -1)
    fi
    
    if [ -z "$CKPT" ]; then
        echo "Error: No checkpoint found!"
        exit 1
    fi
    
    echo "Using checkpoint: $CKPT"
    
    python export_onnx.py \
        --checkpoint $CKPT \
        --config $CONFIG \
        --output timbre_restore.onnx \
        --verify \
        --benchmark
}

# 5. 完整流程
run_all() {
    prepare_data
    
    if [ $NUM_GPUS -gt 1 ]; then
        train_ddp
    else
        train_single
    fi
    
    export_model
}

# 主入口
case "${1:-all}" in
    prepare)
        prepare_data
        ;;
    train)
        train_single
        ;;
    train_ddp)
        train_ddp
        ;;
    export)
        export_model
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: $0 [prepare|train|train_ddp|export|all]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
