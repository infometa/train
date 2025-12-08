#!/usr/bin/env python3
"""
训练脚本 for Causal U-Net GAN Timbre Restoration

支持：
- 单卡/多卡 DDP 训练
- 混合精度训练
- TensorBoard 日志
- 检查点保存/恢复
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import yaml
from tqdm import tqdm
import soundfile as sf

# 本地模块
sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import TimbreRestoreDataset, create_dataloader
from model.generator import CausalUNetGenerator, count_parameters
from model.discriminator import MultiScaleDiscriminator, MultiPeriodDiscriminator, CombinedDiscriminator
from model.losses import GeneratorLoss, DiscriminatorLoss


def setup_distributed():
    """初始化分布式训练，GPU 不可用时自动回退单卡 CPU"""
    use_dist = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    has_cuda = torch.cuda.is_available()
    
    if use_dist and has_cuda:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return rank, world_size, local_rank, device
    
    if use_dist and not has_cuda:
        print("Warning: 设置了分布式环境变量但未检测到 CUDA，回退到 CPU 单卡。")
    
    device = torch.device('cuda:0' if has_cuda else 'cpu')
    return 0, 1, 0, device


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config_path: str, resume: str = None):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # 分布式设置
        self.rank, self.world_size, self.local_rank, self.device = setup_distributed()
        self.is_main = self.rank == 0
        
        if self.is_main:
            print(f"World size: {self.world_size}")
            print(f"Device: {self.device}")
        
        # 创建模型
        self._build_models()
        
        # 创建优化器
        self._build_optimizers()
        
        # 创建数据加载器
        self._build_dataloaders()
        
        # 创建损失函数
        self._build_losses()
        
        # 混合精度
        self.scaler_g = GradScaler(device=self.device.type)
        self.scaler_d = GradScaler(device=self.device.type)
        
        # 日志
        if self.is_main:
            self.log_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
            self.ckpt_dir = self.log_dir / "checkpoints"
            self.ckpt_dir.mkdir(exist_ok=True)
            
            # 保存配置
            with open(self.log_dir / "config.yaml", 'w') as f:
                yaml.dump(self.config, f)
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 恢复检查点
        if resume:
            self._load_checkpoint(resume)
    
    def _build_models(self):
        """构建模型"""
        gen_config = self.config['model']['generator']
        disc_config = self.config['model']['discriminator']
        
        # Generator
        self.generator = CausalUNetGenerator(
            in_channels=gen_config['in_channels'],
            out_channels=gen_config['out_channels'],
            channels=gen_config['channels'],
            kernel_size=gen_config['kernel_size'],
            bottleneck_type=gen_config['bottleneck_type'],
            bottleneck_layers=gen_config['bottleneck_layers'],
            use_weight_norm=gen_config['use_weight_norm'],
        ).to(self.device)
        
        # Discriminator: 使用组合判别器（MSD + MPD），同时关注全局与周期性细节
        self.discriminator = CombinedDiscriminator(
            use_msd=True,
            use_mpd=True,
            msd_scales=disc_config.get('scales', 3),
            mpd_periods=disc_config.get('periods', [2, 3, 5, 7, 11]),
            channels=disc_config['channels'],
            kernel_size=disc_config['kernel_size'],
            use_spectral_norm=disc_config['use_spectral_norm'],
        ).to(self.device)
        
        # DDP
        if self.world_size > 1:
            if self.device.type != 'cuda':
                raise SystemExit("分布式训练需要 CUDA，请确认 GPU 可用或移除 RANK/WORLD_SIZE 环境变量。")
            self.generator = DDP(
                self.generator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
            self.discriminator = DDP(
                self.discriminator,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        
        if self.is_main:
            g_params = count_parameters(self.generator)
            d_params = count_parameters(self.discriminator)
            print(f"Generator params: {g_params / 1e6:.2f}M")
            print(f"Discriminator params: {d_params / 1e6:.2f}M")
    
    def _build_optimizers(self):
        """构建优化器"""
        opt_config = self.config['training']['optimizer']
        sched_config = self.config['training']['scheduler']
        epochs = self.config['training']['epochs']
        self.warmup_epochs = sched_config.get('warmup_epochs', 0)
        
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=opt_config['lr_g'],
            betas=tuple(opt_config['betas']),
            weight_decay=opt_config['weight_decay'],
        )
        
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=opt_config['lr_d'],
            betas=tuple(opt_config['betas']),
            weight_decay=opt_config['weight_decay'],
        )
        self.base_lr_g = opt_config['lr_g']
        self.base_lr_d = opt_config['lr_d']
        
        # 学习率调度 - 支持 CosineAnnealingLR 和 CosineAnnealingWarmRestarts
        sched_type = sched_config.get('type', 'CosineAnnealingLR')
        
        if sched_type == 'CosineAnnealingLR':
            T_max = sched_config.get('T_max', epochs)
            eta_min = sched_config.get('eta_min', 1e-6)
            
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_g,
                T_max=T_max,
                eta_min=eta_min,
            )
            self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_d,
                T_max=T_max,
                eta_min=eta_min,
            )
            self.scheduler_step_per_epoch = True
        else:
            # CosineAnnealingWarmRestarts
            T_0 = sched_config.get('T_0', 10)
            T_mult = sched_config.get('T_mult', 2)
            eta_min = sched_config.get('eta_min', 1e-6)
            
            self.scheduler_g = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_g,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
            )
            self.scheduler_d = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_d,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
            )
            self.scheduler_step_per_epoch = False
        # 若配置了 warmup，在 warmup 结束后再启用调度
        self.enable_scheduler = self.warmup_epochs == 0
    
    def _build_dataloaders(self):
        """构建数据加载器"""
        data_config = self.config['data']
        train_config = self.config['training']

        align_df_delay = data_config.get('align_df_delay', False)
        # 与配置保持一致，覆盖 DeepFilterNet 可能的较大延迟
        align_max_shift = data_config.get('align_max_shift', 8000)
        align_sample_count = data_config.get('align_sample_count', 32)
        
        output_dir = Path(data_config['output_dir'])
        train_file = output_dir / "train.txt"
        val_file = output_dir / "val.txt"
        
        # 训练集
        train_dataset = TimbreRestoreDataset(
            file_list=str(train_file),
            segment_length=data_config['segment_length'],
            sample_rate=data_config['sample_rate'],
            augment=True,
            align_df_delay=align_df_delay,
            align_max_shift=align_max_shift,
            align_sample_count=align_sample_count,
        )
        
        # 验证集
        val_dataset = TimbreRestoreDataset(
            file_list=str(val_file),
            segment_length=data_config['segment_length'],
            sample_rate=data_config['sample_rate'],
            augment=False,
            align_df_delay=align_df_delay,
            align_max_shift=align_max_shift,
            align_sample_count=align_sample_count,
        )
        
        if len(train_dataset) == 0:
            raise SystemExit(f"训练集为空，请检查数据生成是否成功: {train_file}")
        if len(val_dataset) == 0:
            raise SystemExit(f"验证集为空，请检查数据生成是否成功: {val_file}")
        
        self.train_len = len(train_dataset)
        self.val_len = len(val_dataset)
        
        # 分布式采样器
        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=data_config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if data_config['num_workers'] > 0 else False,
            prefetch_factor=2 if data_config['num_workers'] > 0 else None,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            sampler=None,  # 验证集不分片，确保所有进程（或主进程）看到完整验证集
            num_workers=data_config['num_workers'],
            pin_memory=True if self.device.type == 'cuda' else False,
            persistent_workers=True if data_config['num_workers'] > 0 else False,
            prefetch_factor=2 if data_config['num_workers'] > 0 else None,
            drop_last=False,
        )
        
        if self.is_main:
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
    
    def _build_losses(self):
        """构建损失函数"""
        loss_config = self.config['training']['loss_weights']
        stft_config = self.config['training'].get('stft_config', {})
        
        self.g_loss_fn = GeneratorLoss(
            l1_weight=loss_config['l1'],
            stft_weight=loss_config['multi_stft'],
            adv_weight=loss_config['adversarial'],
            fm_weight=loss_config['feature_matching'],
            stft_config=stft_config if stft_config else None,
        ).to(self.device)
        
        self.d_loss_fn = DiscriminatorLoss().to(self.device)
    
    def _save_checkpoint(self, tag: str = "latest"):
        """保存检查点"""
        if not self.is_main:
            return
        
        gen_state = self.generator.module.state_dict() if self.world_size > 1 else self.generator.state_dict()
        disc_state = self.discriminator.module.state_dict() if self.world_size > 1 else self.discriminator.state_dict()
        
        ckpt = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'generator': gen_state,
            'discriminator': disc_state,
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'scaler_g': self.scaler_g.state_dict(),
            'scaler_d': self.scaler_d.state_dict(),
        }
        
        path = self.ckpt_dir / f"checkpoint_{tag}.pt"
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path: str):
        """加载检查点"""
        print(f"Loading checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        
        gen = self.generator.module if self.world_size > 1 else self.generator
        disc = self.discriminator.module if self.world_size > 1 else self.discriminator
        
        gen.load_state_dict(ckpt['generator'])
        disc.load_state_dict(ckpt['discriminator'])
        self.optimizer_g.load_state_dict(ckpt['optimizer_g'])
        self.optimizer_d.load_state_dict(ckpt['optimizer_d'])
        self.scheduler_g.load_state_dict(ckpt['scheduler_g'])
        self.scheduler_d.load_state_dict(ckpt['scheduler_d'])
        self.scaler_g.load_state_dict(ckpt['scaler_g'])
        self.scaler_d.load_state_dict(ckpt['scaler_d'])
        
        self.epoch = ckpt['epoch'] + 1  # 从下一个 epoch 开始
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        
        print(f"Resumed from epoch {ckpt['epoch']}, step {self.global_step}")
    
    def train_step(self, degraded: torch.Tensor, clean: torch.Tensor) -> dict:
        """单步训练"""
        degraded = degraded.to(self.device)
        clean = clean.to(self.device)
        
        train_config = self.config['training']
        use_gan = self.epoch >= train_config['gan_start_epoch']
        # 关闭混合精度，避免数值不稳定
        amp_enabled = False
        
        losses = {}
        
        # ============ Discriminator ============
        if use_gan:
            self.optimizer_d.zero_grad()
            
            with autocast(device_type=self.device.type, enabled=amp_enabled):
                # 生成假样本
                with torch.no_grad():
                    fake = self.generator(degraded)
                
                # 判别
                real_out, real_feats = self.discriminator(clean)
                fake_out, _ = self.discriminator(fake.detach())
                
                # 损失
                d_loss, d_losses = self.d_loss_fn(real_out, fake_out)
            
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.unscale_(self.optimizer_d)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), train_config['grad_clip'])
            self.scaler_d.step(self.optimizer_d)
            self.scaler_d.update()
            
            losses.update(d_losses)
        
        # ============ Generator ============
        self.optimizer_g.zero_grad()
        
        with autocast(device_type=self.device.type, enabled=amp_enabled):
            fake = self.generator(degraded)
            
            if use_gan:
                fake_out, fake_feats = self.discriminator(fake)
                g_loss, g_losses = self.g_loss_fn(
                    fake, clean,
                    disc_fake_outputs=fake_out,
                    disc_fake_features=fake_feats,
                    disc_real_features=real_feats,
                )
            else:
                g_loss, g_losses = self.g_loss_fn(fake, clean)
        
        self.scaler_g.scale(g_loss).backward()
        self.scaler_g.unscale_(self.optimizer_g)
        nn.utils.clip_grad_norm_(self.generator.parameters(), train_config['grad_clip'])
        self.scaler_g.step(self.optimizer_g)
        self.scaler_g.update()
        
        losses.update(g_losses)
        
        return losses
    
    @torch.no_grad()
    def validate(self) -> dict:
        """验证"""
        self.generator.eval()
        self.discriminator.eval()
        
        total_loss = 0.0
        total_l1 = 0.0
        total_stft = 0.0
        total_adv = 0.0
        count = 0
        train_config = self.config['training']
        use_gan = self.epoch >= train_config['gan_start_epoch']
        
        for degraded, clean in self.val_loader:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            fake = self.generator(degraded)

            if use_gan:
                real_out, real_feats = self.discriminator(clean)
                fake_out, fake_feats = self.discriminator(fake)
                loss, losses = self.g_loss_fn(
                    fake, clean,
                    disc_fake_outputs=fake_out,
                    disc_fake_features=fake_feats,
                    disc_real_features=real_feats,
                )
            else:
                loss, losses = self.g_loss_fn(fake, clean)
            
            total_loss += loss.item()
            total_l1 += losses.get('l1', 0)
            total_stft += losses.get('stft', 0)
            total_adv += losses.get('adv', 0)
            count += 1
        
        self.generator.train()
        self.discriminator.train()
        
        return {
            'val_loss': total_loss / max(count, 1),
            'val_l1': total_l1 / max(count, 1),
            'val_stft': total_stft / max(count, 1),
            'val_adv': total_adv / max(count, 1),
        }
    
    @torch.no_grad()
    def save_samples(self, num_samples: int = 3):
        """保存音频样本"""
        if not self.is_main:
            return
        
        self.generator.eval()
        sample_rate = self.config['data']['sample_rate']
        
        sample_dir = self.log_dir / "samples" / f"epoch_{self.epoch:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (degraded, clean) in enumerate(self.val_loader):
            if i >= num_samples:
                break
            
            degraded = degraded.to(self.device)
            fake = self.generator(degraded)
            
            # 保存
            for j in range(min(degraded.size(0), 2)):
                sf.write(
                    sample_dir / f"{i}_{j}_degraded.wav",
                    degraded[j, 0].cpu().numpy(),
                    sample_rate
                )
                sf.write(
                    sample_dir / f"{i}_{j}_restored.wav",
                    fake[j, 0].cpu().numpy(),
                    sample_rate
                )
                sf.write(
                    sample_dir / f"{i}_{j}_clean.wav",
                    clean[j, 0].numpy(),
                    sample_rate
                )
        
        self.generator.train()
    
    def train(self):
        """训练主循环"""
        train_config = self.config['training']
        epochs = train_config['epochs']
        log_every = train_config['log_every']
        save_every = train_config['save_every']
        
        if self.is_main:
            print(f"\nStarting training from epoch {self.epoch}")
            print(f"Loss weights: L1={train_config['loss_weights']['l1']}, "
                  f"STFT={train_config['loss_weights']['multi_stft']}, "
                  f"Adv={train_config['loss_weights']['adversarial']}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # 线性 warmup：按 epoch 设置当前 lr
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                warmup_scale = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer_g.param_groups:
                    param_group['lr'] = self.base_lr_g * warmup_scale
                for param_group in self.optimizer_d.param_groups:
                    param_group['lr'] = self.base_lr_d * warmup_scale
            else:
                if not self.enable_scheduler:
                    self.enable_scheduler = True
            
            # 设置分布式采样器的 epoch
            if self.world_size > 1:
                self.train_loader.sampler.set_epoch(epoch)
            
            num_batches = len(self.train_loader)
            
            # 进度条
            if self.is_main:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
            else:
                pbar = self.train_loader
            
            epoch_losses = {}
            
            for batch_idx, (degraded, clean) in enumerate(pbar):
                losses = self.train_step(degraded, clean)
                self.global_step += 1
                
                # 累积损失
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = 0.0
                    epoch_losses[k] += v
                
                # 日志
                if self.is_main and self.global_step % log_every == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                    
                    lr = self.optimizer_g.param_groups[0]['lr']
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    
                    if isinstance(pbar, tqdm):
                        pbar.set_postfix({
                            'loss': f"{losses.get('total', 0):.4f}",
                            'l1': f"{losses.get('l1', 0):.4f}",
                        })
                
                # CosineAnnealingWarmRestarts 按 step 更新
                if not self.scheduler_step_per_epoch and num_batches > 0:
                    if self.enable_scheduler:
                        # 让调度在 warmup 结束后从 0 开始累计
                        step_frac = max(0, epoch - self.warmup_epochs) + batch_idx / num_batches
                        self.scheduler_g.step(step_frac)
                        self.scheduler_d.step(step_frac)
            
            # CosineAnnealingLR 按 epoch 更新
            if self.scheduler_step_per_epoch:
                if self.enable_scheduler:
                    self.scheduler_g.step()
                    self.scheduler_d.step()
            
            # 验证
            if self.is_main:
                val_losses = self.validate()
                for k, v in val_losses.items():
                    self.writer.add_scalar(f"val/{k}", v, self.global_step)
                
                print(f"\nEpoch {epoch} - Val Loss: {val_losses['val_loss']:.4f}, "
                      f"L1: {val_losses['val_l1']:.4f}, STFT: {val_losses['val_stft']:.4f}")
                
                # 保存最佳模型
                if val_losses['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['val_loss']
                    self._save_checkpoint("best")
                    print(f"New best model! Val Loss: {self.best_val_loss:.4f}")
            
            # 保存检查点
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}")
                self.save_samples()
            
            self._save_checkpoint("latest")
        
        # 训练结束
        if self.is_main:
            self._save_checkpoint("final")
            self.writer.close()
            print("\nTraining completed!")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Timbre Restoration Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    trainer = Trainer(args.config, args.resume)
    
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
