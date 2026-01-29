# train_resume_fast.py
# 8x3090 极速续训版：集成 RAM Cache, AMP, TF32, Compile

import torch
# 🔥 1. 开启 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from time import time
import argparse
import logging
import os
from copy import deepcopy
from collections import OrderedDict
from models import DiT_models
from diffusion import create_diffusion

#################################################################################
#                           极速内存数据集 (RAM Cache)                          #
#################################################################################

class CachedLatentDataset(Dataset):
    def __init__(self, features_dir):
        super().__init__()
        self.files = sorted(glob(os.path.join(features_dir, "**", "*.pt"), recursive=True))
        
        # 建立类别映射字典
        all_folders = sorted([d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(all_folders)}
        
        if dist.get_rank() == 0:
            print(f"🚀 [RAM Cache] Pre-loading {len(self.files)} latents. Classes: {len(all_folders)}")
        
        self.latents = []
        self.labels = []
        for path in self.files:
            latent = torch.load(path, map_location="cpu")
            folder_name = path.split(os.sep)[-2]
            label = self.class_to_idx[folder_name] # 关键修复：字符串转数字
            self.latents.append(latent)
            self.labels.append(label)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

#################################################################################
#                                  Utilities                                    #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        return logging.getLogger(__name__)
    return logging.getLogger(__name__)

#################################################################################
#                                Training Loop                                  #
#################################################################################

def main(args):
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    # 建立新的实验文件夹 (022-...)
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-resume"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
    else:
        logger = create_logger(None)

    # 1. 初始化模型
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ema = deepcopy(model).to(device)
    
    # 2. 🔥 核心：加载 Checkpoint (续训逻辑)
    start_step = 0
    if args.resume:
        if rank == 0: logger.info(f"🚀 Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        
        # 加载 Model
        model.load_state_dict(ckpt["model"], strict=True) # 既然是续训自己的模型，strict=True 更安全
        # 加载 EMA
        if "ema" in ckpt:
            if rank == 0: logger.info("✅ EMA weights loaded.")
            ema.load_state_dict(ckpt["ema"])
            
        # 尝试恢复步数 (根据文件名，例如 0004000.pt)
        try:
            basename = os.path.basename(args.resume)
            start_step = int(basename.replace(".pt", ""))
            if rank == 0: logger.info(f"🔄 Continuing from step: {start_step}")
        except:
            if rank == 0: logger.info("⚠️ Could not parse step count from filename, starting count from 0.")
            start_step = 0

    # 3. 🔥 编译加速 (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        if rank == 0: logger.info("⚡️ Enabling torch.compile (Fused Kernels)...")
        model = torch.compile(model)
    
    model = DDP(model, device_ids=[rank])
    
    # 4. 优化器 & 混合精度
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # 如果你也想加载优化器状态（完全复原），取消下面注释。
    # 但改变 Batch Size 后建议重置优化器，所以这里默认不加载 opt state。
    # if "opt" in ckpt: opt.load_state_dict(ckpt["opt"])

    scaler = torch.cuda.amp.GradScaler() # FP16 混合精度管理器
    diffusion = create_diffusion(timestep_respacing="") 
    
    # 5. 加载数据 (RAM Cache)
    dataset = CachedLatentDataset(args.features_dir)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=0, # RAM 模式必须为 0
        pin_memory=True,
        drop_last=True
    )
    
    model.train()
    ema.eval()
    
    train_steps = start_step
    start_time = time()
    
    if rank == 0: logger.info(f"🔥 Training Start. Global Batch Size: {args.global_batch_size}")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            # 🔥 AMP 混合精度前向
            with torch.cuda.amp.autocast():
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            # 🔥 AMP 混合精度反向
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            update_ema(ema, model.module)

            train_steps += 1
            
            # 日志
            if train_steps % args.log_every == 0:
                end_time = time()
                steps_per_sec = args.log_every / (end_time - start_time)
                avg_loss = torch.tensor(loss.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} | Speed: {steps_per_sec:.2f} it/s")
                start_time = time()

            # 保存
            if train_steps % args.ckpt_every == 0:
                if rank == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    # 解包 DDP 和 compile
                    raw_model = model.module._orig_mod if hasattr(model.module, '_orig_mod') else model.module
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
                dist.barrier()

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=str, required=True, help="Path to .pt latents")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=64) # 3090 开 128 没问题
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=2000) # 每 2000 步存一次
    parser.add_argument("--resume", type=str, required=True, help="Path to the checkpoint to resume from")
    
    args = parser.parse_args()
    main(args)