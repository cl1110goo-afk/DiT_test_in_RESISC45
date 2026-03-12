import torch
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


class CachedLatentDataset(Dataset):
    def __init__(self, features_dir):
        super().__init__()
        self.files = sorted(glob(os.path.join(features_dir, "**", "*.pt"), recursive=True))
        
        # build class mapping from folder names
        all_folders = sorted([d for d in os.listdir(features_dir) if os.path.isdir(os.path.join(features_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(all_folders)}
        
        if dist.get_rank() == 0:
            print(f"[RAM Cache] Pre-loading {len(self.files)} latents across {len(all_folders)} classes.")
        
        self.latents = []
        self.labels = []
        for path in self.files:
            latent = torch.load(path, map_location="cpu")
            folder_name = path.split(os.sep)[-2]
            label = self.class_to_idx[folder_name] # map folder string to int idx
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
    
    # setup experiment directory
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-resume"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at: {experiment_dir}")
    else:
        logger = create_logger(None)

    # init models
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ema = deepcopy(model).to(device)
    
    # load checkpoint for resume
    start_step = 0
    if args.resume:
        if rank == 0: logger.info(f"Resuming training from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        
        # load model weights
        model.load_state_dict(ckpt["model"], strict=True) 
        
        # load ema weights
        if "ema" in ckpt:
            if rank == 0: logger.info("EMA weights loaded successfully.")
            ema.load_state_dict(ckpt["ema"])
            
        # parse step count from filename
        try:
            basename = os.path.basename(args.resume)
            start_step = int(basename.replace(".pt", ""))
            if rank == 0: logger.info(f"Resuming from global step: {start_step}")
        except:
            if rank == 0: logger.info("Warning: Could not parse step count from filename. Starting from 0.")
            start_step = 0

    # torch.compile for training speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        if rank == 0: logger.info("Enabling torch.compile...")
        model = torch.compile(model)
    
    model = DDP(model, device_ids=[rank])
    
    # optimizer & mixed precision
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    # note: uncomment below if you want to resume optimizer state, 
    # but usually better to reset if batch size changes
    # if "opt" in ckpt: opt.load_state_dict(ckpt["opt"])

    scaler = torch.cuda.amp.GradScaler() 
    diffusion = create_diffusion(timestep_respacing="") 
    
    # dataset & dataloader
    dataset = CachedLatentDataset(args.features_dir)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=0, # must be 0 for RAM cache
        pin_memory=True,
        drop_last=True
    )
    
    model.train()
    ema.eval()
    
    train_steps = start_step
    start_time = time()
    
    if rank == 0: logger.info(f"Training started. Global batch size: {args.global_batch_size}")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            # forward pass with AMP
            with torch.cuda.amp.autocast():
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()

            opt.zero_grad()
            
            # backward pass with AMP
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            update_ema(ema, model.module)

            train_steps += 1
            
            # logging
            if train_steps % args.log_every == 0:
                end_time = time()
                steps_per_sec = args.log_every / (end_time - start_time)
                avg_loss = torch.tensor(loss.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                if rank == 0:
                    logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f} | Speed: {steps_per_sec:.2f} it/s")
                start_time = time()

            # checkpoint saving
            if train_steps % args.ckpt_every == 0:
                if rank == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    # unwrap DDP and compile
                    raw_model = model.module._orig_mod if hasattr(model.module, '_orig_mod') else model.module
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
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
    parser.add_argument("--ckpt-every", type=int, default=2000) 
    parser.add_argument("--resume", type=str, required=True, help="Path to the checkpoint to resume from")
    
    args = parser.parse_args()
    main(args)
