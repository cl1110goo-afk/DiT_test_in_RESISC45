# extract_latents.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from diffusers.models import AutoencoderKL
from PIL import Image
import numpy as np
import argparse
import os
from glob import glob
from tqdm import tqdm

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that also returns file paths."""
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)

def main(args):
    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"🚀 Starting Latent Extraction on {dist.get_world_size()} GPUs...")
        os.makedirs(args.features_dir, exist_ok=True)

    # Load VAE (Frozen, BF16 for speed)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    # Define Transform (Must match training logic)
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolderWithPaths(args.data_path, transform=transform)
    
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=8, 
        pin_memory=True
    )

    if rank == 0:
        print(f"📂 Data Source: {args.data_path}")
        print(f"💾 Save Target: {args.features_dir}")
        print(f"🖼 Total Images: {len(dataset)}")

    # Extraction Loop
    for inputs, labels, paths in tqdm(loader, disable=(rank != 0)):
        inputs = inputs.to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16): # 3090 Use BF16
                # Encode -> Sample -> Scale
                # We sample ONCE here to create static latents. 
                # This is standard practice for speeding up training.
                latents = vae.encode(inputs).latent_dist.sample().mul_(0.18215)

        # Save each latent individually
        for i, latent in enumerate(latents):
            # Get original relative path (e.g., "airport/airport_001.jpg")
            original_path = paths[i]
            relative_path = os.path.relpath(original_path, args.data_path)
            # Change extension to .pt
            save_path = os.path.join(args.features_dir, os.path.splitext(relative_path)[0] + ".pt")
            
            # Create sub-directory if not exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save as a small tensor clone to save space
            torch.save(latent.clone(), save_path)

    dist.barrier()
    if rank == 0:
        print("✅ All latents extracted successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-dir", type=str, required=True) # Output folder
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32) # Single GPU batch size
    parser.add_argument("--vae", type=str, default="ema")
    args = parser.parse_args()
    main(args)