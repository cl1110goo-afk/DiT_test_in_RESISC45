# Diffusion Transformer (DiT) Exploration on Remote Sensing Imagery

### [Original Paper](http://arxiv.org/abs/2212.09748) | [Original DiT Project Page](https://www.wpeebles.com/DiT) 

This repository contains the PyTorch implementation, custom data loaders, and training configurations for adapting Diffusion Transformers (DiTs) to large-scale geographical and structural image datasets. 

Building upon the foundational DiT architecture, this project specifically leverages the **NWPU-RESISC45** dataset (a large-scale benchmark for remote sensing image scene classification) to explore the generative capabilities of vision transformers on complex aerial imagery. The ultimate goal of this exploration is to set a preliminary technical foundation for generating high-fidelity training data for **Structural Digital Twins**.

## 🌟 Core Implementations & Modifications

Unlike standard image synthesis (e.g., ImageNet), generating remote sensing and structural imagery requires handling unique spatial complexities. This repository includes the following custom modifications:

* **Custom Data Pipeline:** Developed tailored `Dataset` and `DataLoader` Python scripts specifically designed to parse, process, and load the NWPU-RESISC45 dataset efficiently.
* **Architecture Adaptation:** Adapted the state-of-the-art generative AI architecture to handle the specific latent features of large-scale geographical images.
* **Environment & Training Loop:** Configured a local training environment with a managed training loop optimized for testing generative capabilities on structural datasets.

## ⚙️ Setup

First, download and set up the repo:

```bash
git clone [https://github.com/your-username/DiT_test_in_RESISC45.git](https://github.com/your-username/DiT_test_in_RESISC45.git)
cd DiT_test_in_RESISC45
```

We provide an `environment.yml` file that can be used to create a Conda environment.

```bash
conda env create -f environment.yml
conda activate DiT
```

## 🚀 Training on NWPU-RESISC45

We provide a modified training script in `train_ddp_cached.py`. This script integrates the custom dataloaders for the remote sensing dataset. To launch training with `N` GPUs on one node:

```bash
torchrun --nnodes=1 --nproc_per_node=N train_ddp_cached.py --model DiT-XL/2 --data-path /path/to/NWPU-RESISC45
```
*Note: Ensure your `--data-path` points to the correctly formatted directory of the NWPU-RESISC45 dataset.*

## 🖼️ Sampling

You can sample from your fine-tuned custom DiT models using the `sample_ddp_custom.py` script. Simply use the `--ckpt` argument to point to your saved weights. 

For example, to sample from the EMA weights of your custom 256x256 model, run:

```bash
python sample_ddp_custom.py --model DiT-XL/2 --image-size 256 --ckpt /path/to/your/custom_model.pt
```

## 🙏 Acknowledgments

This project is built upon the phenomenal work by William Peebles and Saining Xie in their paper [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748). 

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```
