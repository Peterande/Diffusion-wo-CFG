# Diffusion Models Without Classifier-free Guidance

Zhicong Tang $^1$, Jianmin Bao $^2$, Dong Chen $^2$, Baining Guo $^2$

$^1$ Tsinghua University \
$^2$ Microsoft Research Asia

## TL;DR

We propose Model-guidance (MG) for training diffusion models, remove the commmonly used Classifier-free guidance (CFG), and achieve SOTA on ImageNet-256 conditional generation with FID=1.34.

## Code

### 1. Environment setup

```bash
conda create -n mg python=3.12 -y
conda activate mg
pip install -r requirements.txt
```

### 2. Evaluation

We provide FID stat files (from [ADM](https://github.com/openai/guided-diffusion/tree/main/evaluations)) and the checkpoint of our final SiT-XL/2 model. To download and evaluate them, simply run

```bash
bash download.sh
torchrun --nnodes=2 --nproc_per_node=8 test.py
```

You can adjust `nnodes` and `nproc_per_node` according to your environment. However, they may affect evaluation results as the random seed is relative to GPU ranks (see [`train.py`](https://github.com/tzco/diffusion-wo-cfg/blob/main/train.py#L37)).

### 3. Training

We list the hyper-paremeters for the final SiT-XL/2 model in our paper as the defaults of `train.py`. Through the following command your can train your own models

```bash
torchrun --nnodes=2 --nproc_per_node=8 train.py
# Or using REPA checkpoints as initialization. This does not affect final performances.
# torchrun --nnodes=2 --nproc_per_node=8 train.py --ckpt-path output/SiT-XL-2-REPA.pt
```

## Acknowledgement

This code is mainly built upon [DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT), and [REPA](https://github.com/sihyun-yu/REPA) repositories.
