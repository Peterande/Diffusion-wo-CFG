import argparse
import torch
import diffusers
from models import SiT_models
from train import log, setup_ddp, evaluate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    device, rank, local_rank, world_size, node_rank, master_addr, master_port, seed = setup_ddp(args)
    
    vae = diffusers.models.AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    model = SiT_models[args.model](
        input_size=args.image_size // 8,
        class_dropout_prob=0.0,
        learn_sigma=False,
        num_classes=args.num_classes
    )

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])
    log(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    checkpoint = checkpoint['ema'] if 'ema' in checkpoint else checkpoint['model']
    missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint, strict=False)
    log(f'Loaded model from {args.ckpt_path}')
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        log(f'Missing keys: {missing_keys}')
        log(f'Unexpected keys: {unexpected_keys}')

    model.eval()

    log(f"Evaluation Only")

    evaluate(args, model, vae, world_size, rank, device, 0, args.results_dir, cfgw=args.cfgw, fid=True, num_samples=args.num_eval)

    log("Evaluation Done")
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="output/evalulation/try")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=248)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument('--ckpt-path', type=str, default='output/SiT-XL-2-MG.pt')
    parser.add_argument('--num-eval', type=int, default=50000)
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--fid-every', type=int, default=50000)
    parser.add_argument('--start-step', type=int, default=100000)
    parser.add_argument('--data-ratio', type=float, nargs='+', default=[0.2, 0.1])
    parser.add_argument('--mgw', type=float, nargs='+', default=[1.45, 1.45])
    parser.add_argument('--mg-high', type=float, default=0.75)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--contrastive', action='store_true', default=False)
    parser.add_argument('--sampling-steps', type=int, default=250)
    parser.add_argument('--cfgw', type=float, default=1.0)
    args = parser.parse_args()
    main(args)
