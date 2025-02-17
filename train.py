import os
import copy
import shutil
import zipfile
import argparse
import tqdm
import time
import datetime
import random
import numpy as np
import torch
import torchvision
import torch_fidelity
import diffusers
import datasets

from models import SiT_models, ScaleAwareSiT
from sampler import euler_maruyama_sampler

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def log(msg):
    if torch.distributed.get_rank() == 0:
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print('\n'.join([f'[{now}] {m}' for m in msg.split('\n')]))

def setup_ddp(args):
    torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(seconds=86400))
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    node_rank = int(os.environ['GROUP_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    device = f'cuda:{local_rank}'
    seed = args.global_seed + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f'Starting global rank {global_rank}, node rank {node_rank}, local rank {local_rank}, seed {seed}, world_size {world_size}; Connecting to {master_addr}:{master_port}.')
    return device, global_rank, local_rank, world_size, node_rank, master_addr, master_port, seed

def get_dataset(args, world_size):
    import torchvision.transforms.v2 as v2
    import PIL, io
    transforms = torchvision.transforms.Compose([
        v2.RGB(),
        v2.Resize(args.image_size),
        v2.CenterCrop(args.image_size),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ])
    def map_fn(item):
        item['image'] = transforms(PIL.Image.open(io.BytesIO(item['image']['bytes'])))
        return item
    dataset_train = datasets.load_dataset(
        'ILSVRC/imagenet-1k',
        split='train',
        trust_remote_code=True
    )
    loader_len = len(dataset_train) // args.global_batch_size
    dataset_train = dataset_train.to_iterable_dataset(num_shards=world_size*args.num_workers).map(map_fn)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.global_batch_size // world_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader_train, loader_len

def save_ckpt(args, model, ema, opt, epoch, step, checkpoint_dir, global_rank):
    if global_rank == 0:
        checkpoint = {
            'model': model.module.state_dict(),
            'ema': ema.state_dict(),
            'opt': opt.state_dict(),
            'args': args,
            'epoch': epoch,
            'step': step,
        }
        checkpoint_path = f'{checkpoint_dir}/{step:07d}.pt'
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, f'{checkpoint_dir}/latest.pt')
        log(f'Saved checkpoint to {checkpoint_path}')
    torch.distributed.barrier()

def load_ckpt(args, model, ema, opt):
    if args.ckpt_path is None:
        return 0, 0

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')

    start_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    start_step = checkpoint['step'] if 'step' in checkpoint else 0

    if 'model' in checkpoint:
        missing_keys, unexpected_keys = model.module.load_state_dict(checkpoint['model'], strict=False)
        log(f'Loaded model from {args.ckpt_path} at epoch {start_epoch} and step {start_step}')
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            log(f'Missing keys: {missing_keys}')
            log(f'Unexpected keys: {unexpected_keys}')

    if 'ema' in checkpoint:
        missing_keys, unexpected_keys = ema.load_state_dict(checkpoint['ema'])
        log(f'Loaded ema model from {args.ckpt_path}')
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            log(f'Missing keys: {missing_keys}')
            log(f'Unexpected keys: {unexpected_keys}')
    else:
        update_ema(ema, model.module, decay=0)

    if 'opt' in checkpoint:
        opt.load_state_dict(checkpoint['opt'])
        log(f'Loaded optimizer from {args.ckpt_path}')

    return start_epoch, start_step

@torch.no_grad()
def encode_image(x, vae):
    z = vae.encode(x.to(vae.dtype))['latent_dist'].sample().to(x.dtype)
    z = z * 0.18215
    return z.to(x.dtype)

@torch.no_grad()
def decode_image(z, vae):
    z = z / 0.18215
    x = vae.decode(z.to(vae.dtype))['sample'].clamp(-1, 1).to(z.dtype)
    return x.to(z.dtype)

def evaluate(args, model, vae, world_size, global_rank, device, step, visualize_dir, noise=None, cond=None, start=1000, cfgw=1, fid=False, num_samples=5000):
    tmp_dir = os.path.join('output', 'temp', 'fid_image', 'generated')
    if fid and global_rank == 0:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

    save_dir = f'{visualize_dir}/sample_{step:07d}'
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    torch.distributed.barrier()
    with torch.no_grad():
        eval_batch_size = 8 if not fid else 25

        num_collect = eval_batch_size * world_size
        x_hat = torch.zeros((num_collect, 3, args.image_size, args.image_size), device=device)
        label = torch.zeros((num_collect,), device=device, dtype=torch.int64)

        if fid:
            conds = torch.arange(1000, device=device, dtype=torch.int64).repeat(100)

        num_batches = int(np.ceil((args.num_eval if not fid else num_samples) / eval_batch_size / world_size))
        bar = tqdm.trange(num_batches, desc=f'FID', disable=(global_rank != 0 or not fid))
        for i in bar:
            x_noise = torch.randn(eval_batch_size, 4, args.image_size // 8, args.image_size // 8, device=device) if (noise is None or fid) else noise
            if fid:
                cond = conds[i * num_collect + global_rank * eval_batch_size:i * num_collect + (global_rank + 1) * eval_batch_size]
            elif cond is None:
                cond = torch.randint(0, 1000, (eval_batch_size,), device=device, dtype=torch.int64)

            x_noise = euler_maruyama_sampler(model, x_noise, cond, num_steps=args.sampling_steps, cfg_scale=cfgw)
            _x_hat = (decode_image(x_noise, vae) + 1) / 2
            x_hat[global_rank * eval_batch_size:(global_rank + 1) * eval_batch_size] = _x_hat
            label[global_rank * eval_batch_size:(global_rank + 1) * eval_batch_size] = cond

            if fid:
                torch.distributed.all_reduce(x_hat, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(label, op=torch.distributed.ReduceOp.SUM)
                if global_rank == 0:
                    for j in range(num_collect):
                        if i * num_collect + j >= num_samples:
                            break
                        img = torchvision.transforms.functional.to_pil_image(x_hat[j].cpu().float())
                        img.save(f'{tmp_dir}/{i * num_collect + j:05d}.png')
                torch.distributed.barrier()
                x_hat.zero_()
                label.zero_()

        ret = 0
        if not fid:
            torch.distributed.all_reduce(x_hat, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(label, op=torch.distributed.ReduceOp.SUM)
            if global_rank == 0:
                img = torchvision.utils.make_grid(x_hat[:64], nrow=8)
                img = torchvision.transforms.functional.to_pil_image(img.cpu().float())
                img.save(f'{save_dir}/t{start:04d}_cfgw{cfgw}.png')
                open(f'{save_dir}/t{start:04d}_cfgw{cfgw}.txt', 'w').write('\n'.join([str(l.item()) for l in label[:64]]))
            log(f'(step={step}) <t={start}> Images saved at {save_dir}/t{start:04d}_cfgw{cfgw}.png')
        else:
            torch.distributed.barrier()
            if global_rank == 0:
                log(f"zipping images ......")
                with zipfile.ZipFile(f'{save_dir}/eval_{num_samples}_cfgw{cfgw}.zip', 'w') as hat_zip:
                    for i in tqdm.trange(len(os.listdir(tmp_dir))):
                        hat_zip.write(f"{tmp_dir}/{i:05d}.png", f"{i:05d}.png")
                log(f"{save_dir}/eval_{num_samples}_cfgw{cfgw}.zip saved {len(os.listdir(tmp_dir))} files")

                # calculate FID
                stat = f'fid-{int(np.ceil(num_samples / 1000))}k-{args.image_size}.npz'
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=tmp_dir, input2=None,
                    fid_statistics_file=f'output/{stat}',
                    cuda=True, isc=True, fid=True, kid=False, prc=False, verbose=False,
                )
                inception_score = metrics_dict['inception_score_mean']
                inception_score_std = metrics_dict['inception_score_std']
                ret = fid_score = metrics_dict['frechet_inception_distance']

                log(f'(step={step}) <t={start}> FID={fid_score:.4f}, IS={inception_score:.4f}±{inception_score_std:.4f}, Images saved at {save_dir}/eval_{num_samples}_cfgw{cfgw}.zip')
            torch.distributed.barrier()

    return ret

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            encoders=[],
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias

    def interpolant(self, t):
        alpha_t = 1 - t
        sigma_t = t
        d_alpha_t = -1
        d_sigma_t =  1
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, zs=None):
        if model_kwargs == None:
            model_kwargs = {}

        time_input = torch.rand((images.shape[0], 1, 1, 1))
        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        model_target = d_alpha_t * images + d_sigma_t * noises
        model_output  = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = (model_output - model_target) ** 2

        loss_dict = {
            'loss': denoising_loss,
            't': time_input,
            'x_t': model_input,
            'target': model_target,
            'model_output': model_output,
        }

        return loss_dict

def main(args):
    device, rank, local_rank, world_size, node_rank, master_addr, master_port, seed = setup_ddp(args)

    checkpoint_dir = f'{args.results_dir}/checkpoint'
    visualize_dir = f'{args.results_dir}/visualize'

    args.num_eval = 8 * world_size if args.num_eval < 0 else args.num_eval

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(visualize_dir, exist_ok=True)
        log(f'Experiment directory created at {args.results_dir}')

    if os.path.exists(os.path.join(args.results_dir, 'checkpoint', 'latest.pt')):
        args.ckpt_path = os.path.join(args.results_dir, 'checkpoint', 'latest.pt')
        log(f">>>>>> Auto-resume from {args.ckpt_path} <<<<<<")

    log('args:\n' + '\n'.join([f'\t{arg}: {getattr(args, arg)}' for arg in vars(args)]))

    loader, loader_len = get_dataset(args, world_size)
    log(f"Dataset loaded")

    vae = diffusers.models.AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    model = SiT_models[args.model](
        input_size=args.image_size // 8,
        class_dropout_prob=0.0,
        learn_sigma=False,
        num_classes=args.num_classes
    )
    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False

    model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[local_rank])
    log(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    Loss = SILoss()

    start_epoch, start_step = load_ckpt(args, model, ema, opt)

    model.train()
    ema.eval()

    eval_noise = torch.randn(8, 4, args.image_size // 8, args.image_size // 8, device=device)
    eval_cond = torch.randint(0, 1000, (8,), device=device, dtype=torch.int64)
    
    round = lambda x: int(np.ceil(x)) if (random.random() < x - int(np.floor(x))) else int(np.floor(x))

    epoch, train_steps, log_steps, running_loss, start_time = start_epoch, start_step, 0, 0, time.time()

    log(f"Training for {args.epochs} epochs...")
    for data in loader:

        if train_steps % args.ckpt_every == 0 and train_steps > start_step:
            save_ckpt(args, model, ema, opt, train_steps, checkpoint_dir, rank)

        if train_steps % args.eval_every == 0:
            evaluate(args, ema, vae, world_size, rank, device, train_steps, visualize_dir, noise=eval_noise, cond=eval_cond, cfgw=args.cfgw)

        if train_steps % args.fid_every == 0 and train_steps > start_step:
            evaluate(args, ema, vae, world_size, rank, device, train_steps, visualize_dir, cfgw=args.cfgw, fid=True)

        if train_steps - start_step > (epoch - start_epoch + 1) * loader_len:
            epoch += 1
            if epoch > args.epochs:
                break
            log(f"Beginning epoch {epoch}...")

        x, y = encode_image(data['image'].to(device), vae), data['label'].to(device)

        use_mg_loss = args.start_step >= 0 and train_steps >= args.start_step

        num_mg, num_drop = 0, round(args.data_ratio[1] * len(y))
        if use_mg_loss:
            num_mg = round(args.data_ratio[0] * len(y)) if isinstance(model.module, ScaleAwareSiT) else (len(y) - num_drop)

        scale = torch.ones(len(x), device=device)
        scale[:num_mg] = torch.rand(num_mg, device=device) * (args.mgw[1] - args.mgw[0]) + args.mgw[0]

        y_drop = y.clone()
        y_drop[num_mg:num_mg+num_drop] = 1000
        scale[num_mg:num_mg+num_drop] = 0

        model_kwargs = dict(y=y_drop, s=scale) if isinstance(model.module, ScaleAwareSiT) else dict(y=y_drop)
        loss_dict = Loss(model, x, model_kwargs)

        if use_mg_loss:
            t, x_t, target, model_output = loss_dict['t'].reshape(-1), loss_dict['x_t'], loss_dict['target'], loss_dict['model_output']
            with torch.no_grad():
                if args.contrastive:
                    new_y = (y[:num_mg] + torch.randint(1, 1000, (num_mg,), device=device)) % 1000
                else:
                    new_y = torch.ones_like(y[:num_mg]) * 1000
                if isinstance(ema, ScaleAwareSiT):
                    pred_w_cond = ema(x_t[:num_mg], t[:num_mg], y[:num_mg], 1)[:, :4]
                    pred_wo_cond = ema(x_t[:num_mg], t[:num_mg], new_y, 0)[:, :4]
                else:
                    pred_w_cond = ema(x_t[:num_mg], t[:num_mg], y[:num_mg])[:, :4]
                    pred_wo_cond = ema(x_t[:num_mg], t[:num_mg], new_y)[:, :4]
                w = torch.where(t[:num_mg] < args.mg_high, scale[:num_mg] - 1, 0)
                target[:num_mg] = target[:num_mg] + w.view(-1, 1, 1, 1) * (pred_w_cond - pred_wo_cond)
            loss_dict['loss'] = (target - model_output) ** 2

        loss = loss_dict["loss"].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model.module, decay=args.ema_decay)

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time.time()
            steps_per_sec = log_steps / (end_time - start_time)
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size
            log(f"(step={train_steps}) Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
            running_loss = 0
            log_steps = 0
            start_time = time.time()

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="output/try")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50000)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--num-eval', type=int, default=-1)
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
