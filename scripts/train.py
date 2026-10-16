import argparse
import json
import os
import time

import cv2
import torch
import torch.nn.functional as F
import vaezip
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vztbench.pipelines.pipeline import Pipeline
from vztbench.text_encoders.sdxl_text_encoder import SDXLTextEncoder
from vztbench.transformers.transformer import TransformerModel
from vztbench.utils.fid import FIDEvaluator
from vztbench.utils.image_tools import torch_to_cv2
from vztbench.utils.model_ema import ModelEMA
from vztbench.vaes.flux_2_vae import Flux2VAE

DATASET_ROOT = "datasets/coco-2017"
IMAGE_SIZE = 256
AUTOCAST_DTYPE = torch.float16


class FP32Dataset(Dataset):
    def __init__(self):
        self.dataset_dir = f"{DATASET_ROOT}/fp32_encoded_train2017"
        self.samples = sorted(os.listdir(self.dataset_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = load_file(os.path.join(self.dataset_dir, self.samples[idx]))
        # Samples are stored with a leading batch dim of 1.
        return sample_dict["latents"][0], sample_dict["text_embed"][0]


class VaeZipDataset(Dataset):
    def __init__(self, codec: str):
        self.dataset_dir = f"{DATASET_ROOT}/{codec}_encoded_train2017"
        self.samples = sorted(os.listdir(self.dataset_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = vaezip.load(os.path.join(self.dataset_dir, self.samples[idx]))
        latents = sample_dict["latents"].decode(device="cpu")[0]
        text_embed = sample_dict["text_embed"].decode(device="cpu")[0]

        return latents, text_embed


DATASETS = {
    "fp32": FP32Dataset,
    "lbrq_3bit": lambda: VaeZipDataset("lbrq_3bit"),
    "lbrq_4bit": lambda: VaeZipDataset("lbrq_4bit"),
    "lbrq_6bit": lambda: VaeZipDataset("lbrq_6bit"),
}


def load_prompts(num_prompts: int) -> list[str]:
    """First caption of the first num_prompts val2017 images, in image id order."""
    with open(f"{DATASET_ROOT}/annotations/captions_val2017.json") as f:
        annotations = json.load(f)["annotations"]

    captions = {}
    for ann in sorted(annotations, key=lambda a: (a["image_id"], a["id"])):
        captions.setdefault(ann["image_id"], ann["caption"].strip())

    assert num_prompts <= len(captions), f"Only {len(captions)} val prompts available"
    return [captions[image_id] for image_id in sorted(captions)[:num_prompts]]


class TestSampleGenerator:
    """Generates the same images (fixed prompts and seeds) every time it is called."""

    def __init__(
        self,
        pipeline: Pipeline,
        text_embeds: torch.Tensor,
        batch_size: int,
        num_inference_steps: int,
        seed: int,
    ):
        assert len(text_embeds) % batch_size == 0, (
            "Number of samples must be divisible by the generation batch size"
        )
        self.pipeline = pipeline
        self.text_embeds = text_embeds
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps
        self.seed = seed

    @torch.no_grad()
    def batches(self):
        """
        yields:
            images: [B, 3, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        starts = range(0, len(self.text_embeds), self.batch_size)
        for batch_idx, start in enumerate(starts):
            yield self.pipeline.generate_image(
                width=IMAGE_SIZE,
                height=IMAGE_SIZE,
                batch_size=self.batch_size,
                num_inference_steps=self.num_inference_steps,
                conditioning={
                    "text_embed": self.text_embeds[start : start + self.batch_size]
                },
                seed=self.seed + batch_idx,
                autocast_dtype=AUTOCAST_DTYPE,
            )


class FlowMatchingTrainer:
    def __init__(
        self,
        model,
        model_ema,
        loader,
        device,
        lr: float,
        grad_clip: float,
        seed: int,
    ):
        self.model = model
        self.model_ema = model_ema
        self.loader = loader
        self.device = device
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.scaler = torch.amp.GradScaler(device)
        # Noise and timesteps come from a dedicated generator, so they match across runs.
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def train_step(
        self, latents: torch.Tensor, text_embed: torch.Tensor
    ) -> torch.Tensor:
        noise = torch.randn(latents.shape, device=self.device, generator=self.generator)
        # Logit-normal timestep sampling, sigma = 1 is pure noise.
        sigma = torch.sigmoid(
            torch.randn(len(latents), device=self.device, generator=self.generator)
        )
        sigma_b = sigma[:, None, None, None]

        noisy_latents = noise * sigma_b + (1 - sigma_b) * latents
        target_velocity = noise - latents

        with torch.autocast(self.device, dtype=AUTOCAST_DTYPE):
            pred = self.model(
                sample=noisy_latents,
                conditioning={"text_embed": text_embed},
                timesteps=sigma * 1000.0,
            )
        loss = F.mse_loss(pred.float(), target_velocity)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model_ema.update(self.model)

        return loss.detach()

    def train_epoch(self, epoch: int) -> float:
        """Returns the mean training loss of the epoch."""
        self.model.train()
        loss_sum = torch.zeros((), device=self.device)
        progress = tqdm(self.loader, desc=f"epoch {epoch}")
        for step, (latents, text_embed) in enumerate(progress, start=1):
            latents = latents.to(self.device, non_blocking=True)
            text_embed = text_embed.to(self.device, non_blocking=True)

            loss_sum += self.train_step(latents, text_embed)
            # Reading the loss syncs with the GPU, so do it rarely.
            if step % 50 == 0:
                progress.set_postfix(loss=(loss_sum / step).item())

        return (loss_sum / len(self.loader)).item()


def save_samples(images: torch.Tensor, prompts: list[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for i, image in enumerate(images):
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}.png"), torch_to_cv2(image[None]))
    with open(os.path.join(out_dir, "prompts.txt"), "w") as f:
        f.write("\n".join(f"{i:03d}: {p}" for i, p in enumerate(prompts)) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-source", choices=list(DATASETS), required=True)
    parser.add_argument("--run-name", default=None, help="default: dataset source")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fid-samples", type=int, default=2048)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--num-saved-samples", type=int, default=16)
    parser.add_argument(
        "--fid-stats-path",
        default=f"{DATASET_ROOT}/fid_stats_train2017_{IMAGE_SIZE}.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or args.dataset_source
    device = "cuda"
    assert args.num_saved_samples <= args.gen_batch_size

    dataset = DATASETS[args.dataset_source]()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    vae = Flux2VAE(
        path_to_model="models/FLUX.2-klein-4B/vae", device=device, dtype=torch.float32
    )

    # Fixed prompts are embedded once, then the text encoder is released.
    prompts = load_prompts(args.fid_samples)
    text_encoder = SDXLTextEncoder(
        path_to_model="models/stable-diffusion-xl-base-1.0",
        device=device,
        dtype=torch.float32,
    )
    text_embeds = torch.cat(
        [
            text_encoder.encode_text_batch(prompts[i : i + 256])
            for i in range(0, len(prompts), 256)
        ]
    )
    del text_encoder
    torch.cuda.empty_cache()

    fid_evaluator = FIDEvaluator(
        real_image_dir=f"{DATASET_ROOT}/train2017",
        stats_path=args.fid_stats_path,
        image_size=IMAGE_SIZE,
        device=device,
        num_workers=args.num_workers,
    )

    # Seed right before creation so the init does not depend on earlier RNG use.
    torch.manual_seed(args.seed)
    model = TransformerModel().to(device)
    model_ema = ModelEMA(model, decay=args.ema_decay, device=device)
    model.compile()
    model_ema.ema_model.compile()

    pipeline = Pipeline(
        model=model_ema.ema_model,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    test_generator = TestSampleGenerator(
        pipeline=pipeline,
        text_embeds=text_embeds,
        batch_size=args.gen_batch_size,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )

    trainer = FlowMatchingTrainer(
        model=model,
        model_ema=model_ema,
        loader=loader,
        device=device,
        lr=args.lr,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )

    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    writer.add_text("args", json.dumps(vars(args), indent=2))
    samples_dir = os.path.join("samples", run_name)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss = trainer.train_epoch(epoch)
        train_seconds = time.time() - start_time

        for i, images in enumerate(test_generator.batches()):
            fid_evaluator.update(images)
            if i == 0:
                n = args.num_saved_samples
                epoch_dir = os.path.join(samples_dir, f"epoch_{epoch:03d}")
                save_samples(images[:n], prompts[:n], epoch_dir)
                writer.add_images("samples", images[:n].clamp(0, 1), epoch)
        fid = fid_evaluator.compute()

        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("eval/fid", fid, epoch)
        writer.add_scalar("time/train_seconds", train_seconds, epoch)
        writer.flush()
        print(
            f"epoch {epoch}: loss {loss:.5f}, fid {fid:.2f}, train {train_seconds:.0f}s"
        )

    writer.close()


if __name__ == "__main__":
    main()
