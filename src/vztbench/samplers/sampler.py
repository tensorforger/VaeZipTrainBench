import torch

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from vztbench.transformers.transformer import TransformerModel


class Sampler:
    """
    Parallel sampler for 2d flow matching models.
    No mutable states after __init__.
    """

    def __init__(
        self,
        model: TransformerModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        num_inference_steps: int = 32,
        num_latent_channels: int = 32,
    ):
        self.model = model
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.num_latent_channels = num_latent_channels

        self.scheduler.set_timesteps(self.num_inference_steps, mu=1.0)
        self.timesteps = torch.cat([self.scheduler.timesteps, torch.zeros(1)])
        self.timesteps = self.timesteps.to(model.device, model.dtype)

    @torch.no_grad()
    def denoising_step(
        self,
        latents: torch.Tensor,
        from_timesteps: torch.Tensor,
        to_timesteps: torch.Tensor,
        conditioning: dict | None = None,
        cfg: float | None = None,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W]
            from_timesteps: [B]
            to_timesteps: [B]
            conditioning: according to the model

        returns:
            latents: [B, C, H, W]
        """
        assert not self.model.training, (
            "Model in training mode was passed to inference logic"
        )

        velocity = self.model(
            sample=latents,
            timesteps=from_timesteps,
            conditioning=conditioning,
        )

        if cfg is not None:
            negative_conditioning = {
                "text_embed": torch.zeros_like(conditioning["text_embed"])
            }

            negative_velocity = self.model(
                sample=latents,
                timesteps=from_timesteps,
                conditioning=negative_conditioning,
            )

            velocity = negative_velocity + (velocity - negative_velocity) * cfg

        deltas = (from_timesteps - to_timesteps) / 1000.0
        deltas = deltas[:, None, None, None]
        latents = latents - velocity * deltas

        return latents

    def get_generator(
        self,
        seed: int | None = None,
        device: torch.device | None = None,
    ):
        """
        returns:
            generator: torch.Generator
        """
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        return generator

    def sample_noise(
        self,
        width: int,
        height: int,
        batch_size: int = 1,
        seed: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        returns:
            noise: [B, C, H, W]
        """
        if device is None:
            device = self.model.device

        if dtype is None:
            dtype = self.model.dtype

        generator = self.get_generator(seed, device)
        noise = torch.randn(
            [batch_size, self.num_latent_channels, height, width],
            device=device,
            dtype=dtype,
            generator=generator,
        )

        return noise

    def sample_noise_like(self, tensor: torch.Tensor, seed: int | None = None):
        generator = self.get_generator(seed=seed, device=tensor.device)
        noise = torch.randn_like(tensor, generator=generator)

        return noise

    def invert(
        self,
        latents: torch.Tensor,
        timestep_idx: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W]
            noise: [B, C, H, W]
            timestep_idx: [B]
                timestep = 0 -> clear image
                timestep = num_inference_steps -> pure noise
        returns:
            latents: [B, C, H, W]
        """
        if noise is None:
            noise = self.sample_noise_like(latents)

        t = self.timestep_by_idx(timestep_idx)
        t = t[:, None, None, None]
        sigma = t / 1000.0
        latents = noise * sigma + (1 - sigma) * latents

        return latents

    def timestep_by_idx(self, idx: torch.Tensor) -> torch.Tensor:
        """
        args:
            idx: [B]
        returns:
            timestep: [B]
        """
        t = self.timesteps[idx]

        return t

    def sample_zero_timesteps(
        self,
        batch_size: int,
    ):
        idx = torch.zeros([batch_size], dtype=torch.long)
        return idx

    def denoise_to_cleans(
        self,
        latents: torch.Tensor,
        idx: torch.Tensor | None = None,
        conditioning: dict | None = None,
        cfg: float | None = None,
    ) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W]
            idx: [B], if none all set to num_inference_steps

        returns:
            edit_batch: [B, C, H, W]
        """
        B, C, H, W = latents.shape
        if idx is None:
            idx = self.sample_zero_timesteps(B)

        while idx.min() < self.num_inference_steps:
            from_timesteps = self.timestep_by_idx(idx)
            idx = idx + 1
            idx = torch.clamp(idx, 0, self.num_inference_steps)
            to_timesteps = self.timestep_by_idx(idx)

            latents = self.denoising_step(
                latents=latents,
                from_timesteps=from_timesteps,
                to_timesteps=to_timesteps,
                conditioning=conditioning,
                cfg=cfg,
            )

        return latents
