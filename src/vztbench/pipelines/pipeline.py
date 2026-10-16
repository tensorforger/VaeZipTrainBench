import torch

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vztbench.samplers.sampler import Sampler
from vztbench.transformers.transformer import TransformerModel
from vztbench.vaes.flux_2_vae import Flux2VAE


class Pipeline:
    """
    Parallel Generation Pipeline for flow matching models.
    No mutable states after __init__.
    """

    def __init__(
        self,
        model: TransformerModel,
        vae: Flux2VAE,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        self.model = model
        self.vae = vae
        self.scheduler = scheduler

    def generate_image(
        self,
        width: int,
        height: int,
        batch_size: int,
        num_inference_steps: int,
        conditioning: dict | None = None,
        seed: int | None = None,
        autocast_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """
        Generates a random sample where all frames are initialized as pure noise.

        returns:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        latent_width = width // self.vae.get_spatial_compression()
        latent_height = height // self.vae.get_spatial_compression()
        self.sampler = Sampler(
            model=self.model,
            scheduler=self.scheduler,
            num_inference_steps=num_inference_steps,
            num_latent_channels=self.vae.get_latent_channels(),
        )
        noise = self.sampler.sample_noise(
            width=latent_width,
            height=latent_height,
            batch_size=batch_size,
            seed=seed,
        )
        # Autocast covers only the denoiser, the VAE decode stays in its own dtype.
        with torch.autocast(
            device_type=self.model.device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None,
        ):
            latents = self.sampler.denoise_to_cleans(
                latents=noise,
                conditioning=conditioning,
            )

        image = self.vae.decode(latents=latents)

        return image
