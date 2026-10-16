from abc import ABC, abstractmethod
import torch


class ImageVAE(torch.nn.Module):
    """
    Abstract class and common interface for all Image VAE wrappers.
    Intended for VAE inference only.
    VAE should handle by itself:
    - pixel and latent normalization
    - image and latent device and dtype handling according to weights
    Subclasses must define a class constant NAME equal to the config entry.
    """

    NAME: str

    @abstractmethod
    def encode(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        returns:
            latents: [B, C, H, W], normalized to model format
        """
        pass

    @abstractmethod
    def decode(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
            latents: [B, C, H, W], normalized to model format
        returns:
            image: [B, C, H, W], pixel_range = [0.0, 1.0], channel_order = RGB
        """
        pass

    @abstractmethod
    def get_latent_channels(self) -> int:
        """
        returns:
            latent_channels of the VAE
        """
        pass

    @abstractmethod
    def get_spatial_compression(self) -> int:
        """
        returns:
            spatial_compression of the VAE
        """
        pass

    @abstractmethod
    def get_vae_name(self) -> str:
        pass
