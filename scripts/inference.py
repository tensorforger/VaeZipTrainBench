from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vztbench.transformers.transformer import TransformerModel
from vztbench.pipelines.pipeline import Pipeline
from vztbench.utils.image_tools import *
from vztbench.text_encoders.sdxl_text_encoder import SDXLTextEncoder
from vztbench.vaes.flux_2_vae import Flux2VAE


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    vae = Flux2VAE(
        path_to_model="models/FLUX.2-klein-4B/vae", device=device, dtype=dtype
    )

    text_encoder = SDXLTextEncoder(
        path_to_model="models/stable-diffusion-xl-base-1.0", device=device, dtype=dtype
    )

    model = TransformerModel()
    model.to(device=device).eval()

    scheduler = FlowMatchEulerDiscreteScheduler()

    text_embed = text_encoder.encode_text_batch(["A dog on the green grass"])

    print(text_embed.shape)

    conditioning = {"text_embed": text_embed}

    pipeline = Pipeline(model=model, vae=vae, scheduler=scheduler)

    image = pipeline.generate_image(
        width=256,
        height=256,
        batch_size=1,
        num_inference_steps=32,
        conditioning=conditioning,
        seed=52,
    )

    show_torch_image(image)


if __name__ == "__main__":
    main()
