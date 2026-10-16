import json
import torch
import os
import cv2
from safetensors.torch import save_file
from tqdm import tqdm

from vztbench.utils.image_tools import *
from vztbench.vaes.flux_2_vae import Flux2VAE
from vztbench.text_encoders.sdxl_text_encoder import SDXLTextEncoder
import vaezip


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    vae = Flux2VAE(
        path_to_model="models/FLUX.2-klein-4B/vae", device=device, dtype=dtype
    )
    vae.compile()

    text_encoder = SDXLTextEncoder(
        path_to_model="models/stable-diffusion-xl-base-1.0", device=device, dtype=dtype
    )

    text_encoder.compile()

    in_dir = "datasets/coco-2017/train2017"
    fp32_out_dir = "datasets/coco-2017/fp32_encoded_train2017"
    lbrq_6bit_out_dir = "datasets/coco-2017/lbrq_6bit_encoded_train2017"
    lbrq_4bit_out_dir = "datasets/coco-2017/lbrq_4bit_encoded_train2017"
    lbrq_3bit_out_dir = "datasets/coco-2017/lbrq_3bit_encoded_train2017"

    os.makedirs(fp32_out_dir, exist_ok=True)
    os.makedirs(lbrq_6bit_out_dir, exist_ok=True)
    os.makedirs(lbrq_4bit_out_dir, exist_ok=True)
    os.makedirs(lbrq_3bit_out_dir, exist_ok=True)
    files = os.listdir(in_dir)

    with open("datasets/coco-2017/annotations/captions_train2017.json") as f:
        annotations = json.load(f)["annotations"]
    captions = {}
    for ann in annotations:
        captions.setdefault(ann["image_id"], ann["caption"])

    for file in tqdm(files):
        name = file.split(".")[0]
        image = cv2.imread(os.path.join(in_dir, file))
        image = crop_maximal_rectangle(image, target_height=256, target_width=256)
        image = cv2_to_torch(image)
        latents = vae.encode(image)

        caption = captions[int(name)]
        embed = text_encoder.encode_text_batch([caption])

        # fp32
        out_path = os.path.join(fp32_out_dir, f"{name}.safetensors")
        sample_dict = {"latents": latents.cpu(), "text_embed": embed.cpu()}
        save_file(tensors=sample_dict, filename=out_path)

        # lbrq_6bit
        out_path = os.path.join(lbrq_6bit_out_dir, f"{name}.vz.safetensors")
        sample_dict = {
            "latents": vaezip.encode(latents, codec="lbrq_6bit"),
            "text_embed": vaezip.encode(embed, codec="cast_fp32"),
        }
        vaezip.save(sample_dict, path=out_path)

        # lbrq_4bit
        out_path = os.path.join(lbrq_4bit_out_dir, f"{name}.vz.safetensors")
        sample_dict = {
            "latents": vaezip.encode(latents, codec="lbrq_4bit"),
            "text_embed": vaezip.encode(embed, codec="cast_fp32"),
        }
        vaezip.save(sample_dict, path=out_path)

        # lbrq_3bit
        out_path = os.path.join(lbrq_3bit_out_dir, f"{name}.vz.safetensors")
        sample_dict = {
            "latents": vaezip.encode(latents, codec="lbrq_3bit"),
            "text_embed": vaezip.encode(embed, codec="cast_fp32"),
        }
        vaezip.save(sample_dict, path=out_path)


if __name__ == "__main__":
    main()
