import torch
from torch import nn
import os

from transformers import CLIPTextModel, CLIPTokenizer


class SDXLTextEncoder(nn.Module):
    def __init__(
        self, path_to_model: str, device: torch.device = "cuda", dtype=torch.float32
    ):
        super().__init__()
        text_encoder_path = os.path.join(path_to_model, "text_encoder")
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        self.text_encoder.to(device, dtype)

        tokenizer_path = os.path.join(path_to_model, "tokenizer")
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @torch.no_grad()
    def encode_text_batch(self, text_batch) -> torch.Tensor:
        inputs = self.tokenizer(
            text_batch,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to("cuda")

        embed = self.text_encoder(inputs.input_ids, return_dict=True)
        text_embeds = embed.last_hidden_state

        return text_embeds
