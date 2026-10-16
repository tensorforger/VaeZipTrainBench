import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.embeddings import (
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)

NUM_ROPE_AXES = 4  # (modality, h, w, text position)
MODALITY_TEXT = 0
MODALITY_IMAGE = 1


class PosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            ids: [seq_len, num_axes]

        returns:
            cos, sin: [seq_len, head_dim]
        """
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float64,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class RoPESelfAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(
        self,
        qkv: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        args:
            qkv: [batch, seq_len, num_heads * head_dim * 3]
            rotary_emb: tuple(
                cos: [seq_len, head_dim],
                sin: [seq_len, head_dim]
            )

        returns:
            x: [batch, seq_len, num_heads * head_dim]
        """
        b, l, _ = qkv.shape

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_emb(q, rotary_emb)
        k = apply_rotary_emb(k, rotary_emb)

        h = F.scaled_dot_product_attention(q, k, v)

        return h.transpose(1, 2).reshape(b, l, self.num_heads * self.head_dim)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: [batch, ..., dim * 2]

        returns:
            x: [batch, ..., dim]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class FusedBlock(nn.Module):
    """Self attention and SwiGLU MLP computed in parallel from one input projection."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: int,
    ):
        super().__init__()
        self.attention_inner_dim = num_heads * head_dim
        self.qkv_dim = self.attention_inner_dim * 3
        self.mlp_inner_dim = channels * mlp_ratio
        self.mlp_2x_dim = self.mlp_inner_dim * 2

        self.in_proj = nn.Linear(channels, self.qkv_dim + self.mlp_2x_dim)
        self.out_proj = nn.Linear(
            self.attention_inner_dim + self.mlp_inner_dim, channels
        )

        self.self_attention = RoPESelfAttention(num_heads=num_heads, head_dim=head_dim)
        self.act_fn = SwiGLU()

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        args:
            x: [batch, seq_len, channels]
            rotary_emb: tuple(cos, sin), each [seq_len, head_dim]

        returns:
            x: [batch, seq_len, channels]
        """
        qkv, mlp_state = torch.split(
            self.in_proj(x), [self.qkv_dim, self.mlp_2x_dim], dim=-1
        )

        attn_output = self.self_attention(qkv, rotary_emb)
        mlp_state = self.act_fn(mlp_state)

        return self.out_proj(torch.cat([attn_output, mlp_state], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: int,
    ):
        super().__init__()

        self.fused_block = FusedBlock(
            channels=channels,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
        )

        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Linear(channels, channels * 3)

        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        modulation_conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            x: [batch, seq_len, channels]
            rotary_emb: tuple(cos, sin), each [seq_len, head_dim]
            modulation_conditioning: [batch, channels]

        returns:
            x: [batch, seq_len, channels]
        """
        modulation = self.modulation(modulation_conditioning)
        scale, shift, gate = modulation[:, None, :].chunk(3, dim=-1)

        residual = x

        x = self.norm(x)
        x = (1 + scale) * x + shift
        x = self.fused_block(x, rotary_emb)

        return residual + x * gate


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128, max_period: int = 1000):
        super().__init__()
        self.dim = dim

        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period))
            * torch.arange(half, dtype=torch.float32)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(dtype=timesteps.dtype)

        args = timesteps[..., None] * freqs
        return torch.cat((torch.sin(args), torch.cos(args)), dim=-1)


class ModulationConditioning(nn.Module):
    def __init__(
        self,
        dim: int,
        time_embed_dim: int = 128,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(dim=time_embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(time_embed_dim, dim * mlp_ratio),
            nn.SiLU(),
            nn.Linear(dim * mlp_ratio, dim),
            nn.SiLU(),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        args:
            timesteps: [batch]

        returns:
            modulation_conditioning: [batch, dim]
        """
        return self.ff(self.time_embed(timesteps))


class Transformer(nn.Module):
    def __init__(
        self,
        base_dim: int,
        num_heads: int,
        head_dim: int,
        num_blocks: int,
        mlp_ratio: int,
        rope_theta: int,
    ):
        super().__init__()

        assert head_dim % NUM_ROPE_AXES == 0
        self.pos_embed = PosEmbed(
            theta=rope_theta, axes_dim=[head_dim // NUM_ROPE_AXES] * NUM_ROPE_AXES
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels=base_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

        self.modulation_conditioning = ModulationConditioning(dim=base_dim)

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            x: [batch, seq_len, base_dim]
            ids: [seq_len, 4]
            timesteps: [batch]

        returns:
            x: [batch, seq_len, base_dim]
        """
        rotary_emb = self.pos_embed(ids)
        modulation_conditioning = self.modulation_conditioning(timesteps)

        for block in self.blocks:
            x = block(x, rotary_emb, modulation_conditioning)

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        sample_dim: int = 32,
        text_dim: int = 768,
        base_dim: int = 512,
        num_heads: int = 8,
        head_dim: int = 64,
        num_blocks: int = 16,
        mlp_ratio: int = 4,
        rope_theta: int = 1000,
        use_patchify: bool = False,
    ):
        super().__init__()

        latents_dim = sample_dim * 4 if use_patchify else sample_dim

        self.image_in_proj = nn.Linear(latents_dim, base_dim)
        self.text_in_proj = nn.Linear(text_dim, base_dim)
        self.out_proj = nn.Linear(base_dim, latents_dim)

        self.transformer = Transformer(
            base_dim, num_heads, head_dim, num_blocks, mlp_ratio, rope_theta
        )

        self.num_blocks = num_blocks
        self.use_patchify = use_patchify

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def prepare_text_ids(self, length: int, device: torch.device) -> torch.Tensor:
        """
        returns:
            text_ids: [L, 4] (modality, h, w, position)
        """
        ids = torch.zeros(length, NUM_ROPE_AXES, device=device)
        ids[:, 0] = MODALITY_TEXT
        ids[:, 3] = torch.arange(length, device=device)
        return ids

    def prepare_image_ids(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """
        returns:
            image_ids: [H*W, 4] (modality, h, w, position)
        """
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        hw = torch.cartesian_prod(h, w)
        ids = torch.zeros(height * width, NUM_ROPE_AXES, device=device)
        ids[:, 0] = MODALITY_IMAGE
        ids[:, 1:3] = hw.to(ids.dtype)
        return ids

    def pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W]

        returns:
            latents: [B, H * W, C]
        """
        return latents.flatten(2).transpose(1, 2)

    def unpack_latents(self, latents: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        args:
            latents: [B, H * W, C]

        returns:
            latents: [B, C, H, W]
        """
        B, _, C = latents.shape
        return latents.transpose(1, 2).reshape(B, C, H, W)

    def patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C, H, W]

        returns:
            latents: [B, C * 4, H // 2, W // 2]
        """
        B, C, H, W = latents.shape
        assert H % 2 == 0 and W % 2 == 0

        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(B, C * 4, H // 2, W // 2)

    def unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        args:
            latents: [B, C * 4, H, W]

        returns:
            latents: [B, C, H * 2, W * 2]
        """
        B, C4, H, W = latents.shape
        assert C4 % 4 == 0

        latents = latents.reshape(B, C4 // 4, 2, 2, H, W)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(B, C4 // 4, H * 2, W * 2)

    def forward(
        self,
        sample: torch.Tensor,
        conditioning: dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        args:
            sample: [B, C, H, W]
            timesteps: [B], scale [0...1000]
            conditioning: dict
                text_embed: [B, L, D]

        returns:
            pred: [B, C, H, W]
        """
        text_embed = conditioning["text_embed"]

        x = self.patchify_latents(sample) if self.use_patchify else sample
        _, _, H, W = x.shape

        text_ids = self.prepare_text_ids(text_embed.shape[1], x.device)
        image_ids = self.prepare_image_ids(H, W, x.device)
        ids = torch.cat([text_ids, image_ids], dim=0)

        text_tokens = self.text_in_proj(text_embed)
        image_tokens = self.image_in_proj(self.pack_latents(x))
        tokens = torch.cat([text_tokens, image_tokens], dim=1)

        tokens = self.transformer(tokens, ids, timesteps)

        image_tokens = tokens[:, text_tokens.shape[1] :]
        x = self.unpack_latents(self.out_proj(image_tokens), H, W)

        if self.use_patchify:
            x = self.unpatchify_latents(x)

        return x
