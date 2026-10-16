import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        warmup_updates: int = 512,
        device=None,
    ):
        self.decay = decay

        self.ema_model = copy.deepcopy(model)
        self.ema_model = self.ema_model.to(dtype=torch.float32)
        self.ema_model.eval()
        self.training = False

        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        if device is not None:
            self.ema_model.to(device)

        self._cache_params_and_buffers(model)

        self.updates = 0
        self.warmup_updates = warmup_updates

    def _cache_params_and_buffers(self, model):
        raw_model = model.module if hasattr(model, "module") else model

        self.ema_params = list(self.ema_model.parameters())
        self.ema_buffers = list(self.ema_model.buffers())

        self.model_params = list(raw_model.parameters())
        self.model_buffers = list(raw_model.buffers())

        assert len(self.ema_params) == len(self.model_params), (
            "Parameter count mismatch!"
        )
        assert len(self.ema_buffers) == len(self.model_buffers), (
            "Buffer count mismatch!"
        )

    @torch.no_grad()
    def update(self, model):
        decay = self.decay
        self.updates += 1
        if self.updates < self.warmup_updates:
            decay = 0.0
        raw_model = model.module if hasattr(model, "module") else model
        if len(self.model_params) != len(list(raw_model.parameters())):
            self._cache_params_and_buffers(model)

        for ema_p, model_p in zip(self.ema_params, self.model_params):
            ema_p.mul_(decay).add_(model_p.data.float(), alpha=1.0 - decay)

        for ema_b, model_b in zip(self.ema_buffers, self.model_buffers):
            ema_b.copy_(model_b.data)

    @property
    def device(self):
        return next(self.ema_model.parameters()).device

    @property
    def dtype(self):
        return next(self.ema_model.parameters()).dtype

    def eval(self):
        self.ema_model.eval()

    def train(self, mode=True):
        self.ema_model.train(mode)

    def to(self, *args, **kwargs):
        self.ema_model.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        self.ema_model.cuda(device)
        return self

    def cpu(self):
        self.ema_model.cpu()
        return self

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
