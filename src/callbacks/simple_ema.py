from typing import Any, Dict

import torch
import torch.nn as nn
import threading
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src.utils.copy import swap_tensors

class SimpleEMA(Callback):
    def __init__(self, net:nn.Module, ema_net:nn.Module,
                 decay: float = 0.9999,
                 every_n_steps: int = 1,
                 eval_original_model:bool = False
                 ):
        super().__init__()
        self.decay = decay
        self.every_n_steps = every_n_steps
        self.eval_original_model = eval_original_model
        self._stream = torch.cuda.Stream()

        self.net_params = list(net.parameters())
        self.ema_params = list(ema_net.parameters())

    def swap_model(self):
        for ema_p, p, in zip(self.ema_params, self.net_params):
            swap_tensors(ema_p, p)

    def ema_step(self):
        @torch.no_grad()
        def ema_update(ema_model_tuple, current_model_tuple, decay):
            torch._foreach_mul_(ema_model_tuple, decay)
            torch._foreach_add_(
                ema_model_tuple, current_model_tuple, alpha=(1.0 - decay),
            )

        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            ema_update(self.ema_params, self.net_params, self.decay)


    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step % self.every_n_steps == 0:
            self.ema_step()

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.eval_original_model:
            self.swap_model()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.eval_original_model:
            self.swap_model()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.eval_original_model:
            self.swap_model()

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.eval_original_model:
            self.swap_model()


    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
            "eval_original_model": self.eval_original_model,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay = state_dict["decay"]
        self.every_n_steps = state_dict["every_n_steps"]
        self.eval_original_model = state_dict["eval_original_model"]

