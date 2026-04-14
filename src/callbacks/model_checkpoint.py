import os.path
from typing import Optional, Dict, Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


# class CheckpointHook(ModelCheckpoint):
#     """Save checkpoint with only the incremental part of the model"""
#     def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
#         self.dirpath = trainer.default_root_dir
#         self.exception_ckpt_path = os.path.join(self.dirpath, "on_exception.pt")
#         pl_module.strict_loading = False

#     def on_save_checkpoint(
#             self, trainer: "pl.Trainer",
#             pl_module: "pl.LightningModule",
#             checkpoint: Dict[str, Any]
#     ) -> None:
#         del checkpoint["callbacks"]

import os
from typing import Dict, Any
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


class CheckpointHook(ModelCheckpoint):
    """Checkpoint hook for periodic last.ckpt saving and resume-friendly layout."""

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        # Put ckpts under run_dir/checkpoints/
        self.dirpath = os.path.join(trainer.default_root_dir, "checkpoints")
        os.makedirs(self.dirpath, exist_ok=True)
        pl_module.strict_loading = False

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ) -> None:
        checkpoint.pop("callbacks", None)

        # Also persist global_step in a sidecar text file for easier auto-resume audit
        if trainer.is_global_zero:
            step_file = os.path.join(self.dirpath, "last_step.txt")
            with open(step_file, "w", encoding="utf-8") as f:
                f.write(f"{int(trainer.global_step)}\n")