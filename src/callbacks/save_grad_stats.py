# new document, for gradient logging
import os
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info

class SaveGradStatsHook(Callback):
    """
    Append all grad_log_text records generated during training to the same file:
      {default_root_dir}/{save_dir}/grad_stats.txt
    """
    def __init__(self, save_dir="train_logs", log_interval=10, filename="grad_stats.txt"):
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.filename = filename

    def _target_dir(self, trainer: "pl.Trainer"):
        return os.path.join(trainer.default_root_dir, self.save_dir)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            target_dir = self._target_dir(trainer)
            os.makedirs(target_dir, exist_ok=True)
            log_path = os.path.join(target_dir, self.filename)
            rank_zero_info(f"Grad stats will be appended to {log_path}")



    # on_train_step_end: The step hasn't been updated yet, meaning it's still at step 9.
    # The batch update will update the step, which is now at step 10.

    # training_step
    # ↓
    # on_train_step_end
    # ↓
    # (backward / optimizer.step / lr_scheduler.step)

    # training_step
    # ↓
    # backward
    # ↓
    # optimizer.step (maybe)
    # ↓
    # lr_scheduler.step (maybe)
    # ↓
    # on_train_batch_end
    
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if trainer.global_step % self.log_interval != 0:
            return
        if not isinstance(outputs, dict):
            return

        # return {"loss": ..., "outputs": outputs}
        text = outputs.get("outputs", {}).get("grad_log_text", None)
        if not text:
            return

        target_dir = self._target_dir(trainer)
        os.makedirs(target_dir, exist_ok=True)

        log_path = os.path.join(target_dir, self.filename)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")

