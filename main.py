import time
from typing import Any, Optional, Tuple

from src.utils.patch_bugs import *

import os
import re
import torch
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback
from src.lightning_data import DataModule
from src.lightning_model import LightningModel
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback

import logging
logger = logging.getLogger("lightning.pytorch")


class SaveOnInterruptCallback(Callback):
    """
    Best-effort interrupt checkpoint saver.

    Notes:
    - We do NOT add manual barriers here, otherwise Ctrl+C under DDP can hang.
    - We save both checkpoints/last.ckpt and a tag-specific ckpt.
    - We also save checkpoints/last_step.txt for resume auditing.
    """

    @staticmethod
    def _write_step_meta(ckpt_dir: str, step: int) -> None:
        step_file = os.path.join(ckpt_dir, "last_step.txt")
        with open(step_file, "w", encoding="utf-8") as f:
            f.write(f"{int(step)}\n")

    def _try_save(self, trainer: Trainer, tag: str) -> None:
        if trainer is None:
            return

        ckpt_dir = os.path.join(trainer.default_root_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        step = int(getattr(trainer, "global_step", 0))
        ckpt_last = os.path.join(ckpt_dir, "last.ckpt")
        ckpt_tag = os.path.join(ckpt_dir, f"last-{tag}.ckpt")

        try:
            # Let all ranks participate according to strategy implementation.
            trainer.save_checkpoint(ckpt_last)
            trainer.save_checkpoint(ckpt_tag)

            if trainer.is_global_zero:
                self._write_step_meta(ckpt_dir, step)
                print(f"[interrupt-save] saved {ckpt_last} and {ckpt_tag} at step={step}")
        except Exception as e:
            if trainer.is_global_zero:
                print(f"[interrupt-save] failed: {e}")

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        self._try_save(trainer, "exception")

    def on_keyboard_interrupt(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._try_save(trainer, "keyboard-interrupt")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Always refresh last.ckpt at train end as well.
        self._try_save(trainer, "train-end")


class ReWriteRootSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        stamp = time.strftime("%y%m%d%H%M")
        file_path = os.path.join(trainer.default_root_dir, f"config-{stage}-{stamp}.yaml")
        self.parser.save(
            self.config,
            file_path,
            skip_none=False,
            overwrite=self.overwrite,
            multifile=self.multifile,
        )


class ReWriteRootDirCli(LightningCLI):
    @staticmethod
    def _safe_read_int(path: str) -> Optional[int]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except Exception:
            return None

    @staticmethod
    def _extract_step_from_filename(filename: str) -> Optional[int]:
        """
        e.g.:
            last-step12345.ckpt -> 12345
            step=12345.ckpt      -> 12345 (fallback)
        """
        patterns = [
            r"step(\d+)",
            r"step=(\d+)",
        ]
        for p in patterns:
            m = re.search(p, filename)
            if m:
                return int(m.group(1))
        return None

    @classmethod
    def _find_latest_resume_run(
        cls,
        root_base: str,
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """
        Scan all run dirs under root_base and return:
          (run_dir, ckpt_path, step)

        Priority:
        1) checkpoints/last.ckpt + checkpoints/last_step.txt
        2) checkpoints/last.ckpt + mtime
        3) newest *.ckpt in checkpoints with parsed step or mtime

        Sorting preference:
        - prefer runs that contain last.ckpt
        - prefer valid step metadata
        - prefer larger step
        - prefer newer mtime
        """
        root_base = os.path.abspath(root_base)
        if not os.path.isdir(root_base):
            return None, None, None

        candidates = []

        for entry in os.scandir(root_base):
            if not entry.is_dir():
                continue

            run_dir = entry.path
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                continue

            last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
            step_file = os.path.join(ckpt_dir, "last_step.txt")

            if os.path.isfile(last_ckpt):
                step = cls._safe_read_int(step_file)
                mtime = os.path.getmtime(last_ckpt)
                candidates.append((run_dir, last_ckpt, step, mtime, 1))
                continue

            # fallback: any ckpt inside checkpoints/
            for fn in os.listdir(ckpt_dir):
                if not fn.endswith(".ckpt"):
                    continue
                ckpt_path = os.path.join(ckpt_dir, fn)
                step = cls._extract_step_from_filename(fn)
                mtime = os.path.getmtime(ckpt_path)
                candidates.append((run_dir, ckpt_path, step, mtime, 0))

        if not candidates:
            return None, None, None

        def sort_key(x):
            run_dir, ckpt_path, step, mtime, has_last = x
            step_valid = 1 if step is not None else 0
            step_value = step if step is not None else -1
            return (has_last, step_valid, step_value, mtime)

        best = max(candidates, key=sort_key)
        run_dir, ckpt_path, step, _, _ = best
        return run_dir, ckpt_path, step

    def before_instantiate_classes(self) -> None:
        super().before_instantiate_classes()

        config_trainer = self._get(self.config, "trainer", default={})

        # predict: disable logger
        if self.subcommand == "predict":
            config_trainer.logger = None

        should_resume = bool(self._get(self.config, "resume", default=False))

        # IMPORTANT:
        # In LightningCLI subcommand mode, ckpt_path belongs to fit/predict/test/... config,
        # not the top-level root config.
        subcfg = self.config[self.subcommand]
        has_ckpt_path = subcfg.get("ckpt_path") is not None

        if should_resume and not has_ckpt_path and self.subcommand == "fit":
            root_base = config_trainer.get("default_root_dir", None) or os.path.join(os.getcwd(), "workdirs")
            run_dir, ckpt, step = self._find_latest_resume_run(root_base)

            if ckpt is None:
                logger.warning(
                    "Resume requested, but no checkpoint found under %s",
                    os.path.abspath(root_base),
                )
            else:
                # IMPORTANT:
                # Must write into fit.ckpt_path, otherwise resume won't really take effect.
                subcfg["ckpt_path"] = ckpt
                logger.info(
                    "Auto resume checkpoint: %s (run_dir=%s, last_step=%s)",
                    ckpt, run_dir, str(step),
                )

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        class TagsClass:
            def __init__(self, exp: str):
                ...

        parser.add_class_arguments(TagsClass, nested_key="tags")

    def add_default_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_default_arguments_to_parser(parser)
        parser.add_argument("--torch_hub_dir", type=str, default=None, help="torch hub dir")
        parser.add_argument("--huggingface_cache_dir", type=str, default=None, help="huggingface hub dir")
        parser.add_argument("--resume", action="store_true", help="auto resume from latest run checkpoint")

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        config_trainer = self._get(self.config_init, "trainer", default={})
        default_root_dir = config_trainer.get("default_root_dir", None)

        if default_root_dir is None:
            default_root_dir = os.path.join(os.getcwd(), "workdirs")

        dirname = ""
        for v, k in self._get(self.config, "tags", default={}).items():
            dirname += f"{v}_{k}"
        default_root_dir = os.path.join(default_root_dir, dirname)

        # IMPORTANT:
        # Read ckpt_path from current subcommand config, e.g. fit.ckpt_path
        ckpt_path = self.config[self.subcommand].get("ckpt_path")

        if ckpt_path:
            # Resume: continue in the exact same run directory as checkpoint.
            default_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
            logger.info("Resume mode: reuse existing run directory: %s", default_root_dir)
        else:
            # New run: avoid overwriting non-empty existing dirs
            if os.path.exists(default_root_dir) and os.listdir(default_root_dir):
                base = default_root_dir
                i = 1
                while os.path.exists(default_root_dir) and os.listdir(default_root_dir):
                    default_root_dir = f"{base}-{i:03d}"
                    i += 1

        config_trainer.default_root_dir = default_root_dir

        trainer = super().instantiate_trainer(**kwargs)
        trainer.callbacks.append(SaveOnInterruptCallback())

        if trainer.is_global_zero:
            os.makedirs(default_root_dir, exist_ok=True)

        return trainer

    def instantiate_classes(self) -> None:
        torch_hub_dir = self._get(self.config, "torch_hub_dir")
        huggingface_cache_dir = self._get(self.config, "huggingface_cache_dir")

        if huggingface_cache_dir is not None:
            os.environ["HUGGINGFACE_HUB_CACHE"] = huggingface_cache_dir
        if torch_hub_dir is not None:
            os.environ["TORCH_HOME"] = torch_hub_dir
            torch.hub.set_dir(torch_hub_dir)

        super().instantiate_classes()


if __name__ == "__main__":
    cli = ReWriteRootDirCli(
        LightningModel,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=ReWriteRootSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )