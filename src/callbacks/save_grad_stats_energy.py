import csv
import math
import os
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info
import matplotlib.pyplot as plt


class SaveGradStatsHook(Callback):
    """
    训练时：
      每隔 log_interval 把 step / fm_loss / alignment loss
      以及各 frequency band 的 energy weight / energy / loss 指标追加写入 CSV:

          {default_root_dir}/{save_dir}/{filename}

    验证开始时：
      读取截至当前为止的全部日志，
      画出 loss / band weight / band energy / band loss 曲线并保存到:

          {default_root_dir}/{save_dir}/plots/
    """

    def __init__(
        self,
        save_dir="train_logs",
        log_interval=10,
        filename="loss_stats.csv",
        plot_on_validation=True,
        plot_every_val=True,
        dpi=180,
    ):
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.filename = filename
        self.plot_on_validation = plot_on_validation
        self.plot_every_val = plot_every_val
        self.dpi = dpi
        self._enc_bank_names = []
        self._dec_bank_names = []

    def _to_float(self, x, default=float("nan")):
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default

    def _ordered_unique(self, values):
        seen = set()
        ordered = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _names_from_payload(self, payload, prefix):
        return [
            key[len(prefix):]
            for key in payload.keys()
            if isinstance(key, str) and key.startswith(prefix)
        ]

    def _sync_bank_names(self, pl_module=None, payload=None):
        enc_names = list(self._enc_bank_names)
        dec_names = list(self._dec_bank_names)

        diffusion_trainer = getattr(pl_module, "diffusion_trainer", None)
        if diffusion_trainer is not None:
            enc_names.extend(list(getattr(diffusion_trainer, "enc_bank_names", []) or []))
            dec_names.extend(list(getattr(diffusion_trainer, "dec_bank_names", []) or []))

        if isinstance(payload, dict):
            enc_names.extend(self._names_from_payload(payload, "enc_band_weight_"))
            enc_names.extend(self._names_from_payload(payload, "enc_band_energy_"))
            enc_names.extend(self._names_from_payload(payload, "enc_band_loss_"))

            dec_names.extend(self._names_from_payload(payload, "dec_band_weight_"))
            dec_names.extend(self._names_from_payload(payload, "dec_band_energy_"))
            dec_names.extend(self._names_from_payload(payload, "dec_band_loss_"))

        self._enc_bank_names = self._ordered_unique(enc_names)
        self._dec_bank_names = self._ordered_unique(dec_names)

    def _extra_log_fields(self):
        """
        这里集中定义 alignment / energy-based frequency weighting 相关字段。
        bank name 从 REPATrainer 同步，避免和 config 里的 enc_bank_names / dec_bank_names 脱节。
        """
        fields = [
            "loss_enc_align",
            "loss_dec_align",
        ]

        fields.extend([f"enc_band_weight_{name}" for name in self._enc_bank_names])
        fields.extend([f"enc_band_energy_{name}" for name in self._enc_bank_names])
        fields.extend([f"enc_band_loss_{name}" for name in self._enc_bank_names])

        fields.extend([f"dec_band_weight_{name}" for name in self._dec_bank_names])
        fields.extend([f"dec_band_energy_{name}" for name in self._dec_bank_names])
        fields.extend([f"dec_band_loss_{name}" for name in self._dec_bank_names])

        return fields

    def _all_log_fields(self):
        base_fields = [
            "step",
            "fm_loss",
        ]
        return base_fields + self._extra_log_fields()

    def _target_dir(self, trainer: "pl.Trainer"):
        return os.path.join(trainer.default_root_dir, self.save_dir)

    def _log_path(self, trainer: "pl.Trainer"):
        return os.path.join(self._target_dir(trainer), self.filename)

    def _plot_dir(self, trainer: "pl.Trainer"):
        return os.path.join(self._target_dir(trainer), "plots")

    def _ensure_log_header(self, log_path: str):
        fields = self._all_log_fields()

        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            with open(log_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            return

        with open(log_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            if old_fields == fields:
                return
            rows = list(reader)

        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})

    def on_train_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        if not trainer.is_global_zero:
            return

        target_dir = self._target_dir(trainer)
        os.makedirs(target_dir, exist_ok=True)

        log_path = self._log_path(trainer)

        self._sync_bank_names(pl_module=pl_module)
        self._ensure_log_header(log_path)

        rank_zero_info(f"Loss stats will be appended to {log_path}")

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

        if not isinstance(outputs, dict):
            return

        payload = outputs.get("outputs", outputs)

        if not isinstance(payload, dict):
            return

        step = payload.get("step", None)
        fm_loss = payload.get("fm_loss", None)

        # 没到 trainer 内部 log step 时，不会有这些字段
        if step is None or fm_loss is None:
            return

        if int(step) % int(self.log_interval) != 0:
            return

        target_dir = self._target_dir(trainer)
        os.makedirs(target_dir, exist_ok=True)

        log_path = self._log_path(trainer)

        # 如果中途文件被删了，自动重建 header；如果沿用旧实验文件，也会补齐新字段。
        self._sync_bank_names(pl_module=pl_module, payload=payload)
        self._ensure_log_header(log_path)

        row = {
            "step": int(step),
            "fm_loss": f"{self._to_float(fm_loss):.8f}",
        }

        for key in self._extra_log_fields():
            row[key] = f"{self._to_float(payload.get(key, None)):.8f}"

        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._all_log_fields())
            writer.writerow(row)

    def _read_loss_log(self, log_path: str):
        """读取 CSV 日志，返回 steps 和每个数值字段的序列。"""
        if not os.path.exists(log_path):
            return [], {}

        steps = []
        series = {}

        with open(log_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            value_fields = [field for field in fieldnames if field != "step"]
            series = {field: [] for field in value_fields}

            for row in reader:
                try:
                    steps.append(int(float(row.get("step", "nan"))))
                except ValueError:
                    continue

                for field in value_fields:
                    series[field].append(self._to_float(row.get(field, None)))

        return steps, series

    def _has_any_finite(self, values):
        return any(math.isfinite(value) for value in values)

    def _save_series_plot(
        self,
        plot_dir,
        filename_base,
        title,
        ylabel,
        steps,
        series_items,
        trainer,
    ):
        series_items = [
            (label, values)
            for label, values in series_items
            if values and self._has_any_finite(values)
        ]
        if not series_items:
            return None, None

        latest_path = os.path.join(plot_dir, f"{filename_base}_latest.png")
        step_path = os.path.join(plot_dir, f"{filename_base}_step_{trainer.global_step}.png")

        plt.figure(figsize=(8, 5))
        for label, values in series_items:
            plt.plot(steps, values, label=label)

        plt.xlabel("step")
        plt.ylabel(ylabel)
        plt.title(f"{title} up to validation (global_step={trainer.global_step})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(latest_path, dpi=self.dpi)
        plt.savefig(step_path, dpi=self.dpi)
        plt.close()

        return latest_path, step_path

    def _save_loss_plot(
        self,
        trainer: "pl.Trainer",
        steps,
        series,
    ):
        if len(steps) == 0:
            return None, None

        plot_dir = self._plot_dir(trainer)
        os.makedirs(plot_dir, exist_ok=True)

        return self._save_series_plot(
            plot_dir=plot_dir,
            filename_base="loss_curve",
            title="Training losses",
            ylabel="loss",
            steps=steps,
            series_items=[
                (field, series.get(field, []))
                for field in [
                    "fm_loss",
                    "loss_enc_align",
                    "loss_dec_align",
                ]
            ],
            trainer=trainer,
        )

    def _save_metric_plots(self, trainer: "pl.Trainer", steps, series):
        plot_dir = self._plot_dir(trainer)
        os.makedirs(plot_dir, exist_ok=True)

        saved_paths = []

        prefix_groups = [
            ("enc_band_weight_", "enc_band_weight"),
            ("enc_band_energy_", "enc_band_energy"),
            ("enc_band_loss_", "enc_band_loss"),
            ("dec_band_weight_", "dec_band_weight"),
            ("dec_band_energy_", "dec_band_energy"),
            ("dec_band_loss_", "dec_band_loss"),
        ]

        for prefix, filename_base in prefix_groups:
            fields = [
                field
                for field in series.keys()
                if isinstance(field, str) and field.startswith(prefix)
            ]

            fields.sort(key=lambda field: field[len(prefix):])

            latest_path, step_path = self._save_series_plot(
                plot_dir=plot_dir,
                filename_base=filename_base,
                title=filename_base,
                ylabel="value",
                steps=steps,
                series_items=[
                    (field[len(prefix):], series.get(field, []))
                    for field in fields
                ],
                trainer=trainer,
            )

            if latest_path is not None:
                saved_paths.append((latest_path, step_path))

        return saved_paths

    def on_validation_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """
        每次进入 validation 时，读取当前累计日志并画图。
        """
        if not self.plot_on_validation:
            return

        if not trainer.is_global_zero:
            return

        # 跳过 Lightning 的 sanity validation
        if getattr(trainer, "sanity_checking", False):
            return

        log_path = self._log_path(trainer)

        steps, series = self._read_loss_log(log_path)

        if len(steps) == 0:
            return

        latest_path, step_path = self._save_loss_plot(
            trainer,
            steps,
            series,
        )
        saved_paths = self._save_metric_plots(trainer, steps, series)

        if latest_path is not None:
            rank_zero_info(
                f"Saved loss curves to {latest_path} and {step_path}"
            )
        if saved_paths:
            rank_zero_info(
                f"Saved {len(saved_paths)} energy-weighted band metric curve groups to {self._plot_dir(trainer)}"
            )