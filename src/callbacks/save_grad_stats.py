import os
import csv
import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info
import matplotlib.pyplot as plt


class SaveGradStatsHook(Callback):
    """
    训练时：
      把 step / fm_loss / inv_loss / sigreg_loss / loss
      追加写入:
          {default_root_dir}/{save_dir}/{filename}

    验证开始时：
      读取截至当前为止的全部日志，
      画出三条曲线:
          fm_loss / inv_loss / sigreg_loss
      保存到:
          {default_root_dir}/{save_dir}/plots/
    """

    def __init__(
        self,
        save_dir="train_logs",
        log_interval=10,
        filename="loss_stats.txt",
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

    def _to_float(self, x, default=float("nan")):
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default

    def _extra_log_fields(self):
        """
        这里集中定义你想额外写入日志的 collapse / shortcut 字段。
        后续想增删字段，只改这里即可。
        """
        return [
            # mid1_mean vs mid2_mean cross-view
            "mid_img_cross_pos_cos",
            "mid_img_cross_neg_cos",
            "mid_img_cross_gap",
            # p1 vs p2 cross-view
            "p_img_cross_pos_cos",
            "p_img_cross_neg_cos",
            "p_img_cross_gap",

            # proj1 前（mid1_mean）collapse
            "mid1_mean_effective_rank",
            "mid1_mean_lambda1_ratio",
            "mid1_mean_offdiag_mean",
            "mid1_mean_dead_dim_ratio",
            "mid1_mean_std_mean",

            # proj1 后（p1）collapse
            "p1_effective_rank",
            "p1_lambda1_ratio",
            "p1_offdiag_mean",
            "p1_dead_dim_ratio",
            "p1_std_mean",
        ]

    def _target_dir(self, trainer: "pl.Trainer"):
        return os.path.join(trainer.default_root_dir, self.save_dir)

    def _log_path(self, trainer: "pl.Trainer"):
        return os.path.join(self._target_dir(trainer), self.filename)

    def _plot_dir(self, trainer: "pl.Trainer"):
        return os.path.join(self._target_dir(trainer), "plots")

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not trainer.is_global_zero:
            return

        target_dir = self._target_dir(trainer)
        os.makedirs(target_dir, exist_ok=True)

        log_path = self._log_path(trainer)
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            with open(log_path, "w", encoding="utf-8") as f:
                base_fields = ["step", "fm_loss", "similarity_loss", "sigreg_loss", "var_loss", "cov_loss", "disp_loss", "loss"]
                extra_fields = self._extra_log_fields()
                all_fields = base_fields + extra_fields
                f.write("\t".join(all_fields) + "\n")
            # with open(log_path, "w", encoding="utf-8") as f:
            #     f.write("step\tfm_loss\tsimilarity_loss\tsigreg_loss\tloss\n")

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
        similarity_loss = payload.get("similarity_loss", None)
        sigreg_loss = payload.get("sigreg_loss", None)
        var_loss = payload.get("var_loss", None)
        cov_loss = payload.get("cov_loss", None)
        disp_loss = payload.get("disp_loss", None)
        total_loss = payload.get("loss", None)

        if step is None or fm_loss is None or similarity_loss is None or var_loss is None:
            return
        if step % self.log_interval != 0:
            return

        target_dir = self._target_dir(trainer)
        os.makedirs(target_dir, exist_ok=True)
        log_path = self._log_path(trainer)

        # with open(log_path, "a", encoding="utf-8") as f:
        #     f.write(
        #         f"{int(step)}\t"
        #         f"{float(fm_loss):.8f}\t"
        #         f"{float(similarity_loss):.8f}\t"
        #         f"{float(sigreg_loss):.8f}\t"
        #         f"{float(total_loss) if total_loss is not None else float('nan'):.8f}\n"
        #     )
        with open(log_path, "a", encoding="utf-8") as f:
            base_values = [
                str(int(step)),
                f"{self._to_float(fm_loss):.8f}",
                f"{self._to_float(similarity_loss):.8f}",
                f"{self._to_float(sigreg_loss):.8f}",
                f"{self._to_float(var_loss):.8f}",
                f"{self._to_float(cov_loss):.8f}",
                f"{self._to_float(disp_loss):.8f}",
                f"{self._to_float(total_loss):.8f}",
            ]

            extra_values = []
            for key in self._extra_log_fields():
                extra_values.append(f"{self._to_float(payload.get(key, None)):.8f}")

            f.write("\t".join(base_values + extra_values) + "\n")

    def _read_loss_log(self, log_path: str):
        """
        读取 TSV 日志，返回:
            steps, fm_losses, inv_losses, sigreg_losses
        """
        if not os.path.exists(log_path):
            return [], [], [], []

        steps = []
        fm_losses = []
        similarity_losses = []
        sigreg_losses = []
        var_losses = []
        cov_losses = []
        disp_losses = []

        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)  # 跳过表头
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    steps.append(int(row[0]))
                    fm_losses.append(float(row[1]))
                    similarity_losses.append(float(row[2]))
                    sigreg_losses.append(float(row[3]))
                    var_losses.append(float(row[4]))
                    cov_losses.append(float(row[5]))
                    disp_losses.append(float(row[6]))
                except ValueError:
                    continue

        return steps, fm_losses, similarity_losses, sigreg_losses, var_losses, cov_losses, disp_losses

    def _save_loss_plot(
        self,
        trainer: "pl.Trainer",
        steps,
        fm_losses,
        similarity_losses,
        sigreg_losses,
        var_losses,
        cov_losses,
        disp_losses,
    ):
        if len(steps) == 0:
            return None, None

        plot_dir = self._plot_dir(trainer)
        os.makedirs(plot_dir, exist_ok=True)

        latest_path = os.path.join(plot_dir, "loss_curve_latest.png")
        step_path = os.path.join(plot_dir, f"loss_curve_step_{trainer.global_step}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(steps, fm_losses, label="fm_loss")
        plt.plot(steps, similarity_losses, label="similarity_loss")
        plt.plot(steps, sigreg_losses, label="sigreg_loss")
        plt.plot(steps, var_losses, label="var_loss")
        plt.plot(steps, cov_losses, label="cov_loss")
        plt.plot(steps, disp_losses, label="disp_loss")

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"Training losses up to validation (global_step={trainer.global_step})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(latest_path, dpi=self.dpi)
        plt.savefig(step_path, dpi=self.dpi)
        plt.close()

        return latest_path, step_path

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
        steps, fm_losses, similarity_losses, sigreg_losses, var_losses, cov_losses, disp_losses = self._read_loss_log(log_path)

        if len(steps) == 0:
            return

        latest_path, step_path = self._save_loss_plot(
            trainer,
            steps,
            fm_losses,
            similarity_losses,
            sigreg_losses,
            var_losses,
            cov_losses,
            disp_losses
        )

        if latest_path is not None:
            rank_zero_info(
                f"Saved loss curves to {latest_path} and {step_path}"
            )