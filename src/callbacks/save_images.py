import sys
import os
sys.path.append("path_to_your_evaluations")   # need tobe revised
import lightning.pytorch as pl
from lightning.pytorch import Callback
import argparse
import io
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple
from evaluator import Evaluator
import numpy as np
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from scipy import linalg
from tqdm.auto import tqdm
import torch
import os.path
import numpy
from PIL import Image
from typing import Sequence, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_utilities.core.rank_zero import rank_zero_info
from datetime import datetime

def process_fn(image, path):
    Image.fromarray(image).save(path)

class SaveImagesHook(Callback):
    def __init__(self, save_dir="val", max_save_num=100, compressed=True):
       self.save_dir = save_dir
       self.max_save_num = max_save_num
       self.compressed = compressed

    def save_start(self, target_dir):
        self.target_dir = target_dir
        self.executor_pool = ThreadPoolExecutor(max_workers=8)
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)
        else:
            if os.listdir(target_dir) and "debug" not in str(target_dir):
                raise FileExistsError(f'{self.target_dir} already exists and not empty!')
        self.samples = []
        self._have_saved_num = 0
        rank_zero_info(f"Save images to {self.target_dir}")

    def save_image(self, images, filenames):
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        for sample, filename in zip(images, filenames):
            if isinstance(filename, Sequence):
                filename = filename[0]
            path = f'{self.target_dir}/{filename}'
            if self._have_saved_num >= self.max_save_num:
                break
            self.executor_pool.submit(process_fn, sample, path)
            self._have_saved_num += 1

    def process_batch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: STEP_OUTPUT,
        batch: Any,
    ) -> None:
        b, c, h, w = samples.shape
        xT, y, metadata = batch
        all_samples = pl_module.all_gather(samples).view(-1, c, h, w)
        self.save_image(samples, metadata)
        if trainer.is_global_zero:
            all_samples = all_samples.permute(0, 2, 3, 1).cpu().numpy()
            self.samples.append(all_samples)

    # revised, supports metric computing after generating npz
    def save_end(self):
        if self.compressed and len(self.samples) > 0:
            samples = numpy.concatenate(self.samples)
            sample_path = f'{self.target_dir}/output.npz'
            numpy.savez(sample_path, arr_0=samples)

            if torch.distributed.is_available():
                is_main = torch.distributed.get_rank() == 0
            else:
                is_main = True

            if is_main:
                # rank0
                rank_zero_info("Starting evaluation...")

                ref_path = path_to_your_npz       # need to be revised

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                evaluator = Evaluator(device)

                ref_acts = evaluator.read_activations(ref_path)
                ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_path, ref_acts)

                sample_acts = evaluator.read_activations(sample_path)
                sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_path, sample_acts)

                is_score = evaluator.compute_inception_score(
                    sample_acts[0], split_size=sample_acts[0].shape[0]
                )
                fid = sample_stats.frechet_distance(ref_stats)
                sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
                prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])


                print("========== Evaluation Results ==========")
                print("Inception Score:", is_score)
                print("FID:", fid)
                print("sFID:", sfid)
                print("Precision:", prec)
                print("Recall:", recall)
                print("========================================")

                # ========= 保存结果 =========
                metrics = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_samples": int(sample_acts[0].shape[0]),
                    "Inception Score": float(is_score),
                    "FID": float(fid),
                    "sFID": float(sfid),
                    "Precision": float(prec),
                    "Recall": float(recall),
                }

                # 保存 txt
                txt_path = os.path.join(self.target_dir, "metrics.txt")
                with open(txt_path, "a") as f:
                    f.write("========== Evaluation Results ==========\n")
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")
                    f.write("========================================\n\n")

                # 保存 json
                json_path = os.path.join(self.target_dir, "metrics.json")
                with open(json_path, "w") as f:
                    json.dump(metrics, f, indent=4)

                rank_zero_info(f"Metrics saved to {self.target_dir}")

        self.executor_pool.shutdown(wait=True)
        self.samples = []
        self.target_dir = None
        self._have_saved_num = 0
        self.executor_pool = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, f"iter_{trainer.global_step}")
        self.save_start(target_dir)

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, outputs, batch)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        target_dir = os.path.join(trainer.default_root_dir, self.save_dir, "predict")
        self.save_start(target_dir)

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        samples: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.process_batch(trainer, pl_module, samples, batch)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.save_end()