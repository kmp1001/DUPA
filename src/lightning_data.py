from typing import Any
import torch
import copy
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from src.data.dataset.randn import RandomNDataset
from src.data.var_training import VARTransformEngine

def collate_fn(batch):
    new_batch = copy.deepcopy(batch)
    new_batch = list(zip(*new_batch))
    for i in range(len(new_batch)):
        if isinstance(new_batch[i][0], torch.Tensor):
            try:
                new_batch[i] = torch.stack(new_batch[i], dim=0)
            except:
                print("Warning: could not stack tensors")
    return new_batch

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_root,
                 test_nature_root,
                 test_gen_root,
                train_image_size=64,
                train_batch_size=64,
                train_num_workers=8,
                var_transform_engine: VARTransformEngine = None,
                train_prefetch_factor=2,
                train_dataset: str = None,
                eval_batch_size=32,
                eval_num_workers=4,
                eval_max_num_instances=50000,
                pred_batch_size=32,
                pred_num_workers=4,
                pred_seeds:str=None,
                pred_selected_classes=None,
                num_classes=1000,
                latent_shape=(4,64,64),
    ):
        super().__init__()
        pred_seeds = list(map(lambda x: int(x), pred_seeds.strip().split(","))) if pred_seeds is not None else None

        self.train_root = train_root
        self.train_image_size = train_image_size
        self.train_dataset = train_dataset
        # stupid data_convert override, just to make nebular happy
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.train_prefetch_factor = train_prefetch_factor

        self.test_nature_root = test_nature_root
        self.test_gen_root = test_gen_root
        self.eval_max_num_instances = eval_max_num_instances
        self.pred_seeds = pred_seeds
        self.num_classes = num_classes
        self.latent_shape = latent_shape

        self.eval_batch_size = eval_batch_size
        self.pred_batch_size = pred_batch_size

        self.pred_num_workers = pred_num_workers
        self.eval_num_workers = eval_num_workers

        self.pred_selected_classes = pred_selected_classes

        self._train_dataloader = None
        self.var_transform_engine = var_transform_engine

    def setup(self, stage: str) -> None:
        if stage == "fit":
           assert self.train_dataset is not None
           if self.train_dataset == "pix_imagenet64":
               from src.data.dataset.imagenet import PixImageNet64
               self.train_dataset = PixImageNet64(
                   root=self.train_root,
               )
           elif self.train_dataset == "pix_imagenet128":
               from src.data.dataset.imagenet import PixImageNet128
               self.train_dataset = PixImageNet128(
                   root=self.train_root,
               )
           elif self.train_dataset == "imagenet256":
               from src.data.dataset.imagenet import ImageNet256
               self.train_dataset = ImageNet256(
                   root=self.train_root,
               )
           elif self.train_dataset == "pix_imagenet256":
               from src.data.dataset.imagenet import PixImageNet256
               self.train_dataset = PixImageNet256(
                   root=self.train_root,
               )
           elif self.train_dataset == "imagenet512":
               from src.data.dataset.imagenet import ImageNet512
               self.train_dataset = ImageNet512(
                   root=self.train_root,
               )
           elif self.train_dataset == "pix_imagenet512":
               from src.data.dataset.imagenet import PixImageNet512
               self.train_dataset = PixImageNet512(
                   root=self.train_root,
               )
           else:
             raise NotImplementedError("no such dataset")

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.var_transform_engine and self.trainer.training:
            batch = self.var_transform_engine(batch)
        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        self._train_dataloader = DataLoader(
            self.train_dataset,
            self.train_batch_size,
            timeout=6000,
            num_workers=self.train_num_workers,
            prefetch_factor=self.train_prefetch_factor,
            sampler=sampler,
            collate_fn=collate_fn,
        )
        return self._train_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.eval_dataset = RandomNDataset(
            latent_shape=self.latent_shape,
            num_classes=self.num_classes,
            max_num_instances=self.eval_max_num_instances,
        )
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.eval_dataset, self.eval_batch_size,
                          num_workers=self.eval_num_workers,
                          prefetch_factor=2,
                          collate_fn=collate_fn,
                          sampler=sampler
                )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        global_rank = self.trainer.global_rank
        world_size = self.trainer.world_size
        self.pred_dataset = RandomNDataset(
            seeds= self.pred_seeds,
            max_num_instances=50000,
            num_classes=self.num_classes,
            selected_classes=self.pred_selected_classes,
            latent_shape=self.latent_shape,
        )
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(self.pred_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
        return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size,
                          num_workers=self.pred_num_workers,
                          prefetch_factor=4,
                          collate_fn=collate_fn,
                          sampler=sampler
               )
