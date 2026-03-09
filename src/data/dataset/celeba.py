from typing import Callable
from torchvision.datasets import CelebA


class LocalDataset(CelebA):
    def __init__(self, root:str,  ):
        super(LocalDataset, self).__init__(root, "train")

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data