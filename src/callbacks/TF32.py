import torch
from lightning.pytorch import Callback

class TF32Callback(Callback):
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def setup(self, trainer, pl_module, stage=None):
        if self.enabled and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True