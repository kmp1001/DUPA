import time

import torch
import torch.nn as nn

class BaseTrainer(nn.Module):
    def __init__(self,
                 null_condition_p=0.1,
                 log_var=False,
        ):
        super(BaseTrainer, self).__init__()
        self.null_condition_p = null_condition_p
        self.log_var = log_var

    def preproprocess(self, raw_iamges, x, condition, uncondition):
        # Replace the conditional embeddings of a subset of samples with unconditional embeddings using probability null_condition_p for training the CFG.
        # raw_images: (B,3,H,W)
        # x: (B,3,H,W)
        bsz = x.shape[0]
        if self.null_condition_p > 0:
            # mask.shape = (B,)
            mask = torch.rand((bsz), device=condition.device) < self.null_condition_p
            mask = mask.expand_as(condition)
            # For the selected samples, directly replace their conditional embeddings with unconditional embeddings.
            condition[mask] = uncondition[mask]
        return raw_iamges, x, condition

    def _impl_trainstep(self, net, ema_net, raw_images, x, y, step):
        raise NotImplementedError

    def __call__(self, net, ema_net, raw_images, x, condition, uncondition, step):
        # raw_images, x, condition = self.preproprocess(raw_images, x, condition, uncondition)
        return self._impl_trainstep(net, ema_net, raw_images, x, condition, step)

