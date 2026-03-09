from diffusers import AutoencoderKL

import torch
from typing import Callable
from torchvision.datasets import ImageFolder, ImageNet

import cv2
import os
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from PIL import Image
import pathlib

import torch
import random
from torchvision.io.image import read_image
import torchvision.transforms as tvtf
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet

IMG_EXTENSIONS = (
    "*.png",
    "*.JPEG",
    "*.jpeg",
    "*.jpg"
)

class NewImageFolder(ImageFolder):
    def __getitem__(self, item):
        path, target = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


import time
class CenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        def center_crop_arr(pil_image, image_size):
            """
            Center cropping implementation from ADM.
            https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
            """
            while min(*pil_image.size) >= 2 * image_size:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = image_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            arr = np.array(pil_image)
            crop_y = (arr.shape[0] - image_size) // 2
            crop_x = (arr.shape[1] - image_size) // 2
            return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

        return center_crop_arr(image, self.size)

def save(obj, path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(obj, f'{path}')

if __name__ == "__main__":
    writer_pool = ThreadPoolExecutor(8)
    for split in ['train']:
        train = split == 'train'
        transforms = tvtf.Compose([
            CenterCrop(512),
            # tvtf.RandomHorizontalFlip(p=1),
            tvtf.ToTensor(),
            tvtf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = NewImageFolder(root='/mnt/bn/wangshuai6/data/ImageNet/train', transform=transforms,)
        B = 8
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=False, prefetch_factor=16, num_workers=16)
        vae = AutoencoderKL.from_pretrained("/mnt/bn/wangshuai6/models/sd-vae-ft-ema")#.to('cuda')
        vae = vae.to(torch.float16)
        from accelerate import Accelerator

        accelerator = Accelerator()

        vae, dataloader = accelerator.prepare(vae, dataloader)
        rank = accelerator.process_index
        with torch.no_grad():
            for i, (image, label, path_list) in enumerate(dataloader):
                 print(i/len(dataloader))
                 flag = False
                 new_path_list = []
                 for p in path_list:
                     p = p + ".pt"
                     p = p.replace("/mnt/bn/wangshuai6/data/ImageNet/train",
                                   "/mnt/bn/wangshuai6/data/ImageNet/train_512_latent")
                     new_path_list.append(p)
                     if not os.path.exists(p):
                         print(p)
                         flag = True

                 if flag:
                     image = image.to("cuda")
                     image = image.to(torch.float16)
                     distribution = vae.module.encode(image).latent_dist
                     mean = distribution.mean
                     logvar = distribution.logvar

                     for j in range(len(path_list)):
                         out = dict(
                             mean=mean[j].cpu(),
                             logvar=logvar[j].cpu(),
                         )
                         writer_pool.submit(save, out, new_path_list[j])
        writer_pool.shutdown(wait=True)
        accelerator.wait_for_everyone()