import torch
import torch.nn as nn
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import copy

class DINOv2(nn.Module):
    def __init__(self, weight_path:str):
        super(DINOv2, self).__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', weight_path)
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.precomputed_pos_embed = dict()

    def fetch_pos(self, h, w):
        key = (h, w)
        if key in self.precomputed_pos_embed:
            return self.precomputed_pos_embed[key]
        value = timm.layers.pos_embed.resample_abs_pos_embed(
            self.pos_embed.data, [h, w],
        )
        self.precomputed_pos_embed[key] = value
        return value

    def forward(self, x):
        b, c, h, w = x.shape
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (int(224*h/256), int(224*w/256)), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        pos_embed_data = self.fetch_pos(patch_num_h, patch_num_w)
        self.encoder.pos_embed.data = pos_embed_data
        feature = self.encoder.forward_features(x)['x_norm_patchtokens']
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        return feature

class MAE(nn.Module):
    def __init__(self, model_id, weight_path:str):
        super(MAE, self).__init__()
        if os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, "pytorch_model.bin")
        self.encoder = timm.create_model(
            model_id,
            checkpoint_path=weight_path,
            num_classes=0,
        )
        self.pos_embed = copy.deepcopy(self.encoder.pos_embed)
        self.encoder.head = torch.nn.Identity()
        self.patch_size = self.encoder.patch_embed.patch_size
        self.shifts = nn.Parameter(torch.tensor([0.0
        ]), requires_grad=False)
        self.scales = nn.Parameter(torch.tensor([1.0
        ]), requires_grad=False)

    def forward(self, x):
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic')
        b, c, h, w = x.shape
        patch_num_h, patch_num_w = h//self.patch_size[0], w//self.patch_size[1]
        feature = self.encoder.forward_features(x)[:, self.encoder.num_prefix_tokens:]
        feature = feature.transpose(1, 2)
        feature = feature.view(b, -1, patch_num_h, patch_num_w).contiguous()
        feature = (feature - self.shifts.view(1, -1, 1, 1)) / self.scales.view(1, -1, 1, 1)
        return feature


from diffusers import AutoencoderKL

from torchvision.datasets import ImageFolder, ImageNet

import cv2
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from PIL import Image
import torch
import torchvision.transforms as tvtf

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
    torch_hub_dir = '/mnt/bn/wangshuai6/torch_hub'
    os.environ["TORCH_HOME"] = torch_hub_dir
    torch.hub.set_dir(torch_hub_dir)

    for split in ['train']:
        train = split == 'train'
        transforms = tvtf.Compose([
            CenterCrop(256),
            tvtf.ToTensor(),
            # tvtf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = NewImageFolder(root='/mnt/bn/wangshuai6/data/ImageNet/train', transform=transforms,)
        B = 4096
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True, prefetch_factor=None, num_workers=0)
        dino = DINOv2("dinov2_vitb14")
        # dino = MAE("vit_base_patch16_224.mae", "/mnt/bn/wangshuai6/models/vit_base_patch16_224.mae")
        # dino = CLIP("/mnt/bn/wangshuai6/models/vit_base_patch16_clip_224.openai")
        from accelerate import Accelerator

        accelerator = Accelerator()
        dino, dataloader = accelerator.prepare(dino, dataloader)
        rank = accelerator.process_index

        acc_mean = torch.zeros((768, ), device=accelerator.device)
        acc_num = 0
        with torch.no_grad():
            for i, (images, labels, path_list) in enumerate(dataloader):
                acc_num += len(images)
                feature = dino(images)
                stds = torch.std(feature, dim=[0, 2, 3]).tolist()
                for std in stds:
                    print("{:.2f},".format(std), end='')
                print()
                means = torch.mean(feature, dim=[0, 2, 3]).tolist()
                for mean in means:
                    print("{:.2f},".format(mean), end='')
                break

        accelerator.wait_for_everyone()