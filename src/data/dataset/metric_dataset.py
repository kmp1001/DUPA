import pathlib

import torch
import random
import numpy as np
from torchvision.io.image import read_image
import torchvision.transforms as tvtf
from torch.utils.data import Dataset

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


from PIL import Image
IMG_EXTENSIONS = (
    "*.png",
    "*.JPEG",
    "*.jpeg",
    "*.jpg"
)

def test_collate(batch):
    return torch.stack(batch)

class ImageDataset(Dataset):
    def __init__(self, root, image_size=(224, 224)):
        self.root = pathlib.Path(root)
        images =  []
        for ext in IMG_EXTENSIONS:
            images.extend(self.root.rglob(ext))
        random.shuffle(images)
        self.images = list(map(lambda x: str(x), images))
        self.transform = tvtf.Compose(
            [
                CenterCrop(image_size[0]),
                tvtf.ToTensor(),
                tvtf.Lambda(lambda x: (x*255).to(torch.uint8)),
                tvtf.Lambda(lambda x: x.expand(3, -1, -1))
            ]
        )
        self.size = image_size

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx])
            image = self.transform(image)
        except Exception as e:
            print(self.images[idx])
            image = torch.zeros(3, self.size[0], self.size[1], dtype=torch.uint8)

        # print(image)
        metadata = dict(
            path = self.images[idx],
            root = self.root,
        )
        return image #, metadata

    def __len__(self):
        return len(self.images)