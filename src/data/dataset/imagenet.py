import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize
import os
from src.data.dataset.metric_dataset import CenterCrop

class LocalCachedDataset(ImageFolder):
    def __init__(self, root, resolution=256):
        super().__init__(root)
        self.transform = CenterCrop(resolution)
        self.cache_root = None

    def load_latent(self, latent_path):
        pk_data = torch.load(latent_path)
        mean = pk_data['mean'].to(torch.float32)
        logvar = pk_data['logvar'].to(torch.float32)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(mean) * std
        return latent

# revised
# The old implementation accesses the exact indexed sample and expects its latent cache to be present.
# The new implementation introduces a fallback mechanism: if the latent file is missing, it searches forward for the next valid sample and returns that one instead.
# it is useful in the practical cases

    def __getitem__(self, idx: int):
        while True:
            image_path, target = self.samples[idx]
            latent_path = image_path.replace(self.root, self.cache_root) + ".pt"

            if self.cache_root is not None and not os.path.exists(latent_path):
                # jump to next
                idx = (idx + 1) % len(self.samples)
                continue

            # latent exists
            raw_image = Image.open(image_path).convert('RGB')
            raw_image = self.transform(raw_image)
            raw_image = to_tensor(raw_image)

            if self.cache_root is not None:
                latent = self.load_latent(latent_path)
            else:
                latent = raw_image

            return raw_image, latent, target


class ImageNet256(LocalCachedDataset):
    def __init__(self, root, ):
        super().__init__(root, 256)
        self.cache_root = root + "_256_latent"

class ImageNet512(LocalCachedDataset):
    def __init__(self, root, ):
        super().__init__(root, 512)
        self.cache_root = root + "_512_latent"

class PixImageNet(ImageFolder):
    def __init__(self, root, resolution=256):
        super().__init__(root)
        self.transform = CenterCrop(resolution)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)

        normalized_image = self.normalize(raw_image)
        return raw_image, normalized_image, target

class PixImageNet64(PixImageNet):
    def __init__(self, root, ):
        super().__init__(root, 64)

class PixImageNet128(PixImageNet):
    def __init__(self, root, ):
        super().__init__(root, 128)


class PixImageNet256(PixImageNet):
    def __init__(self, root, ):
        super().__init__(root, 256)

class PixImageNet512(PixImageNet):
    def __init__(self, root, ):
        super().__init__(root, 512)





