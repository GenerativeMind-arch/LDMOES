import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from ldm.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from ldm.data.helper_types import Image
from ldm.modules.image_degradation.bsrgan import degradation_bsrgan


class FacesBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.keys = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        ex = {}
        if self.keys is not None:
            for k in self.keys:
                ex[k] = example[k]
        else:
            ex = example
        return ex

augmentations = A.Compose([
    A.RandomCrop(width=128, height=128),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(256, 256),
    ToTensorV2()
])

class CelebAHQTrain(FacesBase):
    def __init__(self, size=None, keys=None, degradation=None,config=None):
        self.size = size
        self.config = config
        self.degradation = degradation
        super().__init__()
        root = "/data/LDMOES/PixelSpace/stable_diffusion/data/celebahq"
        with open("/data/LDMOES/PixelSpace/stable_diffusion/data/celebahq/celebahqtrain.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]

        self.data = ImagePaths(paths=paths, size=size, random_crop=False,augmentations=augmentations)
        self.keys = keys


class CelebAHQValidation(FacesBase):
    def __init__(self, size=None, keys=None,degradation=None,config=None):
        self.degradation = degradation
        self.size = size
        self.config = config
        super().__init__()
        root = "/data/LDMOES/PixelSpace/stable_diffusion/data/celebahq"
        with open("/data/LDMOES/PixelSpace/stable_diffusion/data/celebahq/celebahqvalidation.txt", "r") as f:
            relpaths = f.read().splitlines()
        paths = [os.path.join(root, relpath) for relpath in relpaths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False,augmentations=augmentations)
        self.keys = keys
