import torch
import os 
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm





def resize_church():
    path = 'exp_church_base/datasets/lsun/church_imgs'
    save_dir = 'exp_church_base/datasets/lsun/church_imgs_resized'
    transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                    ]
                )
    for file in tqdm(os.listdir(path)):
        img_path = os.path.join(path,file)
        img = Image.open(img_path)
        img = transform(img)
        img.save(os.path.join(save_dir,file))

if __name__ == '__main__':
    resize_church()


