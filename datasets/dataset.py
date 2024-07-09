import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
import random
import torch
import torchvision.transforms.functional as F

TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def get_transforms(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class CelebAHQDataset(Dataset):
    def __init__(self, dataset_root, mode="test",
                 img_transform=TO_TENSOR, fraction=1.0,
                 flip_p=-1):  # negative number for no flipping

        self.mode = mode
        self.root = dataset_root
        self.img_transform = img_transform
        self.fraction = fraction
        self.flip_p = flip_p

        if mode == "train":
            self.imgs = sorted([os.path.join(self.root, "CelebA-HQ-img", "%d.jpg" % idx) for idx in range(28000)])
        else:
            self.imgs = sorted(
                [os.path.join(self.root, "CelebA-HQ-img", "%d.jpg" % idx) for idx in range(28000, 30000)])
        self.imgs = self.imgs[:int(len(self.imgs) * self.fraction)]

        # image pairs indices
        self.indices = np.arange(len(self.imgs))

    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, inlcuding
            - the image,

        Args:
            index (int): index of the sample
        Return:
            img: RGB image

        """
        img1 = self.imgs[index]
        img2 = self.imgs[index + 1] if index != len(self.imgs) - 1 else self.imgs[index - 1]
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        if self.img_transform is not None:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
        return img1, img2

    def __getitem__(self, idx):
        index = self.indices[idx]

        img1, img2 = self.load_single_image(index)

        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img1 = F.hflip(img1)
                img2 = F.hflip(img2)
        return img1, img2


class FFHQDataset(Dataset):
    def __init__(self, dataset_root,
                 img_transform=TO_TENSOR,
                 fraction=1.0,
                 flip_p=-1):

        self.root = dataset_root
        self.img_transform = img_transform
        self.fraction = fraction
        self.flip_p = flip_p

        self.imgs = sorted([os.path.join(self.root, f"{idx:05d}.png") for idx in range(70000)])
        self.imgs = self.imgs[:int(len(self.imgs) * self.fraction)]

        self.indices = np.arange(len(self.imgs))

    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, including
            - the image,

        Args:
            index (int): index of the sample
        Return:
            img: RGB image
        """
        img1 = self.imgs[index]
        img2 = self.imgs[index + 1] if index != len(self.imgs) - 1 else self.imgs[index - 1]
        img1 = Image.open(img1).convert('RGB')
        img2 = Image.open(img2).convert('RGB')
        if self.img_transform is not None:
            img1 = self.img_transform(img1)
            img2 = self.img_transform(img2)
        return img1, img2

    def __getitem__(self, idx):
        index = self.indices[idx]

        img1, img2 = self.load_single_image(index)

        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img1 = F.hflip(img1)
                img2 = F.hflip(img2)

        return img1, img2
