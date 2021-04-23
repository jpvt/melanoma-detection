import cv2
import torch
import numpy as np

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataset import Dataset

class DataLoader:

    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations=None,
        backend="pil",
        channel_first=True,
    ):

        """
        :param image_paths: list of paths to images
        :param targets: numpy array that represents the images classes
        :param resize: tuple or None
        :param backend: string "pil" or "cv2"
        :param augmentations: albumentations augs
        """

        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations
        self.backend = backend
        self.channel_first = channel_first

        self.dataset = Dataset(
            image_paths=self.image_paths, 
            targets=self.targets, 
            resize= self.resize,
            augmentations=self.augmentations,
            backend=self.backend,
            channel_first=self.channel_first,
        )

    def fetch(
        self,
        batch_size,
        num_workers,
        drop_last=False,
        shuffle=True,
        sampler=None,
    ):

        """
        :param batch_size: int batch size
        :param num_workers: int number of processes
        :param drop_last: bool drop the last batch?
        :param shuffle: bool
        """

        data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return data_loader
