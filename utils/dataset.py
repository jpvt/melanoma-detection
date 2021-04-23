import cv2
import torch
import numpy as np

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset:

    def __init__(
        self,
        image_paths,
        targets,
        resize,
        augmentations = None,
        backend = "pil",
        channel_first=True
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


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        targets = self.targets[idx]

        if self.backend == "pil":
            image = Image.open(self.image_paths[idx])

            if self.resize is not None:
                image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)
            
            image = np.array(image)

            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]

        elif self.backend == "cv2":
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.resize is not None:
                image = cv2.resize(image, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_CUBIC,)

            if self.augmentations is not None:
                augmented = self.augmentations(image=image)
                image = augmented["image"]

        else:
            raise Exception("Backend not implemented")

        if self.channel_first:
            image = np.tranpose(image, (2, 0, 1)).astype(np.float32)

        
        return {
            "image": torch.tensor(image),
            "targets": torch.tensor(targets),
        }


