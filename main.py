import os
import torch

import albumentations
import pretrainedmodels

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch.nn import functional as F

from utils.data.dataset import Dataset
from utils.early_stopping import EarlyStopping

from utils.engine import Engine

from sklearn import metrics

class SEResNext50_32x4d(nn.Module):
    """
        SEResNext50_32x4d pretrained model.
    """

    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):

        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))

        return out, loss


def train(fold):
    training_data_path = "data_storage/working/train/"
    model_path = f"data_storage/model_{fold}.bin"
    df = pd.read_csv("data_storage/train_folds.csv")
    device = "cuda"
    epochs = 10
    train_bs = 16 # accumulation step: 2 -> simulates batch_size: 32
    valid_bs = 8 # accumulation step: 2 -> simulates batch_size: 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = Dataset(
        image_paths=train_images,
        targets=train_targets, 
        resize=None,
        augmentations=train_aug)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = Dataset(
        image_paths=valid_images,
        targets=valid_targets, 
        resize=None,
        augmentations=valid_aug
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )


    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        patience=3,
        mode="max",
    )

    es = EarlyStopping(patience=5, mode="max")

    engine = Engine(model=model, optimizer=optimizer, device=device,accumulation_steps=2)
    for epoch in range(epochs):
        training_loss = engine.train(train_loader)
        
        valid_loss, predictions = engine.evaluate(valid_loader)

        try:
            predictions = np.vstack((predictions)).ravel()
        except Exception as e:
            print(e)
            print('Predictions: \n', predictions)

        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"Epoch: {epoch}, auc: {auc}")

        es(auc, model, model_path)
        if es.early_stop:
            print("Early Stopping ...")
            break


def predict(fold):
    test_data_path = "data_storage/working/test/"
    model_path = f"data_storage/model_{fold}.bin"
    df = pd.read_csv("data_storage/test.csv")
    df.loc[:, "target"] = 0

    device = "cuda"
    epochs = 50
    test_bs = 4 # accumulation step: 4 -> simulates batch_size: 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply=True),
        ]
    )

    test_images = df.image_name.values.tolist()
    test_images = [os.path.join(training_data_path, i + ".jpg") for i in test_images]
    test_targets = df.target.values

    test_dataset = Dataset(
        image_paths=test_images,
        targets=test_targets, 
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=4
    )


    model = SEResNext50_32x4d(pretrained="imagenet")
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    engine = Engine(model=model, optimizer=optimizer, device=device, accumulation_steps=4)

    predictions = engine.predict(test_loader)

    return np.vstack((predictions)).ravel()
        
if __name__ == "__main__":
    train(fold=0)
    #predict(fold=0)