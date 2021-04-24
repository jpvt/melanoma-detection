import os
from flask import Flask, render_template, request

import torch

import albumentations
import pretrainedmodels
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
from utils.engine import Engine
from utils.data.dataset import Dataset


app = Flask(__name__)
UPLOAD_FOLDER = "/home/detector/app/static/"
DEVICE = "cpu"
MODEL = None
MODEL_PATH = "model.bin"

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
        out = F.sigmoid(self.out(x))
        loss = 0
        return out, loss

def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value = 255.0, always_apply=True),
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = Dataset(
        image_paths=test_images,
        targets=test_targets, 
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )


    engine = Engine(model=model, optimizer=None, device=DEVICE)

    predictions = engine.predict(test_loader)

    return np.vstack((predictions)).ravel()


@app.route('/', methods = ['GET', 'POST'])
def upload_predict():
    
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)
            pred = predict(image_path, MODEL)[0]
            print(pred)
            return render_template('index.html', prediction=round(pred, 3), image_loc = image_path)

    return render_template('index.html', prediction=0, image_loc = None)


if __name__ == '__main__':

    MODEL = SEResNext50_32x4d(pretrained=None)
    MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)


    app.run(host="0.0.0.0", port=12000, debug=True)