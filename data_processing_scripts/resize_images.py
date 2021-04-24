import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(image_path, output_folder, resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample = Image.BILINEAR)

    img.save(outpath)


input_folder = "../data_storage/jpeg/train/"
output_folder = "../data_storage/working/train/"

images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(8)(
    delayed(resize_image)(
        i, 
        output_folder, 
        (224, 224)
    ) for i in tqdm(images)
)

input_folder = "../data_storage/jpeg/test/"
output_folder = "../data_storage/working/test/"

images = glob.glob(os.path.join(input_folder, "*.jpg"))
Parallel(8)(
    delayed(resize_image)(
        i, 
        output_folder, 
        (224, 224)
    ) for i in tqdm(images)
)