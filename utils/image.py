import os
import cv2 as cv
import numpy as np
from torchvision import transforms

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path, target_shape=None):
    '''
    Load and resize the image.
    '''
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'Path not found: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB when reading
    if target_shape:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    return img

def prepare_img(img_path, target_shape, device):
   '''
   Normalize the image.
   '''
   img = load_image(img_path, target_shape)
   transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Lambda(lambda x: x.mul(255)),
       transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])
   img = transform(img).to(device).unsqueeze(0)
   return img

def save_image(img, img_path):
   if len(img.shape) == 2:
       img = np.stack((img,) * 3, axis=-1)
   cv.imwrite(img_path, img[:, :, ::-1])  # Convert RGB to BGR while writing