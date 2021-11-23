from glob import glob
import os
import pathlib
import random

import cv2
import numpy as np
import imgaug
from imgaug import augmenters as iaa
import torchvision
from torchvision import transforms


class RandomVerticalLine():

    def __call__(self, img: np.array):
        i = random.randint(0, 28)
        img[:, i-1:i+2] = 0
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

PWD = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":

    folder = "/home/ych/tmp/mnist/test"
    data = list()
    for label in range(10):
        paths = glob(os.path.join(folder, str(label), "*.jpg"))
        for path in paths:
            img = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)  # BGR
            data.append((img, label))

    tfs1 = transforms.Compose([
         transforms.ToTensor(),
    ])

    tfs2 = transforms.Compose([
        iaa.Affine(rotate=(-45, 45)).augment_image,
        iaa.Affine(translate_percent={"x": (-0.3, 0.3)}).augment_image,
        iaa.MultiplyBrightness(mul=(0.65, 1.35)).augment_image,
        iaa.AddToHueAndSaturation(value_hue=(-50, 50), value_saturation=(-200, 200)).augment_image,
        iaa.GaussianBlur(sigma=(0.0, 0.5)).augment_image,
        iaa.AdditiveGaussianNoise(scale=0.05*255).augment_image,
        iaa.Salt(0.1).augment_image,
        iaa.Pepper(0.1).augment_image,
        RandomVerticalLine(),
        # iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1)).augment_image,
        # transforms.ToPILImage(),
        # transforms.ColorJitter(),
        # transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.RandomErasing(p=1.0, scale=(0.02, 0.12), ratio=(0.1, 0.11), value=0),
        # transforms.RandomErasing(p=1.0, scale=(0.02, 0.12), ratio=(10.0, 10.01), value=0),
    ])

    output = list()
    s = 2000
    for i in range(s, s+5):
        output.append(tfs1(data[i][0]))
    for i in range(s, s+5):
        output.append(tfs2(data[i][0]))

    img_saved = torchvision.utils.make_grid(output, nrow=5, padding=1, pad_value=1)
    img_saved = img_saved.permute(1,2,0).numpy() * 255
    img_saved = img_saved.astype(np.uint8)

    # torchvision.utils.save_image(tensor=img_saved, fp="./output.png")
    cv2.imwrite(filename=os.path.join(PWD, "output.png"), img=img_saved)
