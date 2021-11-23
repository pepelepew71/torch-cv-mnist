"""
Setup MNIST dataset for Torch
"""

from glob import glob
import os

import cv2
import torch


class DataSetMnist(torch.utils.data.Dataset):

    def __init__(self, folder: str, transform=None):
        self.folder = folder
        self.transform = transform
        self.data = list()
        self._setup()

    def _setup(self):
        assert os.path.exists(path=self.folder), print(f'{self.folder} is not exists')
        for label in range(10):
            paths = glob(os.path.join(self.folder, str(label), "*.jpg"))
            for path in paths:
                img = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)  # BGR
                self.data.append((img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


if __name__ == "__main__":
    folder = "/home/ych/tmp/mnist/train"
    dataset_mnist = DataSetMnist(folder=folder)
    print()
