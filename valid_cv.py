from glob import glob
import os
import pathlib
from random import randint

import cv2
import numpy as np

PWD = str(pathlib.Path(__file__).parent.absolute())

if __name__ == "__main__":

    folder = "/home/ych/tmp/mnist/test"
    data = list()
    for label in range(10):
        paths = glob(os.path.join(folder, str(label), "*.jpg"))
        for path in paths:
            img = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)  # BGR
            data.append((img, label))

    net = cv2.dnn.readNetFromONNX(os.path.join(PWD, "net_aug.onnx"))

    for _ in range(5):
        img, label = data[randint(0, 10000)]
        img = img.astype(np.float32)

        scale = 1.0/255.0
        # mean = np.array([0.5, 0.5, 0.5])
        # std = np.array([0.5, 0.5, 0.5])

        img *= scale
        # img -= mean
        # img /= std

        input_blob = cv2.dnn.blobFromImage(
            image=img,
            size=(28,28),
            swapRB=False,  # BGR -> RGB
        )

        net.setInput(input_blob)
        out = net.forward()
        print([f'{i:.3e}' for i in out[0].tolist()], f", Pred: {np.argmax(out):d}, Label: {label:d}")
