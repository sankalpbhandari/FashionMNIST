import struct

import cv2
import os
import pandas as pd
import numpy as np


def loadlocal_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def generate_data(datatype="train"):
    data = []
    images = "dataset/" + datatype + "-images-idx3-ubyte"
    labels = "dataset/" + datatype + "-labels-idx1-ubyte"
    x, y = loadlocal_mnist(images, labels)
    path = "dataset/" + datatype + "/"
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(x.shape[0]):
        im = np.reshape(x[i], (28, 28))
        cv2.imwrite(path + str(i) + ".png", im)
        data.append([path + str(i) + ".png", y[i]])
    filename = datatype + ".csv"
    pd.DataFrame(data, columns=['Image', 'Label']).to_csv(filename, index=False)


generate_data("train")
generate_data("t10k")
