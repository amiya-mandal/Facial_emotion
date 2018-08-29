from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import torch


class FaceDataLoader(Dataset):

    def __init__(self):
        self.filedata = os.listdir("process")
        self.lenval = len(self.filedata)

    def  _encode(self, val: int):
        if val == 0:
            return np.array([1,0,0,0,0,0,0])
        elif val == 1:
            return np.array([0,1,0,0,0,0,0])
        elif val == 2:
            return np.array([0,0,1,0,0,0,0])
        elif val == 3:
            return np.array([0,0,0,1,0,0,0])
        elif val == 4:
            return np.array([0,0,0,0,1,0,0])
        elif val == 5:
            return np.array([0,0,0,0,0,1,0])
        elif val == 6:
            return np.array([0,0,0,0,0,0,1])
    def __nameparser(self, strval: str):
        nameval = strval.split("_")
        nameval = int(nameval[0])
        return self._encode(nameval)

    def __getitem__(self, item):
        filename = self.filedata[item]
        path = os.path.join("process", filename)

        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        image = np.reshape(image,(1,48,48))
        img_data = image.astype('float32')
        image = torch.from_numpy(img_data)
        lable = torch.from_numpy(self.__nameparser(filename))

        return image, lable

    def __len__(self):
        return self.lenval

