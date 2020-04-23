from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


class OIDdataset(object):
    """docstring for OIDdataset"""
    def __init__(self, root, transform, classes):
        super(OIDdataset, self).__init__()
        self.root = Path(root)
        self.transform = transform
        self.classes = classes

        self.labels = []

        # label format
        '''
        _______________ x
        |
        |    o___
        |    |   |
        |    |   |
        |    |   |
        |    ----o
        y
        '''
        # class, left, top, right, bottom

        self.images = list(self.root.glob("*/*.jpg"))

        for image in self.images:
            filename = image.parent / "Label" / Path(str(image.stem) + ".txt")
            self.labels.append(filename)

        assert len(self.images) == len(self.labels)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        ymax, xmax, channels = image.shape[0], image.shape[1], image.shape[2]
        if channels < 3:
            image = np.stack((image, image, image), axis=2)

        # image = image.transpose((2, 0, 1))
        # image = torch.as_tensor(image, dtype=torch.float32)

        target = {}

        with open(self.labels[idx], "r") as fin:
            lines = fin.readlines()
            if len(lines) > 1:
                boxes = []
                labels = []
                areas = []
                for line in lines:
                    line = line.split()
                    className = line[0]
                    label = self.classes.index(className)
                    label = torch.as_tensor([label], dtype=torch.int64)
                    left = float(line[1])
                    top = float(line[2])
                    right = float(line[3])
                    bottom = float(line[4])
                    bbox = [left, top, right, bottom]
                    area = np.abs(right - left) * np.abs(bottom - top)
                    areas.append(area)
                    boxes.append(bbox)
                    labels.append(label)
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
                target["area"] = torch.as_tensor(areas, dtype=torch.float32)

            else:
                lines = lines[0].split()
                className = lines[0]
                label = self.classes.index(className)
                label = torch.as_tensor([label], dtype=torch.int64)
                left = float(lines[1])
                top = float(lines[2])
                right = float(lines[3])
                bottom = float(lines[4])
                bbox = [left, top, right, bottom]
                area = np.abs(right - left) * np.abs(bottom - top)
                bbox = torch.as_tensor([bbox], dtype=torch.float32)
                target["boxes"] = torch.as_tensor(bbox, dtype=torch.float32)
                target["labels"] = torch.as_tensor(label, dtype=torch.int64)
                target["area"] = torch.as_tensor([area], dtype=torch.float32)

        target["iscrowd"] = torch.as_tensor(0, dtype=torch.int64)
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def __len__(self):
        return len(self.labels)
