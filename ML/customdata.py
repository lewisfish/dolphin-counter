from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

__all__ = ["OIDdataset", "DolphinDataset"]


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
                    label = self.classes.index(className)+1
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
                label = self.classes.index(className)+1
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

# 錄製_2019_11_20_10_13_54_900.mp4, 43472, 229, 885, 235, 896, 10
# 錄製_2019_11_29_16_14_59_189.mp4, 31677, 127, 606, 141, 658, 10
# 錄製_2019_11_20_10_13_54_900.mp4, 44691, 88, 1325, 93, 1332, 10
# 錄製_2019_11_28_12_05_07_124.mp4, 30440, 749, 550, 758, 556, 10

# filename, framenumber, x0, y0, x1, y1, label
# cavet is that y1 and y2 offset by 130 due to cropping of screen recording
# this wont be true for all data after deploy though
# just true of the train, test, validation sets.


class DolphinDataset(object):
    """docstring for DolphinDataset"""
    def __init__(self, root, transforms, file):
        super(DolphinDataset, self).__init__()
        self.root = Path(root)
        self.transforms = transforms
        self.datafile = file
        self.videoFiles = list(self.root.glob("**/*.mp4"))

        self.labels = []
        self.frameNumbers = []
        self.bboxs = []
        self.videoFileNames = []

        # load label file into memory
        with open(self.datafile, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split(",")
                videoName = self._getFullFileName(parts[0])
                self.videoFileNames.append(videoName)
                self.frameNumbers.append(int(parts[1]))
                self.bboxs.append([int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])
                self.labels.append(int(parts[6]))

    def _getFullFileName(self, target):
        '''Get the full filename path'''

        for file in self.videoFiles:
            if target in str(file):
                return file

    def __getitem__(self, idx):
        print(self.videoFileNames[idx])
        cap = cv2.VideoCapture(str(self.videoFileNames[idx]))  # converts to RGB by default
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumbers[idx])
        _, image = cap.read()
        cap.release()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        ymax, xmax = image.shape[0], image.shape[1]

        target = {}

        # y0, x0, y1, x1
        # boxes = [[self.bboxs[idx][1]/xmax, (self.bboxs[idx][0]+130)/ymax, self.bboxs[idx][3]/xmax, (self.bboxs[idx][2]+130)/ymax]]
        left = self.bboxs[idx][0]
        top = self.bboxs[idx][1] + 130
        right = self.bboxs[idx][2]
        bottom = self.bboxs[idx][3] + 130
        bbox = [left, top, right, bottom]
        bbox = torch.as_tensor([bbox], dtype=torch.float32)

        area = np.abs(right - left) * np.abs(bottom - top)

        label = self.labels[idx] + 1  # as 0 is background
        label = torch.as_tensor([label], dtype=torch.int64)

        target["boxes"] = torch.as_tensor(bbox, dtype=torch.float32)
        target["labels"] = torch.as_tensor(label, dtype=torch.int64)
        target["area"] = torch.as_tensor([area], dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(0, dtype=torch.int64)
        target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.labels)
