from pathlib import Path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


class Dataset(object):
    """docstring for Dataset"""
    def __init__(self, root, transforms, classes):
        super(Dataset, self).__init__()
        self.root = Path(root)
        self.transforms = transforms
        self.classes = classes
        print(self.classes)
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

        image = image.transpose((2, 0, 1))
        image = torch.as_tensor(image, dtype=torch.float32).to(device)

        if self.transforms:
            image = self.transforms(image)

        target = {}

        with open(self.labels[idx], "r") as fin:
            lines = fin.readlines()
            if len(lines) > 1:
                boxes = []
                labels = []
                for line in lines:
                    line = line.split()
                    className = line[0]
                    label = self.classes.index(className)
                    left = float(line[1]) / xmax
                    top = float(line[2]) / ymax
                    right = float(line[3]) / xmax
                    bottom = float(line[4]) / ymax
                    bbox = [left, top, right, bottom]
                    boxes.append(bbox)
                    labels.append(label)
                target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).to(device)
                target["labels"] = torch.as_tensor(labels, dtype=torch.int64).to(device)

            else:
                lines = lines[0].split()
                className = lines[0]
                label = self.classes.index(className)
                label = torch.as_tensor([label], dtype=torch.int64).to(device)
                left = float(lines[1])
                top = float(lines[2])
                right = float(lines[3])
                bottom = float(lines[4])
                bbox = [left, top, right, bottom]
                bbox = torch.as_tensor([bbox], dtype=torch.float32).to(device)
                target["boxes"] = bbox
                target["labels"] = label

        return image, target

    def __len__(self):
        return len(self.labels)


def train(model, trainloader, log_interval, path='./test.pth'):

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_steps = [16, 22]
    lr_gamma = 0.1
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)

    epochs = 5

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, data in enumerate(trainloader):

            inputs, labels = data
            inputs.unsqueeze_(0)

            # zero the parameters gradients
            optimiser.zero_grad()

            # forward + backward + optimise
            outputs = model(inputs, [labels])
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimiser.step()
            lr_scheduler.step()

            if batch_idx % log_interval == 0:
                print(f"Train epoch: {epoch} [{batch_idx*len(data)}/{len(trainloader)} ({100.*batch_idx / len(trainloader):.0f}%)]\t Loss:{loss.item():.6f}")

        if epoch % 2 == 0:
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimiser.state_dict(),
                          "loss": loss}

            torch.save(checkpoint, "chkpt.pth")

    torch.save(model.state_dict(), path)
    print("donzo washington")


def validation(model, validationLoader):
    model.eval()
    validation_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in validationLoader:
            data.unsqueeze_(0)
            output = model(data)
            plot(data, output[0])
            # validation_loss += sum(loss for loss in output.values())
            # print(output)
            sys.exit()


def calcIOU(boxA, boxB):
    ''' from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc'''

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def plot(image, box):
    fig, ax = plt.subplots(1)

    img = image.cpu().numpy()
    img = img[0, :, :, :]
    img = np.transpose(img, (1, 2, 0))
    ymax, xmax, chan = img.shape[0], img.shape[1], img.shape[2]
    img[:, :, 0] = ((img[:, :, 0] * 0.5) + .5)
    img[:, :, 1] = ((img[:, :, 1] * 0.5) + .5)
    img[:, :, 2] = ((img[:, :, 2] * 0.5) + .5)
    ax.imshow(img)

    # for b, score in zip(box["boxes"].data.cpu().numpy(), box["scores"].data.cpu().numpy()):
    idx = box["scores"].data.argmax().cpu().numpy().argmax()
    x0, x1, y0, y1 = box["boxes"].data[idx][0].cpu().numpy()*xmax, box["boxes"].data[idx][1].cpu().numpy()*ymax, box["boxes"].data[idx][2].cpu().numpy()*xmax, box["boxes"].data[idx][3].cpu().numpy()*ymax
    print(x0)
    w = max(x1, x0) - min(x1, x0)
    h = max(y1, y0) - min(y1, y0)

    rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor="r", facecolor="none")
    ax.add_patch(rect)

    plt.savefig("test.png", dpi=96)


if __name__ == '__main__':

    # setup using gpu 1
    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    normalise = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
    trainData = Dataset("Dataset/validation", normalise, ["Dolphin", "Guitar", "Apple", "Orange"])

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 4 + 1  # +1 for background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    model.load_state_dict(torch.load("test.pth"))

    validation(model, trainData)
    # train(model, trainData, 100)
