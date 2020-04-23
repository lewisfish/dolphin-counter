import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from customdata import OIDdataset
from engine import train_one_epoch, evaluate
import transforms as T
import utils


def get_model_instance(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

    num_classes = 4

    dataset = OIDdataset("Dataset/train", get_transform(train=True), ["Dolphin", "Guitar", "Apple", "Orange"])
    dataset_test = OIDdataset("Dataset/validation", get_transform(train=False), ["Dolphin", "Guitar", "Apple", "Orange"])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4,
                                                   collate_fn=utils.collate_fn)

    model = get_model_instance(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9,
                                weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,
                                                   gamma=0.1)

    if args.continue_train:
        checkpoint = torch.load("checkpoint_state.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    num_epochs = args.epochs

    for epoch in range(start_epoch, num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, "checkpoint_state.pth")
        evaluate(model, data_loader_test, device=device)

    torch.save(model, "final-output.pth")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch pipeline for train\
                                     and detection of 4 image classes.')

    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of\
                        epochs to train for (default: 10)", metavar="N")
    parser.add_argument("-c", "--continue_train", action="store_true",
                        default=False, help="For continuing training.")
    parser.add_argument("-i", "--evaluate", action="store_true", help="Infere on a bunch of images.")

    args = parser.parse_args()

    main(args)
