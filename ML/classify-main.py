import argparse
import sys

import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from customdata import DolphinDatasetClass
from engine import train_classify, class_evaluate
import transforms as T


def get_resnet50(num_classes):

    # get pretrained model
    model = torchvision.models.resnet50(pretrained=True)
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # swap out final layer so has the correct number of classes.
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, num_classes),
                             nn.LogSoftmax(dim=1))

    return model


def imageTransfroms(train):
    transforms = []
    transforms.append(torchvision.transforms.Resize((224, 224)))
    if train:
        transforms.append(torchvision.transforms.RandomRotation(180))
        transforms.append(torchvision.transforms.RandomHorizontalFlip(180))

    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize([.485, .456, .406],
                                                       [.229, .224, .225]))
    return torchvision.transforms.Compose(transforms)


def main(args):
    # import json
    torch.manual_seed(1)

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    print(device)

    classes = ["dolphin", "not dolphin"]
    num_classes = len(classes)

    root = "/data/lm959/data/ETP trial survey/Jason's Computer"
    labelfile = "train.csv"

    batch_size = 64

    dataset = DolphinDatasetClass(root, imageTransfroms(train=True), labelfile)
    dataset_test = DolphinDatasetClass(root, imageTransfroms(train=False), "valid.csv")
    class_weights = np.loadtxt("class_weights.csv", delimiter=",")

    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_weights, len(dataset), replacement=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                              drop_last=True, sampler=sampler)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
                                                   drop_last=True)

    # create class weights.
    # easier to do once then read in saved weights as reading images takes
    # too long

    # labels = []
    # weights = 1000. / torch.tensor([1602, 11688.], dtype=torch.float)
    # for i, (img, targ) in enumerate(data_loader):
    #     labels.append(targ.item())
    # class_weights = np.array(weights[labels])

    model = get_resnet50(num_classes)
    model.to(device)

    # penalize not dolphin class
    weight = torch.FloatTensor([7.2959, 1.]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0003)

    if args.continue_train:
        checkpoint = torch.load("checkpoint_state_DC.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    num_epochs = args.epochs

    if not args.evaluate:
        writer = SummaryWriter("dolphin/classify/train")
        train_classify(model, criterion, optimizer, data_loader, data_loader_test, device, num_epochs, writer, 50)
        torch.save(model, "final-model_DC1.pth")
    else:
        model = torch.load("final-model_DC1.pth")
        model.eval()
        val_losses = 0
        class_evaluate(model, data_loader_test, criterion, val_losses, device)


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
