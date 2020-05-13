import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter

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
    torch.manual_seed(1)

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    print(device)
    num_classes = 4

    dataset = OIDdataset("Dataset/train", get_transform(train=True), ["Dolphin", "Guitar", "Apple", "Orange"])
    dataset_test = OIDdataset("Dataset/validation", get_transform(train=False), ["Dolphin", "Guitar", "Apple", "Orange"])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=4,
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

    if not args.evaluate:
        writer = SummaryWriter("runs/train")
        for epoch in range(start_epoch, num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq=10)
            lr_scheduler.step()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, "checkpoint_state.pth")
            evaluate(model, data_loader_test, device=device)

        torch.save(model, "final-model.pth")
    else:
        model = torch.load("final-model.pth")
        model.eval()
        # evaluate(model, data_loader_test, device)
        gts = {}
        preds = {}
        i = 0
        for image, targets in data_loader_test:
            image = list(img.to(device) for img in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(image)

            # todo make batchable
            # todo turn into dict instaed of list
            imageID = targets[0]["image_id"].detach().cpu().tolist()
            gt_bboxs = targets[0]["boxes"].detach().cpu().tolist()
            gt_labels = targets[0]["labels"].detach().cpu().tolist()
            gts[imageID] = [gt_bboxs, gt_labels]

            p_bboxs = outputs[0]["boxes"].detach().cpu().tolist()
            p_labels = outputs[0]["labels"].detach().cpu().tolist()
            p_scores = outputs[0]["scores"].detach().cpu().tolist()
            preds[imageID] = [p_bboxs, p_labels, p_scores]

        aps = 0
        for IoUThresh in np.linspace(0.5, 0.95, 10):
            scoreThresh = 0.5
            dets = []
            idx = 0
            gtLabelsCounts = [0, 0, 0, 0, 0]
            for imageID in gts:
                if imageID in preds:
                    gt_bboxs = gts[imageID][0]
                    gt_labels = gts[imageID][1]
                    for bboxGT, labelGT in zip(gt_bboxs, gt_labels):
                        gtLabelsCounts[labelGT] += 1
                        det = False
                        p_bbox = preds[imageID][0]
                        p_labels = preds[imageID][1]
                        p_scores = preds[imageID][2]
                        for bbox, label, score in zip(p_bbox, p_labels, p_scores):
                            tmp = [0, 0, 0, 0, 0, 0]
                            tmp[0] = imageID
                            tmp[1] = idx
                            tmp[2] = label
                            tmp[3] = score
                            if score > scoreThresh:
                                iou = _calcIoU(bbox, bboxGT)
                                tmp[4] = iou
                                if iou > IoUThresh:
                                    if label == labelGT and not det:
                                        # TP
                                        det = True
                                        tmp[5] = True
                                    else:
                                        # FP
                                        tmp[5] = False
                                else:
                                    # FP
                                    tmp[5] = False
                            else:
                                # FN
                                tmp[5] = False
                            dets.append(tmp)
                            idx += 1
                else:
                    # FN
                    print("fuck!")

            accTP = [0, 0, 0, 0, 0]
            accFP = [0, 0, 0, 0, 0]

            dets = sorted(dets, key=lambda x: x[2], reverse=True)

            pr = {0: [], 1: [], 2: [], 3: [], 4: []}
            for i, det in enumerate(dets):
                if det[5]:
                    accTP[det[2]] += 1
                else:
                    accFP[det[2]] += 1
                pr[det[2]].append([accTP[det[2]] / (accTP[det[2]] + accFP[det[2]]), accTP[det[2]] / gtLabelsCounts[det[2]]])

            AP = calcAP(pr)
            mAP = sum(AP) / (len(AP)-1)
            aps += mAP
            print(f"AP:{AP}", f"mAP:{mAP}")
            IoUThresh += 0.05
        print(aps / 10)


def calcAP(PRs):
    import bisect

    APs = [0, 0, 0, 0, 0]

    for key, value in PRs.items():
        if len(value) == 0:
            continue
        prec, recall = zip(*value)

        summ = 0
        rs = np.linspace(0, 1, 11)
        for r in rs:
            idx = bisect.bisect_left(recall, r)
            try:
                term = max(prec[idx:])
            except ValueError:
                term = 0.0
            summ += term
        APs[key] = (summ*100) / 11.
    return APs


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep


def _calcIoU(boxA, boxB):
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


def draw_boxes(outputs, names, targets, ax, thresh=0.5):

    outbboxs = []
    for bbox, label, score in zip(outputs[0], outputs[1], outputs[2]):
        if score >= thresh:
            # left, top, right, bottom
            x = bbox[0]
            y = bbox[1]
            width = np.abs(bbox[2] - bbox[0])
            height = np.abs(bbox[3] - bbox[1])
            rect = Rectangle((x, y), width, height, edgecolor="green", fill=False)
            # ax.text(x, y-20, str(score), backgroundcolor="white")
            outbboxs.append(rect)
    opc = PatchCollection(outbboxs, edgecolor="red", facecolor="none")

    outbboxs = []
    for box, label in zip(targets[0], targets[1]):
        x = box[0]
        y = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = Rectangle((x, y), width, height)
        ax.text(x, y-20, str(label))
        outbboxs.append(rect)
    tpc = PatchCollection(outbboxs, edgecolor="green", facecolor="none")

    return opc, tpc


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
