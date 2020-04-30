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
    torch.manual_seed(1)

    gpu = 1
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)

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
        # a = evaluate(model, data_loader_test, device)
        i = 0

        classStats = {"0": [0, 0, 0], "1": [0, 0, 0], "2": [0, 0, 0], "3": [0, 0, 0]}

        det_boxes = list()
        det_labels = list()
        det_scores = list()
        truth_boxes = list()
        truth_labels = list()

        for image, targets in data_loader_test:
            image = list(img.to(device) for img in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(image)
            bboxs = outputs[0]["boxes"].detach().cpu()
            labels = outputs[0]["labels"].detach().cpu()
            scores = outputs[0]["scores"].detach().cpu()
            targetBox = targets[0]["boxes"].cpu()
            targetLabel = targets[0]["labels"].cpu()

            tbox = [b for b in targetBox]
            tlabel = [l for l in targetLabel]

            bbox = [b for b in bboxs]
            labels = [b for b in labels]
            scores = [b for b in scores]

            det_boxes.extend(bboxs)
            det_labels.extend(labels)
            det_scores.extend(scores)
            truth_boxes.extend(tbox)
            truth_labels.extend(tlabel)

            fig, ax = plt.subplots(1, 1)
            image = image[0].detach().cpu().numpy()
            image = np.transpose(image, (1, 2, 0))
            ax.imshow(image)
            # sys.exit()
            targetin = [targetBox, targetLabel]
            bboxs = outputs[0]["boxes"].detach()
            scores = outputs[0]["scores"].detach()
            labels = outputs[0]["labels"].detach()

            # keeps = soft_nms_pytorch(bboxs, scores, sigma=0.5, thresh=0.5, cuda=1)
            # keeps = keeps.cpu().numpy()
            # # print(bboxs)
            # if len(keeps) > 0:
            #     bboxs = bboxs[keeps].cpu().numpy()
            #     scores = scores[keeps].cpu().numpy()

            #     outputin = [bboxs, labels[keeps], scores]
            #     opc, tpc = draw_boxes(outputin, ["Dolphin", "Guitar", "Apple", "Orange"], targetin, ax, thresh=0.0)
            #     ax.add_collection(tpc)
            #     ax.add_collection(opc)

            plt.savefig(f"test{i}.png", dpi=96)
            # sys.exit()
            if i == 4:
                break
                sys.exit()
            i += 1
        print(det_labels)
        print(truth_labels)
        sys.exit()
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, truth_boxes, truth_labels)
        print(APs, mAP)


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels):

    n_classes = 5

    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i])
    true_images = torch.LongTensor(true_images)
    true_boxes = torch.stack(true_boxes, dim=0)
    true_labels = torch.stack(true_labels, dim=0)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i])
    det_images = torch.LongTensor(det_images)
    det_boxes = torch.stack(det_boxes, dim=0)
    det_labels = torch.stack(det_labels, dim=0)
    det_scores = torch.stack(det_scores, dim=0)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    average_precision = torch.zeros((n_classes - 1), dtype=torch.float)

    for c in range(1, n_classes):
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]

        true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8)

        det_class_images = det_images[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        print(c, len(true_class_boxes))
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros((n_class_detections), dtype=torch.float)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)
            this_image = det_class_images[d]

            object_boxes = true_class_boxes[true_class_images == this_image]
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            if max_overlap.item() > 0.5:
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1
                else:
                    false_positives[d] = 1

            else:
                false_positives[d] = 1

        cumul_true_positives = torch.cumsum(true_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
        cumul_recall = cumul_true_positives / len(true_labels)

        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precision[c - 1] = precisions.mean()

    mean_average_precision = average_precision.mean().item()
    rev_label_map = {4: "Dolphin", 3: "Guitar", 2: "Apple", 1: "Orange", 0: "Backgound"}
    average_precision = {rev_label_map[c+1]: v for c, v in enumerate(average_precision.tolist())}

    return average_precision, mean_average_precision


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.
    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


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
