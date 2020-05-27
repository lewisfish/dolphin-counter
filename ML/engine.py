import math
import sys
import time
import torch
from tqdm import tqdm
import torch.nn.functional as F
import inspect
import torchvision.models.detection.mask_rcnn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn import metrics

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_classify(model, criterion, optimizer, train_loader, test_loader, device, epochs, writer, print_freq):
    model.train()
    losses = []
    batches = len(train_loader)
    val_batches = len(test_loader)

    for epoch in range(epochs):
        total_loss = 0
        progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)
        model.train()
        for i, data in progress:

            inputs = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            total_loss += current_loss
            progress.set_description(f"Loss: {total_loss/(i+1):.4f}")
            writer.add_scalar("Loss", total_loss/(i+1), epoch * len(train_loader) + i)

        val_losses = class_evaluate(model, test_loader, criterion, device, epoch, writer)

        print(f"Epoch {epoch+1}/{epochs}, training loss: {total_loss/batches}, validation loss: {val_losses/val_batches}")

        losses.append(total_loss/batches)  # for plotting learning curve
        writer.add_scalar("Loss/train", total_loss/batches, epoch)
        writer.add_scalar("Loss/Test", val_losses/val_batches, epoch)
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "checkpoint_state_DC.pth")


def class_evaluate(model, test_loader, criterion, device, epoch, writer=None):

    val_losses = 0
    precision, recall, f1, accuracy, baccuracy = [], [], [], [],  []
    trues = []
    preds = []

    model.eval()
    with torch.no_grad():
        # start = time.time()
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            outputs = model.forward(X)
            val_losses += criterion(outputs, y)
            predicted_classes = torch.max(outputs, 1)[1]  # get class from network's prediction
            trues.extend(y.cpu().detach().numpy())
            preds.extend(predicted_classes.cpu().detach().numpy())
            for acc, metric in zip((precision, recall, f1, accuracy, baccuracy),
                                   (precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )
        # print(metrics.classification_report(trues, preds))
        # cm = metrics.confusion_matrix(trues, preds)
        # plot_confusion_matrix(cm, ["Dolphin", "Not dolphin"])
        # print_scores(precision, recall, f1, accuracy, baccuracy, 64)
        results = metrics.precision_recall_fscore_support(trues, preds)
        if writer:

            writer.add_scalar("Accuracy/Dolphin", results[0][0], epoch)
            writer.add_scalar("Recall/Dolphin", results[1][0], epoch)
            writer.add_scalar("F1/Dolphin", results[2][0], epoch)

            writer.add_scalar("Accuracy/Not_dolphin", results[0][1], epoch)
            writer.add_scalar("Recall/Not_dolphin", results[1][1], epoch)
            writer.add_scalar("F1/Not_dolphin", results[2][1], epoch)

    # finish = time.time()
    # print(finish-start, (finish-start)/i)
    return val_losses


def train_one_epoch(model, optimizer, data_loader, device, epoch, writer, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    i = 0
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if i % print_freq == 0:
            writer.add_scalar("loss", losses_reduced, epoch * len(data_loader) + i)
            writer.add_scalar("loss_classifier", loss_dict_reduced["loss_classifier"], epoch * len(data_loader) + i)
            writer.add_scalar("loss_box_reg", loss_dict_reduced["loss_box_reg"], epoch * len(data_loader) + i)
            writer.add_scalar("loss_objectness", loss_dict_reduced["loss_objectness"], epoch * len(data_loader) + i)
            writer.add_scalar("loss_rpn_box_reg", loss_dict_reduced["loss_rpn_box_reg"], epoch * len(data_loader) + i)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch * len(data_loader) + i)
        i += 1
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device):
    '''coco evaulation. Gets mAP
    '''
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


def calculate_metric(metric_fn, true_y, pred_y):
    # multi class problems need to have averaging method
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)


def print_scores(p, r, f1, a, ba, batch_size):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy", "baccuracy"), (p, r, f1, a, ba)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")


def plot_confusion_matrix(cm,
                          labels,
                          title='Confusion matrix',
                          cmap=None,
                          norm=False):
    """
    given a sklearn confusion matrix (cm), create a matplotlib figure
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    labels:       given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    norm:         If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm     = cm,                  # confusion matrix created by
                                                        # sklearn.metrics.confusion_matrix
                          norm   = True,                # show proportions
                          labels = y_labels_vals,       # list of names of the classes
                          title  = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """
    from matplotlib import colorbar, rcParams
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    # Set font params
    rcParams['font.size'] = 20
    rcParams['font.weight'] = 'normal'

    # Calculate accuracy and max value
    accuracy = np.trace(cm) / float(np.sum(cm))
    maximum = 1 if norm else cm.max()

    # Set default colourmap (purple is nice)
    if cmap is None:
        cmap = plt.get_cmap('Purples')

    # Normalise values
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    im = plt.imshow(norm_cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title, fontweight='bold')
    pos = ax.get_position()

    # Add values to figure
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > cm[i].sum() / 2 else 'black'
        text = f"{norm_cm[i,j]:0.4f}" if norm else f"{cm[i,j]:0.0f}"
        plt.text(j, i, text, horizontalalignment='center', va='center', color=color, fontsize=25)
        ax.axhline(i-.5, color='black', linewidth=1.5)
        ax.axvline(j-.5, color='black', linewidth=1.5)

    # Add primary axes
    tick_marks = np.arange(len(labels))

    ax.tick_params(
        axis='both',
        which='both',
        labeltop=False,
        labelbottom=False,
        length=0)
    ax.set_ylabel(f'True\n')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    ax.set_xticks(tick_marks)
    ax.set_xlabel(f'\nOverall Accuracy={accuracy:0.4f}')
    ax.tick_params(axis='both', which='both', pad=15)
    ax.tick_params(axis='y', which='minor', labelrotation=90)

    # Add secondary axes displaying at top of figure
    ax2 = ax.twiny()
    ax2.tick_params(
        axis='both',
        which='both',
        labelbottom=False,
        length=0)
    ax2.tick_params(axis='both', which='both', pad=15)
    ax2.set_xticks(tick_marks)
    ax2.set_xlim(ax.get_xlim())

    ax.autoscale(False)
    ax2.autoscale(False)

    ax2.set_xlabel('\nPredicted\n')
    ax2.set_xticklabels(labels)

    # Add colourbar
    cbax = fig.add_axes([pos.x0+pos.width+.15, pos.y0, 0.08, pos.height])
    cb = colorbar.ColorbarBase(cbax, cmap=cmap, orientation='vertical')
    cb.set_label('Accuracy per label')

    plt.savefig("cm.png")