import json

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from myutils.common import file_util
from utils import yolo_util


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    iou = intersection / ua
    return iou


def evaluate_coco(dataset, model, device, output_file_path, threshold=0.05, log_size=100):
    model.eval()
    with torch.no_grad():
        # start collecting results
        results = []
        image_ids = []
        num_samples = len(dataset)
        unit_size = num_samples // log_size
        for index in range(num_samples):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = model(data['img'].permute(2, 0, 1).to(device).float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            # correct boxes for image scale
            boxes /= scale

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                # compute predicted labels and scores
                # for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            if (index + 1) % unit_size == 0:
                # print progress
                print('{}/{}'.format(index + 1, num_samples), end='\r')

        if not len(results):
            return

        # write output
        file_util.make_parent_dirs(output_file_path)
        json.dump(results, open(output_file_path, 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(output_file_path)

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        result = coco_eval.eval
        model.train()
        return result


def evaluate_coco4yolo(dataset, model, device, output_file_path='tmp.json'):
    """
    COCO average precision (AP) Evaluation. Iterate inference on the test dataset
    and the results are evaluated by COCO API.
    Args:
        model : model object
    Returns:
        ap50_95 (float) : calculated COCO AP for IoU=50:95
        ap50 (float) : calculated COCO AP for IoU=50
    """
    model.eval()
    ids = []
    data_dict = []
    data_iterator = iter(yolo_util.get_data_loader(dataset, batch_size=1))
    # all the data in val2017
    for img, _, info_img, id_ in data_iterator:
        info_img = [float(info) for info in info_img]
        id_ = int(id_)
        ids.append(id_)
        with torch.no_grad():
            img = img.to(device)
            outputs = model(img)
            if outputs[0] is None:
                continue

            outputs = outputs[0].cpu().data

        for output in outputs:
            x1 = float(output[0])
            y1 = float(output[1])
            x2 = float(output[2])
            y2 = float(output[3])
            label = dataset.class_ids[int(output[6])]
            box = yolo_util.yolobox2label((y1, x1, y2, x2), info_img)
            bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
            score = float(output[4].data.item() * output[5].data.item())  # object score * class score
            result_dict = {"image_id": id_, "category_id": label, "bbox": bbox,
                           "score": score, "segmentation": []}  # COCO json format
            data_dict.append(result_dict)

    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_dict) > 0:
        coco = dataset.coco
        file_util.make_parent_dirs(output_file_path)
        json.dump(data_dict, open(output_file_path, 'w'))
        coco_detection = coco.loadRes(output_file_path)
        result = COCOeval(coco, coco_detection, 'bbox')
        result.params.imgIds = ids
        result.evaluate()
        result.accumulate()
        result.summarize()
        return result.stats[0], result.stats[1]
    else:
        return 0, 0


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    intersection = iw * ih
    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, model, device, score_threshold=0.05, max_detections=100):
    """ Get the detections from the model using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for _ in range(dataset.num_classes())] for _ in range(len(dataset))]
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            # run network
            scores, labels, boxes = model(data['img'].permute(2, 0, 1).to(device).float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')
    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for _ in range(generator.num_classes())] for _ in range(len(generator))]
    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
        print('{}/{}'.format(i + 1, len(generator)), end='\r')
    return all_annotations


def evaluate_csv(generator, model, device, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    """ Evaluate a given dataset using a given model.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations
    all_detections = _get_detections(generator, model, device, score_threshold=score_threshold,
                                     max_detections=max_detections)
    all_annotations = _get_annotations(generator)
    average_precisions = {}
    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0
        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []
            for d in detections:
                scores = np.append(scores, d[4])
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\nmAP:')
    for label in range(generator.num_classes()):
        label_name = generator.label_to_name(label)
        print('{}: {}'.format(label_name, average_precisions[label][0]))
    return average_precisions

