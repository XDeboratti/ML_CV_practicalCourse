import math
import sys
import torch
import frcnn_utils
import time


import numpy as np

def calculate_iou(box_a, box_b):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    box_a and box_b are in the format [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    #calculate the intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    #calculate the area of each box
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    #calculate the union area
    union_area = area_a + area_b - intersection_area

    # avoid division by zero
    if union_area == 0:
        return 0

    return intersection_area / union_area

def compute_average_precision(pred_boxes, pred_labels, target_boxes, target_labels, iou_threshold=0.5):
    """
    Compute the Average Precision (AP) for a single image.
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0

    matched = [False] * len(target_boxes)
    true_positives = 0
    false_positives = 0

    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        best_iou = 0
        best_match_idx = -1

        for idx, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
            if matched[idx]:
                continue

            iou = calculate_iou(pred_box, target_box)
            if iou >= iou_threshold and pred_label == target_label and iou > best_iou:
                best_iou = iou
                best_match_idx = idx

        if best_match_idx != -1:
            matched[best_match_idx] = True
            true_positives += 1
        else:
            false_positives += 1

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

def generate_loss_metric(predictions, targets):
    # Ensure predictions and targets are of the same length
    assert len(predictions) == len(targets)

    mean_iou_acc = 0
    mean_map_acc = 0

    for label_pred, label_target in zip(predictions, targets):
        # Extract predicted and ground truth boxes and labels
        pred_boxes = label_pred['boxes']
        pred_labels = label_pred['labels']
        target_boxes = label_target['boxes']
        target_labels = label_target['labels']

        # Compute IoU
        iou_sum = 0
        count = 0
        for target_box in target_boxes:
            best_iou = 0
            for pred_box in pred_boxes:
                iou = calculate_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
            iou_sum += best_iou
            count += 1

        iou = iou_sum / count if count > 0 else 0

        # Compute mAP
        map = compute_average_precision(pred_boxes, pred_labels, target_boxes, target_labels)

        mean_iou_acc += iou
        mean_map_acc += map

    res = {
        'iou': mean_iou_acc,
        'map': mean_map_acc,
        'num_images': len(predictions)
    }
    return res



def generate_loss_metric3(predictions, targets):
    #calculation of losses
    
    assert len(predictions) == len(targets)

    
    print("predictions", predictions)
    
    mean_iou_acc = 0
    mean_map_acc = 0   
    
    for label_pred, label_target in zip(predictions, targets):
        
        # what we want to predict
        expected_num_preds = label_target['labels'].shape[0]
        
        expected_box = label_target['boxes']
        expected_labels = label_target['labels']


        # what we predicted
        num_preds = label_pred['labels'].shape[0]
        print("num_preds", num_preds)
        boxes = label_pred['boxes']
        labels = label_pred['labels']

       
        iou = 0
        map = 0
        
        mean_iou_acc += iou
        mean_map_acc += map
    

    res = {
        'iou': mean_iou_acc,
        'map': mean_map_acc,
        'num_images' : len(predictions)
    }


    return res



def accumulate_metric(metric):  
    # TODO
    iou_acc = 0
    map_acc = 0
    num_images = 0

    for m in metric:
        iou_acc += m['iou']
        map_acc += m['map']
        num_images += m['num_images']
    
    iou = iou_acc / num_images
    map = map_acc / num_images

    metric = {
        'mean_iou': iou.item() if isinstance(iou, torch.Tensor) else iou,
        'mean_map': map
    }
    return metric


def evaluate(model ,val_loader, device, confidence_threshold=0.0):

    #load model
    model.to(device)

    #evaluation
    print("start evaluation...")
    results = []
    validation_loss = 0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


            # Berechnung der Verluste im Trainingsmodus
            # Temporär ins Training setzen
            predictions = model(images)
            # Filter predictions based on the confidence threshold
            filtered_predictions = []
            for pred in predictions:
                scores = pred['scores'].tolist()
                indices = [i for i, score in enumerate(scores) if score >= confidence_threshold]

                # Filter boxes, labels, and scores
                filtered_predictions.append({
                    'boxes': pred['boxes'][indices],
                    'labels': pred['labels'][indices],
                    'scores': pred['scores'][indices]
                })
            metric = generate_loss_metric(filtered_predictions, targets)
            results.append(metric)
            
            total_samples += 1

    # accumulate the metric over batches
    # metric -> iou, map
    metric_acc = accumulate_metric(results)
        
    res = {
        'validation_iou': metric_acc['mean_iou'],
        'validation_map': metric_acc['mean_map'],
    }
  
    return res


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1165):
    model.train()
    metric_logger = frcnn_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", frcnn_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    epoch_loss = 0  

    start_time = time.time()
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        batch_size = len(images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        epoch_loss += losses.item()    

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses.item(), **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        time_taken = time.time() - start_time
        time_per_image = time_taken / batch_size
        #print(f"Epoch: {epoch}, Time: {time_taken:.4f}, Loss: {losses.item()}, time_per_image : {time_per_image:.4f}")
        start_time = time.time()

    
    #evaluate the model on training data
    model.eval()
    
    
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
            metric = generate_loss_metric(predictions, targets)
            results.append(metric)

    # accumulate the metric over batches
    # metric -> iou, map
    metric_acc = accumulate_metric(results)


    res = {
        'train_loss': epoch_loss / len(data_loader),
        'train_iou': metric_acc['mean_iou'],
        'train_map': metric_acc['mean_map']
    }

    return res