import torch
import torchvision 
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from frcnn_data import CustomDataset
import frcnn_utils
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from frcnn_train import evaluate

#Change these
MODEL_NAME = "model_2_aug_long_run_95.pth" #Modelname
VAL_FILE = "val"                           # Training Folder


#dont touch
MODEL_PATH = os.path.join("checkpoints", MODEL_NAME)
VAL_ANNOTATIONS = f"{VAL_FILE}/coco_anno_{VAL_FILE}.json"
VAL_IMAGES = os.path.join(VAL_FILE, "images")
input_annotations_dir = os.path.join(VAL_FILE, "annotations")

# Device (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1


#
#load model
#
print("laod model...")
weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=3)


NUM_CLASSES = 4  # 3 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, NUM_CLASSES
)

#laod weights
print(f"load model weights {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])  # Lade nur den Modellzustand

model.to(DEVICE)

#
# load validation data
# 
print("load validationdata...")
from frcnn_data import CustomDataset
val_images = os.path.join(VAL_FILE, "images")
val_annotations = os.path.join(VAL_FILE, "annotations")
val_dataset = CustomDataset(val_images, val_annotations, use_target_size=False, target_size=1024, augmentation=False)


val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=frcnn_utils.collate_fn,
    num_workers=8,
    drop_last=False
)

#save the image as a jpg with the predicted bounding boxes and labels as red box on the image
def draw_boxes(image, boxes, labels, scores):
    image = image.mul(255).permute(1, 2, 0).byte().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for box, label, score in zip(boxes, labels, scores):
        box = [int(b) for b in box]
        color = (0, 0, 255)  # Red
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        image = cv2.putText(image, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if label == 0:
            print('Found label 0 background in image')
    return image




current_model_name = MODEL_NAME

print("start evaluation...")
model.eval()
results = []
i = 0



eval_metrics = evaluate(model, val_loader, DEVICE, confidence_threshold=0.0)

print(f"Validation: {eval_metrics}")   

#all images with their predictions of validation data
'''
for images, targets in val_loader:
    images = [img.to(DEVICE) for img in images]
    
    #assert len(images) == 1, "Die Batch-Größe sollte 1 sein."
    
    with torch.no_grad():
        outputs = model(images)
    
    #assert len(outputs) == 1, "Die Anzahl der Ausgaben sollte 1 sein."
    output = outputs[0]
    print(f"Out für Bild {i}: {output}")
    boxes = output["boxes"].tolist()

    num_boxes = len(boxes)
    if num_boxes == 0:
        print(f"Keine Bounding Boxes für Bild {i}")
        i += 1
        continue
    else:
        print(f"{num_boxes} Bounding Boxes für Bild {i}")

    scores = output["scores"].tolist()
    labels = output["labels"].tolist()

    # draw the boxes on the image
    img = images[0].detach().cpu()
    image = draw_boxes(img, boxes, labels, scores)

    # Save the image

    image_path = f"/home/marek/Dokumente/Train_model_1/eval_images/image_{i}.jpg"
    cv2.imwrite(image_path, image)
    i += 1

# -------------------------
'''


