import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
import json
import numpy as np
import random
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


# Mapillary Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None, target_size=1024, use_target_size=True, augmentation=True):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.json')])
        self.target_size = target_size
        self.use_target_size = use_target_size
        self.augmentation = augmentation

        assert len(self.image_files) == len(self.annotation_files), \
            "Number of Images does not fit to number of annotation files."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # === Lade Bild ===
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # === Lade Annotation ===
        annotation_path = os.path.join(self.annotations_dir, self.annotation_files[idx])
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # === Extrahiere Bounding Boxes ===
        boxes = []
        labels = []
        label_mapping = {
            "regulatory--stop--g1": 1,
            "regulatory--yield--g1": 2,
            "regulatory--priority-road--g4": 3
        }


        for obj in annotation["objects"]:
            original_label = obj["label"]
            if original_label in label_mapping:
                bbox = obj["bbox"]
                xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_mapping[original_label])

        # === Augmentation: Flip ===
        if self.augmentation and np.random.rand() > 0.5:
            img = F.hflip(img)
            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]
                new_xmin = img.width - xmax
                new_xmax = img.width - xmin
                boxes[i] = [new_xmin, ymin, new_xmax, ymax]

        # === Augmentation: Cropping ===
        if self.augmentation:
            width, height = img.size
            crop_left = int(width * random.uniform(0.01, 0.25))
            crop_bottom = int(height * random.uniform(0.01, 0.25))

            new_width = width - crop_left
            new_height = height - crop_bottom

            img = F.crop(img, top=crop_bottom, left=crop_left, height=new_height, width=new_width)

            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]
                xmin = max(0, xmin - crop_left)
                xmax = max(0, xmax - crop_left)
                ymin = max(0, ymin - crop_bottom)
                ymax = max(0, ymax - crop_bottom)
                boxes[i] = [xmin, ymin, xmax, ymax]

        #Resizing
        if self.use_target_size:
            current_width, current_height = img.size
            scale_x = self.target_size / current_width
            scale_y = self.target_size / current_height

            img = img.resize((self.target_size, self.target_size))

            for i in range(len(boxes)):
                xmin, ymin, xmax, ymax = boxes[i]
                xmin *= scale_x
                ymin *= scale_y
                xmax *= scale_x
                ymax *= scale_y
                boxes[i] = [xmin, ymin, xmax, ymax]

        #Filter invalid Bounding Boxes
        valid_boxes = []
        valid_labels = []
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            if xmax > xmin and ymax > ymin:  #keep only valid BBoxes 
                valid_boxes.append([xmin, ymin, xmax, ymax])
                valid_labels.append(labels[i])

        #convert picture to Tensor
        img = F.to_tensor(img)

        #convert bboxes and labels to tensors
        if len(valid_boxes) > 0:
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            labels = torch.as_tensor(valid_labels, dtype=torch.int64)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        #create Target Dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return img, target


# DFG Dataset
class DFG_Dataset(Dataset):
    def __init__(self, data_dir, phase='train', target_size=800, use_target_size=True, augmentation=True):
        super().__init__()
        self.target_size = target_size
        self.use_target_size = use_target_size
        self.augmentation = augmentation
        self.data_dir = data_dir
        self.phase = phase

        with open(os.path.join(data_dir, f'DFG-tsd-annot-json/{phase}.json')) as f:
            labels_file = json.load(f)

        self.image_files = []
        self.image_ids = []
        self.labels = {}

        for image in labels_file['images']:
            img_id = image['id']
            if img_id in [ann['image_id'] for ann in labels_file['annotations']]:  
                self.image_files.append(image['file_name'])
                self.image_ids.append(img_id)

        self.compute_labels(labels_file)

    def compute_labels(self, labels_file):
        """Reads the annotations and stores valid bounding boxes in a dictionary"""
        for annotation in labels_file['annotations']:
            img_id = annotation['image_id']
            transformed_bbox = self.transform_bbox(annotation['bbox'])

            if transformed_bbox is None:  #skip invalid box 
                continue

            if img_id not in self.labels:
                self.labels[img_id] = {'boxes': [], 'labels': []}

            self.labels[img_id]['boxes'].append(transformed_bbox)
            self.labels[img_id]['labels'].append(annotation['category_id'])

    def transform_bbox(self, bbox):
        """Transforms COCO-Bounding Box to (x1, y1, x2, y2)-format and validates it."""
        x1, y1, w, h = bbox
        if w <= 1 or h <= 1:
            return None  
        x2, y2 = x1 + w, y1 + h
        if x1 >= x2 or y1 >= y2:
            return None  

        return [x1, y1, x2, y2]

    def get_image(self, image_file):
        """Loads image or returns a blank image if not found"""
        img_path = os.path.join(self.data_dir, 'JPEGImages', image_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Warnung: Bild nicht gefunden: {img_path}")
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB Conversion

    def augment_image(self, image, boxes):
        """Applies random augmentations to the image and bounding boxes"""
        height, width = image.shape[:2]

        # === Random Cropping ===
        if self.augmentation:
            crop_x = int(width * random.uniform(0.01, 0.2))
            crop_y = int(height * random.uniform(0.01, 0.2))

            image = image[crop_y:, crop_x:]
            valid_boxes = []
            for xmin, ymin, xmax, ymax in boxes:
                xmin = max(0, xmin - crop_x)
                xmax = max(0, xmax - crop_x)
                ymin = max(0, ymin - crop_y)
                ymax = max(0, ymax - crop_y)

                if xmax > xmin and ymax > ymin:  #check if Box is still valid
                    valid_boxes.append([xmin, ymin, xmax, ymax])

            boxes = valid_boxes

        # === Gaussian Noise ===
        if self.augmentation and random.random() > 0.5:
            noise = np.random.normal(0, 0.03 * 255, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)

        # === Gaussian Blur ===
        if self.augmentation and random.random() > 0.5:
            image = cv2.GaussianBlur(image, (5, 5), 0.8)  

        return image, boxes


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """loads image and according labels"""
        image_file = self.image_files[index]
        image = self.get_image(image_file)

        #if there are no labels, skip the image
        label = self.labels.get(self.image_ids[index], None)
        if label is None or len(label['boxes']) == 0:
            return self.__getitem__((index + 1) % len(self))

        boxes = label['boxes']
        labels = label['labels']

        #apply augmentations
        image, boxes = self.augment_image(image, boxes)


        current_height, current_width = image.shape[:2]

        # === scalation for Target Size ===
        if self.use_target_size and (current_width != self.target_size or current_height != self.target_size):
            scale_x = self.target_size / current_width
            scale_y = self.target_size / current_height

            #scale bboxes
            boxes = [[xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y] for xmin, ymin, xmax, ymax in boxes]

            #resize image
            image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        #convert to PyTorch-Tensors
        image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.empty((0,), dtype=torch.long)

        #if there is no valid bbox left, skip image
        if boxes.shape[0] == 0:
            return self.__getitem__((index + 1) % len(self))

        target = {'boxes': boxes, 'labels': labels}
        return image, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
