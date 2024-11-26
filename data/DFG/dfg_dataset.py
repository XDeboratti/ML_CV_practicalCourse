import numpy as np
import cv2
import json
import torch

from torchvision.models.detection import ssd300_vgg16

from torch.utils.data import DataLoader, Dataset
import lightning as pl
from torchmetrics.detection import MeanAveragePrecision
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.fabric.utilities.seed import seed_everything

seed_everything(1234)


class DFG_Dataset(Dataset):
    def __init__(self, data_dir, phase='train'):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase

        with open(data_dir+'DFG-tsd-annot-json/'+phase+'.json') as f:
            labels_file = json.load(f)

        #the json is split into 
        # - images[id, height, width, file_name]
        # - categories[id, name, supercategory] 
        # - annotations[id, area, bbox, category_id, segmentation, image_id, ignore, iscrowd]
        #we need the labels in the following form: [List[Dict{str, Tensor}]]
        # - where the list index corresponds to the index of the described image in the image list
        # - the dictionary has 'boxes' and 'class'
        # - 'boxes' is the key for a tensor containing the boxes
        # - 'class' is the key for a tensor containing the classes for the boxes 
        annotations = labels_file['annotations']
        self.labels = {}
        for annotation in annotations:
            if self.labels.get(annotation['image_id'], None) is None: #we didn't see any annotations for this image yet
                self.labels[annotation['image_id']] = {'boxes': [self.transform_bbox(annotation['bbox'])], 'class': [annotation['category_id']]}
            else: #we already saw annotations for this image
                self.labels[annotation['image_id']]['boxes'].append(self.transform_bbox(annotation['bbox']))
                self.labels[annotation['image_id']]['class'].append(annotation['category_id'])

        #store filenames and ids (filenames for ids) if we have id 0, the name of the file can be found in image_files[0]
        #ToDo: What if an ID/name is missing?! What if there ar not equally many?
        self.image_files = [image['file_name'] for image in labels_file['images']]
        self.image_ids = [image['id'] for image in labels_file['images']]
    
    #boxes are provided in coco format bbx[x, y, w, h] where x & y are the coordinates of the upper left corner
    #we need the boxes in bbx[x1, y1, x2, y2] where 1 is the upper left corner and y is the lower right corner of the box
    def transform_bbox(self, bbox):
        return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    
    #load and normlize the image
    def get_image(self, image_file):
        return cv2.imread(self.data_dir+'JPEGImages/'+image_file)/255
    
    def __len__(self):
        return len(self.image_ids)
    
    #get image and corresponding target
    def __getitem__(self, index):
        image_file = self.image_files[index]
        #imread gives us a tensor of shape width x length x color, we need color x width x length, cast to float tensor
        image = torch.as_tensor(self.get_image(image_file)).permute((2, 0, 1)).float()
        
        
        label = self.labels.get(self.image_ids[index], None)
        if label is None: #if we don't have any labels move forward to the next picture (there are images in the dataset where there are signs but no labels)
            return self.__getitem__((index+1)%len(self))
        #cast the labels to tensors and return    
        target = {'boxes': torch.as_tensor(label['boxes']).float(), 'labels': torch.as_tensor(label['class']).long()}

        return image, target
