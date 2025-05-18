import numpy as np
import cv2
import json
import torch
import glob
import sys
import os

from torchvision.models.detection import ssd300_vgg16

from torch.utils.data import DataLoader, Dataset
import lightning as pl
from torchmetrics.detection import MeanAveragePrecision
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.fabric.utilities.seed import seed_everything
from ..augmentation import DataAugmentationScale, DataAugmentationGrid, DataAugmentationBlur, DataAugmentationTranslate, DataAugmentation

seed_everything(1234)


class Mapillary_Dataset(Dataset):
    def __init__(self, aug_parameter, data_dir, phase='train'):
        super().__init__()
        self.aug_parameter = aug_parameter
        #self.augmentor = DataAugmentation()
        self.data_dir = data_dir+phase+'ing/'+phase
        self.phase = phase

        self.path_to_annotations = self.data_dir+'/annotations/'
        self.path_to_images = '/graphics/scratch2/students/kornwolfd/ML_CV_practicalCourse/data_RoadSigns/mapillary/images/'

        self.images = []
        self.labels = {}

        self.label_to_id = {"regulatory--stop--g1": 1, "regulatory--yield--g1": 2, "regulatory--priority-road--g4": 3}
        
        for file in glob.glob(os.path.join(self.path_to_annotations, '*.json')):

            base_name = os.path.basename(file).replace('.json', '')
            if not os.path.exists(self.path_to_images + base_name + '.jpg'):
                continue
            self.images.append(base_name)
            with open(file) as f:
                annotation_file = json.load(f)

            for obj in annotation_file['objects']:
                if self.label_to_id.get(obj['label'], None) is not None:
                    if self.labels.get(base_name, None) is None:
                        self.labels[base_name] = {'class': [self.label_to_id[obj['label']]], 'boxes': [(obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax'])]}
                    else:
                        self.labels[base_name]['class'].append(self.label_to_id[obj['label']])
                        self.labels[base_name]['boxes'].append((obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']))
    
    def __len__(self):
        return len(self.images)
    
    #get image and corresponding target
    def __getitem__(self, index):    
        label = self.labels.get(self.images[index], None)
        if label is None: #if we don't have any labels move forward to the next picture (there are images in the dataset where there are signs but no labels)
            return self.__getitem__((index+1)%len(self))
        #cast the labels to tensors and return    
        target = {'boxes': torch.as_tensor(label['boxes']).float(), 'labels': torch.as_tensor(label['class']).long()}
        image_path = self.path_to_images + self.images[index] + '.jpg'
        #try:
            #assert os.path.exists(image_path)
        #except:
            #print(image_path)
            #return self.__getitem__((index+1)%len(self))
        image = cv2.imread(image_path)/255
        image = torch.as_tensor(image).permute((2, 0, 1)).float()


        #if self.phase == 'train':
            #image, target['boxes'] = self.augmentor(image, target['boxes'])
        return image, target
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    