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
from ..augmentation import DataAugmentationScale, DataAugmentationGrid, DataAugmentationBlur, DataAugmentationTranslate

seed_everything(1234)


class DFG_Dataset(Dataset):
    def __init__(self, aug_parameter, data_dir, phase='train'):
        super().__init__()
        self.data_dir = data_dir
        self.phase = phase

        with open(data_dir+'DFG-tsd-annot-json/'+phase+'.json') as f:
            labels_file = json.load(f)

        #store filenames and ids (filenames for ids) if we have id 0, the name of the file can be found in image_files[0]
        #ToDo: What if an ID/name is missing?! What if there ar not equally many?
        self.image_files = [image['file_name'] for image in labels_file['images']]
        self.image_ids = [image['id'] for image in labels_file['images']]

        #the json is split into 
        # - images[id, height, width, file_name]
        # - categories[id, name, supercategory] 
        # - annotations[id, area, bbox, category_id, segmentation, image_id, ignore, iscrowd]
        #we need the labels in the following form: [List[Dict{str, Tensor}]]
        # - where the list index corresponds to the index of the described image in the image list
        # - the dictionary has 'boxes' and 'class'
        # - 'boxes' is the key for a tensor containing the boxes
        # - 'class' is the key for a tensor containing the classes for the boxes 
        self.labels = {}
        self.compute_labels(labels_file)
        self.aug_parameter = aug_parameter
        self.augmentor = DataAugmentationTranslate(self.aug_parameter)

    def compute_labels(self, labels_file):
        annotations = labels_file['annotations']
        for annotation in annotations:
            if self.labels.get(annotation['image_id'], None) is None: #we didn't see any annotations for this image yet
                self.labels[annotation['image_id']] = {'boxes': [self.transform_bbox(annotation['bbox'])], 'class': [annotation['category_id']]}
            else: #we already saw annotations for this image
                self.labels[annotation['image_id']]['boxes'].append(self.transform_bbox(annotation['bbox']))
                self.labels[annotation['image_id']]['class'].append(annotation['category_id'])

    def compute_labels_reducedClasses(self, labels_file):
        annotations = labels_file['annotations']
        classes = {} 
        annotation_count = [x['category_id'] for x in annotations]
        annotation_count = np.bincount(np.array(annotation_count))
        merge_index = annotation_count <= 20
        merge_index[38:82] = False
        merge_index[161:171] = False
        merge_id_mapping = np.arange(0, 200)
        merge_id_mapping[merge_index] = 11

        class_shift = np.insert(np.cumsum(merge_index), 0, 0)[:len(merge_id_mapping)]
        class_shift[merge_index] = 0
        class_shift -= 1
        class_shift[class_shift < 0] = 0
        merge_id_mapping -= class_shift

        for annotation in annotations:
            if classes.get(annotation['category_id'], None) is None:
                classes[annotation['category_id']] = 1
            else:
                classes[annotation['category_id']] += 1

        sortedClasses = dict(sorted(classes.items(), key = lambda item: item[0]))

        rightOfWayClasses = []
        for x in range(38,82):
            rightOfWayClasses.append(x)
        for x in range(161,171):
            rightOfWayClasses.append(x)

        intermedClassMapping = {}
        for c in sortedClasses:
            if c in rightOfWayClasses:
                    intermedClassMapping[c] = c
            elif sortedClasses[c] <= 20:
                    intermedClassMapping[c] = 300
            else:
                    intermedClassMapping[c] = c

        newSet = set(intermedClassMapping.values())
        numDifferentClasses = len(newSet)

        classMapping = {}
        counter = 0
        for c in intermedClassMapping:
            if intermedClassMapping[c] != 300:
                    classMapping[c] = counter
                    counter += 1
            elif intermedClassMapping[c] == 300 and counter == 11:
                    classMapping[c] = 11
                    counter += 1
            else:
                    classMapping[c] = 11
        
        labels_new = {}

        for annotation in annotations:
            if self.labels.get(annotation['image_id'], None) is None: #we didn't see any annotations for this image yet
                self.labels[annotation['image_id']] = {'boxes': [self.transform_bbox(annotation['bbox'])], 'class': [classMapping[annotation['category_id']]]}
                labels_new[annotation['image_id']] = {'boxes': [self.transform_bbox(annotation['bbox'])], 'class': [merge_id_mapping[annotation['category_id']]]}
            else: #we already saw annotations for this image
                self.labels[annotation['image_id']]['boxes'].append(self.transform_bbox(annotation['bbox']))
                self.labels[annotation['image_id']]['class'].append(classMapping[annotation['category_id']])

                labels_new[annotation['image_id']]['boxes'].append(self.transform_bbox(annotation['bbox']))
                labels_new[annotation['image_id']]['class'].append(merge_id_mapping[annotation['category_id']])
        #aaaaaa = self.labels == labels_new
        #print()
    
    #boxes are provided in coco format bbx[x, y, w, h] where x & y are the coordinates of the upper left corner
    #we need the boxes in bbx[x1, y1, x2, y2] where 1 is the upper left corner and y is the lower right corner of the box
    def transform_bbox(self, bbox):
        return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
    
    #load and normlize the image
    def get_image(self, image_file):
        return cv2.imread(self.data_dir+'JPEGImages/'+image_file)/255
    
    #draw bounding boxes into the respective image
    def drawImageWBbox(self, image_file, label):
        #just for viewing what's happening
        # Reading an image in default mode
        showImage = self.get_image(image_file)*255
        
        # Window name in which image is displayed
        window_name = 'Image'
        
        for box in label['boxes']:
            # Start coordinate, here (5, 5)
            # represents the top left corner of rectangle
            start_point = (box[0], box[1])

            # Ending coordinate, here (220, 220)
            # represents the bottom right corner of rectangle
            end_point = (box[2], box[3])

            # Green color in BGR
            color = (0, 255, 0)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            showImage = cv2.rectangle(showImage, start_point, end_point, color, thickness)

        # Displaying the image 
        cv2.imwrite(self.data_dir+'imagesBbox/'+image_file, showImage) 
        #cv2.imshow(window_name, showImage) 

    
    def __len__(self):
        return len(self.image_ids)
    
    #get image and corresponding target
    def __getitem__(self, index):
        image_file = self.image_files[index]
        #imread gives us a tensor of shape width x length x color, we need color x length x width for the augmentation, cast to float tensor
        image = torch.as_tensor(self.get_image(image_file)).permute((2, 0, 1)).float()
        
        
        label = self.labels.get(self.image_ids[index], None)
        if label is None: #if we don't have any labels move forward to the next picture (there are images in the dataset where there are signs but no labels)
            return self.__getitem__((index+1)%len(self))
        #cast the labels to tensors and return    
        target = {'boxes': torch.as_tensor(label['boxes']).float(), 'labels': torch.as_tensor(label['class']).long()}

        # for x in target['labels']:
        #      if x == 0:
        #           self.drawImageWBbox(image_file, label)

        if self.phase == 'train':
            image, target['boxes'] = self.augmentor(image, target['boxes'])
        return image, target
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    