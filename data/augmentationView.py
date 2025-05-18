import torch.nn as nn
import torch
import cv2
import json
import numpy as np
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import RandomGaussianNoise, RandomGaussianBlur, RandomAffine, AugmentationSequential
from torch import Tensor, Generator, rand
from lightning.fabric.utilities.seed import seed_everything

seed_everything(1234)
        
class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.myGenerator = Generator()
        self.uniqueFilename = 0
        self.transforms = nn.ModuleList([
        nn.Sequential(
            RandomGaussianNoise(mean=0.0, std=0.15, p=0.0, keepdim=True),
            RandomGaussianBlur(kernel_size=(7,7), sigma=(1., 1.), border_type='constant', p=0.0, keepdim=True),
            RandomAffine(degrees=0.0, translate=(0.5, 0.5), padding_mode=0, p=0.0, keepdim=True),
            RandomAffine(degrees=0.0, scale=(0.5, 1.5), padding_mode=0, p=1.0, keepdim=True)
            )])

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        tensor = self.transforms[0](x)

        np_image = tensor.numpy()*255
        # Convert the numpy array to a cv2 image
        cv2_image = np.transpose(np_image, (1, 2, 0))

        cv2.imwrite('./aug'+'.jpg', cv2_image)
        self.uniqueFilename += 1

    

    


data_dir = '.'
phase = 'train'

#load and normlize the image
def get_image(image_file):
    return cv2.imread(data_dir+'JPEGImages/'+image_file)/255


with open(data_dir+'DFG-tsd-annot-json/'+phase+'.json') as f:
    labels_file = json.load(f)

#store filenames and ids (filenames for ids) if we have id 0, the name of the file can be found in image_files[0]
#ToDo: What if an ID/name is missing?! What if there ar not equally many?
image_files = [image['file_name'] for image in labels_file['images']]
image_ids = [image['id'] for image in labels_file['images']]

augmentor = DataAugmentation()


#imread gives us a tensor of shape width x length x color, we need color x length x width for the augmentation, cast to float tensor
image = torch.as_tensor(get_image('0000197.jpg')).permute((2, 0, 1)).float()
image = augmentor(image)
