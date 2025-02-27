import torch.nn as nn
import torch
import cv2
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
        self.transforms = AugmentationSequential(
            RandomGaussianNoise(mean=0.0, std=0.07, p=1.0, keepdim=True),
            RandomGaussianBlur((5,5), (0.8, 0.8), p=1.0, keepdim=True),
            RandomAffine(degrees=0.0, translate=(0.1, 0.1), padding_mode=0, p=1.0, keepdim=True),
            RandomAffine(degrees=0.0, scale=(0.9, 1.1), padding_mode=0, p=1.0, keepdim=True),
            data_keys=['input', 'bbox_xyxy'])

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.transforms(x,y)
        
class DataAugmentationGrid(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, stddev) -> None:
        super().__init__()
        self.stddev = stddev
        self.myGenerator = Generator()
        self.uniqueFilename = 0
        self.transforms = nn.ModuleList([
        nn.Sequential(
            RandomGaussianNoise(mean=0.0, std=self.stddev, p=1., keepdim=True),
            RandomGaussianBlur((5,5), (0.1, 4.), p=0., keepdim=True)
            )])

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        randomNumber = rand(1, generator=self.myGenerator)
        if randomNumber < 0.5 and False:
            # transform to BxCxWxH for the net
            return x, y
        else:
            transform_idx = 0 #np.random.randint(low=0, high=len(self.transforms)+1)
            #tensor = self.transforms[transform_idx](x)
    
            #np_image = tensor.numpy()*255
            # Convert the numpy array to a cv2 image
           # cv2_image = np.transpose(np_image, (1, 2, 0))
            #cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

            #cv2.imwrite('/graphics/scratch2/students/kornwolfd/data_RoadSigns/dfg/augmentedImages/image_'+str(self.uniqueFilename)+'.jpg', cv2_image)
            #self.uniqueFilename += 1
            return self.transforms[transform_idx](x), y

class DataAugmentationBlur(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, sigma) -> None:
        super().__init__()
        self.transforms = RandomGaussianBlur(kernel_size=(3,3), sigma=(sigma, sigma), border_type='constant', p=1.0, keepdim=True)
            

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.transforms(x), y

class DataAugmentationTranslate(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, translationFactor) -> None:
        super().__init__()
        self.transforms = AugmentationSequential(
            RandomAffine(degrees=0.0, translate=(translationFactor, translationFactor), padding_mode=0, p=1.0, keepdim=True),
            data_keys=['input', 'bbox_xyxy']
        )
            

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.transforms(x, y)
    
class DataAugmentationScale(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, scalingFactor) -> None:
        super().__init__()
        self.transforms = AugmentationSequential(
            RandomAffine(degrees=0.0, scale=(1.0-scalingFactor, 1.0+scalingFactor), padding_mode=0, p=1.0, keepdim=True),
            data_keys=['input', 'bbox_xyxy']
        )
            

    @torch.no_grad()
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.transforms(x, y)