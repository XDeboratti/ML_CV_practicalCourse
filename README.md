# Right-of-Way Sign Detection for Autonomous Driving

## Project Overview
This project focused on developing an object detection model to recognize right-of-way traffic signs, a crucial component for autonomous vehicles to understand and follow traffic rules. Our goal is to build a minimum viable product (MVP) that detects right-of-way signs, with potential for future extensions to include traffic light detection and state recognition.

## Objectives
- Detect and classify right-of-way signs with high accuracy.

## Datasets
We were exploring several datasets for traffic signs:
- [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset)
- [DFG Traffic Sign Dataset](https://www.vicos.si/resources/dfg/)

## Models
We tested and evaluated the following models:
- [Faster R-CNN](https://pytorch.org/vision/main/models/faster_rcnn.html) with a [ResNet50](https://pytorch.org/vision/0.18/models/generated/torchvision.models.resnet50.html) backbone
- [SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/), with a [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) backbone
- 
## Timeline
The project was structured into phases:
1. **Dataset Collection and Analysis** (18.11 - 24.11)
2. **Model Selection and Initial Training** (25.11 - 01.12)
3. **Augmentation and Hyperparameter Tuning** (02.12 - 15.12)
4. **Fine-Tuning and Problem Solving** (16.12 - 22.12)
5. **Evaluation and Intermediate Presentation** (30.12 - 08.01)
6. **Final Extensions and Presentation Preparation** (09.01 - Project End)

## Team
This project is developed by a team of two members: [@XDeboratti](https://github.com/XDeboratti) and [@mqrek](https://github.com/mqrek), with equal contribution. Both were responsible for training one model on both datasets, augmentations, and evaluation steps.
