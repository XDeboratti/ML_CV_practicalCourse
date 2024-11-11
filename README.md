# Right-of-Way Sign Detection for Autonomous Driving

## Project Overview
This project focuses on developing an object detection model to recognize right-of-way traffic signs, a crucial component for autonomous vehicles to understand and follow traffic rules. Our goal is to build a minimum viable product (MVP) that detects right-of-way signs, with future extensions to include traffic light detection and state recognition.

## Objectives
- Detect and classify right-of-way signs with high accuracy.
- Potentially extend to additional traffic rule components, including traffic light detection and implementing complex right-of-way scenarios (e.g., "right before left" rule in German traffic).

## Datasets
We are exploring several datasets for traffic signs:
- [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset)
- [DFG Traffic Sign Dataset]([https://www.dfg.de/traffic-sign-dataset](https://www.vicos.si/resources/dfg/))
- EVOTEGRA (pending access)

## Models
We plan to test and evaluate multiple models, including:
- Convolutional Neural Networks (CNN): Faster R-CNN, with backbone options of [VGG16]([https://arxiv.org/abs/1409.1556](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)) or [ResNet50]([https://arxiv.org/abs/1512.03385](https://pytorch.org/vision/0.18/models/generated/torchvision.models.resnet50.html))
- Transformer-based model: [DETR](https://github.com/facebookresearch/detr)

The initial model testing phase will guide us in selecting the best dataset-model combinations for further training and fine-tuning.

## Timeline
The project is structured into phases:
1. **Dataset Collection and Analysis** (18.11 - 24.11)
2. **Model Selection and Initial Training** (25.11 - 01.12)
3. **Augmentation and Hyperparameter Tuning** (02.12 - 15.12)
4. **Fine-Tuning and Problem Solving** (16.12 - 22.12)
5. **Evaluation and Intermediate Presentation** (30.12 - 08.01)
6. **Final Extensions and Presentation Preparation** (09.01 - Project End)

## Team
This project is developed by a team of four members, including [@XDeboratti](https://github.com/XDeboratti),[@mqrek](https://github.com/mqrek),[@sh1negg](https://github.com/sh1negg),[] each responsible for specific model-dataset combinations, augmentations, and evaluation steps.

