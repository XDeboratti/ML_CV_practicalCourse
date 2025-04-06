import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from frcnn_engine import train_one_epoch, evaluate
from torchvision.transforms import functional as F
import frcnn_utils
from frcnn_data import CustomDataset #Mapillary
from frcnn_data import DFG_Dataset #DFG

import os
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from frcnn_continue_training import continue_training
from torch.optim.lr_scheduler import ReduceLROnPlateau



def main(model_name, backbone_layers, num_epochs, start_lr, every_num_epoch, lr_updater, augmentation):

    # fix the seeds for torch and numpy (where applicable) to ensure reproducibility
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)


    #paths
    TRAIN_FILE = "train"
    VAL_FILE = "val"
    MODEL_NAME = model_name

    # Hyperparameter
    batch_size = 12
    num_epochs = num_epochs
    learning_rate = start_lr
    #num_classes = 4 = 3 Classes + Background #12 Experiment
    num_classes = 200
    num_workers = 6
    optimizer_type = "SGD"
    schedular = "none"


    train_images = os.path.join(TRAIN_FILE, "images")
    train_annotations = os.path.join(TRAIN_FILE, "annotations")
    val_images = os.path.join(VAL_FILE, "images")
    val_annotations = os.path.join(VAL_FILE, "annotations")
    checkpoints_file = "checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset und DataLoader
    # Mapillary Dataset
    #train_dataset = CustomDataset(train_images, train_annotations, use_target_size=True, augmentation=augmentation)
    #val_dataset = CustomDataset(val_images, val_annotations, use_target_size=True, augmentation=False)

    # Dataset and DataLoader
    # DFG Dataset
    train_dataset = DFG_Dataset(data_dir="DFG/", phase="train", target_size=1200, use_target_size=True, augmentation=True)
    val_dataset = DFG_Dataset(data_dir="DFG/", phase="test", target_size=800, use_target_size=False, augmentation=False)
    #Mapillary
    '''
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.collate_fn,
        num_workers=num_workers,
        drop_last=False
    )
    '''
    #DFG
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DFG_Dataset.collate_fn,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DFG_Dataset.collate_fn,
        num_workers=num_workers,
    )

    #load model and adjust
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, trainable_backbone_layers=backbone_layers)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    print('model on device',device)
    model.to(device)

    # Optimizer und Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    # Scheduler: StepLR
    


    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    #scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5,min_lr=1e-6, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 75], gamma=0.1)


    validation_losses = {}
    all_model_metrics = {}
    new_lr = learning_rate
    #training und Evaluierung
    for epoch in range(num_epochs):
        #half the learning rate every 10 epochs
        
        #if epoch > 0 and epoch % every_num_epoch == 0: #half the learning rate every 5 epochs
        #    new_lr = new_lr * lr_updater

        #print(f"\nEpoch {epoch + 1}/{num_epochs}")
        #print(f"Learning Rate: {new_lr}")
        # update optimizer learning rate
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = new_lr

        # Training
        epoch_train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=874)
        print(f"Training: {epoch_train_loss}")
        #training_losses[f"epoch_{epoch}"] = epoch_train_loss

        # save model
        if epoch % 5 == 0: #every epoch
            current_model_name = f"{MODEL_NAME}_{epoch}"
            checkpoint_path = os.path.join(checkpoints_file, f"{current_model_name}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }, checkpoint_path)
        
        # evaluierung
        eval_metrics = evaluate(model, val_loader, device,confidence_threshold=0.0)
        
        validation_map = eval_metrics.get('validation_map', 0)

        print(f"Validation: {eval_metrics}")

        #scheduler.step(validation_map)
        scheduler.step()
        
        validation_loss = eval_metrics.pop("validation_loss", None) 
        validation_losses[f"epoch_{epoch}"] = validation_loss

        # save metrics
        all_model_metrics[f"epoch_{epoch}"] = {
            "training_loss": epoch_train_loss,
            "validation_loss": validation_loss,
            **eval_metrics
        }
        epoch_metrics_path = os.path.join("evaluation_results", f"metrics_epoch_{epoch}.json")
        os.makedirs("evaluation_results", exist_ok=True)
        with open(epoch_metrics_path, "w") as f:
            json.dump(all_model_metrics[f"epoch_{epoch}"], f, indent=4)

        print(f"metrics for epoch {epoch} saved in: {epoch_metrics_path}")
        

    #save all metrics
    model_metrics_path = os.path.join("evaluation_results", f"metrics_{MODEL_NAME}.json")
    with open(model_metrics_path, "w") as f:
        json.dump(all_model_metrics, f, indent=4)

    print(f"All metrics for model {MODEL_NAME} saved in: {model_metrics_path}")


    model_details = {
        "model_name": MODEL_NAME,
        "train_file": TRAIN_FILE,
        "val_file": VAL_FILE,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": {
            "type": optimizer_type,
            "lr": learning_rate,    
        },
        "scheduler": {
            "type": schedular,
            "t_max": num_epochs,
            "eta_min": 0,
        },
        "device": str(device),
        "pretrained_weights": str(weights)
    }

    TRAINING_PARAMETERS_FILE = "training_parameters"
    json_path = os.path.join(TRAINING_PARAMETERS_FILE, f"{MODEL_NAME}_details.json")
    with open(json_path, "w") as json_file:
        json.dump(model_details, json_file, indent=4)
    
    
    print(f"Model details saved to {json_path}")


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_random_samples(data_loader, num_samples=5):
    """ Shows random images with bounding boxes from the DataLoader. """
    for i, (images, targets) in enumerate(data_loader):
        if i >= num_samples:
            break

        img = images[0].permute(1, 2, 0).numpy()  # (C, H, W) zu (H, W, C)
        target = targets[0]
        boxes = target['boxes'].numpy()

        # Plot image
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # add bboxes
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

        plt.show()

if __name__ == '__main__':
    print("-----starting training-----")
    '''
    train_loader = DataLoader(
        CustomDataset("train/images", "train/annotations", use_target_size=True, augmentation=True),
        batch_size=12,
        shuffle=True,
        collate_fn=utils.collate_fn,
        num_workers=8,
    )
    
    #visualize_random_samples(train_loader, num_samples=10)
    '''
    #main("model_1_aug_higher_res",5, 80, 0.02, 1, 1, True)
    #main("model_1_noaug",5, 50, 0.02, 1, 1, False)
    #main("model_1_noaug",5, 50, 0.02, 1, 1, False)
    #main("model_2_aug_long_run_moreclasses",5, 100, 0.02, 1, 1, True)
    
    main("DFG_AUG_5_LAYERS_COSINE_ANNEALING_LONG_RUN", 5, 100, 0.02, 1, 1, True)
