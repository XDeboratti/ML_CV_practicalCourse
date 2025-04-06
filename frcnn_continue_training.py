import torch
import os
import json
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from frcnn_engine import train_one_epoch, evaluate
from frcnn_data import CustomDataset
import frcnn_utils
import numpy as np

def continue_training(checkpoint_path,backbone_layers, model_name, num_epochs, start_lr, every_num_epoch, lr_updater, augmentation):

    #fix seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #paths and parameters
    TRAIN_FILE = "train"
    VAL_FILE = "val"
    MODEL_NAME = model_name

    batch_size = 12
    num_classes = 4  # 3 Classes + Background
    num_workers = 8

    train_images = os.path.join(TRAIN_FILE, "images")
    train_annotations = os.path.join(TRAIN_FILE, "annotations")
    val_images = os.path.join(VAL_FILE, "images")
    val_annotations = os.path.join(VAL_FILE, "annotations")
    checkpoints_file = "checkpoints"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    train_dataset = CustomDataset(train_images, train_annotations, use_target_size=True, augmentation=False)
    val_dataset = CustomDataset(val_images, val_annotations, use_target_size=True, augmentation=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=frcnn_utils.collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=frcnn_utils.collate_fn,
        num_workers=num_workers,
        drop_last=False
    )

    #load model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn_v2(weights=weights,trainable_backbone_layers=backbone_layers)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    model.to(device)


    #load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    #optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=start_lr, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    #load existing metrics if available
    metrics_path = os.path.join("evaluation_results", f"metrics_{MODEL_NAME}.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            all_model_metrics = json.load(f)
    else:
        all_model_metrics = {}

    #training and evaluation loop
    new_lr = start_lr
    for epoch in range(start_epoch, start_epoch + num_epochs):
        if epoch > 0 and epoch % every_num_epoch == 0:
            new_lr *= lr_updater
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        print(f"\nEpoch {epoch}/{start_epoch + num_epochs - 1}")
        print(f"Learning Rate: {new_lr}")

        #training
        epoch_train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1000)
        print(f"Training Loss: {epoch_train_loss}")

        #save checkpoint
        if epoch % 1 == 0:
            current_model_name = f"{MODEL_NAME}_{epoch}"
            checkpoint_path = os.path.join(checkpoints_file, f"{current_model_name}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }, checkpoint_path)

        #evaluation
        eval_metrics = evaluate(model, val_loader, device, confidence_threshold=0.0)
        print(f"Validation Metrics: {eval_metrics}")

        validation_loss = eval_metrics.pop("validation_loss", None)
        all_model_metrics[f"epoch_{epoch}"] = {
            "training_loss": epoch_train_loss,
            "validation_loss": validation_loss,
            **eval_metrics,
        }

        #save metrics
        epoch_metrics_path = os.path.join("evaluation_results", f"metrics_epoch_{epoch}.json")
        os.makedirs("evaluation_results", exist_ok=True)
        with open(epoch_metrics_path, "w") as f:
            json.dump(all_model_metrics[f"epoch_{epoch}"], f, indent=4)

        print(f"Metrics for epoch {epoch} saved to: {epoch_metrics_path}")

    #save all metrics
    with open(metrics_path, "w") as f:
        json.dump(all_model_metrics, f, indent=4)
    print(f"All metrics saved to: {metrics_path}")


