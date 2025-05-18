from data.DFG.dfg_dataset import DFG_Dataset
from data.augmentation import DataAugmentation
import numpy as np
import torch

from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights, _utils
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.utils.data import DataLoader

from torchmetrics.detection import MeanAveragePrecision

import lightning as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.fabric.utilities.seed import seed_everything

seed_everything(1234)

def create_model(size=300):
    # Load the Torchvision pretrained model.
    model = ssd300_vgg16(num_classes=91, weights=SSD300_VGG16_Weights.COCO_V1)
    # Retrieve the list of input channels. 
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    # List containing number of anchors based on aspect ratios.
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        #200 with all classes, 165 with only right of way and classes >= 20 elements
        num_classes=200,
    )
    # Image size for transforms.
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model

class Detector(pl.LightningModule):
    def __init__(self, num_epochs = 50, **kwargs):
        super().__init__()

        #set model to ssd with vgg16 backbone, batch size & Metric for the evaluation
        #ToDo: we should choose the same metric (experiment) & sane batch size (experiment with batch size & epochs)
        self.model = create_model(1080)
        self.transform = DataAugmentation()  # per batch augmentation_kornia
        self.batch_size = 16
        self.num_epochs = num_epochs
        self.metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)


    
    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.train_dataset = DFG_Dataset('/graphics/scratch2/students/kornwolfd/data_RoadSigns/dfg/', phase='train')
        self.val_dataset = DFG_Dataset('/graphics/scratch2/students/kornwolfd/data_RoadSigns/dfg/', phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, collate_fn=self.train_dataset.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.val_dataset.collate_fn)
    
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     x, y = batch
    #     if self.trainer.training:
    #         for i, img in enumerate(x):
    #             x[i] = self.transform(img)  # => we perform GPU/Batched data augmentation
    #     return x, y

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('Loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = net.model(images)
        self.metric.update(outputs, targets)
        return outputs
    
    #!!!!!!!!!!!!!!!!!!!metric.reset to avoid memory leak!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def on_validation_epoch_end(self):
        result = self.metric.compute()
        print(result)
        #ToDo: Look at AP of classes?!
        self.log('mAP_50', result['map_50'])
        self.metric.reset()

    def total_steps(self) -> int:
        return int((np.ceil(len(self.train_dataset)/self.batch_size))*self.num_epochs)


    def configure_optimizers(self):
        #ToDo: read Adam
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)#what about eps, weight_decay, maximize
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        #ToDo: read about scheduler, experiment
        #after every batch. step should be called after a batch has been used for training.

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.total_steps(), max_lr=0.01, pct_start=0.1, anneal_strategy='cos', cycle_momentum= True, base_momentum= 0.85, max_momentum= 0.95, div_factor= 25.0, final_div_factor= 1000.0, last_epoch=-1)
        return [optimizer], [scheduler]
        #return [optimizer]


net = Detector(50)
checkpoint_callback = ModelCheckpoint(dirpath='/graphics/scratch2/students/kornwolfd/checkpoint_RoadSigns/DFG_firstExperiments', monitor="mAP_50", filename='{epoch}-{mAP_50:.3f}', mode='max')
lr_monitor = LearningRateMonitor(logging_interval='step')
tb_logger = pl_loggers.TensorBoardLogger(save_dir="/graphics/scratch2/students/kornwolfd/lightningLogs_RoadSigns", version='AdamExp01')
trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=net.num_epochs, num_sanity_val_steps=2, callbacks=[checkpoint_callback, lr_monitor], gradient_clip_val=None, deterministic=True, logger=tb_logger)
trainer.fit(net)
