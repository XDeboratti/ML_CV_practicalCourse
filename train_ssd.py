from data.DFG.dfg_dataset import DFG_Dataset
import numpy as np
import torch

from torchvision.models.detection import ssd300_vgg16

from torch.utils.data import DataLoader
import lightning as pl
from torchmetrics.detection import MeanAveragePrecision
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.fabric.utilities.seed import seed_everything

seed_everything(1234)

#ToDo: geht das nicht in die detector klasse? muss ich dann self übergeben? wie bei transform box?
def collate_fn(batch):
    return tuple(zip(*batch))

class Detector(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        #set model to ssd with vgg16 backbone, batch size & Metric for the evaluation
        #ToDo: we should choose the same metric (experiment) & sane batch size (experiment with batch size & epochs)
        self.model = ssd300_vgg16(num_classes=200)
        self.batch_size = 16
        self.metric = MeanAveragePrecision(iou_type="bbox")
    
    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.train_dataset = DFG_Dataset('/graphics/scratch2/students/kornwolfd/data_RoadSigns/dfg/', phase='train')
        self.val_dataset = DFG_Dataset('/graphics/scratch2/students/kornwolfd/data_RoadSigns/dfg/', phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=8, collate_fn=collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    #ToDo: What does batch_idx do here?
    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('Loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = net.model(images)
        self.metric.update(outputs, targets)
        return outputs
    
    def on_validation_epoch_end(self):
        result = self.metric.compute()
        print(result)
        self.log('AP_50', result['map_50'])

    def total_steps(self) -> int:
        #why *50? sieht der so nicht manche Bilder öfter?
        return int((np.ceil(len(self.train_dataset)/self.batch_size))*50)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=self.total_steps(), max_lr=0.01, pct_start=0.1, anneal_strategy='cos', cycle_momentum= True, base_momentum= 0.85, max_momentum= 0.95, div_factor= 25.0, final_div_factor= 10000.0, last_epoch=-1)
        return [optimizer], [scheduler]


net = Detector()
checkpoint_callback = ModelCheckpoint(dirpath='/graphics/scratch2/students/kornwolfd/checkpoint_RoadSigns/DFG_firstExperiments', save_top_k=5, monitor="AP_50", filename='{epoch}-{AP_50:.3f}', mode='max')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(accelerator='gpu', devices=[0], max_epochs=50, num_sanity_val_steps=2, callbacks=[checkpoint_callback, lr_monitor], gradient_clip_val=0, deterministic=True)
trainer.fit(net)