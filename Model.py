import lightning as L
import torch
import torch.nn as nn
import time

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 16, 3, 1, 1)
    
    def forward(self, x):
        return self.layer(x)
    
class TestModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = Network()
        self.lr = 4e-4
        self.weight_decay = 1e-2
        self.eta_min = self.lr * 1e-3

    def forward(self, x):
        return self.layer(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.trainer.max_epochs, T_mult=1, eta_min=self.eta_min)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "name": 'AdamW'
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, batch, batch_idx):
        time.sleep(0.01)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        time.sleep(0.01)
        self.log('hp_metric', self.current_epoch)
    