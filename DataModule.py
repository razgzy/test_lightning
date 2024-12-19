import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

class TestDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 100

    def __getitem__(self, index):
        x = torch.tensor([1.0], dtype=torch.float32)
        return x

class TestDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 4, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = TestDataset()

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=3)
        return dataloader
    
    def val_dataloader(self):
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True)
        return dataloader
    
