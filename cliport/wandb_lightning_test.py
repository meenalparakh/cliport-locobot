from tqdm.notebook import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision
from math import floor
import pytorch_lightning as pl
from torch.utils.data import random_split
# from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class ToyDataset(Dataset):
    def __init__(self, length = 4096):
        self.length = length
        self.x = np.random.rand(self.length, 2).astype(float)
        self.y = np.sum(self.x, axis = 1) + np.random.normal(loc = 0, scale = 0.01, size = (self.length,)).astype(float)

        #todo normaiee here
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class BasicNetwork(pl.LightningModule):
    def __init__(self, lr):
        super(BasicNetwork, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(2, 8),
                        nn.ReLU(),
                        nn.Linear(8, 1)
                        )
        self.lr = lr
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, x):
        # print(f'In forward step, shape of inuput: {ob.shape}')
        out = self.net(x).squeeze(1)
        return out

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                        lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        train_x, train_y = train_batch
        output = self.forward(train_x.float())
        loss = self.loss_fn(output, train_y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        val_x, val_y = val_batch
        output = self.forward(val_x.float())
        loss = self.loss_fn(output, val_y.float())
        self.log("val_loss", loss)

    def test_step(self, test_batch, batch_idx):
        test_x, test_y = test_batch
        output = self.forward(test_x.float())
        loss = self.loss_fn(output, test_y.float())
        self.log("test_loss", loss)


def train_agent(logs_filename, max_epochs=10, batch_size=16, lr=0.001,
                device="cpu", delta_error=0.0001, patience=5):

    model = BasicNetwork(lr)
    model = model.float()

    dataset_train = ToyDataset(length = 4096)
    dataset_val = ToyDataset(length = 256)
    dataset_test = ToyDataset(length = 1024)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=8)

    wandb_logger = WandbLogger(project="test-script")
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

    trainer = pl.Trainer(callbacks=[early_stop_callback],
                         logger=wandb_logger,
                         gpus=0,
                         # precision='bf16',
                         limit_train_batches=0.5,
                         max_epochs=max_epochs)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader, ckpt_path = 'best')

    return model

if __name__ == '__main__':
    train_agent('lightning_logs_fname')
