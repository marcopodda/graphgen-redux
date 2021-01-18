import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl

from core.hparams import HParams


class GraphgenWrapper(pl.LightningModule):
    model_class = None

    def __init__(self, hparams, mapper):
        super().__init__()
        self.hparams = HParams.load(hparams)
        self.model = self.model_class(hparams, mapper)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-5)
        scheduler = MultiStepLR(optimizer, milestones=self.hparams.milestones)
        return [optimizer], [scheduler]

    def shared_step(self, batch, batch_idx):
        y_pred, y, lengths = self.model(batch)
        loss_sum = F.binary_cross_entropy(y_pred, y, reduction='none')
        loss = torch.mean(torch.sum(loss_sum, dim=[1, 2]) / (lengths.float() + 1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)