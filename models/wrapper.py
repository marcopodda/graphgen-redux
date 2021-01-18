from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl

from core.hparams import HParams


class BaseWrapper(pl.LightningModule):
    model_class = None

    def __init__(self, hparams, mapper):
        super().__init__()
        self.hparams = HParams.load(hparams)
        self.model = self.model_class(hparams, mapper)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-5)
        scheduler = MultiStepLR(optimizer, milestones=self.hparams.milestones)
        return [optimizer], [scheduler]

    def forward(self, batch):
        return self.model(batch)

    def shared_step(self, batch, batch_idx):
        loss = self(batch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)