import torch
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from core.serialization import save_yaml
from core.hparams import HParams
from datasets.loaders import DataLoader
from modules.base import Base
from modules.wrapper import Wrapper


class Trainer(Base):
    @classmethod
    def from_args(cls, args):
        return cls(
            exp_name=args.exp_name,
            root_dir=args.root_dir,
            dataset_name=args.dataset_name,
            reduced=args.reduced,
            hparams=HParams.from_file(args.hparams_file),
            gpu=args.gpu if torch.cuda.is_available() else None,
            debug=args.debug)

    def __init__(self, exp_name, root_dir, dataset_name, reduced, hparams, gpu, debug):
        super().__init__(exp_name, root_dir, dataset_name, reduced, hparams, gpu, debug)
        self.dump()

    def train(self):
        logger = TensorBoardLogger(save_dir=self.dirs.exp, name="", version="logs")
        ckpt_callback = ModelCheckpoint(filepath=self.dirs.ckpt, save_top_k=-1)
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=10,
            verbose=False,
            mode='min'
        )

        wrapper = Wrapper(
            hparams=self.hparams,
            mapper=self.dataset.mapper,
            reduced=self.reduced)

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stop_callback],
            checkpoint_callback=ckpt_callback,
            max_epochs=self.hparams.num_epochs,
            gradient_clip_val=self.hparams.clipping,
            progress_bar_refresh_rate=10,
            gpus=[self.gpu] if self.gpu is not None else None)

        loader = DataLoader(self.hparams, self.dataset)
        train_loader = loader(partition="train", shuffle=True)
        val_loader = loader(partition="val", shuffle=False)

        trainer.fit(wrapper, train_dataloader=train_loader, val_dataloaders=val_loader)

    def dump(self):
        config = {
            "exp_name": self.exp_name,
            "root_dir": self.dirs.root.as_posix(),
            "dataset_name": self.dataset_name,
            "reduced": self.reduced,
            "hparams": self.hparams.__dict__,
            "gpu": self.gpu,
            "debug": self.debug
        }
        save_yaml(config, self.dirs.exp / "config.yml")