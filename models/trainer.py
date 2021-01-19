import time
from pathlib import Path
from argparse import Namespace

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

from core.hparams import HParams
from core.module import BaseModule
from core.utils import get_or_create_dir, time_elapsed


class TimeElapsedCallback(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        elapsed = time_elapsed(self.start_time, time.time())
        path = self.log_dir / "time_elapsed.txt"
        with open(path, "w") as f:
            print(f"Time elapsed: {elapsed}", file=f)


class Trainer(BaseModule):
    loader_class = None

    @classmethod
    def from_args(cls, args):
        return cls(
            model_name=args.model_name,
            root_dir=args.root_dir,
            dataset_name=args.dataset_name,
            hparams=HParams.from_file(args.hparams_file),
            gpu=args.gpu if torch.cuda.is_available() else None)

    def _setup_dirs(self, root_dir):
        dirs = super()._setup_dirs(root_dir)
        dirs.logs = get_or_create_dir(dirs.exp / "logs")
        return dirs

    def __init__(self, model_name, root_dir, dataset_name, hparams, gpu):
        super().__init__(model_name, root_dir, dataset_name, hparams, gpu)
        self.dataset = self.dataset_class(dataset_name)
        self.dump()

    def train(self):
        logger = TensorBoardLogger(save_dir=self.dirs.exp, name="", version=self.dirs.logs.parts[-1])
        ckpt_callback = ModelCheckpoint(filepath=self.dirs.ckpt, save_top_k=-1)
        time_elapsed_callback = TimeElapsedCallback(log_dir=self.dirs.logs)

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            patience=1000,
            verbose=False,
            mode='min'
        )

        wrapper = self.get_wrapper()

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stop_callback, time_elapsed_callback],
            checkpoint_callback=ckpt_callback,
            max_epochs=self.hparams.num_epochs,
            gradient_clip_val=self.hparams.clipping,
            progress_bar_refresh_rate=10,
            gpus=[self.gpu] if self.gpu is not None else None)

        loader = self.loader_class(self.hparams, self.dataset)
        train_loader = loader(partition="train", shuffle=True)
        val_loader = loader(partition="val", shuffle=False)

        trainer.fit(wrapper, train_dataloader=train_loader, val_dataloaders=val_loader)

    def get_wrapper(self):
        raise NotImplementedError