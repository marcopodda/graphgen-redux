import torch
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from core.utils import get_or_create_dir
from core.serialization import load_yaml, save_yaml
from core.hparams import HParams
from datasets.datasets import Dataset, ReducedDataset
from datasets.loaders import TrainDataLoader, TrainReducedDataLoader
from trainer.wrapper import Wrapper


class Trainer:
    @classmethod
    def load(cls, exp_dir):
        exp_dir = Path(exp_dir)
        config = load_yaml(exp_dir / "config.yml")
        return cls(**config)

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
        self.exp_name = exp_name
        self.dataset_name = dataset_name
        self.reduced = reduced
        self.hparams = HParams.load(hparams)
        self.gpu = gpu
        self.debug = debug
        self.dirs = self._setup_dirs(root_dir)
        self.dump()

        if self.reduced:
            self.dataset = ReducedDataset(self.dataset_name)
        else:
            self.dataset = Dataset(self.dataset_name)

    def _setup_dirs(self, root_dir):
        root_dir = get_or_create_dir(root_dir)
        base_dir = get_or_create_dir(root_dir / self.dataset_name)
        exp_dir = get_or_create_dir(base_dir / self.exp_name)

        dirs = Namespace(
            root=get_or_create_dir(root_dir),
            base=get_or_create_dir(base_dir),
            exp=get_or_create_dir(exp_dir),
            ckpt=get_or_create_dir(exp_dir / "checkpoints"),
            logs=get_or_create_dir(exp_dir / "logs"),
            embeddings=get_or_create_dir(exp_dir / "embeddings"),
            samples=get_or_create_dir(exp_dir / "samples"))

        return dirs

    def train(self):
        logger = TensorBoardLogger(save_dir=self.dirs.exp, name="", version="logs")
        ckpt_callback = ModelCheckpoint(filepath=self.dirs.ckpt, save_top_k=-1)

        wrapper = Wrapper(
            hparams=self.hparams,
            mapper=self.dataset.mapper,
            reduced=self.reduced)

        trainer = pl.Trainer(
            logger=logger,
            checkpoint_callback=ckpt_callback,
            max_epochs=self.hparams.num_epochs,
            gradient_clip_val=self.hparams.clipping,
            progress_bar_refresh_rate=10,
            gpus=[self.gpu] if self.gpu is not None else None)

        if self.reduced:
            loader = TrainReducedDataLoader(self.hparams, self.dataset)
        else:
            loader = TrainDataLoader(self.hparams, self.dataset)
        loader = loader(shuffle=True)

        trainer.fit(wrapper, train_dataloader=loader)

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