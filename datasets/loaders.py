import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader as DL, Subset

from sklearn.model_selection import train_test_split

from core.serialization import load_pickle, save_pickle


class BaseDataLoader:
    partition = None

    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset
        self.mapper = dataset.mapper
        self.root_dir = dataset.root_dir

        if not (self.root_dir / "splits.pkl").exists():
            self._split_dataset()

        self.indices = load_pickle(self.root_dir / "splits.pkl")

    def _split_dataset(self):
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=0.1)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1)
        indices = {"train": train_indices.tolist(), "val": val_indices.tolist(), "test": test_indices.tolist()}
        save_pickle(indices, self.root_dir / "splits.pkl")

    def __call__(self, shuffle=False):
        dataset = Subset(self.dataset, self.indices[self.partition])
        return DL(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

    def collate(self, batch):
        raise NotImplementedError


class DataLoader(BaseDataLoader):
    def collate(self, batch):
        return {
            "t1": torch.stack([b['t1'] for b in batch]),
            "t2": torch.stack([b['t2'] for b in batch]),
            "v1": torch.stack([b['v1'] for b in batch]),
            "e": torch.stack([b['e'] for b in batch]),
            "v2": torch.stack([b['v2'] for b in batch]),
            "len": torch.LongTensor([b['len'] for b in batch])
        }

class TrainDataLoader(DataLoader):
    partition = "train"


class ValDataLoader(DataLoader):
    partition = "val"


class TestDataLoader(DataLoader):
    partition = "test"


class ReducedDataLoader(BaseDataLoader):
    def collate(self, batch):
        return {
            "t1": torch.stack([b['t1'] for b in batch]),
            "t2": torch.stack([b['t2'] for b in batch]),
            "tok": torch.stack([b['tok'] for b in batch]),
            "len": torch.LongTensor([b['len'] for b in batch])
        }


class TrainReducedDataLoader(ReducedDataLoader):
    partition = "train"


class ValReducedDataLoader(ReducedDataLoader):
    partition = "val"


class TestReducedDataLoader(ReducedDataLoader):
    partition = "test"