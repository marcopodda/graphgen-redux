import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader as DL, Subset


def collate_reduced(batch):
    return {
        "t1": torch.stack([b['t1'] for b in batch]),
        "t2": torch.stack([b['t2'] for b in batch]),
        "tok": torch.stack([b['tok'] for b in batch]),
        "len": torch.LongTensor([b['len'] for b in batch])
    }

def collate(batch):
    return {
        "t1": torch.stack([b['t1'] for b in batch]),
        "t2": torch.stack([b['t2'] for b in batch]),
        "v1": torch.stack([b['v1'] for b in batch]),
        "e": torch.stack([b['e'] for b in batch]),
        "v2": torch.stack([b['v2'] for b in batch]),
        "len": torch.LongTensor([b['len'] for b in batch])
    }


class DataLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset
        self.reduced = dataset.reduced
        self.mapper = dataset.mapper
        self.root_dir = dataset.root_dir
        self.collate = collate_reduced if self.reduced else collate

    def __call__(self, partition, shuffle=False):
        dataset = Subset(self.dataset, self.dataset.indices[partition])
        return DL(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)