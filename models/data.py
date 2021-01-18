import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset

from core.settings import DATA_DIR
from core.serialization import load_pickle, save_pickle


class BaseDataset:
    def __init__(self, name):
        self.name = name
        self.root_dir = DATA_DIR / name

        self.graphs = load_pickle(self.root_dir / "graphs.pkl")

        if not (self.root_dir / "splits.pkl").exists():
            self._split_dataset()
        self.indices = load_pickle(self.root_dir / "splits.pkl")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        code = self.codes[index]
        return self.tensorizer(code)

    def _split_dataset(self):
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=0.1)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1)
        indices = {"train": train_indices.tolist(), "val": val_indices.tolist(), "test": test_indices.tolist()}
        save_pickle(indices, self.root_dir / "splits.pkl")

    def select_graphs(self, indices):
        return [self.graphs[i] for i in indices]

    def tensorizer(self, dfscode):
        raise NotImplementedError


class BaseLoader:
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset
        self.mapper = dataset.mapper
        self.root_dir = dataset.root_dir

    def __call__(self, partition, shuffle=False):
        dataset = Subset(self.dataset, self.dataset.indices[partition])
        return DataLoader(
            dataset=dataset,
            collate_fn=lambda b: self.collate(b),
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.hparams.num_workers)

    def collate(self, batch):
        raise NotImplementedError