import torch

from core.serialization import load_pickle
from models.data import BaseDataset, BaseLoader


class Dataset(BaseDataset):
    def __init__(self, name):
        super().__init__(name)

        self.mapper = load_pickle(self.root_dir / "reduced_map.dict")
        self.codes = load_pickle(self.root_dir / "reduced_dfs_codes.pkl")

    def tensorizer(self, dfscode):
        max_nodes, max_edges = self.mapper['max_nodes'], self.mapper['max_edges']
        reduced_forward = self.mapper['reduced_forward']
        num_nodes_feat = len(self.mapper['reduced_forward'])

        # max_nodes, num_nodes_feat and num_edges_feat are end token labels
        # So ignore tokens are one higher
        dfscode_tensors = {
            't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            'tok': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            'len': len(dfscode)
        }

        for i, code in enumerate(dfscode):
            dfscode_tensors['t1'][i] = int(code[0])
            dfscode_tensors['t2'][i] = int(code[1])
            dfscode_tensors['tok'][i] = int(reduced_forward[code[2]])

        # Add end token
        dfscode_tensors['t1'][len(dfscode)] = max_nodes
        dfscode_tensors['t2'][len(dfscode)] = max_nodes
        dfscode_tensors['tok'][len(dfscode)] = num_nodes_feat

        return dfscode_tensors


class Loader(BaseLoader):
    def collate(self, batch):
        return {
            "t1": torch.stack([b['t1'] for b in batch]),
            "t2": torch.stack([b['t2'] for b in batch]),
            "tok": torch.stack([b['tok'] for b in batch]),
            "len": torch.LongTensor([b['len'] for b in batch])
        }