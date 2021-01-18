import torch

from core.serialization import load_pickle
from models.data import BaseDataset, BaseLoader


class Dataset(BaseDataset):
    def __init__(self, name):
        super().__init__(name)

        self.mapper = load_pickle(self.root_dir / "map.dict")
        self.codes = load_pickle(self.root_dir / "dfs_codes.pkl")

    def tensorizer(self, dfscode):
        max_nodes, max_edges = self.mapper['max_nodes'], self.mapper['max_edges']
        node_forward_dict, edge_forward_dict = self.mapper['node_forward'], self.mapper['edge_forward']
        num_nodes_feat, num_edges_feat = len(self.mapper['node_forward']), len(self.mapper['edge_forward'])

        # max_nodes, num_nodes_feat and num_edges_feat are end token labels
        # So ignore tokens are one higher
        dfscode_tensors = {
            't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            'v1': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            'e': (num_edges_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            'v2': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
            'len': len(dfscode)
        }

        for i, code in enumerate(dfscode):
            dfscode_tensors['t1'][i] = int(code[0])
            dfscode_tensors['t2'][i] = int(code[1])
            dfscode_tensors['v1'][i] = int(node_forward_dict[code[2]])
            dfscode_tensors['e'][i] = int(edge_forward_dict[code[3]])
            dfscode_tensors['v2'][i] = int(node_forward_dict[code[4]])

        # Add end token
        dfscode_tensors['t1'][len(dfscode)] = max_nodes
        dfscode_tensors['t2'][len(dfscode)] = max_nodes
        dfscode_tensors['v1'][len(dfscode)] = num_nodes_feat
        dfscode_tensors['v2'][len(dfscode)] = num_nodes_feat
        dfscode_tensors['e'][len(dfscode)] = num_edges_feat

        return dfscode_tensors


class Loader(BaseLoader):
    def collate(self, batch):
        return {
            "t1": torch.stack([b['t1'] for b in batch]),
            "t2": torch.stack([b['t2'] for b in batch]),
            "v1": torch.stack([b['v1'] for b in batch]),
            "e": torch.stack([b['e'] for b in batch]),
            "v2": torch.stack([b['v2'] for b in batch]),
            "len": torch.LongTensor([b['len'] for b in batch])
        }