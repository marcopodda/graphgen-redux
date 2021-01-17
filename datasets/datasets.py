import numpy as np
import torch
from sklearn.model_selection import train_test_split

from core.settings import DATA_DIR
from core.serialization import load_pickle, save_pickle


def dfscode_to_tensor(dfscode, feature_map):
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    node_forward_dict, edge_forward_dict = feature_map['node_forward'], feature_map['edge_forward']
    num_nodes_feat, num_edges_feat = len(feature_map['node_forward']), len(feature_map['edge_forward'])

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


def reduced_dfscode_to_tensor(dfscode, feature_map):
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    reduced_forward = feature_map['reduced_forward']
    num_nodes_feat = len(feature_map['reduced_forward'])

    # max_nodes, num_nodes_feat and num_edges_feat are end token labels
    # So ignore tokens are one higher
    reduced_dfscode_tensors = {
        't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'tok': (num_nodes_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'len': len(dfscode)
    }

    for i, code in enumerate(dfscode):
        reduced_dfscode_tensors['t1'][i] = int(code[0])
        reduced_dfscode_tensors['t2'][i] = int(code[1])
        reduced_dfscode_tensors['tok'][i] = int(reduced_forward[code[2]])

    # Add end token
    reduced_dfscode_tensors['t1'][len(dfscode)] = max_nodes
    reduced_dfscode_tensors['t2'][len(dfscode)] = max_nodes
    reduced_dfscode_tensors['tok'][len(dfscode)] = num_nodes_feat

    return reduced_dfscode_tensors


class Dataset:
    mapper_filename = None
    codes_filename = None
    tensorizer = None

    def __init__(self, name, reduced=True):
        self.name = name
        self.reduced = reduced
        self.root_dir = DATA_DIR / name

        graphs_filename = self.root_dir / "graphs.pkl"
        self.graphs = load_pickle(graphs_filename)

        mapper_filename = "reduced_map.dict" if reduced else "map.dict"
        self.mapper = load_pickle(self.root_dir / mapper_filename)

        codes_filename = "reduced_dfs_codes.pkl" if reduced else "dfs_codes.pkl"
        self.codes = load_pickle(self.root_dir / codes_filename)

        self.tensorizer = reduced_dfscode_to_tensor if reduced else dfscode_to_tensor

        if not (self.root_dir / "splits.pkl").exists():
            self._split_dataset()
        self.indices = load_pickle(self.root_dir / "splits.pkl")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        code = self.codes[index]
        return self.tensorizer(code, self.mapper)

    def _split_dataset(self):
        indices = np.arange(len(self.dataset))
        train_indices, test_indices = train_test_split(indices, test_size=0.1)
        train_indices, val_indices = train_test_split(train_indices, test_size=0.1)
        indices = {"train": train_indices.tolist(), "val": val_indices.tolist(), "test": test_indices.tolist()}
        save_pickle(indices, self.root_dir / "splits.pkl")

    def select_graphs(self, indices):
        return [self.graphs[i] for i in indices]