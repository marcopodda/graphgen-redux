import torch

from core.settings import DATA_DIR
from core.serialization import load_pickle


def dfscode_to_tensor(self, dfscode, feature_map):
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


def reduced_dfscode_to_tensor(self, dfscode, feature_map):
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


class BaseDataset:
    mapper_filename = None
    codes_filename = None
    tensorizer = None

    def __init__(self, name):
        self.name = name
        self.root_dir = DATA_DIR / name
        self.mapper = load_pickle(self.root_dir / self.mapper_filename)
        self.codes = load_pickle(self.root_dir / self.codes_filename)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        code = self.codes[index]
        return self.tensorizer(code, self.mapper)


class Dataset(BaseDataset):
    mapper_filename = "map.dict"
    codes_filename = "dfs_codes.pkl"
    tensorizer = dfscode_to_tensor


class ReducedDataset(BaseDataset):
    mapper_filename = "reduced_map.dict"
    codes_filename = "reduced_dfs_codes.pkl"
    tensorizer = reduced_dfscode_to_tensor