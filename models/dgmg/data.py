"""
Code adapted from https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg
"""

import pickle
import torch

import networkx as nx

from core.serialization import load_pickle
from models.data import BaseDataset, BaseLoader


class Dataset(BaseDataset):
    def __init__(self, name):
        super().__init__(name)
        self.mapper = load_pickle(self.root_dir / "map.dict")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        node_map, edge_map = self.mapper['node_forward'], self.mapper['edge_forward']

        perm = torch.randperm(len(graph.nodes())).numpy()
        perm_map = {i: perm[i] for i in range(len(perm))}
        graph = nx.relabel_nodes(graph, perm_map)

        actions = []
        for v in range(len(graph.nodes())):
            actions.append(1 + node_map[graph.nodes[v]['label']])  # Add node

            for u, val in graph[v].items():
                if u < v:
                    # Add edge
                    actions.append(1)
                    actions.append(
                        int(u * len(edge_map) + edge_map[val['label']]))

            actions.append(0)  # Stop Edge

        actions.append(0)  # Stop Node

        return actions

    def collate_batch(self, batch):
        return batch


class Loader(BaseLoader):
    def collate(self, batch):
        return batch