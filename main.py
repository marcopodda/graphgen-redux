import random
import time
import pickle
from torch.utils.data import DataLoader

from graphgen.model import GraphGen
from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
from datasets.preprocess import calc_max_prev_node, dfscodes_weights
from baselines.dgmg.data import DGMG_Dataset_from_file
from baselines.graph_rnn.data import Graph_Adj_Matrix_from_file
from graphgen.data import Graph_DFS_code_from_file
from graphgen.model import create_model
from graphgen.train import train


if __name__ == '__main__':
    args = Args()
    args = args.update_args()

    if args.note != 'DFScodeRNN':
        raise AttributeError('Use this main to train DFScodeRNN')

    create_dirs(args)

    random.seed(123)

    graphs = create_graphs(args)

    random.shuffle(graphs)
    graphs_train = graphs[: int(0.80 * len(graphs))]
    graphs_validate = graphs[int(0.80 * len(graphs)): int(0.90 * len(graphs))]

    # show graphs statistics
    print('Model:', args.note)
    print('Device:', args.device)
    print('Graph type:', args.graph_type)
    print('Training set: {}, Validation set: {}'.format(
        len(graphs_train), len(graphs_validate)))

    # Loading the feature map
    with open(args.current_dataset_path + 'map.dict', 'rb') as f:
        feature_map = pickle.load(f)

    with open(args.min_dfscode_reduced_path + 'token_map.dict', 'rb') as f:
        reduced_map = pickle.load(f)

    print('Max number of nodes: {}'.format(feature_map['max_nodes']))
    print('Max number of edges: {}'.format(feature_map['max_edges']))
    print('Min number of nodes: {}'.format(feature_map['min_nodes']))
    print('Min number of edges: {}'.format(feature_map['min_edges']))
    print('Max degree of a node: {}'.format(feature_map['max_degree']))
    print('No. of node labels: {}'.format(len(feature_map['node_forward'])))
    print('No. of edge labels: {}'.format(len(feature_map['edge_forward'])))
    print('No. of token labels: {}'.format(len(reduced_map['dfs_to_reduced'])))

    if args.weights:
        feature_map = {
            **feature_map,
            **dfscodes_weights(args.min_dfscode_path, graphs_train, feature_map, args.device)
        }

    dataset_train = Graph_DFS_code_from_file(
        args, graphs_train, feature_map, reduced_map)
    dataset_validate = Graph_DFS_code_from_file(
        args, graphs_validate, feature_map, reduced_map)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers)
    dataloader_validate = DataLoader(
        dataset_validate, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers)

    model = GraphGen(args, feature_map, reduced_map)

    train(args, dataloader_train, model, feature_map, reduced_map, dataloader_validate)
