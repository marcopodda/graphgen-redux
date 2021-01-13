from pathlib import Path
import argparse
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from core.utils import get_n_jobs, get_or_create_dir, flatten
from datasets.utils import sample_subgraphs


DATA_DIR = Path("DATA")


DATASETS = [
    "All",
    "Breast",
    "citeseer",
    "cora",
    "ENZYMES",
    "Leukemia",
    "Lung",
    "Yeast"
]


def process_molecule_dataset(name):
    root_dir = DATA_DIR / name
    data_file = root_dir / f"{name.lower()}.txt"

    lines = []
    with open(data_file, 'r') as fr:
        for line in fr:
            line = line.rstrip("\n").split()
            lines.append(line)

    index, count, graphs = 0, 0, []

    while index < len(lines):
        graph_id = lines[index][0][1:]
        index += 1

        G = nx.Graph(id=graph_id)

        num_nodes = int(lines[index][0])
        index += 1

        for n in range(num_nodes):
            label = lines[index][0]
            G.add_node(n, label=label)
            index += 1

        num_edges = int(lines[index][0])
        index += 1

        for _ in range(num_edges):
            e1 = int(lines[index][0])
            e2 = int(lines[index][1])
            label = lines[index][2]
            G.add_edge(e1, e2, label=label)
            index += 1

        if nx.is_connected(G):
            graphs.append(G)
            count += 1

        index += 1

    return graphs


def process_citation_dataset(name, num_factor=5, iterations=150):
    root_dir = DATA_DIR / name
    output_dir = get_or_create_dir(root_dir / "graphs")
    content_file = root_dir / f"{name}.content"
    cites_file = root_dir / f"{name}.cites"

    print('Producing random_walk graphs - num_factor - {}'.format(num_factor))
    G = nx.Graph()

    d = {}
    count = 0
    with open(content_file, 'r') as f:
        for line in f.readlines():
            spp = line.rstrip("\n").split('\t')
            G.add_node(count, label=spp[-1])
            d[spp[0]] = count
            count += 1

    count = 0
    with open(cites_file, 'r') as f:
        for line in f.readlines():
            spp = line.rstrip("\n").split('\t')
            if spp[0] in d and spp[1] in d:
                G.add_edge(d[spp[0]], d[spp[1]], label='DEFAULT_LABEL')
            else:
                count += 1

    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)
    node_idxs = list(range(G.number_of_nodes()))

    P = Parallel(n_jobs=get_n_jobs(), verbose=1)
    graph_list = P(delayed(sample_subgraphs)(i, G, iterations, num_factor) for i in node_idxs)
    return flatten(graph_list)


def process_tud_dataset(name, min_num_nodes=20, max_num_nodes=100):
    root_dir = DATA_DIR / name
    adj_file = root_dir / f"{name}_A.txt"
    graph_ind_file = root_dir / f"{name}_graph_indicator.txt"
    node_lab_file = root_dir / f"{name}_node_labels.txt"

    data_adj = np.loadtxt(adj_file, delimiter=',').astype(int)
    data_node_label = np.loadtxt(node_lab_file, delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(graph_ind_file, delimiter=',').astype(int)
    data_tuple = list(map(tuple, data_adj))

    # add edges
    G = nx.Graph()
    G.add_edges_from(data_tuple)

    # add node attributes
    for i in range(data_node_label.shape[0]):
        G.add_node(i + 1, label=str(data_node_label[i]))

    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    num_graphs = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0]) + 1

    count, graphs = 0, []
    for i in range(num_graphs):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator == i + 1]
        G_sub = G.subgraph(nodes)
        num_nodes = G_sub.number_of_nodes()

        if num_nodes >= min_num_nodes and num_nodes <= max_num_nodes:
            if nx.is_connected(G_sub):
                G_sub = nx.convert_node_labels_to_integers(G_sub)
                G_sub.remove_edges_from(nx.selfloop_edges(G_sub))

                for node in G_sub.nodes():
                    G_sub.nodes[node]['label'] = str(G_sub.nodes[node]['label'])

                nx.set_edge_attributes(G_sub, 'DEFAULT_LABEL', 'label')
                graphs.append(G_sub)
                count += 1

    return graphs


def create_dataset(dataset_name):
    np.random.seed(123)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=DATASETS, dest="dataset_name", required=True)
    args = parser.parse_args()
    dataset_name = args.dataset_name

    if dataset_name in ["All", "Breast", "Leukemia", "Lung", "Yeast"]:
        process_molecule_dataset(dataset_name)
    elif dataset_name in ["cora", "citeseer"]:
        process_citation_dataset(dataset_name)
    elif dataset_name in ["ENZYMES"]:
        process_tud_dataset(dataset_name)
    else:
        raise Exception("Unknown dataset name!")