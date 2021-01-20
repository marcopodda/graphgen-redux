
import numpy as np
import networkx as nx
import torch

from models.generator import Generator
from models.graphrnn.data import Dataset
from models.graphrnn.model import GraphRNN

EPS = 1e-8


class GraphRNNGenerator(Generator):
    dataset_class = Dataset

    def load_wrapper(self, ckpt_path):
        return GraphRNN.load_from_checkpoint(
            checkpoint_path=ckpt_path.as_posix(),
            hparams=self.hparams,
            mapper=self.dataset.mapper)

    def get_samples(self, model, device):
        model = model.eval()
        model = model.to(device)
        mapper = self.dataset.mapper

        batch_size = self.hparams.batch_size
        num_samples = self.hparams.num_samples
        num_iter = num_samples // batch_size
        num_runs = self.hparams.num_runs

        all_graphs = []

        max_nodes = mapper['max_nodes']
        min_nodes = mapper['min_nodes']
        num_node_features = len(mapper['node_forward']) + 2
        num_edge_features = len(mapper['edge_forward']) + 3
        max_prev_node = mapper['max_prev_node']
        feature_len = num_node_features + max_prev_node * num_edge_features

        all_graphs = []

        for run in range(num_runs):
            for _ in range(num_iter):
                model.node_level_rnn.hidden = model.node_level_rnn.init_hidden(batch_size=batch_size, device=device)
                # [batch_size] * [num of nodes]
                x_pred_node = np.zeros((batch_size, max_nodes), dtype=np.int32)
                # [batch_size] * [num of nodes] * [num_nodes_to_consider]
                x_pred_edge = np.zeros((batch_size, max_nodes, max_prev_node), dtype=np.int32)

                node_level_input = torch.zeros(batch_size, 1, feature_len, device=device)
                # Initialize to node level start token
                node_level_input[:, 0, num_node_features - 2] = 1

                for i in range(max_nodes):
                    # [batch_size] * [1] * [hidden_size_node_level_rnn]
                    node_level_output = model.node_level_rnn(node_level_input)
                    # [batch_size] * [1] * [node_feature_len]
                    node_level_pred = model.output_node(node_level_output)
                    # [batch_size] * [node_feature_len] for torch.multinomial
                    node_level_pred = node_level_pred.reshape(batch_size, num_node_features)
                    # [batch_size]: Sampling index to set 1 in next node_level_input and x_pred_node
                    # Add a small probability for each node label to avoid zeros
                    node_level_pred[:, :-2] += EPS
                    # Start token should not be sampled. So set it's probability to 0
                    node_level_pred[:, -2] = 0
                    # End token should not be sampled if i less than min_num_node
                    if i < min_nodes:
                        node_level_pred[:, -1] = 0

                    sample_node_level_output = torch.multinomial(node_level_pred, 1).reshape(-1)
                    node_level_input = torch.zeros(batch_size, 1, feature_len, device=device)
                    node_level_input[torch.arange(batch_size), 0, sample_node_level_output] = 1

                    # [batch_size] * [num of nodes]
                    x_pred_node[:, i] = sample_node_level_output.cpu().data

                    # [batch_size] * [1] * [hidden_size_edge_level_rnn]
                    hidden_edge = model.embedding_node_to_edge(node_level_output)

                    hidden_edge_rem_layers = torch.zeros(self.hparams.num_layers-1, batch_size, hidden_edge.size(2), device=device)
                    # [num_layers] * [batch_size] * [hidden_len]
                    model.edge_level_rnn.hidden = torch.cat((hidden_edge.permute(1, 0, 2), hidden_edge_rem_layers), dim=0)

                    # [batch_size] * [1] * [edge_feature_len]
                    edge_level_input = torch.zeros(batch_size, 1, num_edge_features, device=device)
                    # Initialize to edge level start token
                    edge_level_input[:, 0, num_edge_features - 2] = 1

                    for j in range(min(max_prev_node, i)):
                        # [batch_size] * [1] * [edge_feature_len]
                        edge_level_emb = model.edge_level_rnn(edge_level_input)
                        edge_level_output = model.output_edge(edge_level_emb)

                        # [batch_size] * [edge_feature_len] needed for torch.multinomial
                        edge_level_output = edge_level_output.reshape(batch_size, num_edge_features)

                        # [batch_size]: Sampling index to set 1 in next edge_level input and x_pred_edge
                        # Add a small probability for no edge to avoid zeros
                        edge_level_output[:, -3] += EPS
                        # Start token and end should not be sampled. So set it's probability to 0
                        edge_level_output[:, -2:] = 0
                        sample_edge_level_output = torch.multinomial(edge_level_output, 1).reshape(-1)
                        edge_level_input = torch.zeros(batch_size, 1, num_edge_features, device=device)
                        edge_level_input[:, 0, sample_edge_level_output] = 1

                        # Setting edge feature for next node_level_input
                        start = num_node_features + j * num_edge_features
                        end = num_node_features + (j + 1) * num_edge_features
                        node_level_input[:, 0, start:end] = edge_level_input[:, 0, :]

                        # [batch_size] * [num of nodes] * [num_nodes_to_consider]
                        x_pred_edge[:, i, j] = sample_edge_level_output.cpu()

                # Save the batch of graphs
                for k in range(batch_size):
                    G = nx.Graph()

                    for v in range(max_nodes):
                        # End node token
                        if x_pred_node[k, v] == num_node_features - 1:
                            break
                        elif x_pred_node[k, v] < len(mapper['node_forward']):
                            G.add_node(
                                v, label=mapper['node_backward'][x_pred_node[k, v]])
                        else:
                            print('Error in sampling node features')
                            exit()

                    for u in range(len(G.nodes())):
                        for p in range(min(max_prev_node, u)):
                            if x_pred_edge[k, u, p] < len(mapper['edge_forward']):
                                v = u - p - 1
                                G.add_edge(u, v, label=mapper['edge_backward'][x_pred_edge[k, u, p]])
                            elif x_pred_edge[k, u, p] == len(mapper['edge_forward']):
                                # No edge
                                pass
                            else:
                                print('Error in sampling edge features')
                                exit()

                    # Take maximum connected component
                    if len(G.nodes()):
                        max_comp = max(nx.connected_components(G), key=len)
                        G = nx.Graph(G.subgraph(max_comp))

                    all_graphs.append(G)

        return all_graphs