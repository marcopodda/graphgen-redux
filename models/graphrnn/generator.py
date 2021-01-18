
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

        max_num_node = self.hparams.max_num_node
        len_node_vec = len(mapper['node_forward']) + 2
        len_edge_vec = len(mapper['edge_forward']) + 3
        max_prev_node = mapper['max_prev_node']
        feature_len = len_node_vec + max_prev_node * len_edge_vec

        all_graphs = []

        for run in range(num_runs):
            for _ in range(num_iter):

                model.node_level_rnn.hidden = model.node_level_rnn.init_hidden(batch_size=batch_size, device=device)

                # [batch_size] * [num of nodes]
                x_pred_node = np.zeros((batch_size, max_num_node), dtype=np.int32)
                # [batch_size] * [num of nodes] * [max_prev_node]
                x_pred_edge = np.zeros((batch_size, max_num_node, max_prev_node), dtype=np.int32)

                node_level_input = torch.zeros(batch_size, 1, feature_len, device=device)
                # Initialize to node level start token
                node_level_input[:, 0, len_node_vec - 2] = 1
                for i in range(max_num_node):
                    # [batch_size] * [1] * [hidden_size_node_level_rnn]
                    node_level_output = model.node_level_rnn(node_level_input)
                    # [batch_size] * [1] * [node_feature_len]
                    node_level_pred = model.output_node(node_level_output)
                    # [batch_size] * [node_feature_len] for torch.multinomial
                    node_level_pred = node_level_pred.reshape(batch_size, len_node_vec)
                    # [batch_size]: Sampling index to set 1 in next node_level_input and x_pred_node
                    # Add a small probability for each node label to avoid zeros
                    node_level_pred[:, :-2] += EPS
                    # Start token should not be sampled. So set it's probability to 0
                    node_level_pred[:, -2] = 0
                    # End token should not be sampled if i less than min_num_node
                    if i < self.hparams.min_num_node:
                        node_level_pred[:, -1] = 0

                    sample_node_level_output = torch.multinomial(node_level_pred, 1).reshape(-1)
                    node_level_input = torch.zeros(batch_size, 1, feature_len, device=device)
                    node_level_input[torch.arange(batch_size), 0, sample_node_level_output] = 1

                    # [batch_size] * [num of nodes]
                    x_pred_node[:, i] = sample_node_level_output.cpu().data

                    # [batch_size] * [1] * [hidden_size_edge_level_rnn]
                    hidden_edge = model.embedding_node_to_edge(node_level_output)

                    hidden_edge_rem_layers = torch.zeros(self.hparams.num_layers - 1, batch_size, hidden_edge.size(2), device=device)
                    # [num_layers] * [batch_size] * [hidden_len]
                    model.edge_level_rnn.hidden = torch.cat((hidden_edge.permute(1, 0, 2), hidden_edge_rem_layers), dim=0)

                    # [batch_size] * [1] * [edge_feature_len]
                    edge_level_input = torch.zeros(batch_size, 1, len_edge_vec, device=device)
                    # Initialize to edge level start token
                    edge_level_input[:, 0, len_edge_vec - 2] = 1
                    for j in range(min(max_prev_node, i)):
                        # [batch_size] * [1] * [edge_feature_len]
                        edge_level_emb = model.edge_level_rnn(edge_level_input)
                        edge_level_output = self.output_edge(edge_level_emb)
                        # [batch_size] * [edge_feature_len] needed for torch.multinomial
                        edge_level_output = edge_level_output.reshape(batch_size, len_edge_vec)

                        # [batch_size]: Sampling index to set 1 in next edge_level input and x_pred_edge
                        # Add a small probability for no edge to avoid zeros
                        edge_level_output[:, -3] += EPS
                        # Start token and end should not be sampled. So set it's probability to 0
                        edge_level_output[:, -2:] = 0
                        sample_edge_level_output = torch.multinomial(edge_level_output, 1).reshape(-1)
                        edge_level_input = torch.zeros(batch_size, 1, len_edge_vec, device=device)
                        edge_level_input[:, 0, sample_edge_level_output] = 1

                        # Setting edge feature for next node_level_input
                        node_level_input[:, 0, len_node_vec + j * len_edge_vec: len_node_vec + (j + 1) * len_edge_vec] = \
                            edge_level_input[:, 0, :]

                        # [batch_size] * [num of nodes] * [max_prev_node]
                        x_pred_edge[:, i, j] = sample_edge_level_output.cpu().data

                # Save the batch of graphs
                for k in range(batch_size):
                    G = nx.Graph()

                    for v in range(max_num_node):
                        # End node token
                        if x_pred_node[k, v] == len_node_vec - 1:
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
                                if self.hparams.max_prev_node is not None:
                                    v = u - p - 1
                                elif self.hparams.max_head_and_tail is not None:
                                    if p < self.hparams.max_head_and_tail[1]:
                                        v = u - p - 1
                                    else:
                                        v = p - self.hparams.max_head_and_tail[1]
                                else:
                                    print('Error in sampling edge features')
                                    exit()

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