import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import GRU, MLP, SoftmaxMLP
from models.wrapper import BaseWrapper
from models.trainer import Trainer
from models.graphrnn.data import Dataset, Loader


class Model(nn.Module):
    def __init__(self, hparams, mapper):
        super().__init__()
        self.hparams = hparams
        self.mapper = mapper

        self.num_node_features = len(mapper['node_forward']) + 2
        self.num_edge_features = len(mapper['edge_forward']) + 3
        self.max_prev_node = mapper['max_prev_node']
        self.num_features = self.num_node_features + self.max_prev_node * self.num_edge_features

        self.node_level_rnn = GRU(
            input_size=self.num_features,
            embedding_size=hparams.embedding_size_node_level_rnn,
            hidden_size=hparams.hidden_size_node_level_rnn,
            num_layers=hparams.num_layers)

        self.embedding_node_to_edge = MLP(
            input_size=hparams.hidden_size_node_level_rnn,
            hidden_size=hparams.embedding_size_node_level_rnn,
            output_size=hparams.hidden_size_edge_level_rnn)

        self.edge_level_rnn = GRU(
            input_size=self.num_edge_features,
            embedding_size=hparams.embedding_size_edge_level_rnn,
            hidden_size=hparams.hidden_size_edge_level_rnn,
            num_layers=hparams.num_layers)

        self.output_node = SoftmaxMLP(
            input_size=hparams.hidden_size_node_level_rnn,
            hidden_size=hparams.embedding_size_node_output,
            output_size=self.num_node_features)

        self.output_edge = SoftmaxMLP(
            input_size=hparams.hidden_size_edge_level_rnn,
            hidden_size=hparams.embedding_size_edge_output,
            output_size=self.num_edge_features)

    def forward(self, batch):
        x_unsorted = batch['x']
        x_len_unsorted = batch['len']
        x_len_max = x_len_unsorted.max()
        x_unsorted = x_unsorted[:, :x_len_max, :]
        batch_size = x_unsorted.size(0)

        # sort input for packing variable length sequences
        x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)
        x = torch.index_select(x_unsorted, 0, sort_indices)

        # initialize node_level_rnn hidden according to batch size
        self.node_level_rnn.hidden = self.node_level_rnn.init_hidden(batch_size=batch_size, device=x.device)

        # Teacher forcing: Feed the target as the next input
        # Start token for graph level RNN decoder is node feature second last bit is 1
        node_level_input = torch.cat((torch.zeros(batch_size, 1, x.size(2), device=x.device), x), dim=1)
        node_level_input[:, 0, self.num_node_features - 2] = 1

        # Forward propogation
        node_level_output = self.node_level_rnn(node_level_input, x_len=x_len + 1)

        # Evaluating node predictions
        x_pred_node = self.output_node(node_level_output)

        # Evaluating edge predictions
        # Make a 2D matrix of edge feature vectors with size = [sum(x_len)] x [min(x_len_max - 1, self.max_prev_node) * self.num_edge_features]
        # 2D matrix will have edge vectors sorted by time_stamp in graph level RNN
        edge_mat_packed = pack_padded_sequence(
            x[:, :, self.num_node_features: min(x_len_max - 1, self.max_prev_node) * self.num_edge_features + self.num_node_features],
            x_len, batch_first=True)

        edge_mat, _ = edge_mat_packed.data, edge_mat_packed.batch_sizes

        # Time stamp 'i' corresponds to edge feature sequence of length i (including start token added later)
        # Reverse the matrix in dim 0 (for packing purposes)
        idx = torch.LongTensor([i for i in range(edge_mat.size(0) - 1, -1, -1)]).to(x.device)
        edge_mat = edge_mat.index_select(0, idx)

        # Start token of edge level RNN is 1 at second last position in vector of length self.num_edge_featurestor
        # End token of edge level RNN is 1 at last position in vector of length self.num_edge_featurestor
        # Convert the edge_mat in a 3D tensor of size
        # [sum(x_len)] x [min(x_len_max, self.max_prev_node + 1)] x [self.num_edge_features]
        edge_mat = edge_mat.reshape(edge_mat.size(0), min(x_len_max - 1, self.max_prev_node), self.num_edge_features)
        edge_level_input = torch.cat((torch.zeros(sum(x_len), 1, self.num_edge_features, device=x.device), edge_mat), dim=1)
        edge_level_input[:, 0, self.num_edge_features - 2] = 1

        # Compute descending list of lengths for y_edge
        x_edge_len = []
        # Histogram of y_len
        x_edge_len_bin = torch.bincount(x_len)
        for i in range(len(x_edge_len_bin) - 1, 0, -1):
            # count how many x_len is above and equal to i
            count_temp = torch.sum(x_edge_len_bin[i:]).item()

            # put count_temp of them in x_edge_len each with value min(i, self.max_prev_node + 1)
            x_edge_len.extend([min(i, self.max_prev_node + 1)] * count_temp)

        x_edge_len = torch.LongTensor(x_edge_len).to(x.device)

        # Get edge-level RNN hidden state from node-level RNN output at each timestamp
        # Ignore the last hidden state corresponding to END
        hidden_edge = self.embedding_node_to_edge(node_level_output[:, 0:-1, :])

        # Prepare hidden state for edge level RNN similiar to edge_mat
        # Ignoring the last graph level decoder END token output (all 0's)
        hidden_edge = pack_padded_sequence(hidden_edge, x_len, batch_first=True).data
        idx = torch.LongTensor([i for i in range(hidden_edge.size(0) - 1, -1, -1)]).to(x.device)
        hidden_edge = hidden_edge.index_select(0, idx)

        # Set hidden state for edge-level RNN
        # shape of hidden tensor (num_layers, batch_size, hidden_size)
        hidden_edge = hidden_edge.view(1, hidden_edge.size(0), hidden_edge.size(1))
        hidden_edge_rem_layers = torch.zeros(self.hparams.num_layers - 1, hidden_edge.size(1), hidden_edge.size(2), device=x.device)
        self.edge_level_rnn.hidden = torch.cat((hidden_edge, hidden_edge_rem_layers), dim=0)

        # Run edge level RNN
        x_pred_edge = self.edge_level_rnn(edge_level_input, x_len=x_edge_len)

        # cleaning the padding i.e setting it to zero
        x_pred_node = pack_padded_sequence(x_pred_node, x_len + 1, batch_first=True)
        x_pred_node, _ = pad_packed_sequence(x_pred_node, batch_first=True)
        x_pred_edge = pack_padded_sequence(x_pred_edge, x_edge_len, batch_first=True)
        x_pred_edge, _ = pad_packed_sequence(x_pred_edge, batch_first=True)

        # Loss evaluation & backprop
        x_node = torch.cat((x[:, :, :self.num_node_features], torch.zeros(batch_size, 1, self.num_node_features, device=x.device)), dim=1)
        x_node[torch.arange(batch_size), x_len, self.num_node_features - 1] = 1

        x_edge = torch.cat((edge_mat, torch.zeros(sum(x_len), 1, self.num_edge_features, device=x.device)), dim=1)
        x_edge[torch.arange(sum(x_len)), x_edge_len - 1, self.num_edge_features - 1] = 1

        loss1 = F.binary_cross_entropy(x_pred_node, x_node, reduction='sum')
        loss2 = F.binary_cross_entropy(x_pred_edge, x_edge, reduction='sum')

        # Avg (node prediction + edge prediction) error per example
        loss = (loss1 + loss2) / batch_size

        return loss


class GraphRNN(BaseWrapper):
    model_class = Model


class GraphRNNTrainer(Trainer):
    dataset_class = Dataset
    loader_class = Loader

    def get_wrapper(self):
        return GraphRNN(self.hparams, self.dataset.mapper)