
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.trainer import Trainer

from models.layers import LSTM, SoftmaxMLP
from models.wrapper import BaseWrapper
from models.graphgen.vanilla.data import Dataset, Loader


class Model(nn.Module):
    def __init__(self, hparams, mapper):
        super().__init__()

        self.max_nodes = mapper['max_nodes']
        self.len_node_vec  = len(mapper['node_forward']) + 1
        self.len_edge_vec = len(mapper['edge_forward']) + 1
        self.feature_len = 2 * self.max_nodes + 2 * self.len_node_vec + self.len_edge_vec

        self.rnn = LSTM(
            input_size=self.feature_len,
            embedding_size=hparams.embedding_size,
            hidden_size=hparams.rnn_hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout)

        self.output_t1 = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.max_nodes + 1,
            dropout=hparams.dropout)

        self.output_t2 = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.max_nodes + 1,
            dropout=hparams.dropout)

        self.output_v1 = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.len_node_vec,
            dropout=hparams.dropout)

        self.output_e = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.len_edge_vec,
            dropout=hparams.dropout)

        self.output_v2 = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.len_node_vec,
            dropout=hparams.dropout)

    def forward(self, batch):
        x_len_unsorted = batch['len']
        x_len_max = x_len_unsorted.max() + 1
        batch_size = x_len_unsorted.size(0)

        # sort input for packing variable length sequences
        x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)

        t1 = torch.index_select(batch['t1'][:, :x_len_max + 1], 0, sort_indices)
        t2 = torch.index_select(batch['t2'][:, :x_len_max + 1], 0, sort_indices)
        v1 = torch.index_select(batch['v1'][:, :x_len_max + 1], 0, sort_indices)
        e = torch.index_select(batch['e'][:, :x_len_max + 1], 0, sort_indices)
        v2 = torch.index_select(batch['v2'][:, :x_len_max + 1], 0, sort_indices)

        x_t1 = F.one_hot(t1, num_classes=self.max_nodes + 2)[:, :, :-1]
        x_t2 = F.one_hot(t2, num_classes=self.max_nodes + 2)[:, :, :-1]
        x_v1= F.one_hot(v1, num_classes=self.len_node_vec + 1)[:, :, :-1]
        x_v2 = F.one_hot(v2, num_classes=self.len_node_vec + 1)[:, :, :-1]
        x_e = F.one_hot(e, num_classes=self.len_edge_vec + 1)[:, :, :-1]
        x_target = torch.cat((x_t1, x_t2, x_v1, x_e, x_v2), dim=2).float()

        # initialize dfs_code_rnn hidden according to batch size
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size, device=x_target.device)

        # Teacher forcing: Feed the target as the next input
        # Start token is all zeros
        zeros = torch.zeros(batch_size, 1, self.feature_len, device=x_t1.device)
        rnn_input = torch.cat([zeros, x_target[:, :-1, :]], dim=1)

        # Forward propogation
        rnn_output = self.rnn(rnn_input, input_len=x_len + 1)

        # Evaluating dfscode tuple
        out_t1 = self.output_t1(rnn_output)
        out_t2 = self.output_t2(rnn_output)
        out_v1 = self.output_v1(rnn_output)
        out_e = self.output_e(rnn_output)
        out_v2 = self.output_v2(rnn_output)

        x_pred = torch.cat([out_t1, out_t2, out_v1, out_e, out_v2], dim=2)

        # Cleaning the padding i.e setting it to zero
        x_pred = pack_padded_sequence(x_pred, x_len + 1, batch_first=True)
        x_pred, _ = pad_packed_sequence(x_pred, batch_first=True)

        loss_sum = F.binary_cross_entropy(x_pred, x_target, reduction='none')
        loss = torch.mean(torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))
        return loss


class Graphgen(BaseWrapper):
    model_class = Model


class GraphgenTrainer(Trainer):
    dataset_class = Dataset
    loader_class = Loader

    def get_wrapper(self):
        return Graphgen(self.hparams, self.dataset.mapper)
