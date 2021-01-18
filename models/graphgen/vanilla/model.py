
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.trainer import Trainer

from models.graphgen.modules import RNN, MLP
from models.graphgen.wrapper import GraphgenWrapper
from models.graphgen.vanilla.data import Dataset, Loader


class Model(nn.Module):
    def __init__(self, hparams, mapper):
        super().__init__()

        self.dim_ts_out = mapper['max_nodes'] + 1
        self.dim_vs_out  = len(mapper['node_forward']) + 1
        self.dim_e_out = len(mapper['edge_forward']) + 1
        self.dim_input = 2 * self.dim_ts_out + 2 * self.dim_vs_out + self.dim_e_out

        self.rnn = RNN(
            input_size=self.dim_input,
            embedding_size=hparams.embedding_size,
            hidden_size=hparams.rnn_hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout)

        self.output_t1 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out,
            dropout=hparams.dropout)

        self.output_t2 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out,
            dropout=hparams.dropout)

        self.output_v1 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_vs_out,
            dropout=hparams.dropout)

        self.output_e = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_e_out,
            dropout=hparams.dropout)

        self.output_v2 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_vs_out,
            dropout=hparams.dropout)

    def forward(self, batch):
        lengths = batch['len']
        max_length = lengths.max() + 1
        batch_size = lengths.size(0)

        # sort input for packing variable length sequences
        lengths, sort_indices = torch.sort(lengths, dim=0, descending=True)

        # Prepare targets with end_tokens already there
        t1 = torch.index_select(batch['t1'][:, :max_length], 0, sort_indices)
        t2 = torch.index_select(batch['t2'][:, :max_length], 0, sort_indices)
        v1 = torch.index_select(batch['v1'][:, :max_length], 0, sort_indices)
        e = torch.index_select(batch['e'][:, :max_length], 0, sort_indices)
        v2 = torch.index_select(batch['v2'][:, :max_length], 0, sort_indices)

        # One-hot encode sequences
        x_t1 = F.one_hot(t1, num_classes=self.dim_ts_out + 1)[:, :, :-1]
        x_t2 = F.one_hot(t2, num_classes=self.dim_ts_out + 1)[:, :, :-1]
        x_v1 = F.one_hot(v1, num_classes=self.dim_vs_out + 1)[:, :, :-1]
        x_e = F.one_hot(e, num_classes=self.dim_e_out + 1)[:, :, :-1]
        x_v2 = F.one_hot(v2, num_classes=self.dim_vs_out + 1)[:, :, :-1]
        y = torch.cat((x_t1, x_t2, x_v1, x_e, x_v2), dim=2).float()

        # Init rnn
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size, device=y.device)

        # Teacher forcing: Feed the target as the next input
        # Start token is all zeros
        sos = torch.zeros(batch_size, 1, self.dim_input, device=y.device)
        rnn_input = torch.cat([sos, y[:, :-1, :]], dim=1)

        # Forward propogation
        rnn_output = self.rnn(rnn_input, x_len=lengths+1)

        # Evaluating dfscode tuple
        out_t1 = self.output_t1(rnn_output)
        out_t2 = self.output_t2(rnn_output)
        out_v1 = self.output_v1(rnn_output)
        out_e = self.output_e(rnn_output)
        out_v2 = self.output_v2(rnn_output)
        y_pred = torch.cat([out_t1, out_t2, out_v1, out_e, out_v2], dim=2)

        # Cleaning the padding i.e setting it to zero
        y_pred = pack_padded_sequence(y_pred, lengths=lengths+1, batch_first=True)
        y_pred, _ = pad_packed_sequence(y_pred, batch_first=True)

        return y_pred, y, lengths


class Graphgen(GraphgenWrapper):
    model_class = Model


class GraphgenTrainer(Trainer):
    dataset_class = Dataset
    loader_class = Loader

    def get_wrapper(self):
        return Graphgen(self.hparams, self.dataset.mapper)
