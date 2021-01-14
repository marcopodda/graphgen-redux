import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.mlp import MLP
from modules.rnn import RNN


class Model(nn.Module):
    def __init__(self, hparams, mapper):
        super().__init__()

        self.dim_ts_out = mapper['max_nodes']
        self.dim_vs_out  = len(mapper['node_forward']) + 1
        self.dim_e_out = len(mapper['edge_forward']) + 1
        self.dim_input = 2 * (self.dim_ts_out + 1) + 2 * self.dim_vs_out + self.dim_e_out

        self.rnn = RNN(
            input_size=self.dim_input,
            embedding_size=hparams.embedding_size,
            hidden_size=hparams.rnn_hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout)

        self.output_t1 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out+1,
            dropout=hparams.dropout)

        self.output_t2 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out+1,
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
        max_length = max(lengths)
        batch_size = lengths.size(0)

        # sort input for packing variable length sequences
        lengths, sort_indices = torch.sort(lengths, dim=0, descending=True)

        # Prepare targets with end_tokens already there
        t1 = torch.index_select(batch['t1'][:, :max_length + 1], 0, sort_indices)
        t2 = torch.index_select(batch['t2'][:, :max_length + 1], 0, sort_indices)
        v1 = torch.index_select(batch['v1'][:, :max_length + 1], 0, sort_indices)
        e = torch.index_select(batch['e'][:, :max_length + 1], 0, sort_indices)
        v2 = torch.index_select(batch['v2'][:, :max_length + 1], 0, sort_indices)

        # One-hot encode sequences
        x_t1 = F.one_hot(t1, num_classes=self.dim_ts_out + 2)[:, :, :-1]
        x_t2 = F.one_hot(t2, num_classes=self.dim_ts_out + 2)[:, :, :-1]
        x_v1 = F.one_hot(v1, num_classes=self.dim_vs_out + 1)[:, :, :-1]
        x_e = F.one_hot(e, num_classes=self.dim_e_out + 1)[:, :, :-1]
        x_v2 = F.one_hot(v2, num_classes=self.dim_vs_out + 1)[:, :, :-1]
        y = torch.cat((x_t1, x_t2, x_v1, x_e, x_v2), dim=2).float()

        # Init rnn
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size)

        # Teacher forcing: Feed the target as the next input
        # Start token is all zeros
        sos = torch.zeros(batch_size, 1, self.dim_input)
        rnn_input = torch.cat([sos, y[:, :-1, :]], dim=1)

        # Forward propogation
        rnn_output = self.rnn(rnn_input, x_len=lengths + 1)

        # Evaluating dfscode tuple
        out_t1 = self.output_t1(rnn_output)
        out_t2 = self.output_t2(rnn_output)
        out_v1 = self.output_v1(rnn_output)
        out_e = self.output_e(rnn_output)
        out_v2 = self.output_v2(rnn_output)
        y_pred = torch.cat([out_t1, out_t2, out_v1, out_e, out_v2], dim=2)

        # Cleaning the padding i.e setting it to zero
        y_pred = pack_padded_sequence(y_pred, lengths + 1, batch_first=True)
        y_pred, _ = pad_packed_sequence(y_pred, batch_first=True)

        return y_pred, y


class ReducedModel(nn.Module):
    def __init__(self, hparams, mapper):
        super().__init__()

        self.dim_ts_out = mapper['max_nodes']
        self.dim_tok_out  = len(mapper['reduced_forward']) + 1
        self.dim_input = 2 * (self.dim_ts_out + 1) + self.dim_tok_out

        self.rnn = RNN(
            input_size=self.dim_input,
            embedding_size=hparams.embedding_size,
            hidden_size=hparams.rnn_hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout)

        self.output_t1 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out+1,
            dropout=hparams.dropout)

        self.output_t2 = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out+1,
            dropout=hparams.dropout)

        self.output_tok = MLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_tok_out,
            dropout=hparams.dropout)

    def forward(self, batch):
        lengths = batch['len']
        max_length = max(lengths)
        batch_size = lengths.size(0)

        # sort input for packing variable length sequences
        lengths, sort_indices = torch.sort(lengths, dim=0, descending=True)

        # Prepare targets with end_tokens already there
        t1 = torch.index_select(batch['t1'][:, :max_length + 1], 0, sort_indices)
        t2 = torch.index_select(batch['t2'][:, :max_length + 1], 0, sort_indices)
        tok = torch.index_select(batch['tok'][:, :max_length + 1], 0, sort_indices)

        # One-hot encode sequences
        x_t1 = F.one_hot(t1, num_classes=self.dim_ts_out + 2)[:, :, :-1]
        x_t2 = F.one_hot(t2, num_classes=self.dim_ts_out + 2)[:, :, :-1]
        x_tok = F.one_hot(tok, num_classes=self.dim_tok_out + 1)[:, :, :-1]
        y = torch.cat((x_t1, x_t2, x_tok), dim=2).float()

        # Init rnn
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size)

        # Teacher forcing: Feed the target as the next input
        # Start token is all zeros
        sos = torch.zeros(batch_size, 1, self.dim_input)
        rnn_input = torch.cat([sos, y[:, :-1, :]], dim=1)

        # Forward propogation
        rnn_output = self.rnn(rnn_input, x_len=lengths + 1)

        # Evaluating dfscode tuple
        out_t1 = self.output_t1(rnn_output)
        out_t2 = self.output_t2(rnn_output)
        out_tok = self.output_tok(rnn_output)
        y_pred = torch.cat([out_t1, out_t2, out_tok], dim=2)

        # Cleaning the padding i.e setting it to zero
        y_pred = pack_padded_sequence(y_pred, lengths + 1, batch_first=True)
        y_pred, _ = pad_packed_sequence(y_pred, batch_first=True)

        return y_pred, y
