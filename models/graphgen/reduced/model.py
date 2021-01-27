
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from models.trainer import Trainer

from models.layers import LSTM, SoftmaxMLP
from models.wrapper import BaseWrapper
from models.graphgen.reduced.data import Dataset, Loader


class Model(nn.Module):
    def __init__(self, hparams, mapper):
        super().__init__()

        self.dim_ts_out = mapper['max_nodes']
        self.dim_tok_out  = len(mapper['reduced_forward']) + 1
        self.dim_input = 2 * (self.dim_ts_out + 1) + self.dim_tok_out

        self.rnn = LSTM(
            input_size=self.dim_input,
            embedding_size=hparams.embedding_size,
            hidden_size=hparams.rnn_hidden_size,
            num_layers=hparams.num_layers,
            dropout=hparams.dropout)

        self.output_t1 = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out + 1,
            dropout=hparams.dropout)

        self.output_t2 = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_ts_out + 1,
            dropout=hparams.dropout)

        self.output_tok = SoftmaxMLP(
            input_size=hparams.rnn_hidden_size,
            hidden_size=hparams.mlp_hidden_size,
            output_size=self.dim_tok_out,
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
        tok = torch.index_select(batch['tok'][:, :max_length], 0, sort_indices)

        # One-hot encode sequences
        x_t1 = F.one_hot(t1, num_classes=self.dim_ts_out + 2)[:, :, :-1]
        x_t2 = F.one_hot(t2, num_classes=self.dim_ts_out + 2)[:, :, :-1]
        x_tok = F.one_hot(tok, num_classes=self.dim_tok_out + 1)[:, :, :-1]

        y = torch.cat((x_t1, x_t2, x_tok), dim=2).float()

        # Init rnn
        self.rnn.hidden = self.rnn.init_hidden(batch_size=batch_size, device=y.device)

        # Teacher forcing: Feed the target as the next input
        # Start token is all zeros
        sos = torch.zeros(batch_size, 1, self.dim_input, device=y.device)
        rnn_input = torch.cat([sos, y[:, :-1, :]], dim=1)

        # Forward propogation
        rnn_output = self.rnn(rnn_input, x_len=lengths + 1)

        # Evaluating dfscode tuple
        out_t1 = self.output_t1(rnn_output).permute(0, 2, 1)
        out_t2 = self.output_t2(rnn_output).permute(0, 2, 1)
        out_tok = self.output_tok(rnn_output).permute(0, 2, 1)
        # y_pred = torch.cat([out_t1, out_t2, out_tok], dim=2)

        # # Cleaning the padding i.e setting it to zero
        # y_pred = pack_padded_sequence(y_pred, lengths + 1, batch_first=True)
        # y_pred, _ = pad_packed_sequence(y_pred, batch_first=True)
        loss_t1 = F.nll_loss(out_t1, t1, ignore_index=self.dim_ts_out + 1)
        loss_t2 = F.nll_loss(out_t2, t2, ignore_index=self.dim_ts_out + 1)
        loss_tok = F.nll_loss(out_tok, tok, ignore_index=self.dim_tok_out)

        loss = loss_t1 + loss_t2 + loss_tok
        return -loss

        # loss_sum = F.binary_cross_entropy(y_pred, y, reduction='none')
        # loss = torch.mean(torch.sum(loss_sum, dim=[1, 2]) / (lengths.float() + 1))
        # return loss


class ReducedGraphgen(BaseWrapper):
    model_class = Model


class ReducedGraphgenTrainer(Trainer):
    dataset_class = Dataset
    loader_class = Loader

    def get_wrapper(self):
        return ReducedGraphgen(self.hparams, self.dataset.mapper)
