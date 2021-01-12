import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
import networkx as nx

from dfscode.dfs_wrapper import graph_from_dfscode
from graphgen.model import MLP_Plain

class RNN(nn.Module):
    """
    Custom GRU layer
    :param input_size: Size of input vector
    :param embedding_size: Embedding layer size (finally this size is input to RNN)
    :param hidden_size: Size of hidden state of vector
    :param num_layers: No. of RNN layers
    :param rnn_type: Currently only GRU and LSTM supported
    :param dropout: Dropout probability for dropout layers between rnn layers
    :param output_size: If provided, a MLP softmax is run on hidden state with output of size 'output_size'
    :param output_embedding_size: If provided, the MLP softmax middle layer is of this size, else 
        middle layer size is same as 'embedding size'
    :param device: torch device to instanstiate the hidden state on right device
    """

    def __init__(
        self, input_size, embedding_size, hidden_size, num_layers, rnn_type='GRU',
        dropout=0, output_size=None, output_embedding_size=None,
        device=torch.device('cpu')
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.output_size = output_size
        self.device = device

        self.input = nn.Linear(input_size, embedding_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                batch_first=True, dropout=dropout
            )

        # self.relu = nn.ReLU()

        self.hidden = None  # Need initialization before forward run

        if self.output_size is not None:
            if output_embedding_size is None:
                self.output = MLP_Softmax(
                    hidden_size, embedding_size, self.output_size)
            else:
                self.output = MLP_Softmax(
                    hidden_size, output_embedding_size, self.output_size)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform_(
                    param, gain=nn.init.calculate_gain('sigmoid'))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size, x=None):
        if self.rnn_type == 'GRU':
            # h0
            if x is None:
              return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            else:
              return x[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        elif self.rnn_type == 'LSTM':
            # (h0, c0)
            if x is None:
              return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))
            else:
              # Copy last hidden layer's state
              return (x[0][-1].unsqueeze(0).repeat(self.num_layers, 1, 1),
                    x[1][-1].unsqueeze(0).repeat(self.num_layers, 1, 1))

    def forward(self, input, input_len=None):

        input = self.input(input)
        # input = self.relu(input)

        if input_len is not None:
            input = pack_padded_sequence(
                input, input_len, batch_first=True, enforce_sorted=False)

        output, self.hidden = self.rnn(input, self.hidden)

        if input_len is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        if self.output_size is not None:
            output = self.output(output)

        return output
