import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
import torch.nn.functional as F
import networkx as nx

from dfscode.dfs_wrapper import graph_from_dfscode


class MLP_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Log_Softmax(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Log_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Plain(input_size, embedding_size, output_size, dropout),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, input):
        return self.mlp(input)


class MLP_Plain(nn.Module):
    """
    A deterministic linear output layer
    """

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Plain, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            # nn.Linear(embedding_size, embedding_size),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


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

        self.hidden = None  # Need initialization before forward run

        self.output = MLP_Softmax(
                    hidden_size, output_embedding_size, self.output_size)

    def init_hidden(self, batch_size, x=None):
        # h0
        if x is None:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        else:
            # Last layer copied to all the layers
            return x[-1, :, :].repeat(self.num_layers, 1, 1)

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


def create_model(args, feature_map, reduced_map):
    return GraphGen(args, feature_map, reduced_map)


class GraphGen(nn.Module):
    def __init__(self, args, feature_map, reduced_map):
        super().__init__()
        self.max_nodes = feature_map['max_nodes']
        self.len_token = len(reduced_map['dfs_to_reduced']) + 1

        self.feature_len = 2 * (self.max_nodes + 1) + self.len_token

        MLP_layer = MLP_Softmax

        self.dfs_code_rnn = RNN(
            input_size=self.feature_len, embedding_size=args.embedding_size_dfscode_rnn,
            hidden_size=args.hidden_size_dfscode_rnn, num_layers=args.num_layers,
            rnn_type=args.rnn_type, dropout=args.dfscode_rnn_dropout,
            device=args.device).to(device=args.device)

        self.output_timestamp1 = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
            output_size=self.max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

        self.output_timestamp2 = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_timestamp_output,
            output_size=self.max_nodes + 1, dropout=args.dfscode_rnn_dropout).to(device=args.device)

        self.output_token = MLP_layer(
            input_size=args.hidden_size_dfscode_rnn, embedding_size=args.embedding_size_token_output,
            output_size=self.len_token, dropout=args.dfscode_rnn_dropout).to(device=args.device)

    def forward(self, args, x=None, input_len=None, encoder_outputs=None):
        # Start token is all zeros
        # TODO change to random
        if x is None:
            dfscode_rnn_input = torch.zeros(
                (args.batch_size, 1, self.feature_len), device=args.device)
        else:
            dfscode_rnn_input = x
        # Forward propogation
        dfscode_rnn_output = self.dfs_code_rnn(
            dfscode_rnn_input, input_len=input_len)

        # Evaluating dfscode tuple
        timestamp1 = self.output_timestamp1(dfscode_rnn_output)
        timestamp2 = self.output_timestamp2(dfscode_rnn_output)
        token = self.output_token(dfscode_rnn_output)

        return timestamp1, timestamp2, token
