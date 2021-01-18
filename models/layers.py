import torch
from torch import nn
from torch.nn import init

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BaseRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input = nn.Linear(input_size, embedding_size)

        self.hidden = None  # Need initialization before forward run

    def init_weights(self):
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

    def forward(self, x, x_len=None):
        x = self.input(x)

        if x_len is not None:
            x = pack_padded_sequence(x, lengths=x_len, batch_first=True, enforce_sorted=False)

        output, self.hidden = self.rnn(x, self.hidden)

        if x_len is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)

        return output


class GRU(BaseRNN):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0):
        super().__init__(input_size, embedding_size, hidden_size, num_layers, dropout=dropout)

        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.init_weights()

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class LSTM(BaseRNN):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout=0):
        super().__init__(input_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.init_weights()

    def init_hidden(self, batch_size, device):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return self.mlp(x)


class SoftmaxMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=2))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return self.mlp(x)