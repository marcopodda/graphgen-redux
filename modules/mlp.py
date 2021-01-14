from torch import nn
from torch.nn import init


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