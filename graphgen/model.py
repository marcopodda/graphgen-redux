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
                # Last layer copied to all the layers
                return x[-1, :, :].repeat(self.num_layers, 1, 1)
        elif self.rnn_type == 'LSTM':
            # (h0, c0)
            if x is None:
                return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device),
                        torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device))
            else:
                # x is a pair of states
                # Last layer copied to all the layers
                return (x[0][-1, :, :].repeat(self.num_layers, 1, 1),
                        x[-1][-1, :, :].repeat(self.num_layers, 1, 1))

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

        if args.loss_type == 'BCE':
            MLP_layer = MLP_Softmax
        elif args.loss_type == 'NLL':
            MLP_layer = MLP_Log_Softmax

        self.dfs_code_rnn = RNN(
            input_size=self.feature_len, embedding_size=args.embedding_size_dfscode_rnn,
            hidden_size=args.hidden_size_dfscode_rnn, num_layers=args.num_layers,
            rnn_type=args.rnn_type, dropout=args.dfscode_rnn_dropout,
            device=args.device).to(device=args.device)

        # self.attn = nn.Linear(args.hidden_size_dfscode_rnn,
        #                       args.hidden_size_dfscode_rnn).to(device=args.device)
        self.attn_combine = nn.Linear(
            args.hidden_size_dfscode_rnn * 2, args.hidden_size_dfscode_rnn).to(device=args.device)
        """
        # Second attention implementation
        self.attn = nn.Linear(args.hidden_size_dfscode_rnn + args.hidden_size_encoder_rnn, 1).to(device=args.device)
        self.attn_combine = nn.Linear(args.hidden_size_dfscode_rnn + args.hidden_size_encoder_rnn, args.hidden_size_dfscode_rnn).to(device=args.device)
        """

        self.var_attn_mu = nn.Linear(
            args.hidden_size_dfscode_rnn, args.latent_attn_dimension).to(device=args.device)
        self.var_attn_logvar = nn.Linear(
            args.hidden_size_dfscode_rnn, args.latent_attn_dimension).to(device=args.device)
        self.var_attn_out = nn.Linear(
            args.latent_attn_dimension, args.hidden_size_dfscode_rnn).to(device=args.device)

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

        # TODO fix attention
        if (args.attention == 'attn' or args.attention == 'varattn') and encoder_outputs is not None:
            context = encoder_outputs
            query = dfscode_rnn_output

            batch_size, output_len, dimensions = query.size()
            query_len = context.size(1)

            # Dot score function, only dot product
            # query = query.reshape(batch_size * output_len, dimensions)
            # query = self.attn(query)  # It works bettere without the layer
            # query = query.reshape(batch_size, output_len, dimensions)

            # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
            # (batch_size, output_len, query_len)
            attention_scores = torch.bmm(
                query, context.transpose(1, 2).contiguous())

            # Compute weights across every context sequence
            attention_scores = attention_scores.view(
                batch_size * output_len, query_len)
            attention_weights = F.softmax(attention_scores, dim=1)
            attention_weights = attention_weights.view(
                batch_size, output_len, query_len)

            # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
            # (batch_size, output_len, dimensions)
            mix = torch.bmm(attention_weights, context)

            # Variational attention
            if args.attention == 'varattn':
                mu = self.var_attn_mu(mix)
                logvar = self.var_attn_logvar(mix)

                # Reparameterization
                std = torch.exp(0.5*logvar)
                eps = torch.randn_like(std)
                mix = mu + eps*std
                mix = self.var_attn_out(mix)

            # concat -> (batch_size * output_len, 2*dimensions)
            combined = torch.cat((mix, query), dim=2)
            combined = combined.view(batch_size * output_len, 2 * dimensions)

            # Apply linear_out on every 2nd dimension of concat
            # output -> (batch_size, output_len, dimensions)
            output = self.attn_combine(combined).view(
                batch_size, output_len, dimensions)
            output = torch.tanh(output)
            dfscode_rnn_output = output
        """
        # Second implementation (slow)
        #print('first', dfscode_rnn_output.shape)

        # TODO fix attention
        if args.attention == 'attn' or args.attention == 'varattn' and encoder_outputs is not None:

            attention_output = torch.zeros_like(dfscode_rnn_output)

            for i in range(dfscode_rnn_output.shape[1]):
                # Compute attention step by step
                query = dfscode_rnn_output[:, i, :].unsqueeze(1)
                context = encoder_outputs
                # query = dfscode_rnn_output

                batch_size, output_len, dimensions = query.size()
                query_len = context.size(1)

                concat = torch.cat( ( context, query.repeat(1, query_len, 1) ), dim=2)
                # query = query.reshape(batch_size * output_len, dimensions)
                # batch_size * query_len * fixed=1 -> batch_size * fixed=1 * query_len
                score = torch.tanh(self.attn(concat)).permute(0, 2, 1)
                # query = query.reshape(batch_size, output_len, dimensions)

                # TODO: Include mask on PADDING_INDEX?

                # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
                # (batch_size, output_len, query_len)
                # attention_scores = torch.bmm(score, context.transpose(1, 2).contiguous())

                # Compute weights across every context sequence
                # attention_scores = attention_scores.view(batch_size * output_len, query_len)
                attention_weights = F.softmax(score, dim=2)
                # attention_weights = attention_weights.view(batch_size, output_len, query_len)

                # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
                # (batch_size, output_len, dimensions)

                # batch_size * 1 * query_len X batch_size * query_len * enc_out
                # Results batch_size * 1 * enc_out
                # Context vector after encoder steps aggregation
                mix = torch.bmm(attention_weights, context)

                # Variational attention
                if args.attention == 'varattn':
                    mu = self.var_attn_mu(mix)
                    logvar = self.var_attn_logvar(mix)

                    ## Reparameterization
                    std = torch.exp(0.5*logvar)
                    eps = torch.randn_like(std)
                    mix = mu + eps*std
                    mix = self.var_attn_out(mix)

                # concat -> (batch_size * 1 * dec_out+enc_out)
                combined = torch.cat((mix, query), dim=2)
                # combined = combined.view(batch_size * output_len, 2 * dimensions)

                # Apply linear_out on every 2nd dimension of concat
                # output -> (batch_size, 1, dimensions)
                output = self.attn_combine(combined)#.view(batch_size, output_len, dimensions)
                output = torch.tanh(output)

                #print('out[', i, ']', output.shape)
                #print('att', attention_output[:, i, :].shape)

                attention_output[:, i, :] = output.squeeze(1)

            dfscode_rnn_output = attention_output

        #print('later', dfscode_rnn_output.shape)
        """

        # Evaluating dfscode tuple
        timestamp1 = self.output_timestamp1(dfscode_rnn_output)
        timestamp2 = self.output_timestamp2(dfscode_rnn_output)
        token = self.output_token(dfscode_rnn_output)

        if args.attention == 'varattn':
            return timestamp1, timestamp2, token, (mu, logvar)
        return timestamp1, timestamp2, token
