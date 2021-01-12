import time
from tqdm import tqdm
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
import networkx as nx
import random

from graphgen.model import create_model
from dfscode.dfs_wrapper import graph_from_dfscode
from utils import save_model, load_model, get_model_attribute


def train_epoch(
        epoch, args, model, dataloader_train, optimizer,
        scheduler, feature_map, reduced_map, summary_writer=None):
    # Set training mode for modules
    model.train()

    batch_count = len(dataloader_train)
    total_loss = 0.0
    for batch_id, data in enumerate(tqdm(dataloader_train)):

        model.zero_grad()

        loss = evaluate_loss(args, model, data, feature_map,
                             reduced_map, epoch=epoch)

        loss.backward()
        total_loss += loss.data.item()

        # Clipping gradients
        if args.gradient_clipping:
            clip_grad_value_(model.parameters(), 1.0)

        # Update params of rnn and mlp
        optimizer.step()

        scheduler.step()

        if args.log_tensorboard:
            summary_writer.add_scalar('{} {} Loss/train batch'.format(
                args.note, args.graph_type), loss, batch_id + batch_count * epoch)

    return total_loss / batch_count


def test_data(args, model, dataloader, feature_map, reduced_map):
    model.eval()

    batch_count = len(dataloader)
    with torch.no_grad():
        total_loss = 0.0
        for _, data in enumerate(dataloader):
            loss = evaluate_loss(args, model, data, feature_map, reduced_map)
            total_loss += loss.data.item()

    return total_loss / batch_count


# Main training function
def train(args, dataloader_train, model, feature_map, reduced_map, dataloader_validate=None):
    # initialize optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        weight_decay=5e-5)

    scheduler = MultiStepLR(
        optimizer, milestones=args.milestones,
        gamma=args.gamma)

    if args.load_model:
        load_model(args.load_model_path, args.device,
                   model, optimizer, scheduler)
        print('Model loaded')

        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0

    if args.log_tensorboard:
        writer = SummaryWriter(
            log_dir=args.tensorboard_path + args.fname, flush_secs=5)
    else:
        writer = None

    while epoch < args.epochs:
        start_epoch = time.time()
        loss = train_epoch(
            epoch, args, model, dataloader_train, optimizer, scheduler, feature_map, reduced_map, writer)
        elapsed = time.time()-start_epoch
        epoch += 1

        # logging
        if args.log_tensorboard:
            writer.add_scalar('{} {} Loss/train'.format(
                args.note, args.graph_type), loss, epoch)
        # else:
        print('Epoch: {}/{}, train loss: {:.6f}, time: {:.6f}'.format(epoch,
                                                                      args.epochs, loss, elapsed))

        # save model checkpoint
        if args.save_model and epoch != 0 and epoch % args.epochs_save == 0:
            save_model(
                epoch, args, model, optimizer, scheduler, feature_map=feature_map, reduced_map=reduced_map)
            print(
                'Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))

        if dataloader_validate is not None and epoch % args.epochs_validate == 0:
            loss_validate = test_data(
                args, model, dataloader_validate, feature_map, reduced_map)
            if args.log_tensorboard:
                writer.add_scalar('{} {} Loss/validate'.format(
                    args.note, args.graph_type), loss_validate, epoch)
            else:
                print('Epoch: {}/{}, validation loss: {:.6f}'.format(
                    epoch, args.epochs, loss_validate))

    save_model(epoch, args, model, optimizer,
               scheduler, feature_map=feature_map, reduced_map=reduced_map)
    print('Model Saved - Epoch: {}/{}, train loss: {:.6f}'.format(epoch, args.epochs, loss))


def evaluate_loss(args, model, data, feature_map, reduced_map, x=None, epoch=0, enc_out=None):
    x_len_unsorted = data['len'].to(args.device)
    x_len_max = max(x_len_unsorted)
    batch_size = x_len_unsorted.size(0)

    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)

    max_nodes = feature_map['max_nodes']
    len_token = len(reduced_map['reduced_to_dfs']) + 1
    feature_len = 2 * (max_nodes + 1) + len_token

    # Prepare targets with end_tokens already there
    # [:, :x_len_max + 1] removes unnecesary padding
    t1 = torch.index_select(
        data['t1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    t2 = torch.index_select(
        data['t2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    tok = torch.index_select(
        data['token'][:, :x_len_max + 1].to(args.device), 0, sort_indices)

    # Consider EOS and SOS. [:,:,:-1] makes 0,0,0,...,1 -> 0,0,...,0
    # Performing one hot batch-wise is ok because label 3 will alway be 0,0,1,0,... despite the range of values
    x_t1, x_t2 = F.one_hot(t1, num_classes=max_nodes +
                           2)[:, :, :-1], F.one_hot(t2, num_classes=max_nodes + 2)[:, :, :-1]
    x_token = F.one_hot(tok, num_classes=len_token + 1)[:, :, :-1]

    x_target = torch.cat((x_t1, x_t2, x_token), dim=2).float()

    # Realign hidden received from encoder with 
    # Wrong to do it, hid are obtained by feeding x_target already index_selected
    # if x is not None:
    #     x = torch.index_select(x, 0, sort_indices)

    # initialize dfs_code_rnn hidden according to batch size and input from encoder
    model.dfs_code_rnn.hidden = model.dfs_code_rnn.init_hidden(
        batch_size=batch_size, x=x)

    # Teacher forcing: Feed the target as the next input
    # Start token is all zeros
    dfscode_rnn_input = torch.zeros(
        batch_size, 1, feature_len, device=args.device)
    loss_sum = 0

    # Forward propogation
    if epoch < args.teacher_stop and random.random() <= args.teacher_forcing:
        # Teacher forcing
        # Add to SOS the following correct inputs except for EOS
        dfscode_rnn_input = torch.cat(
            (dfscode_rnn_input, x_target[:, :-1, :]), dim=1)
        # Compute feedforward
        if args.attention == 'varattn':
            timestamp1, timestamp2, token, (mu, logvar) = model.forward(
                args, dfscode_rnn_input, x_len+1, enc_out)
        else:
            timestamp1, timestamp2, token = model.forward(
                args, dfscode_rnn_input, x_len+1, enc_out)
    else:
        # Has some issues, deprecated (too slow)
        raise NotImplementedError('Use teacher forcing')
        exit()
        """
        #pred = torch.zeros(
        #    (args.batch_size, x_len_max + 1, feature_len), device=args.device)
        # For each step in the target sequence
        for i in range(x_target.shape[1]):
            if args.attention == 'varattn':
                timestamp1, timestamp2, token, (mu, logvar) = model.forward(args, dfscode_rnn_input, None, enc_out)
            else:
                timestamp1, timestamp2, token = model.forward(args, dfscode_rnn_input, None, enc_out)
            # x_pred already 3D tensor
            x_pred = torch.cat((timestamp1, timestamp2, token), dim=2)
            if i==0:
                pred = x_pred
            else:
                pred = torch.cat((pred, x_pred), dim=1)

            #loss_sum += F.binary_cross_entropy(
            #  x_pred[:, :, :], x_target[:, i, :], reduction='none')

            # Sample for next input
            if args.loss_type == 'BCE':
                timestamp1 = Categorical(timestamp1).sample()
                timestamp2 = Categorical(timestamp2).sample()
                token = Categorical(token).sample()
            elif args.loss_type == 'NLL':
                timestamp1 = Categorical(logits=timestamp1).sample()
                timestamp2 = Categorical(logits=timestamp2).sample()
                token = Categorical(logits=token).sample()

            dfscode_rnn_input = torch.zeros(
                (args.batch_size, 1, feature_len), device=args.device)
            dfscode_rnn_input[torch.arange(args.batch_size), 0, timestamp1] = 1
            dfscode_rnn_input[torch.arange(args.batch_size),
                    0, timestamp2 + max_nodes + 1] = 1
            dfscode_rnn_input[torch.arange(args.batch_size), 0, 2 *
                    max_nodes + 2 + token] = 1
            
            # dfscode_rnn_input = x_pred

        pred = pack_padded_sequence(pred, x_len + 1, batch_first=True)
        pred, _ = pad_packed_sequence(pred, batch_first=True)
        
        loss_sum = F.binary_cross_entropy(
           pred, x_target, reduction='none')
        loss = torch.mean(
            torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))
        if args.attention == 'varattn':
          kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
        else:
          kld = 0
        return loss + args.lamb_attn * kld
        """

    if args.loss_type == 'BCE':
        # Concatenate all the outputs along dimension 2
        x_pred = torch.cat(
            (timestamp1, timestamp2, token), dim=2)

        # Cleaning the padding i.e setting it to zero
        x_pred = pack_padded_sequence(x_pred, x_len + 1, batch_first=True)
        x_pred, _ = pad_packed_sequence(x_pred, batch_first=True)

        # if args.weights:
        #     # Weights for BCE
        #     weight = torch.cat((feature_map['t1_weight'].to(args.device), feature_map['t2_weight'].to(args.device),
        #                         feature_map['token_weight'].to(
        #                             args.device)))

        #     weight = weight.expand(batch_size, x_len_max + 1, -1)
        # else:
        #     weight = None

        loss_sum = F.binary_cross_entropy(
            x_pred, x_target, reduction='none')  # , weight=weight)
        loss = torch.mean(
            torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))

    elif args.loss_type == 'NLL':
        timestamp1 = timestamp1.transpose(dim0=1, dim1=2)
        timestamp2 = timestamp2.transpose(dim0=1, dim1=2)
        token = token.transpose(dim0=1, dim1=2)

        loss_t1 = F.nll_loss(
            timestamp1, t1, ignore_index=max_nodes + 1)  # , weight=feature_map.get('t1_weight'))
        loss_t2 = F.nll_loss(
            timestamp2, t2, ignore_index=max_nodes + 1)  # , weight=feature_map.get('t2_weight'))
        # , weight=feature_map.get('token_weight'))
        loss_token = F.nll_loss(token, tok, ignore_index=len_token)

        loss = loss_t1 + loss_t2 + loss_token

    if args.attention == 'varattn':
        kld = -0.5 * \
            torch.mean(
                torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), dim=-1))
    else:
        kld = 0
    return loss + args.lamb_attn * kld

# Used only for graphgen model, cond_graphgen's method is defined into cond_graphgen/train.py
def predict_graphs(eval_args, x=None):
    train_args = eval_args.train_args
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)
    reduced_map = get_model_attribute(
        'reduced_map', eval_args.model_path, eval_args.device)
    train_args.device = eval_args.device

    model = create_model(train_args, feature_map, reduced_map)
    load_model(eval_args.model_path, eval_args.device, model)

    model.eval()

    max_nodes = feature_map['max_nodes']
    len_node_vec, len_edge_vec = len(
        feature_map['node_forward']) + 1, len(feature_map['edge_forward']) + 1
    len_token = len(reduced_map['dfs_to_reduced']) + 1
    # Length of concatenated triple
    feature_len = 2 * (max_nodes + 1) + len_token

    graphs = []

    for _ in range(eval_args.count // eval_args.batch_size):
        # initialize dfs_code_rnn hidden according to batch size
        model.dfs_code_rnn.hidden = model.dfs_code_rnn.init_hidden(
            batch_size=eval_args.batch_size)

        if x is None:
            # No input from encoder
            rnn_input = torch.zeros(
                (eval_args.batch_size, 1, feature_len), device=eval_args.device)
        else:
            # Input from encoder i.e. hidden state init
            rnn_input = x

        pred = torch.zeros(
            (eval_args.batch_size, eval_args.max_num_edges, 3), device=eval_args.device)

        # Generate triple by triple (edge by edge)
        for i in range(eval_args.max_num_edges):
            # Compute feedforward for current step
            if train_args.attention == 'varattn':
                timestamp1, timestamp2, token, _ = model.forward(
                    eval_args, rnn_input)
            else:
                timestamp1, timestamp2, token = model.forward(
                    eval_args, rnn_input)

            timestamp1, timestamp2, token = timestamp1.reshape(eval_args.batch_size, -1), \
                timestamp2.reshape(eval_args.batch_size, -1), \
                token.reshape(eval_args.batch_size, -1)

            # Sample labels from output distribution
            if train_args.loss_type == 'BCE':
                timestamp1 = Categorical(timestamp1).sample()
                timestamp2 = Categorical(timestamp2).sample()
                token = Categorical(token).sample()

            elif train_args.loss_type == 'NLL':
                timestamp1 = Categorical(logits=timestamp1).sample()
                timestamp2 = Categorical(logits=timestamp2).sample()
                token = Categorical(logits=token).sample()

            # Define new input
            rnn_input = torch.zeros(
                (eval_args.batch_size, 1, feature_len), device=eval_args.device)

            # Insert previous output as new input
            rnn_input[torch.arange(eval_args.batch_size), 0, timestamp1] = 1
            rnn_input[torch.arange(eval_args.batch_size),
                      0, timestamp2 + max_nodes + 1] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, 2 *
                      max_nodes + 2 + token] = 1

            # Add current triple to output sequence
            pred[:, i, 0] = timestamp1
            pred[:, i, 1] = timestamp2
            pred[:, i, 2] = token

        nb = feature_map['node_backward']
        eb = feature_map['edge_backward']
        # For each graph in the batch
        for i in range(eval_args.batch_size):
            dfscode = []
            # For each edge
            for j in range(eval_args.max_num_edges):
                # If EOS, stop
                if pred[i, j, 0] == max_nodes or pred[i, j, 1] == max_nodes \
                        or pred[i, j, 2] == len_token - 1:
                    break
                # Convert token into triple of labels i.e. triple into quintuple
                triple = reduced_map['reduced_to_dfs'][int(pred[i, j, 2].data)]

                # Add quintuple to dfscode
                dfscode.append(
                    (int(pred[i, j, 0].data), int(pred[i, j, 1].data), triple[0],
                     triple[1], triple[2]))

            # Convert dfscode into graph format
            graph = graph_from_dfscode(dfscode)

            # Remove self loops
            graph.remove_edges_from(graph.selfloop_edges())

            # Take maximum connected component
            if len(graph.nodes()):
                max_comp = max(nx.connected_components(graph), key=len)
                graph = nx.Graph(graph.subgraph(max_comp))

            graphs.append(graph)

    return graphs
