'''import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter
import time

from utils import save_model, load_model, get_model_attribute
from graphgen.train import evaluate_loss as eval_loss_dfscode_rnn
from baselines.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from baselines.dgmg.train import evaluate_loss as eval_loss_dgmg


def evaluate_loss(args, model, data, feature_map, reduced_map):
    loss = eval_loss_dfscode_rnn(args, model, data, feature_map, reduced_map)

    return loss


def train_epoch(
        epoch, args, model, dataloader_train, optimizer,
        scheduler, feature_map, reduced_map, summary_writer=None):
    # Set training mode for modules
    model.train()

    batch_count = len(dataloader_train)
    total_loss = 0.0
    for batch_id, data in enumerate(dataloader_train):

        model.zero_grad()

        loss = evaluate_loss(args, model, data, feature_map, reduced_map)

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
        print('Epoch: {}/{}, train loss: {:.6f}, time: {:.6f}'.format(epoch, args.epochs, loss, elapsed))

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
'''