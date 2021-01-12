import torch
from utils import get_model_attribute
import sys
import os
import shutil

from args_evaluate import ArgsEvaluate
from graphgen.train import predict_graphs as gen_graphs_dfscode_rnn
from baselines.graph_rnn.train import predict_graphs as gen_graphs_graph_rnn
from baselines.dgmg.train import predict_graphs as gen_graphs_dgmg
from utils import get_model_attribute, load_graphs, save_graphs

def generate_graphs(eval_args):
    """
    Generate graphs (networkx format) given a trained generative model
    and save them to a directory
    :param eval_args: ArgsEvaluate object
    """

    train_args = eval_args.train_args

    if train_args.note == 'DFScodeRNN':
        gen_graphs = gen_graphs_dfscode_rnn(eval_args)

    if os.path.isdir(eval_args.current_graphs_save_path):
        shutil.rmtree(eval_args.current_graphs_save_path)

    os.makedirs(eval_args.current_graphs_save_path)

    save_graphs(eval_args.current_graphs_save_path, gen_graphs)

def main():
  eval_args = ArgsEvaluate()
  if len(sys.argv) > 1:
    eval_args.count = int(sys.argv[1])
    if len(sys.argv) > 2:
      eval_args.batch_size = int(sys.argv[2])
    else:
      eval_args.batch_size = 1
      
  generate_graphs(eval_args)
  print('Done')

if __name__ == "__main__":
  main()
