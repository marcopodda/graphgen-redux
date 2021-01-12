import torch
from utils import get_model_attribute

class ArgsEvaluate():
    def __init__(self, name=None):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
            
        model = 'DFScodeAE'
        dataset = 'Lung'
        epochs = str(200)
        if name is None:
            self.model_path = 'model_save/' + model + \
                '_' + dataset + '/' + model + '_' + dataset + '_' + \
                epochs + '.dat'
        else:
            self.model_path = name

        self.num_epochs = get_model_attribute(
            'epoch', self.model_path, self.device)

        # Whether to generate networkx format graphs for real datasets
        self.generate_graphs = True
        self.random_seed = False

        self.count = 256
        self.batch_size = 32  # Must be a factor of count

        # Used for mse_latent
        self.return_dfs = False
        self.compare_graphs = False

        self.temperature = 1

        self.metric_eval_batch_size = 256
        self.fast_evaluation = False

        # Specific DFScodeRNN
        self.max_num_edges = 50

        # Specific to GraphRNN
        self.min_num_node = 0
        self.max_num_node = 40

        self.train_args = get_model_attribute(
            'saved_args', self.model_path, self.device)

        # self.reduced_ae_input = 'onehot'    # full | reduced | onehot

        self.graphs_save_path = 'graphs/'
        self.current_graphs_save_path = self.graphs_save_path + self.train_args.fname + '_' + \
            self.train_args.time + '/' + str(self.num_epochs) + '/'

#eval = ArgsEvaluate()
