from datetime import datetime
import torch
from utils import get_model_attribute


class Args:
    """
    Program configuration
    """

    def __init__(self):
        # Can manually select the device too
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        # Clean tensorboard
        self.clean_tensorboard = False
        # Clean temp folder
        self.clean_temp = False

        # Whether to use tensorboard for logging
        self.log_tensorboard = False

        # Algorithm Version - # Algorithm Version - GraphRNN  | DFScodeRNN (GraphGen) | DGMG (Deep GMG)
        self.note = 'DFScodeAE'

        # Check datasets/process_dataset for datasets | Add _inactive and _active with chemical compounds if you want specific category
        # Select dataset to train the model
        self.graph_type = 'Lung'

        # Whether to produce networkx format graphs for real datasets
        self.produce_graphs = True
        # Whether to produce min dfscode and write to files
        self.produce_min_dfscodes = True
        # Whether to map min dfscodes to tensors and save to files
        self.produce_min_dfscode_tensors = True
        # Whether to produce reduced min dfscode and write to files
        self.produce_min_dfscodes_reduced = True
        # Whether to map reduced min dfscode to tensor and save to files
        self.produce_min_dfscodes_reduced_tensors = True

        # if none, then auto calculate
        self.max_prev_node = None  # max previous node that looks back for GraphRNN

        # # Specific to Autoencoder
        # self.reduced_ae_input = 'onehot'    # full | reduced | onehot
        # self.GCN_hidden_features = 1024
        # self.GCN_num_layers=1
        # self.GCN_dropout=0.2

        # Specific to RNN encoder
        self.embedding_size_encoder_rnn = 256
        self.hidden_size_encoder_rnn = 512
        self.encoder_num_layers = 2

        # Specific to RNN VAE-encoder
        self.vae = False
        self.embedding_size_mu_output = 256
        self.embedding_size_logvar_output = 256
        self.latent_dimension = 256
        self.lamb = 1 # Kullback-Leibler Divergence weight

        # Wether to use attention or not
        self.attention = 'attn'  # None | attn | varattn
        self.latent_attn_dimension = 256
        self.lamb_attn = 1

        # Specific to DFScodeRNN
        # Model parameters
        self.embedding_size_dfscode_rnn = 128  # input size for dfscode RNN
        self.hidden_size_dfscode_rnn = 512 # hidden size for dfscode RNN
        # the size for vertex output embedding
        self.embedding_size_timestamp_output = 1024
        self.embedding_size_token_output = 1024 # the size for vertex output embedding
        self.dfscode_rnn_dropout = 0.2  # Dropout layer in between RNN layers
        self.loss_type = 'NLL'  # BCE | NLL (binary cross entropy | negative log likelihood)
        self.weights = False

        # Teacher Forcing
        self.teacher_forcing = 1 # Probability to perform teacher forcing per iterate
        self.teacher_stop = 1000

        # Specific to DFScodeRNN
        self.rnn_type = 'GRU'  # LSTM | GRU
        self.num_layers = 1

        self.batch_size = 32  # normal: 32, and the rest should be changed accordingly

        # training config
        self.num_workers = 4  # num workers to load data, default 4
        self.epochs = 250

        self.lr = 0.005 # Laarning rate
        # Learning rate decay factor at each milestone (no. of epochs)
        self.gamma = 0.3
        self.milestones = [100, 200, 400, 800]  # List of milestones

        # Whether to do gradient clipping
        self.gradient_clipping = True

        # Output config
        self.dir_input = ''
        self.model_save_path = self.dir_input + 'model_save/'
        self.tensorboard_path = self.dir_input + 'tensorboard/'
        self.dataset_path = self.dir_input + 'datasets/'
        self.temp_path = self.dir_input + 'tmp/'

        # Model save and validate parameters
        self.save_model = True
        self.epochs_save = 5
        self.epochs_validate = 1

        # Time at which code is run
        self.time = '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.now())

        # Filenames to save intermediate and final outputs
        self.fname = self.note + '_' + self.graph_type

        # Calcuated at run time
        self.current_model_save_path = self.model_save_path + \
            self.fname + '/' #+ '_' + self.time + '/'
        self.current_dataset_path = None
        self.current_processed_dataset_path = None
        self.current_min_dfscode_path = None
        self.current_temp_path = self.temp_path + self.fname + '/' # '_' + self.time + '/'

        # Model load parameters
        self.load_model = False
        self.load_model_path = 'model_save/'
        self.load_model_path += self.note + '_' + self.graph_type + '/'
        self.load_epochs = str(100)
        self.load_model_path += self.note + '_' + self.graph_type + '_' + self.load_epochs + '.dat'
        self.load_device = self.device
        self.epochs_end = 550

    def update_args(self):
        if self.load_model:
            args = get_model_attribute(
                'saved_args', self.load_model_path, self.load_device)
            args.device = self.load_device
            args.load_model = True
            args.load_model_path = self.load_model_path
            args.epochs = self.epochs_end

            args.clean_tensorboard = False
            args.clean_temp = False

            args.produce_graphs = False
            args.produce_min_dfscodes = False
            args.produce_min_dfscode_tensors = False
            args.produce_min_dfscodes_reduced = False
            args.produce_min_dfscodes_reduced_tensors = False

            return args

        return self