
import networkx as nx
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from datasets.utils import graph_from_reduced_dfscode

from models.generator import Generator
from models.graphgen.reduced.data import Dataset
from models.graphgen.reduced.model import ReducedGraphgen

class ReducedGraphgenGenerator(Generator):
    dataset_class = Dataset

    def load_wrapper(self, ckpt_path):
        return ReducedGraphgen.load_from_checkpoint(
            checkpoint_path=ckpt_path.as_posix(),
            hparams=self.hparams,
            mapper=self.dataset.mapper)

    def get_samples(self, model, device):
        model = model.eval()
        model = model.to(device)
        mapper = self.dataset.mapper

        dim_ts_out = mapper['max_nodes'] + 1
        dim_tok_out  = len(self.mapper['reduced_forward']) + 1
        dim_input = 2 * dim_ts_out + dim_tok_out
        max_edges = mapper['max_edges']
        pred_size = 3

        batch_size = self.hparams.batch_size
        num_samples = self.hparams.num_samples
        num_iter = num_samples // batch_size
        num_runs = self.hparams.num_runs

        all_graphs = []

        for run in range(num_runs):
            for _ in range(num_iter):
                model.rnn.hidden = model.rnn.init_hidden(batch_size=batch_size, device=device)
                rnn_input = torch.zeros((batch_size, 1, dim_input), device=device)
                pred = torch.zeros((batch_size, max_edges, pred_size), device=device)

                for i in range(max_edges):
                    rnn_output = model.rnn(rnn_input)

                    # Evaluating dfscode tuple
                    t1 = model.output_t1(rnn_output).view(batch_size, -1)
                    t2 = model.output_t2(rnn_output).view(batch_size, -1)
                    tok = model.output_tok(rnn_output).view(batch_size, -1)

                    t1 = F.one_hot(Categorical(t1).sample(), num_classes=dim_ts_out)
                    t2 = F.one_hot(Categorical(t2).sample(), num_classes=dim_ts_out)
                    tok = F.one_hot(Categorical(tok).sample(), num_classes=dim_tok_out)

                    rnn_input = torch.cat([t1, t2, tok], dim=-1).view(batch_size, 1, -1).float()

                    pred[:, i, 0] = t1.argmax(dim=-1)
                    pred[:, i, 1] = t2.argmax(dim=-1)
                    pred[:, i, 2] = tok.argmax(dim=-1)

                tb = mapper['reduced_backward']

                for i in range(batch_size):
                    dfscode = []
                    for j in range(max_edges):
                        if (pred[i, j, 0] == dim_ts_out - 1 or
                            pred[i, j, 1] == dim_ts_out - 1 or
                            pred[i, j, 2] == dim_tok_out - 1):
                            break

                        dfscode.append((
                            int(pred[i, j, 0].item()),
                            int(pred[i, j, 1].item()),
                            tb[int(pred[i, j, 2].item())]
                        ))

                    graph = graph_from_reduced_dfscode(dfscode)

                    # Remove self loops
                    graph.remove_edges_from(nx.selfloop_edges(graph))

                    # Take maximum connected component
                    if len(graph.nodes()):
                        max_comp = max(nx.connected_components(graph), key=len)
                        graph = nx.Graph(graph.subgraph(max_comp))

                    all_graphs.append(graph)
        return all_graphs