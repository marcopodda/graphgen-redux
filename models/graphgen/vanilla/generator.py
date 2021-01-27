
import networkx as nx
import torch
from torch.nn import functional as F
from torch.distributions import Categorical

from datasets.utils import graph_from_dfscode
from models.generator import Generator
from models.graphgen.vanilla.data import Dataset
from models.graphgen.vanilla.model import Graphgen


class GraphgenGenerator(Generator):
    dataset_class = Dataset

    def load_wrapper(self, ckpt_path):
        return Graphgen.load_from_checkpoint(
            checkpoint_path=ckpt_path.as_posix(),
            hparams=self.hparams,
            mapper=self.dataset.mapper)

    def get_samples(self, model, device):
        model = model.eval()
        model = model.to(device)
        mapper = self.dataset.mapper

        dim_ts_out = self.max_nodes + 1
        dim_vs_out  = len(mapper['node_forward']) + 1
        dim_e_out = len(mapper['edge_forward']) + 1
        dim_input = 2 * dim_ts_out + 2 * dim_vs_out + dim_e_out
        max_edges = self.max_edges
        pred_size = 5

        batch_size = self.batch_size
        num_runs = self.num_runs
        all_graphs = []

        for _ in range(num_runs):
            model.rnn.hidden = model.rnn.init_hidden(batch_size=batch_size, device=device)
            rnn_input = torch.zeros((batch_size, 1, dim_input), device=device)
            pred = torch.zeros((batch_size, max_edges, pred_size), device=device)

            for i in range(max_edges):
                rnn_output = model.rnn(rnn_input)

                # Evaluating dfscode tuple
                t1 = model.output_t1(rnn_output).view(batch_size, -1)
                t2 = model.output_t2(rnn_output).view(batch_size, -1)
                v1 = model.output_v1(rnn_output).view(batch_size, -1)
                e = model.output_e(rnn_output).view(batch_size, -1)
                v2 = model.output_v2(rnn_output).view(batch_size, -1)

                t1 = Categorical(t1).sample()
                t2 = Categorical(t2).sample()
                v1 = Categorical(v1).sample()
                e = Categorical(e).sample()
                v2 = Categorical(v2).sample()

                rnn_input = torch.zeros((batch_size, 1, dim_input), device=device)
                rnn_input[torch.arange(batch_size), 0, t1] = 1
                rnn_input[torch.arange(batch_size), 0, dim_ts_out + t2] = 1
                rnn_input[torch.arange(batch_size), 0, 2 * dim_ts_out + v1] = 1
                rnn_input[torch.arange(batch_size), 0, 2 * dim_ts_out + dim_vs_out + e] = 1
                rnn_input[torch.arange(batch_size), 0, 2 * dim_ts_out + dim_vs_out + dim_e_out + v2] = 1

                pred[:, i, 0] = t1
                pred[:, i, 1] = t2
                pred[:, i, 2] = v1
                pred[:, i, 3] = e
                pred[:, i, 4] = v2

            nb = mapper['node_backward']
            eb = mapper['edge_backward']

            for i in range(batch_size):
                dfscode = []
                for j in range(max_edges):
                    if (pred[i, j, 0] == dim_ts_out - 1 or
                        pred[i, j, 1] == dim_ts_out - 1 or
                        pred[i, j, 2] == dim_vs_out - 1 or
                        pred[i, j, 3] == dim_e_out - 1 or
                        pred[i, j, 4] == dim_vs_out - 1):
                        break

                    dfscode.append((
                        int(pred[i, j, 0].item()),
                        int(pred[i, j, 1].item()),
                        nb[int(pred[i, j, 2].item())],
                        eb[int(pred[i, j, 3].item())],
                        nb[int(pred[i, j, 4].item())]
                    ))

                graph = graph_from_dfscode(dfscode)

                # Remove self loops
                graph.remove_edges_from(nx.selfloop_edges(graph))

                # Take maximum connected component
                if len(graph.nodes()):
                    max_comp = max(nx.connected_components(graph), key=len)
                    graph = nx.Graph(graph.subgraph(max_comp))

                all_graphs.append(graph)

        return all_graphs
