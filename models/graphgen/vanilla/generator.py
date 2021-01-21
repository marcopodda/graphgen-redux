
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

        dim_ts_out = mapper['max_nodes'] + 1
        dim_vs_out  = len(mapper['node_forward']) + 1
        dim_e_out = len(mapper['edge_forward']) + 1
        dim_input = 2 * dim_ts_out + 2 * dim_vs_out + dim_e_out
        max_edges = mapper['max_edges']
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

                t1 = F.one_hot(Categorical(t1).sample(), num_classes=dim_ts_out)
                t2 = F.one_hot(Categorical(t2).sample(), num_classes=dim_ts_out)
                v1 = F.one_hot(Categorical(v1).sample(), num_classes=dim_vs_out)
                e = F.one_hot(Categorical(e).sample(), num_classes=dim_e_out)
                v2 = F.one_hot(Categorical(v2).sample(), num_classes=dim_vs_out)

                rnn_input = torch.cat([t1, t2, v1, e, v2], dim=-1).view(batch_size, 1, -1).float()

                pred[:, i, 0] = t1.argmax(dim=-1)
                pred[:, i, 1] = t2.argmax(dim=-1)
                pred[:, i, 2] = v1.argmax(dim=-1)
                pred[:, i, 3] = e.argmax(dim=-1)
                pred[:, i, 4] = v2.argmax(dim=-1)

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
