import torch
from torch.nn import functional as F
from torch.distributions import Categorical
import networkx as nx

from core.serialization import save_pickle
from datasets.utils import graph_from_dfscode, graph_from_reduced_dfscode

from modules.base import Base
from modules.wrapper import Wrapper

class Generator(Base):
    def generate(self, epoch, device):
        ckpt_path = list(self.dirs.ckpt.glob(f"epoch={epoch}-*.ckpt"))[0]
        wrapper = Wrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path.as_posix(),
            hparams=self.hparams,
            mapper=self.mapper,
            reduced=self.reduced)

        generating_fun = self._gen if self.reduced else self._gen_reduced
        samples = generating_fun(wrapper.model, self.mapper, device)
        reduced = "_red" if self.reduced else ""
        save_pickle(samples, self.dirs.samples / f"samples_{epoch:02d}{reduced}.pkl")

    def _gen(self, model, mapper, device):
        model = model.eval()
        dim_ts_out = mapper['max_nodes'] + 1
        dim_vs_out  = len(mapper['node_forward']) + 1
        dim_e_out = len(mapper['edge_forward']) + 1
        dim_input = 2 * dim_ts_out + 2 * dim_vs_out + dim_e_out
        max_edges = mapper['max_edges']
        pred_size = 3 if self.reduced else 5

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

    def _gen_reduced(self, model, mapper, device):
        model = model.eval()
        dim_ts_out = mapper['max_nodes'] + 1
        dim_tok_out  = len(mapper['reduced_forward']) + 1
        dim_input = 2 * dim_ts_out + dim_tok_out
        max_edges = mapper['max_edges']
        pred_size = 3 if self.reduced else 5

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