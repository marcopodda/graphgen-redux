
import networkx as nx

from models.generator import Generator
from models.dgmg.data import Dataset
from models.dgmg.model import DGMG


class DGMGGenerator(Generator):
    dataset_class = Dataset

    def load_wrapper(self, ckpt_path):
        return DGMG.load_from_checkpoint(
            checkpoint_path=ckpt_path.as_posix(),
            hparams=self.hparams,
            mapper=self.dataset.mapper)

    def get_samples(self, model, device):
        model = model.eval()
        model = model.to(device)
        mapper = self.dataset.mapper

        batch_size = self.batch_size
        num_runs = self.num_runs
        all_graphs = []

        for _ in range(num_runs):
            model.prepare(batch_size, training=False)
            sampled_graphs = model.forward_inference()

            nb = mapper['node_backward']
            eb = mapper['edge_backward']
            for sampled_graph in sampled_graphs:
                graph = sampled_graph.to_networkx(
                    node_attrs=['label'], edge_attrs=['label']).to_undirected()

                labeled_graph = nx.Graph()

                for v in graph.nodes():
                    labeled_graph.add_node(
                        v, label=nb[graph.nodes[v]['label'].item() - 1])

                for u, v in graph.edges():
                    labeled_graph.add_edge(
                        u, v, label=eb[graph.edges[u, v]['label'].item()])

                # Take maximum connected component
                if len(labeled_graph.nodes()) > 0:
                    max_comp = max(nx.connected_components(labeled_graph), key=len)
                    labeled_graph = labeled_graph.subgraph(max_comp)

                all_graphs.append(labeled_graph)

        return all_graphs