import random
from statistics import mean

from core.serialization import load_pickle, save_pickle
from core.utils import get_or_create_dir
from modules.base import Base
from metrics import stats


LINE_BREAK = '----------------------------------------------------------------------\n'


def print_stats(
    node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref,
    edge_count_avg_pred, degree_mmd, clustering_mmd, orbit_mmd,
    nspdk_mmd, node_label_mmd, edge_label_mmd, node_label_and_degree
):
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
        mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print('MMD Node label - {:.6f}, MMD Edge label - {:.6f}'.format(
        mean(node_label_mmd), mean(edge_label_mmd)
    ))
    print('MMD Joint Node label and degree - {:.6f}'.format(
        mean(node_label_and_degree)
    ))
    print(LINE_BREAK)


class Evaluator(Base):
    def evaluate(self, partition, epoch):
        indices = self.dataset.indices
        real_graphs = self.dataset.select_graphs(indices[partition])

        reduced = "_red" if self.reduced else ""
        gen_graphs = load_pickle(self.dirs.samples / f"samples_{epoch:02d}{reduced}.pkl")
        tmp_dir = "."

        stats.novelty(real_graphs, gen_graphs, tmp_dir, timeout=60)
        stats.uniqueness(gen_graphs, tmp_dir, timeout=120)

        node_count_avg_ref, node_count_avg_pred = [], []
        edge_count_avg_ref, edge_count_avg_pred = [], []

        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
        node_label_mmd, edge_label_mmd, node_label_and_degree = [], [], []

        batch_size = self.hparams.batch_size

        for i in range(0, len(gen_graphs), batch_size):
            real_sample = random.sample(real_graphs, batch_size)
            gen_sample = gen_graphs[i:i+batch_size]

            node_count_avg_ref.append(mean([len(G.nodes()) for G in real_sample]))
            node_count_avg_pred.append(mean([len(G.nodes()) for G in gen_sample]))

            edge_count_avg_ref.append(mean([len(G.edges()) for G in real_sample]))
            edge_count_avg_pred.append(mean([len(G.edges()) for G in gen_sample]))

            degree_mmd.append(stats.degree_stats(real_sample, gen_sample))
            clustering_mmd.append(stats.clustering_stats(real_sample, gen_sample))
            orbit_mmd.append(stats.orbit_stats_all(real_sample, gen_sample))

            nspdk_mmd.append(stats.nspdk_stats(real_sample, gen_sample))

            node_label_mmd.append(stats.node_label_stats(real_sample, gen_sample))
            edge_label_mmd.append(stats.edge_label_stats(real_sample, gen_sample))
            node_label_and_degree.append(stats.node_label_and_degree_joint_stats(real_sample, gen_sample))

        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
            edge_label_mmd, node_label_and_degree
        )

        results = {
            "Node count avg. ref": node_count_avg_ref,
            "Node count avg. pred": node_count_avg_pred,
            "Edge count avg. ref": edge_count_avg_ref,
            "Edge count avg. pred": edge_count_avg_pred,
            "MMD Degree": degree_mmd,
            "MMD Clustering": clustering_mmd,
            "MMD NSPKD": nspdk_mmd,
            "MMD Node labels": node_label_mmd,
            "MMD Edge labels": edge_label_mmd,
            "MMD Node labels and degrees": node_label_and_degree
        }

        reduced = "_red" if self.reduced else ""
        filename = self.dirs.eval / f"results{reduced}_{partition}_{epoch:02d}.pkl"
        save_pickle(results, filename)

    def _setup_dirs(self, root_dir):
        dirs = super()._setup_dirs(root_dir)
        dirs.eval = get_or_create_dir(dirs.exp / "evaluation")
        return dirs
