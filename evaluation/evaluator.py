import random
from statistics import mean

from core.module import BaseModule
from core.serialization import load_pickle, save_pickle
from core.settings import DATA_DIR
from core.utils import get_or_create_dir
from evaluation.metrics import stats


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


class Evaluator(BaseModule):
    def __init__(self, model_name, root_dir, dataset_name, hparams, gpu):
        super().__init__(model_name, root_dir, dataset_name, hparams, gpu)

        self.graphs = load_pickle(DATA_DIR  / dataset_name / "graphs.pkl")
        self.indices = load_pickle(DATA_DIR / dataset_name / "splits.pkl")
        self.num_samples = 256 if dataset_name != "ENZYMES" else 40
        self.num_runs = 10 if dataset_name != "ENZYMES" else 64

    def evaluate(self, epoch):
        real_graphs = [self.graphs[i] for i in self.indices['test']]
        gen_graphs = load_pickle(self.dirs.samples / f"samples_{epoch:02d}.pkl")
        tmp_dir = "."

        novelty_score = stats.novelty(real_graphs, gen_graphs, tmp_dir, timeout=60)
        uniqueness_score = stats.uniqueness(gen_graphs, tmp_dir, timeout=120)

        node_count_avg_ref, node_count_avg_pred = [], []
        edge_count_avg_ref, edge_count_avg_pred = [], []

        degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
        node_label_mmd, edge_label_mmd, node_label_and_degree = [], [], []

        num_samples = self.num_samples
        num_runs = self.num_runs

        for i in range(num_runs):
            print(f"Evaluating run {i+1}")
            real_sample = random.sample(real_graphs, num_samples)
            gen_sample = gen_graphs[i*num_samples:i*num_samples+num_samples]

            print("Avg. node count...")
            node_count_avg_ref.append(mean([G.number_of_nodes() for G in real_sample]))
            node_count_avg_pred.append(mean([G.number_of_nodes() for G in gen_sample]))

            print("Avg. edge count...")
            edge_count_avg_ref.append(mean([G.number_of_edges() for G in real_sample]))
            edge_count_avg_pred.append(mean([G.number_of_edges() for G in gen_sample]))

            print("Degree distribution...")
            degree_mmd.append(stats.degree_stats(real_sample, gen_sample))
            print("Clustering coefficient distribution...")
            clustering_mmd.append(stats.clustering_stats(real_sample, gen_sample))

            print("Orbit counts distribution...")
            orbit_mmd.append(stats.orbit_stats_all(real_sample, gen_sample))

            print("NSPDK...")
            nspdk_mmd.append(stats.nspdk_stats(real_sample, gen_sample))

            print("Node labels distribution...")
            node_label_mmd.append(stats.node_label_stats(real_sample, gen_sample))

            print("Edge labels distribution...")
            edge_label_mmd.append(stats.edge_label_stats(real_sample, gen_sample))

            print("Node labels + Degree distribution...")
            node_label_and_degree.append(stats.node_label_and_degree_joint_stats(real_sample, gen_sample))

            print_stats(
                node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
                degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
                edge_label_mmd, node_label_and_degree
            )
        print("----------Final Results----------")
        print_stats(
            node_count_avg_ref, node_count_avg_pred, edge_count_avg_ref, edge_count_avg_pred,
            degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd, node_label_mmd,
            edge_label_mmd, node_label_and_degree
        )

        results = {
            "Novelty": novelty_score,
            "Uniqueness": uniqueness_score,
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

        filename = self.dirs.eval / f"results_{epoch:02d}.pkl"
        save_pickle(results, filename)

    def _setup_dirs(self, root_dir):
        dirs = super()._setup_dirs(root_dir)
        dirs.samples = get_or_create_dir(dirs.exp / "samples")
        dirs.eval = get_or_create_dir(dirs.exp / "evaluation")
        return dirs
