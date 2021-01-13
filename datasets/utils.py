import numpy as np
import networkx as nx
import pickle


def random_walk_with_restart_sampling(G, start_node, iterations, fly_back_prob=0.15, max_nodes=None, max_edges=None):
    sampled_graph = nx.Graph()
    sampled_graph.add_node(start_node, label=G.nodes[start_node]['label'])

    curr_node = start_node

    for _ in range(iterations):
        if np.random.rand() < fly_back_prob:
            curr_node = start_node
        else:
            neigh = list(G.neighbors(curr_node))
            chosen_node_id = np.random.choice(len(neigh))
            chosen_node = neigh[chosen_node_id]

            sampled_graph.add_node(
                chosen_node, label=G.nodes[chosen_node]['label'])
            sampled_graph.add_edge(
                curr_node, chosen_node, label=G.edges[curr_node, chosen_node]['label'])

            curr_node = chosen_node

        if max_nodes is not None and sampled_graph.number_of_nodes() >= max_nodes:
            break

        if max_edges is not None and sampled_graph.number_of_edges() >= max_edges:
            break

    return sampled_graph


def sample_subgraphs(node_idx, G, iterations, num_factor):
    graphs = []

    deg = G.degree[node_idx]
    for _ in range(num_factor * int(np.sqrt(deg))):
        G_rw = random_walk_with_restart_sampling(G, node_idx, iterations=iterations)
        G_rw = nx.convert_node_labels_to_integers(G_rw)
        G_rw.remove_edges_from(nx.selfloop_edges(G_rw))

        if nx.is_connected(G_rw):
            graphs.append(G_rw)

    return graphs