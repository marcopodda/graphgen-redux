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


def mapping(graphs):
    """
    :param path: path to folder which contains pickled networkx graphs
    :param dest: place where final dictionary pickle file is stored
    :return: dictionary of 4 dictionary which contains forward
    and backwards mappings of vertices and labels, max_nodes and max_edges
    """

    node_forward, node_backward = {}, {}
    edge_forward, edge_backward = {}, {}
    node_count, edge_count = 0, 0
    max_nodes, max_edges, max_degree = 0, 0, 0
    min_nodes, min_edges = float('inf'), float('inf')

    for G in graphs:
        max_nodes = max(max_nodes, len(G.nodes()))
        min_nodes = min(min_nodes, len(G.nodes()))
        for _, data in G.nodes.data():
            if data['label'] not in node_forward:
                node_forward[data['label']] = node_count
                node_backward[node_count] = data['label']
                node_count += 1

        max_edges = max(max_edges, len(G.edges()))
        min_edges = min(min_edges, len(G.edges()))
        for _, _, data in G.edges.data():
            if data['label'] not in edge_forward:
                edge_forward[data['label']] = edge_count
                edge_backward[edge_count] = data['label']
                edge_count += 1

        max_degree = max(max_degree, max([d for _, d in G.degree()]))

    feature_map = {
        'node_forward': node_forward,
        'node_backward': node_backward,
        'edge_forward': edge_forward,
        'edge_backward': edge_backward,
        'max_nodes': max_nodes,
        'min_nodes': min_nodes,
        'max_edges': max_edges,
        'min_edges': min_edges,
        'max_degree': max_degree
    }

    print('Successfully done node count', node_count)
    print('Successfully done edge count', edge_count)

    return feature_map

def graph_from_dfscode(dfscode):
    graph = nx.Graph()

    for dfscode_egde in dfscode:
        i, j, l1, e, l2 = dfscode_egde
        graph.add_node(int(i), label=l1)
        graph.add_node(int(j), label=l2)
        graph.add_edge(int(i), int(j), label=e)

    return graph


def graph_from_reduced_dfscode(reduced_dfscode):
    graph = nx.Graph()

    for reduced_dfscode_egde in reduced_dfscode:
        i, j, token = reduced_dfscode_egde
        l1, e, l2 = token
        graph.add_node(int(i), label=l1)
        graph.add_node(int(j), label=l2)
        graph.add_edge(int(i), int(j), label=e)

    return graph
