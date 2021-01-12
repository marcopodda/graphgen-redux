import matplotlib.pyplot as plt
import networkx as nx
from utils import load_graphs
from datasets.process_dataset import create_graphs
from args import Args
import sys
import pickle

def main():
  if len(sys.argv) > 1:
    graphs_path = sys.argv[1]
  else:
    raise Exception('Missing path')

  num = None
  if len(sys.argv) > 2:
    num = int(sys.argv[2])

  print('Loading')
  if num is None:
    graphs_pred = load_graphs(graphs_path)
  else:
    graphs_pred = load_graphs(graphs_path, range(num))

  print('Drawing')  
  for i, G in enumerate(graphs_pred):
    print('Graph', i)

    # pos=nx.spring_layout(G)
    pos=nx.kamada_kawai_layout(G)
    #nx.draw_networkx(G, pos, with_labels=False, node_color='#7FDBFF', alpha=0.9)
    nx.draw_networkx_nodes(G, pos, edgecolors='black', node_color='#C0C0C0')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G, 'label'))
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, 'label'))
    plt.show()

if __name__ == "__main__":
  main()