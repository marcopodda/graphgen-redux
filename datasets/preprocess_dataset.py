import os
import subprocess
import tempfile
from typing import OrderedDict
from joblib import Parallel, delayed

from core.utils import get_n_jobs
from core.serialization import load_pickle, save_pickle
from core.settings import DATA_DIR, BIN_DIR



def get_min_dfscode(idx, G, temp_path=tempfile.gettempdir()):
    input_fd, input_path = tempfile.mkstemp(dir=temp_path)

    with open(input_path, 'w') as f:
        vcount = len(G.nodes)
        f.write(str(vcount) + '\n')
        i = 0
        d = {}
        for x in G.nodes:
            d[x] = i
            i += 1
            f.write(str(G.nodes[x]['label']) + '\n')

        ecount = len(G.edges)
        f.write(str(ecount) + '\n')
        for (u, v) in G.edges:
            f.write(str(d[u]) + ' ' + str(d[v]) +
                    ' ' + str(G[u][v]['label']) + '\n')

    output_fd, output_path = tempfile.mkstemp(dir=temp_path)

    dfscode_bin_path = BIN_DIR / "dfscode"
    with open(input_path, 'r') as f:
        subprocess.call([dfscode_bin_path, output_path, '2'], stdin=f)

    with open(output_path, 'r') as dfsfile:
        dfs_sequence = []
        for row in dfsfile.readlines():
            splitted_row = row.split()
            splitted_row = [splitted_row[2 * i + 1] for i in range(5)]
            dfs_sequence.append(splitted_row)

    os.close(input_fd)
    os.close(output_fd)

    try:
        os.remove(input_path)
        os.remove(output_path)
    except OSError:
        pass

    return (idx, dfs_sequence)


def graphs_to_min_dfscodes(graphs):
    """
    :param graphs_path: Path to directory of graphs in networkx format
    :param min_dfscodes_path: Path to directory to store the min dfscodes
    :param temp_path: path for temporary files
    :return: length of dataset
    """
    P = Parallel(n_jobs=get_n_jobs(), verbose=1)
    dfs_codes = P(delayed(get_min_dfscode)(i, G) for i, G in enumerate(graphs))
    print('Done creating min dfscodes')
    return list(OrderedDict(dfs_codes).values())


def reduce_dfs_codes(dfs_codes):
    reduced = []
    for dfs_code in dfs_codes:
        reduced.append([[e[0], e[1], "-".join(e[2:])] for e in dfs_code])
    return reduced

def get_unique_symbols(reduced_dfs_codes):
    reduced_forward, reduced_backward, count = {}, {}, 0

    symbols = []
    for reduced_dfs_code in reduced_dfs_codes:
        symbols = [e[2] for e in reduced_dfs_code]
        for sym in symbols:
            if sym not in reduced_backward:
                reduced_backward[sym] = count
                reduced_forward[count] = sym
                count += 1

    return reduced_forward, reduced_backward


def preprocess_dataset(name):
    graphs = load_pickle(DATA_DIR / name / "graphs.pkl")
    if not (DATA_DIR / name / "dfs_codes.pkl").exists():
        dfs_codes = graphs_to_min_dfscodes(graphs)
        save_pickle(dfs_codes, DATA_DIR / name / "dfs_codes.pkl")

    if not (DATA_DIR / name / "reduced_dfs_codes.pkl").exists():
        dfs_codes = load_pickle(DATA_DIR / name / "dfs_codes.pkl")
        reduced_dfs_codes = reduce_dfs_codes(dfs_codes)
        save_pickle(reduced_dfs_codes, DATA_DIR / name / "reduced_dfs_codes.pkl")

    if not (DATA_DIR / name / "reduced_map.dict").exists():
        reduced_dfs_codes = load_pickle(DATA_DIR / name / "reduced_dfs_codes.pkl")
        reduced_forward, reduced_backward = get_unique_symbols(reduced_dfs_codes)
        mapping = load_pickle(DATA_DIR / name / "map.dict")
        mapping["reduced_forward"] = reduced_forward
        mapping["reduced_backward"] = reduced_backward
        save_pickle(mapping, DATA_DIR / name / "reduced_map.dict")
