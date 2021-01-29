import os
from pathlib import Path
import time
import multiprocessing
import itertools
import networkx as nx
from rdkit import Chem


BOND_TYPES = {
    "1": Chem.rdchem.BondType.SINGLE,
    "2": Chem.rdchem.BondType.DOUBLE,
    "3": Chem.rdchem.BondType.TRIPLE,
}


def time_elapsed(t):
    return time.strftime("%H:%M:%S", time.gmtime(t))


def get_n_jobs():
    num_cpus = multiprocessing.cpu_count()
    return int(0.75 * num_cpus)


def get_or_create_dir(path):
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
    return path


def dir_is_empty(path):
    return not bool(list(Path(path).rglob("*")))


def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def nx_to_mol(G):
    mol = Chem.RWMol()
    pt = Chem.GetPeriodicTable()
    atomic_symbols = nx.get_node_attributes(G, 'label')
    node_to_idx = {}
    for node in G.nodes():
        sym = atomic_symbols[node]
        atomic_num = pt.GetAtomicNumber(sym)
        a = Chem.Atom(atomic_num)
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'label')
    for bt in bond_types.keys():
        bond_types[bt] = BOND_TYPES[bond_types[bt]]

    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None