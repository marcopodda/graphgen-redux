from statistics import mean, stdev

from core.serialization import load_pickle

KEYS = [
    "Node count avg. pred",
    "Node count avg. ref",
    "Edge count avg. pred",
    "Edge count avg. ref",
    "MMD Degree",
    "MMD Clustering",
    "MMD NSPKD",
    "MMD Node labels",
    "MMD Edge labels",
    "MMD Node labels and degrees"
]


def display_results(path):
    results = load_pickle(path)
    print(f"Uniqueness:              {results['Uniqueness']:.4f}")
    print(f"Novelty:                 {results['Novelty']:.4f}")
    print(f"Node count REF:          {results['Node count avg. ref']:.4f}")
    print(f"Node count GEN:          {results['Node count avg. pred']:.4f}")
    print(f"Edge count REF:          {results['Edge count avg. ref']:.4f}")
    print(f"Edge count GEN:          {results['Edge count avg. pred']:.4f}")
    print(f"MMD Degree:              {results['MMD Degree']:.4f}")
    print(f"MMD Clustering:          {results['MMD Clustering']:.4f}")
    print(f"MMD NSPDK:               {results['MMD NSPDK']:.4f}")
    print(f"MMD Node Labels:         {results['Node Labels']:.4f}")
    print(f"MMD Node Labels:         {results['Edge Labels']:.4f}")
    print(f"MMD Node Labels/Degrees: {results['Edge Labels']:.4f}")
