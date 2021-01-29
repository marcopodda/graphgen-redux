from statistics import mean, stdev

from core.serialization import load_pickle


def fmt_list(values):
    return f"{mean(values):.4f}({stdev(values):.3f})"


def display_results(path):
    results = load_pickle(path)
    print(f"Uniqueness:              {results['Uniqueness']:.4f}")
    print(f"Novelty:                 {results['Novelty']:.4f}")
    print(f"Validity:                {results['Validity']:.4f}")
    print(f"Node count REF:          {fmt_list(results['Node count avg. ref'])}")
    print(f"Node count GEN:          {fmt_list(results['Node count avg. pred'])}")
    print(f"Edge count REF:          {fmt_list(results['Edge count avg. ref'])}")
    print(f"Edge count GEN:          {fmt_list(results['Edge count avg. pred'])}")
    print(f"MMD Degree:              {fmt_list(results['MMD Degree'])}")
    print(f"MMD Clustering:          {fmt_list(results['MMD Clustering'])}")
    print(f"MMD Orbits:              {fmt_list(results['MMD Orbit'])}")
    print(f"MMD NSPDK:               {fmt_list(results['MMD NSPDK'])}")
    print(f"MMD Node Labels:         {fmt_list(results['MMD Node labels'])}")
    print(f"MMD Node Labels:         {fmt_list(results['MMD Edge labels'])}")
    print(f"MMD Node Labels/Degrees: {fmt_list(results['MMD Node labels and degrees'])}")
