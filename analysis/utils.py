from statistics import mean
from pathlib import Path
from core.serialization import load_pickle

RESULTS_DIR = Path("RESULTS")

MODELS = [
    "graphgen-redux",
    "graphgen-full",
    "graphgen-lw"
]

def load_results_on_dataset(dataset_name, last=False):
    dataset_dir = RESULTS_DIR / dataset_name

    dataset_results = {
        "Validity": [],
        "Novelty": [],
        "Uniqueness": [],
        "Node count avg. ref": [],
        "Node count avg. pred": [],
        "Edge count avg. ref": [],
        "Edge count avg. pred": [],
        "MMD Degree": [],
        "MMD Clustering": [],
        "MMD Orbit": [],
        "MMD NSPDK": [],
        "MMD Node labels": [],
        "MMD Edge labels": [],
        "MMD Node labels and degrees": []
    }

    for model_name in MODELS:
        exp_dir = dataset_dir / model_name
        eval_dir = exp_dir / "evaluation"
        idx = 1 if last else 0
        results_path = sorted(eval_dir.glob("*.pkl"))[idx]
        results = load_pickle(results_path)
        for key in results:
            try:
                res = mean(results[key])
            except TypeError:
                res = results[key]
            dataset_results[key].append(res)

    return dataset_results
