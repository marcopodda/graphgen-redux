import os
from pathlib import Path
import time
import multiprocessing
import itertools


def time_elapsed(start, end):
    return time.strftime("%H:%M:%S", time.gmtime(end - start))


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