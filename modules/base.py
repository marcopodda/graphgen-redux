from pathlib import Path
from argparse import Namespace

from datasets.datasets import Dataset
from core.serialization import load_yaml
from core.hparams import HParams
from core.utils import get_or_create_dir

class Base:
    @classmethod
    def initialize(cls, exp_dir):
        exp_dir = Path(exp_dir)
        config = load_yaml(exp_dir / "config.yml")
        return cls(**config)

    def __init__(self, exp_name, root_dir, dataset_name, reduced, hparams, gpu, debug):
        if not exp_name.endswith("_red"):
            exp_name = f"{exp_name}_red" if reduced else exp_name

        self.exp_name = exp_name
        self.dataset_name = dataset_name
        self.reduced = reduced
        self.hparams = HParams.load(hparams)
        self.gpu = gpu
        self.debug = debug
        self.dirs = self._setup_dirs(root_dir)

        self.dataset = Dataset(dataset_name, reduced)
        self.mapper = self.dataset.mapper

    def _setup_dirs(self, root_dir):
        root_dir = get_or_create_dir(root_dir)
        base_dir = get_or_create_dir(root_dir / self.dataset_name)
        exp_dir = get_or_create_dir(base_dir / self.exp_name)

        dirs = Namespace(
            root=get_or_create_dir(root_dir),
            base=get_or_create_dir(base_dir),
            exp=get_or_create_dir(exp_dir),
            ckpt=get_or_create_dir(exp_dir / "checkpoints"),
            logs=get_or_create_dir(exp_dir / "logs"),
            samples=get_or_create_dir(exp_dir / "samples"))

        return dirs