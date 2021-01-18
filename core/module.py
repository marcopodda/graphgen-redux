from pathlib import Path
from argparse import Namespace

from core.serialization import save_yaml, load_yaml
from core.hparams import HParams
from core.utils import get_or_create_dir


class BaseModule:
    dataset_class = None

    @classmethod
    def initialize(cls, exp_dir):
        exp_dir = Path(exp_dir)
        config = load_yaml(exp_dir / "config.yml")
        return cls(**config)

    def __init__(self, model_name, root_dir, dataset_name, hparams, gpu):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hparams = HParams.load(hparams)
        self.gpu = gpu
        self.dirs = self._setup_dirs(root_dir)

    def _setup_dirs(self, root_dir):
        root_dir = get_or_create_dir(root_dir)
        base_dir = get_or_create_dir(root_dir / self.dataset_name)
        exp_dir = get_or_create_dir(base_dir / self.model_name)

        dirs = Namespace(
            root=get_or_create_dir(root_dir),
            base=get_or_create_dir(base_dir),
            exp=get_or_create_dir(exp_dir),
            ckpt=get_or_create_dir(exp_dir / "checkpoints"))

        return dirs

    def dump(self):
        config = {
            "model_name": self.model_name,
            "root_dir": self.dirs.root.as_posix(),
            "dataset_name": self.dataset_name,
            "hparams": self.hparams.__dict__,
            "gpu": self.gpu
        }
        save_yaml(config, self.dirs.exp / "config.yml")