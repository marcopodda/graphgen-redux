from core.module import BaseModule
from core.serialization import save_pickle
from core.utils import get_or_create_dir


class Generator(BaseModule):
    def __init__(self, model_name, root_dir, dataset_name, epochs, hparams, gpu):
        super().__init__(model_name, root_dir, dataset_name, epochs, hparams, gpu)
        self.dataset = self.dataset_class(dataset_name)
        self.num_samples = 2560
        self.num_runs = 10 if dataset_name != "ENZYMES" else 64
        self.batch_size = 256 if dataset_name != "ENZYMES" else 40
        self.max_nodes = self.dataset.mapper['max_nodes']
        self.min_nodes = self.dataset.mapper['min_nodes']
        self.max_edges = self.dataset.mapper['max_edges']
        self.min_edges = self.dataset.mapper['min_edges']

    def generate(self, epoch, device):
        fname = f"epoch={epoch}.ckpt" if epoch else "last.ckpt"
        ckpt_path = self.dirs.ckpt / fname
        wrapper = self.load_wrapper(ckpt_path)
        samples = self.get_samples(wrapper.model, device)
        fname = f"{epoch}" if epoch else "last"
        save_pickle(samples, self.dirs.samples / f"samples_{fname}.pkl")

    def get_samples(self, model, device):
        raise NotImplementedError

    def load_wrapper(self, ckpt_path):
        raise NotImplementedError

    def _setup_dirs(self, root_dir):
        dirs = super()._setup_dirs(root_dir)
        dirs.samples = get_or_create_dir(dirs.exp / "samples")
        return dirs