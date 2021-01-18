from core.module import BaseModule
from core.serialization import save_pickle
from core.utils import get_or_create_dir


class Generator(BaseModule):
    def __init__(self, model_name, root_dir, dataset_name, hparams, gpu):
        super().__init__(model_name, root_dir, dataset_name, hparams, gpu)
        self.dataset = self.dataset_class(dataset_name)

    def generate(self, epoch, device):
        ckpt_path = list(self.dirs.ckpt.glob(f"epoch={epoch}-*.ckpt"))[0]
        wrapper = self.load_wrapper(ckpt_path)
        samples = self.get_samples(wrapper.model, device)
        save_pickle(samples, self.dirs.samples / f"samples_{epoch:02d}.pkl")

    def get_samples(self, model, device):
        raise NotImplementedError

    def load_wrapper(self, ckpt_path):
        raise NotImplementedError

    def _setup_dirs(self, root_dir):
        dirs = super()._setup_dirs(root_dir)
        dirs.samples = get_or_create_dir(dirs.exp / "samples")
        return dirs