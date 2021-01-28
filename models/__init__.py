from models.graphgen.model import ReducedGraphgenTrainer
from models.graphgen.generator import ReducedGraphgenGenerator


MODEL_CONFIG = {
    "reduced-graphgen": {
        "trainer": ReducedGraphgenTrainer,
        "generator": ReducedGraphgenGenerator,
        "hparams": "hparams.yml"
    }
}