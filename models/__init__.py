from models.graphgen.model import ReducedGraphgenTrainer
from models.graphgen.generator import ReducedGraphgenGenerator


MODEL_CONFIG = {
    "graphgen-redux": {
        "trainer": ReducedGraphgenTrainer,
        "generator": ReducedGraphgenGenerator,
        "hparams": "hparams.yml"
    }
}