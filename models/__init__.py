from models.dgmg.model import DGMGTrainer
from models.dgmg.generator import DGMGGenerator

from models.graphgen.vanilla.model import GraphgenTrainer
from models.graphgen.vanilla.generator import GraphgenGenerator

from models.graphgen.reduced.model import ReducedGraphgenTrainer
from models.graphgen.reduced.generator import ReducedGraphgenGenerator



MODEL_CONFIG = {
    "graphgen": {
        "trainer": GraphgenTrainer,
        "generator": GraphgenGenerator,
        "hparams": "hparams.yml"
    },
    "reduced-graphgen": {
        "trainer": ReducedGraphgenTrainer,
        "generator": ReducedGraphgenGenerator,
        "hparams": "hparams.yml"
    },
    "graphrnn": {
        "trainer": None,
        "generator": None,
        "hparams": "hparams_graphrnn.yml"
    },
    "dgmg": {
        "trainer": DGMGTrainer,
        "generator": DGMGGenerator,
        "hparams": "hparams_dgmg.yml"
    },
}