from models.dgmg.model import DGMGTrainer
from models.dgmg.generator import DGMGGenerator

from models.graphgen.vanilla.model import GraphgenTrainer
from models.graphgen.vanilla.generator import GraphgenGenerator

from models.graphgen.reduced.model import ReducedGraphgenTrainer
from models.graphgen.reduced.generator import ReducedGraphgenGenerator

from models.graphrnn.model import GraphRNNTrainer
from models.graphrnn.generator import GraphRNNGenerator



MODEL_CONFIG = {
    "graphgen": {
        "trainer": GraphgenTrainer,
        "generator": GraphgenGenerator,
        "hparams": "hparams_graphgen.yml"
    },
    "reduced-graphgen": {
        "trainer": ReducedGraphgenTrainer,
        "generator": ReducedGraphgenGenerator,
        "hparams": "hparams_reduced-graphgen.yml"
    },
    "graphrnn": {
        "trainer": GraphRNNTrainer,
        "generator": GraphRNNGenerator,
        "hparams": "hparams_graphrnn.yml"
    },
    "dgmg": {
        "trainer": DGMGTrainer,
        "generator": DGMGGenerator,
        "hparams": "hparams_dgmg.yml"
    },
}