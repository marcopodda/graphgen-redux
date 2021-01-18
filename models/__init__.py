from models.graphgen.vanilla.model import GraphgenTrainer
from models.graphgen.vanilla.generator import GraphgenGenerator

from models.graphgen.reduced.model import ReducedGraphgenTrainer
from models.graphgen.reduced.generator import ReducedGraphgenGenerator



MODEL_CONFIG = {
    "graphgen": {
        "trainer": GraphgenTrainer,
        "generator": GraphgenGenerator
    },
    "reduced-graphgen": {
        "trainer": ReducedGraphgenTrainer,
        "generator": ReducedGraphgenGenerator
    },
    "graphrnn": {
        "trainer": None,
        "generator": None
    },
    "DGMG": {
        "trainer": None,
        "generator": None
    },
}