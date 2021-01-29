import argparse
from pathlib import Path

from core.settings import DATASETS
from datasets.create_dataset import create_dataset
from datasets.preprocess_dataset import preprocess_dataset
from evaluation.evaluator import Evaluator

from models import MODEL_CONFIG


def command_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    sub_create = sub.add_parser('create', help="Create a given dataset.")
    sub_create.add_argument("--dataset-name", choices=DATASETS, default="cora")
    sub_create.set_defaults(command='create')

    sub_preprocess = sub.add_parser('preprocess', help="Create DFS codes.")
    sub_preprocess.add_argument("--dataset-name", choices=DATASETS, default="cora")
    sub_preprocess.set_defaults(command='preprocess')

    sub_train = sub.add_parser('train', help="Training.")
    sub_train.add_argument("--model-name", choices=MODEL_CONFIG.keys(), default="graphgen-redux")
    sub_train.add_argument("--dataset-name", choices=DATASETS, default="cora")
    sub_train.add_argument("--epochs", type=int, default=50)
    sub_train.add_argument("--root-dir", default="RESULTS", help="Output folder.")
    sub_train.add_argument("--gpu", default=0, help="GPU number.", type=int)
    sub_train.set_defaults(command='train')

    sub_generate = sub.add_parser('generate', help="Generation.")
    sub_generate.add_argument("--exp-path", help="Experiment path.")
    sub_generate.add_argument("--gpu", default=None, help="GPU number.", type=int)
    sub_generate.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch.")
    sub_generate.set_defaults(command='generate')

    sub_evaluate = sub.add_parser('evaluate', help="Evaluation.")
    sub_evaluate.add_argument("--exp-path", help="Experiment path.")
    sub_evaluate.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch.")
    sub_evaluate.set_defaults(command='evaluate')

    return parser


if __name__ == "__main__":
    parser = command_parser()
    args = parser.parse_args()

    if args.command == "create":
        create_dataset(args.dataset_name)

    elif args.command == "preprocess":
        preprocess_dataset(args.dataset_name)

    elif args.command == "train":
        args.hparams_file = MODEL_CONFIG[args.model_name]["hparams"]
        trainer_class = MODEL_CONFIG[args.model_name]["trainer"]
        trainer = trainer_class.from_args(args)
        trainer.train()

    elif args.command == "generate":
        exp_path = Path(args.exp_path)
        model_name = exp_path.parts[-1]
        generator_class = MODEL_CONFIG[model_name]["generator"]
        generator = generator_class.initialize(exp_path)
        generator.generate(epoch=args.epoch, device=args.gpu)

    elif args.command == "evaluate":
        evaluator = Evaluator.initialize(Path(args.exp_path))
        evaluator.evaluate(epoch=args.epoch)