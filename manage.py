import argparse
from pathlib import Path

from core.settings import DATASETS
from datasets.create_dataset import create_dataset
from datasets.preprocess_dataset import preprocess_dataset


def command_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    sub_create = sub.add_parser('create', help="Create a given dataset.")
    sub_create.add_argument("--dataset-name", choices=DATASETS, default="cora")
    sub_create.set_defaults(command='create')

    sub_dfscode = sub.add_parser('dfscode', help="Create DFS codes.")
    sub_dfscode.add_argument("--dataset-name", choices=DATASETS, default="cora")
    sub_dfscode.set_defaults(command='dfscode')

    sub_train = sub.add_parser('train', help="Train.")
    # sub_train.add_argument("--exp-name", default="Experiment", help="Experiment name.")
    # sub_train.add_argument("--dataset-name", default="drd2", help="Dataset name.")
    # sub_train.add_argument("--hparams-file", default="hparams.yml", help="HParams file.")
    # sub_train.add_argument("--root-dir", default="RESULTS", help="Output folder.")
    # sub_train.add_argument("--gpu", default=0, help="GPU number.", type=int)
    # sub_train.add_argument("--debug", default=False, action="store_true", help="Debug mode.")
    sub_train.set_defaults(command='train')

    # sub_validate = sub.add_parser('validate', help="Validation.")
    # sub_validate.add_argument("--exp-path", help="Experiment path.")
    # sub_validate.add_argument("--epoch", help="Epoch to validate.", type=int)
    # sub_validate.add_argument("--beam-size", default=64, help="Beam size.", type=int)
    # sub_validate.add_argument("--batch-size", default=128, help="Batch size.", type=int)
    # sub_validate.add_argument("--num-samples", default=20, help="Beam size.", type=int)
    # sub_validate.add_argument("--gpu", default=None, help="GPU number.", type=int)
    # sub_validate.set_defaults(command='validate')

    # sub_test = sub.add_parser('test', help="Test.")
    # sub_test.add_argument("--exp-path", help="Experiment path.")
    # sub_test.add_argument("--epoch", help="Epoch to test.", type=int)
    # sub_test.add_argument("--beam-size", default=64, help="Beam size.", type=int)
    # sub_test.add_argument("--batch-size", default=128, help="Batch size.", type=int)
    # sub_test.add_argument("--num-samples", default=20, help="Beam size.", type=int)
    # sub_test.add_argument("--gpu", default=0, help="GPU number.", type=int)
    # sub_test.set_defaults(command='test')

    return parser


if __name__ == "__main__":
    parser = command_parser()
    args = parser.parse_args()

    if args.command == "create":
        create_dataset(args.dataset_name)

    if args.command == "dfscode":
        preprocess_dataset(args.dataset_name)

    # elif args.command == "train":
    #     translator = Translator.from_args(args)
    #     translator.train()

    # elif args.command == "validate":
    #     translator = Translator.load(Path(args.exp_path))
    #     translator.validate(
    #         epoch=args.epoch,
    #         beam_size=args.beam_size,
    #         batch_size=args.batch_size,
    #         num_samples=args.num_samples,
    #         gpu=args.gpu)

    # elif args.command == "test":
    #     translator = Translator.load(Path(args.exp_path))
    #     translator.test(
    #         epoch=args.epoch,
    #         beam_size=args.beam_size,
    #         batch_size=args.batch_size,
    #         num_samples=args.num_samples,
    #         gpu=args.gpu)