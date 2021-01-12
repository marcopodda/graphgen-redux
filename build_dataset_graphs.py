from args import Args
from utils import create_dirs
from datasets.process_dataset import create_graphs
import random

def main():
  args = Args()
  args = args.update_args()

  create_dirs(args)

  random.seed(123)

  print('Building')
  graphs = create_graphs(args)

if __name__ == '__main__':
  main()