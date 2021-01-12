from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import pickle
import os
import torch

MAX_WORKERS = 48

#### DFS reduction into Reduced DFS

def reduce_mindfs_codes(min_dfscodes_path, reduced_dfscode_path):
  min_dfscodes = []
  for filename in os.listdir(min_dfscodes_path):
    if filename.endswith(".dat"):
      min_dfscodes.append(filename)
  # with Pool(processes=MAX_WORKERS) as pool:
  #   for i, _ in tqdm(enumerate(pool.imap_unordered(
  #           partial(reduce_dfs_code_file, min_dfscodes_path=min_dfscodes_path,
  #                   reduced_dfscode_path=reduced_dfscode_path, feature_map=feature_map),
  #           min_dfscodes, chunksize=16), 1)):
  #       pass

  dfs_to_reduced = {}
  reduced_to_dfs = {}

  for el in min_dfscodes:
    reduce_dfs_code_file(el, min_dfscodes_path, reduced_dfscode_path, dfs_to_reduced, reduced_to_dfs)

  # Save both dictionaries
  reduced_map = {}
  # Triple to token
  reduced_map['dfs_to_reduced'] = dfs_to_reduced
  # Token to triple
  reduced_map['reduced_to_dfs'] = reduced_to_dfs

  f = open(reduced_dfscode_path + 'token_map.dict', 'wb')
  pickle.dump(reduced_map, f)
  f.close()

# Reduce files and save
def reduce_dfs_code_file(min_dfscode_file, min_dfscodes_path, reduced_dfscode_path, dfs_to_reduced, reduced_to_dfs):
   
    with open(min_dfscodes_path + min_dfscode_file, 'rb') as f:
      min_dfscode = pickle.load(f)

    reduced_dfscode = reduce_dfscode(min_dfscode, dfs_to_reduced, reduced_to_dfs)

    with open(reduced_dfscode_path + min_dfscode_file, 'wb') as f:
      pickle.dump(reduced_dfscode, f)

# Reduce single file
def reduce_dfscode(min_dfscode, dfs_to_reduced, reduced_to_dfs):
  key = len(dfs_to_reduced)
  reduced_list = []

  for quintuple in min_dfscode:
    if quintuple[2:] in reduced_to_dfs.values():
      # Find token associated to that triple
      k = tuple(quintuple[2:])
      token = dfs_to_reduced[k]
      reduced_list.append(quintuple[:2]+[token])
    else:
      # Create new token
      reduced_to_dfs[key] = quintuple[2:]
      dfs_to_reduced[tuple(quintuple[2:])] = key
      reduced_list.append(quintuple[:2]+[key])
      key+=1
  
  return reduced_list

#### Reduced DFS into tensors

# Reduce to tensors
def dfscode_to_tensor(dfscode, feature_map, reduced_map):
    max_nodes, max_edges = feature_map['max_nodes'], feature_map['max_edges']
    node_forward_dict, edge_forward_dict = feature_map['node_forward'], feature_map['edge_forward']
    num_nodes_feat, num_edges_feat = len(
        feature_map['node_forward']), len(feature_map['edge_forward'])
    num_token_feat = len(reduced_map['reduced_to_dfs'])

    # max_nodes, num_nodes_feat and num_edges_feat are end token labels
    # So ignore tokens are one higher
    reduced_dfscode_tensors = {
        't1': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        't2': (max_nodes + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'token': (num_token_feat + 1) * torch.ones(max_edges + 1, dtype=torch.long),
        'len': len(dfscode)
    }

    for i, code in enumerate(dfscode):
        reduced_dfscode_tensors['t1'][i] = int(code[0])
        reduced_dfscode_tensors['t2'][i] = int(code[1])
        reduced_dfscode_tensors['token'][i] = int(code[2])


    # Add end token
    reduced_dfscode_tensors['t1'][len(dfscode)], reduced_dfscode_tensors['t2'][len(
        dfscode)] = max_nodes, max_nodes
    reduced_dfscode_tensors['token'][len(dfscode)] = num_token_feat

    return reduced_dfscode_tensors


def reduced_dfscode_from_file_to_tensor_to_file(
    reduced_dfscode_file, reduced_dfscodes_path, reduced_dfscode_tensors_path, feature_map, reduced_map
):
    with open(reduced_dfscodes_path + reduced_dfscode_file, 'rb') as f:
        reduced_dfscode = pickle.load(f)

    reduced_dfscode_tensors = dfscode_to_tensor(reduced_dfscode, feature_map, reduced_map)

    with open(reduced_dfscode_tensors_path + reduced_dfscode_file, 'wb') as f:
        pickle.dump(reduced_dfscode_tensors, f)


def reduced_dfscodes_to_tensors(reduced_dfscodes_path, reduced_dfscode_tensors_path, feature_map):
    """
    :param min_dfscodes_path: Path to directory of pickled min dfscodes
    :param min_dfscode_tensors_path: Path to directory to store the min dfscode tensors
    :param feature_map:
    :return: length of dataset
    """
    reduced_dfscodes = []
    for filename in os.listdir(reduced_dfscodes_path):
        if filename.endswith(".dat"):
            reduced_dfscodes.append(filename)

    with open(reduced_dfscodes_path + 'token_map.dict', 'rb') as f:
      reduced_map = pickle.load(f)

    with Pool(processes=MAX_WORKERS) as pool:
        for i, _ in tqdm(enumerate(pool.imap_unordered(
                partial(reduced_dfscode_from_file_to_tensor_to_file, reduced_dfscodes_path=reduced_dfscodes_path,
                        reduced_dfscode_tensors_path=reduced_dfscode_tensors_path, feature_map=feature_map, reduced_map=reduced_map),
                reduced_dfscodes, chunksize=16), 1)):
            pass

            # if i % 10000 == 0:
            #     print('Processed', i, 'graphs')




