import os
import numpy as np
import torch
import argparse

def load_tokens(filepath):
    """ Load tokens from a shard file. """
    npt = np.load(filepath)
    return torch.tensor(npt, dtype=torch.long)

def count_tokens(data_root, split):
    """ Count total tokens in the dataset. """
    shards = sorted([os.path.join(data_root, s) for s in os.listdir(data_root) if split in s])
    total_tokens = 0
    s = 0
    for shard in shards:
        tokens = load_tokens(shard)
        total_tokens += tokens.numel()  # Number of tokens in this shard
        s += 1
    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count total tokens in the dataset.")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--split", type=str, default="train", help="Data split ('train' or 'val')")
    args = parser.parse_args()

    total_tokens = count_tokens(args.data_root, args.split)
    print(f"Total tokens in the {args.split} split: {total_tokens}")
