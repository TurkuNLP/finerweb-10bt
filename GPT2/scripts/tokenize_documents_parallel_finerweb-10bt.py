import os
import argparse
import multiprocessing as mp
from datasets import load_from_disk
from gpt_from_scratch.dataset import Tokenizer, ShardManager

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenizes documents using multiprocessing.")
    parser.add_argument("--local_dir", type=str, default="exquisiteweb", help="Local directory to store data.")
    parser.add_argument("--dataset_path", type=str, default="exquisiteweb", help="Root path to the dataset shards.")
    parser.add_argument("--tokenizer_model", type=str, default="gpt2", help="Tokenizer model to use.")
    parser.add_argument("--shard_size", type=int, default=int(1e8), help="Number of tokens per shard.")
    parser.add_argument("--quality_threshold", type=float, default=0.1, help="Quality threshold for filtering lines.")
    return parser.parse_args()

# Preprocess each row based on the 'text' and 'line_quality' keys
def preprocess_row(row, quality_threshold):
    text_lines = row['text'].splitlines()
    line_quality = row['line_quality']

    # Filter lines based on the quality threshold
    filtered_lines = [line for line, quality in zip(text_lines, line_quality) if quality >= quality_threshold]

    # Rebuild the text by joining the filtered lines with newline characters
    new_text = "\n".join(filtered_lines)

    # Replace the 'text' field with the new preprocessed text
    row['text'] = new_text
    return row

# Generator to stream and preprocess data from multiple shards
def stream_from_shards(root_path, quality_threshold):
    # Find all directories that start with 'shard_' in the root path
    shard_dirs = [os.path.join(root_path, d) for d in os.listdir(root_path) if d.startswith("shard_")]
    
    # Stream rows from each shard directory
    for shard_dir in shard_dirs:
        dataset = load_from_disk(shard_dir)
        for row in dataset:
            # Preprocess the row and yield it
            yield preprocess_row(row, quality_threshold)

def main():
    args = parse_args()

    current_script_path = os.path.dirname(__file__)
    project_root = os.path.dirname(current_script_path)
    data_dir = os.path.join(project_root, 'data')
    DATA_CACHE_DIR = os.path.join(data_dir, args.local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Stream and preprocess rows from all shard directories in 'exquisiteweb'
    fw = stream_from_shards(args.dataset_path, args.quality_threshold)

    # Initialize Tokenizer and ShardManager
    tokenizer = Tokenizer(args.tokenizer_model)
    shard_manager = ShardManager(DATA_CACHE_DIR, args.shard_size)

    # Use multiprocessing to tokenize documents
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(tokenizer.tokenize_doc, fw, chunksize=16):
            shard_manager.add_tokens(tokens)

    shard_manager.finalize()

if __name__ == "__main__":
    main()

