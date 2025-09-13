import os
import torch
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# -----------------------------
# 1. Load trained tokenizer
# -----------------------------
def load_tokenizer(tokenizer_path="../tokenizers/tokenizer.json"):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"✅ Loaded tokenizer from {tokenizer_path}")
    return tokenizer

# -----------------------------
# 2. Load multilingual BabyLM datasets
# -----------------------------
langs = [
    "zho","nld","deu","fra","ind","fas","ukr","bul",
    "yue","est","swe","cym","pol","afr","eus","ita","spa","por","jpn","heb","srp","ara","ell",
    "bug","hun","tur","ces","ace","dan","ban","hrv","mak","nso","ron","nor","isl","zul","sot","xho","kor","rus","sun","jav"
]

def load_all_splits(langs, dev_fraction=0.05):
    splits = {"train": [], "validation": []}
    for lang in langs:
        ds = load_dataset(f"BabyLM-community/babylm-{lang}")
        train_ds = ds["train"]
        if "validation" in ds:
            val_ds = ds["validation"]
        else:
            split = train_ds.train_test_split(test_size=dev_fraction, seed=42)
            train_ds, val_ds = split["train"], split["test"]
        splits["train"].append(train_ds)
        splits["validation"].append(val_ds)

    return DatasetDict({
        split: concatenate_datasets(splits[split]) 
        for split in splits
    })

# -----------------------------
# 3. Encode batch (for multiprocessing)
# -----------------------------
def encode_text_list(tokenizer_path, text_list):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return [tokenizer.encode(text).ids for text in text_list]

# -----------------------------
# 4. Stream encode → shards with multiprocessing and progress
# -----------------------------
def stream_encode_to_shards_mp(dataset, tokenizer_path, output_dir, seq_length=128, shard_size_bytes=100_000_000, batch_size=1000, max_workers=None):
    os.makedirs(output_dir, exist_ok=True)
    buffer = []
    shard_count = 0
    element_size = torch.tensor([0], dtype=torch.long).element_size()
    num_elements_per_shard = max(seq_length, shard_size_bytes // element_size)

    max_workers = max_workers or cpu_count()
    print(f"Encoding dataset using {max_workers} processes and streaming to {output_dir}...")

    # Create batches
    batches = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    total_batches = len(batches)
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(encode_text_list, tokenizer_path, batch["text"]): idx for idx, batch in enumerate(batches)}

        with tqdm(total=total_batches, desc="Encoding batches") as pbar:
            for future in as_completed(futures):
                enc_batch_ids = future.result()
                for ids in enc_batch_ids:
                    buffer.extend(ids)
                    while len(buffer) >= num_elements_per_shard:
                        shard_tensor = torch.tensor(buffer[:num_elements_per_shard], dtype=torch.long)
                        shard_file = os.path.join(output_dir, f"shard_{shard_count:03d}.bin")
                        torch.save(shard_tensor, shard_file)
                        shard_count += 1
                        buffer = buffer[num_elements_per_shard:]

                # Update progress and ETA
                elapsed = time.time() - start_time
                batches_done = pbar.n + 1
                remaining = total_batches - batches_done
                eta = elapsed / batches_done * remaining if batches_done > 0 else 0
                pbar.set_postfix({"shards": shard_count, "ETA(s)": f"{eta:.1f}"})
                pbar.update(1)

    # Save remaining tokens as final shard
    if buffer:
        shard_tensor = torch.tensor(buffer, dtype=torch.long)
        shard_file = os.path.join(output_dir, f"shard_{shard_count:03d}.bin")
        torch.save(shard_tensor, shard_file)
        print(f"[stream] Saved final shard {shard_count} with {shard_tensor.numel()} tokens (~{shard_tensor.numel()*element_size/1e6:.1f} MB)")

    print(f"🎉 Completed streaming encoding. Total shards: {shard_count+1}")

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode BabyLM multilingual dataset with multiprocessing, stream shards, and show progress")
    parser.add_argument("--tokenizer", type=str, default="../tokenizers/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--seq_length", type=int, default=128, help="Minimum sequence length per shard")
    parser.add_argument("--shard_size_bytes", type=int, default=100_000_000, help="Approximate shard size in bytes")
    parser.add_argument("--shard_dir_train", type=str, default="../data/shards/train", help="Directory to save train shards")
    parser.add_argument("--shard_dir_valid", type=str, default="../data/shards/valid", help="Directory to save validation shards")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for encoding")
    parser.add_argument("--max_workers", type=int, default=None, help="Max number of processes for encoding")
    args = parser.parse_args()

    # Step 1: Load datasets
    print("Loading multilingual dataset...")
    multiling_ds = load_all_splits(langs)
    print(multiling_ds)

    # Step 2: Stream encode → shards with multiprocessing
    stream_encode_to_shards_mp(multiling_ds["train"], args.tokenizer, args.shard_dir_train,
                               seq_length=args.seq_length, shard_size_bytes=args.shard_size_bytes,
                               batch_size=args.batch_size, max_workers=args.max_workers)
    stream_encode_to_shards_mp(multiling_ds["validation"], args.tokenizer, args.shard_dir_valid,
                               seq_length=args.seq_length, shard_size_bytes=args.shard_size_bytes,
                               batch_size=args.batch_size, max_workers=args.max_workers)
