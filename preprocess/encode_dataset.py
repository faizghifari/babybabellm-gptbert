# save as encode_and_convert.py
import os
import numpy as np
import torch
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from multiprocessing import cpu_count

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
# 3. Parallel encoding
# -----------------------------
def encode_batch(batch, tokenizer):
    return {"input_ids": [tokenizer.encode(text).ids for text in batch["text"]]}

def parallel_encode(dataset, tokenizer, batch_size=5000):
    num_cpus = cpu_count()
    print(f"Encoding dataset using {num_cpus} cores...")
    tokenized_ds = dataset.map(
        lambda batch: encode_batch(batch, tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=num_cpus
    )
    print("✅ Parallel encoding complete")
    return tokenized_ds

# -----------------------------
# 4. Save as streaming .bin files (uint16)
# -----------------------------
def save_bin_stream(dataset, path, chunk_size=100_000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        buffer = []
        for ids in dataset["input_ids"]:
            buffer.extend(ids)
            if len(buffer) >= chunk_size:
                np.array(buffer, dtype=np.uint16).tofile(f)
                buffer = []
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f)
    print(f"✅ Saved {path} ({len(dataset['input_ids'])} examples)")

# -----------------------------
# 5. Convert uint16 .bin → Torch tensor .bin
# -----------------------------
def convert_uint16_bin_to_torch(input_bin_path, output_bin_path):
    data = np.fromfile(input_bin_path, dtype=np.uint16)
    print(f"✅ Loaded {len(data)} tokens from {input_bin_path}")
    tensor = torch.from_numpy(data).long()
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    torch.save(tensor, output_bin_path)
    print(f"✅ Saved PyTorch-compatible .bin to {output_bin_path}")

# -----------------------------
# 6. Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode BabyLM multilingual dataset and convert .bin to Torch format")
    parser.add_argument("--tokenizer", type=str, default="../tokenizers/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--train_bin", type=str, default="../data/babybabellm_all.bin", help="Path to save train .bin (uint16)")
    parser.add_argument("--val_bin", type=str, default="../data/dev_babybabellm.bin", help="Path to save validation .bin (uint16)")
    parser.add_argument("--train_torch", type=str, default="../data/babybabellm_all_torch.bin", help="Path to save train Torch .bin")
    parser.add_argument("--val_torch", type=str, default="../data/dev_babybabellm_torch.bin", help="Path to save validation Torch .bin")
    args = parser.parse_args()

    # Step 1: Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Step 2: Load datasets
    print("Loading multilingual dataset...")
    multiling_ds = load_all_splits(langs)
    print(multiling_ds)

    # Step 3: Encode
    tokenized_ds = DatasetDict({
        split: parallel_encode(multiling_ds[split], tokenizer)
        for split in ["train", "validation"]
    })

    # Step 4: Save uint16 bins
    save_bin_stream(tokenized_ds["train"], args.train_bin)
    save_bin_stream(tokenized_ds["validation"], args.val_bin)

    # Step 5: Convert to Torch bins
    convert_uint16_bin_to_torch(args.train_bin, args.train_torch)
    convert_uint16_bin_to_torch(args.val_bin, args.val_torch)

    print("🎉 All done! Tokenized + Torch data saved.")

