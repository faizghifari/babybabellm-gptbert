# save as encode_dataset.py
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from multiprocessing import Pool, cpu_count

# -----------------------------
# 1. Load trained tokenizer
# -----------------------------
tokenizer_path = "../tokenizers/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)
print(f"✅ Loaded tokenizer from {tokenizer_path}")

# -----------------------------
# 2. Define languages and load dataset
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
            train_ds = split["train"]
            val_ds = split["test"]
        splits["train"].append(train_ds)
        splits["validation"].append(val_ds)

    return DatasetDict({
        split: concatenate_datasets(splits[split]) 
        for split in splits
    })

print("Loading multilingual dataset...")
multiling_ds = load_all_splits(langs)
print(multiling_ds)

# -----------------------------
# 3. Parallel encoding
# -----------------------------
def encode_batch(batch):
    return {"input_ids": [tokenizer.encode(text).ids for text in batch["text"]]}

def parallel_encode(dataset, batch_size=5000):
    num_cpus = cpu_count()
    print(f"Encoding dataset using {num_cpus} cores...")
    tokenized_ds = dataset.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=num_cpus
    )
    print("✅ Parallel encoding complete")
    return tokenized_ds

tokenized_ds = DatasetDict({
    split: parallel_encode(multiling_ds[split])
    for split in ["train", "validation"]
})

# -----------------------------
# 4. Save as streaming .bin files
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

train_bin = "../data/babybabellm_all.bin"
val_bin = "../data/dev_babybabellm.bin"
save_bin_stream(tokenized_ds["train"], train_bin)
save_bin_stream(tokenized_ds["validation"], val_bin)

print("All done! Tokenized data saved.")
