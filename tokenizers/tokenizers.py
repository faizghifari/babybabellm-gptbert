import os
import json
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, normalizers, decoders
from multiprocessing import Pool, cpu_count

# -----------------------------
# 1. Languages
# -----------------------------
langs = [
    "zho","nld","deu","fra","ind","fas","ukr","bul",
    "yue","est","swe","cym","pol","afr","eus","ita","spa","por","jpn","heb","srp","ara","ell",
    "bug","hun","tur","ces","ace","dan","ban","hrv","mak","nso","ron","nor","isl","zul","sot","xho","kor","rus","sun","jav"
]

# -----------------------------
# 2. Load + merge all splits
# -----------------------------
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
# 3. Tokenizer training
# -----------------------------
special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [f"<special_{i}>" for i in range(11)]
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
trainer = BpeTrainer(vocab_size=32768, special_tokens=special_tokens)

# Streaming iterator for training
def iterator_stream(batch_size=10000):
    ds = multiling_ds["train"]
    for i in range(0, len(ds), batch_size):
        yield ds[i:i+batch_size]["text"]

print("Training tokenizer (streaming)...")
tokenizer.train_from_iterator(iterator_stream(batch_size=10000), trainer)

# -----------------------------
# 4. Save tokenizer in script folder
# -----------------------------
script_dir = os.path.dirname(__file__)
tok_path = os.path.join(script_dir, "tokenizer.json")
tokenizer.save(tok_path)
print(f"✅ Tokenizer saved at {tok_path}")

