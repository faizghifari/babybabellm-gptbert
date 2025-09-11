from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, normalizers, decoders
import numpy as np
import os
import json

# -----------------------------
# Languages
# -----------------------------
langs = [
    "zho","nld","deu","fra","ind","fas","ukr","bul",
    "yue","est","swe","cym","pol","afr","eus","ita","spa","por","jpn","heb","srp","ara","ell",
    "bug","hun","tur","ces","ace","dan","ban","hrv","mak","nso","ron","nor","isl","zul","sot","xho","kor","rus","sun","jav"
]

# -----------------------------
# 1. Load + merge all splits
# -----------------------------
# Train/Dev Split for GPT-BabyBabelLM with 5% dev if no dev set exists and seed 42 for reproducibility
def load_all_splits(langs, dev_fraction=0.05):
    splits = {"train": [], "validation": []}
    for lang in langs:
        ds = load_dataset(f"BabyLM-community/babylm-{lang}")
        
        # If there’s no validation, create one from train
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
# 2. Train tokenizer on train split
# -----------------------------
special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [f"<special_{i}>" for i in range(11)]
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(vocab_size=32768, special_tokens=special_tokens)

def iterator():
    for example in multiling_ds["train"]:
        yield example["text"]

print("Training tokenizer...")
tokenizer.train_from_iterator(iterator(), trainer)

# Save tokenizer
os.makedirs("../tokenizers", exist_ok=True)
tok_path = "../tokenizers/tokenizer.json"
tokenizer.save(tok_path)
print(f"✅ Tokenizer saved at {tok_path}")

# -----------------------------
# 3. Encode datasets
# -----------------------------
def encode(example):
    ids = tokenizer.encode(example["text"]).ids
    return {"input_ids": ids}

print("Encoding full dataset...")
tokenized_ds = multiling_ds.map(encode, remove_columns=["text"])

# -----------------------------
# 4. Save as flat .bin files
# -----------------------------
os.makedirs("../data", exist_ok=True)

def save_bin(dataset, path):
    arr = np.concatenate([np.array(ids, dtype=np.uint16) for ids in dataset["input_ids"]])
    arr.tofile(path)
    print(f"✅ Saved {path} ({arr.shape[0]} tokens)")

train_bin = "../data/babybabellm_all.bin"
val_bin = "../data/dev_babybabellm.bin"
save_bin(tokenized_ds["train"], train_bin)
save_bin(tokenized_ds["validation"], val_bin)

# -----------------------------
# 5. Save meta file
# -----------------------------
meta = {
    "vocab_size": tokenizer.get_vocab_size(),
    "train_path": train_bin,
    "val_path": val_bin,
    "tokenizer_path": tok_path,
    "special_tokens": special_tokens
}

meta_path = "../data/meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"✅ Meta file saved at {meta_path}")

# -----------------------------
# 6. Hugging Face tokenizer usage
# -----------------------------
print("\nYou can load the tokenizer via HF as follows:")
print("from tokenizers import Tokenizer")
print(f"tokenizer = Tokenizer.from_file('{tok_path}')")
