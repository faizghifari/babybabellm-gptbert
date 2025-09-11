from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, normalizers, decoders
import os

# All BabyLM languages (Tier1 + Tier2 + Tier3)
langs = [
    "zho","nld","deu","fra","ind","fas","ukr","bul",
    "yue","est","swe","cym","pol","afr","eus","ita","spa","por","jpn","heb","srp","ara","ell",
    "bug","hun","tur","ces","ace","dan","ban","hrv","mak","nso","ron","nor","isl","zul","sot","xho","kor","rus","sun","jav"
]

# -----------------------------
# 1. Load + merge all splits
# -----------------------------
def load_all_splits(langs):
    splits = {"train": [], "validation": [], "test": []}
    for lang in langs:
        ds = load_dataset(f"BabyLM-community/babylm-{lang}")
        for split in splits.keys():
            splits[split].append(ds[split])
    return DatasetDict({split: concatenate_datasets(splits[split]) for split in splits})

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

os.makedirs("tokenizers", exist_ok=True)
tok_path = "tokenizers/babylm_multilingual.json"
tokenizer.save(tok_path)
print(f"✅ Tokenizer saved at {tok_path}")

# -----------------------------
# 3. Apply tokenizer to dataset
# -----------------------------
def encode(example):
    ids = tokenizer.encode(example["text"]).ids
    return {"input_ids": ids, "length": len(ids)}

print("Tokenizing full dataset...")
tokenized_ds = multiling_ds.map(encode, remove_columns=["text"])

# -----------------------------
# 4. Save tokenized dataset
# -----------------------------
save_dir = "tokenized_babylm_multilingual"
os.makedirs(save_dir, exist_ok=True)
tokenized_ds.save_to_disk(save_dir)

print(f"✅ Tokenized multilingual dataset saved at {save_dir}")
print(tokenized_ds)
