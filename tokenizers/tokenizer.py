import os
import json
import argparse
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
ALL_LANGS = [
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train (multi)lingual BPE tokenizer")
    parser.add_argument("--vocab_size", type=int, default=int(os.environ.get("VOCAB_SIZE", 32768)),
                        help="Target vocabulary size (can also set VOCAB_SIZE env). Default=32768")
    parser.add_argument("--dev_fraction", type=float, default=0.05, help="Fraction for validation split when dataset lacks one")
    parser.add_argument("--output", type=str, default="tokenizer.json", help="Relative output filename inside this folder")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming iterator (load all in memory)")
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Language codes to include (e.g., eng deu). Defaults to the full multilingual set.")
    return parser.parse_args()

args = parse_args()

if args.vocab_size <= 32:  # must exceed special tokens length
    raise ValueError(f"vocab_size too small ({args.vocab_size}); must be > 32")

selected_langs = args.languages if args.languages is not None else ALL_LANGS

print(f"Loading dataset for languages {selected_langs} (dev_fraction={args.dev_fraction})...")
multiling_ds = load_all_splits(selected_langs, dev_fraction=args.dev_fraction)
print(multiling_ds)

# -----------------------------
# 3. Tokenizer training
# -----------------------------
special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<mask>"] + [f"<special_{i}>" for i in range(11)]
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()
trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)

# Streaming iterator for training
if args.no_stream:
    print("Training tokenizer (non-stream / in-memory)...")
    texts = multiling_ds["train"]["text"]
    tokenizer.train_from_iterator(texts, trainer)
else:
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
tok_path = os.path.join(script_dir, args.output)
tokenizer.save(tok_path)
meta = {
    "vocab_size": args.vocab_size,
    "dev_fraction": args.dev_fraction,
    "output": args.output,
    "streaming": not args.no_stream,
    "languages": selected_langs,
    "special_tokens": special_tokens,
}
with open(os.path.join(script_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"✅ Tokenizer saved at {tok_path} (vocab_size={args.vocab_size})")

