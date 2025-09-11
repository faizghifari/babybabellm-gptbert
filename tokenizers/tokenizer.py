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
os.makedirs("../tokenizers", exist_ok=True)
tok_path = "../tokenizers/tokenizer.json"
tokenizer.save(tok_path)
print(f"✅ Tokenizer saved at {tok_path}")

# -----------------------------
# 4. Parallel encoding
# -----------------------------
def encode_batch(batch):
    return {"input_ids": [tokenizer.encode(text).ids for text in batch["text"]]}

print("Encoding dataset in parallel...")
num_cpus = cpu_count()
tokenized_ds = multiling_ds.map(
    encode_batch,
    batched=True,
    batch_size=5000,  # larger batch to reduce overhead
    remove_columns=["text"],
    num_proc=num_cpus
)
print("✅ Parallel encoding complete")

# -----------------------------
# 5. Memory-mapped parallel .bin saving
# -----------------------------
def save_bin_memmap(dataset, path):
    """Save token IDs using memory-mapped array and parallel processing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_examples = len(dataset)
    
    # Compute total tokens
    print("Calculating total token count...")
    total_tokens = sum(len(ids) for ids in dataset["input_ids"])
    
    print(f"Total tokens: {total_tokens}")
    
    # Create memmap
    mmap = np.memmap(path, dtype=np.uint16, mode='w+', shape=(total_tokens,))
    
    # Function to write a slice of data
    def write_slice(start_idx, end_idx, offset):
        for i, ids in enumerate(dataset["input_ids"][start_idx:end_idx]):
            mmap[offset:offset+len(ids)] = ids
            offset += len(ids)
        return offset

    # Split dataset into chunks for parallel writing
    n_chunks = num_cpus * 2
    chunk_size = (n_examples + n_chunks - 1) // n_chunks
    chunks = []
    offsets = []
    current_offset = 0
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_examples)
        token_count = sum(len(ids) for ids in dataset["input_ids"][start:end])
        chunks.append((start, end, current_offset))
        offsets.append(token_count)
        current_offset += token_count

    # Parallel write
    def write_chunk(args):
        start_idx, end_idx, offset = args
        for i, ids in enumerate(dataset["input_ids"][start_idx:end_idx]):
            mmap[offset:offset+len(ids)] = ids
            offset += len(ids)
    
    print(f"Writing {n_examples} examples in {n_chunks} parallel chunks...")
    with Pool(n_chunks) as pool:
        pool.map(write_chunk, chunks)
    
    mmap.flush()
    del mmap
    print(f"✅ Saved {path} ({total_tokens} tokens)")

train_bin = "../data/babybabellm_all.bin"
val_bin = "../data/dev_babybabellm.bin"
save_bin_memmap(tokenized_ds["train"], train_bin)
save_bin_memmap(tokenized_ds["validation"], val_bin)

# -----------------------------
# 6. Save meta file
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
