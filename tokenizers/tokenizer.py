from datasets import load_dataset, concatenate_datasets, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import pre_tokenizers, normalizers, decoders
import numpy as np
import os
import json
from multiprocessing import Pool, cpu_count
from functools import partial

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
# 3. Train tokenizer
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

os.makedirs("../tokenizers", exist_ok=True)
tok_path = "../tokenizers/tokenizer.json"
tokenizer.save(tok_path)
print(f"✅ Tokenizer saved at {tok_path}")

# -----------------------------
# 4. Parallelized encoding
# -----------------------------
def encode_batch(batch):
    return {"input_ids": [tokenizer.encode(text).ids for text in batch["text"]]}

print("Encoding dataset in parallel...")
num_cpus = cpu_count()
tokenized_ds = multiling_ds.map(
    encode_batch,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    num_proc=num_cpus
)
print("✅ Parallelized tokenization complete")

# -----------------------------
# 5. Parallel .bin saving
# -----------------------------
def save_chunk(start_idx, end_idx, dataset, tmp_dir, chunk_id):
    """Encode and save a dataset chunk to a temporary .bin file."""
    arr = np.concatenate([np.array(ids, dtype=np.uint16) for ids in dataset["input_ids"][start_idx:end_idx]])
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_id}.bin")
    arr.tofile(tmp_path)
    return tmp_path, len(arr)

def save_bin_parallel(dataset, path, n_chunks=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_dir = os.path.join(os.path.dirname(path), "tmp_chunks")
    os.makedirs(tmp_dir, exist_ok=True)

    n_examples = len(dataset)
    if n_chunks is None:
        n_chunks = cpu_count() * 2  # more chunks than CPUs for load balancing
    chunk_size = (n_examples + n_chunks - 1) // n_chunks
    chunks = [(i*chunk_size, min((i+1)*chunk_size, n_examples), dataset, tmp_dir, i) for i in range(n_chunks)]

    print(f"Saving {n_examples} examples in {n_chunks} parallel chunks...")
    total_tokens = 0
    tmp_files = []
    with Pool(n_chunks) as pool:
        for tmp_path, n_tokens in pool.starmap(save_chunk, chunks):
            tmp_files.append(tmp_path)
            total_tokens += n_tokens

    # Merge temporary files into final .bin
    with open(path, "wb") as f_out:
        for tmp_file in tmp_files:
            with open(tmp_file, "rb") as f_in:
                f_out.write(f_in.read())
            os.remove(tmp_file)  # remove temp file

    os.rmdir(tmp_dir)
    print(f"✅ Saved {path} ({total_tokens} tokens)")

train_bin = "../data/babybabellm_all.bin"
val_bin = "../data/dev_babybabellm.bin"
save_bin_parallel(tokenized_ds["train"], train_bin)
save_bin_parallel(tokenized_ds["validation"], val_bin)

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

