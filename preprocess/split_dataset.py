# split_dataset.py
import torch
import os

def split_dataset(input_file, output_dir, shard_size_bytes=400_000_000):
    """
    Split a large .bin torch file containing a list of tensors/documents
    into multiple smaller shards of approximately shard_size_bytes each.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[split_dataset] Loading dataset {input_file}...")
    data = torch.load(input_file, map_location="cpu")
    print(f"[split_dataset] Loaded {len(data)} documents")

    shard = []
    current_size = 0
    shard_idx = 0

    for doc in data:
        doc_size = doc.element_size() * doc.numel()
        if current_size + doc_size > shard_size_bytes and len(shard) > 0:
            shard_file = os.path.join(output_dir, f"shard_{shard_idx:03d}.bin")
            torch.save(shard, shard_file)
            print(f"[split_dataset] Saved shard {shard_idx} with {len(shard)} docs (~{current_size/1e6:.1f} MB)")
            shard_idx += 1
            shard = []
            current_size = 0
        shard.append(doc)
        current_size += doc_size

    # Save remaining documents
    if len(shard) > 0:
        shard_file = os.path.join(output_dir, f"shard_{shard_idx:03d}.bin")
        torch.save(shard, shard_file)
        print(f"[split_dataset] Saved shard {shard_idx} with {len(shard)} docs (~{current_size/1e6:.1f} MB)")

if __name__ == "__main__":
    datasets = {
        "../data/babybabellm_all_torch.bin": "../data/shards/train",
        "..data/dev_babybabellm_torch.bin": "../data/shards/valid"
    }

    for infile, outdir in datasets.items():
        split_dataset(infile, outdir, shard_size_bytes=400_000_000)
