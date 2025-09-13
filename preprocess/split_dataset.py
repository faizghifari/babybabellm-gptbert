import torch
import os

def safe_split_dataset_tensor(input_file, output_dir, seq_length=128, shard_size_bytes=100_000_000):
    """
    Split a large .bin tensor into smaller shards, discarding empty shards and
    ensuring each shard has at least `seq_length` tokens.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[safe_split] Loading dataset {input_file}...")
    data = torch.load(input_file, map_location="cpu")
    print(f"[safe_split] Loaded tensor with shape {data.shape} and dtype {data.dtype}")

    if data.numel() == 0:
        raise ValueError(f"Dataset {input_file} is empty!")

    element_size = data.element_size()
    num_elements_per_shard = max(seq_length, shard_size_bytes // element_size)  # ensure enough tokens

    num_elements = data.numel()
    num_shards = (num_elements + num_elements_per_shard - 1) // num_elements_per_shard

    print(f"[safe_split] Each shard ~{num_elements_per_shard} elements (~{num_elements_per_shard*element_size/1e6:.1f} MB)")
    print(f"[safe_split] Total elements: {num_elements}, will create {num_shards} shards")

    shard_count = 0
    for shard_idx in range(num_shards):
        start_idx = shard_idx * num_elements_per_shard
        end_idx = min((shard_idx + 1) * num_elements_per_shard, num_elements)
        shard = data[start_idx:end_idx]

        if shard.numel() == 0:
            print(f"[safe_split] Skipping empty shard {shard_idx}")
            continue

        shard_file = os.path.join(output_dir, f"shard_{shard_count:03d}.bin")
        torch.save(shard, shard_file)
        print(f"[safe_split] Saved shard {shard_count} with {shard.numel()} elements (~{shard.numel()*element_size/1e6:.1f} MB)")
        shard_count += 1

    if shard_count == 0:
        print("[safe_split] Warning: all shards were empty. Adding a dummy shard.")
        dummy = torch.tensor([0], dtype=torch.long)
        torch.save(dummy, os.path.join(output_dir, "shard_000.bin"))

if __name__ == "__main__":
    datasets = {
        "../data/babybabellm_all_torch.bin": "../data/shards/train",
        "../data/dev_babybabellm_torch.bin": "../data/shards/valid"
    }

    for infile, outdir in datasets.items():
        safe_split_dataset_tensor(infile, outdir, seq_length=128, shard_size_bytes=100_000_000)
