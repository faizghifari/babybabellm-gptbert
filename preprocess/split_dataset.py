import torch
import os

def split_dataset_tensor(input_file, output_dir, shard_size_bytes=400_000_000):
    """
    Split a large .bin torch file containing a single huge tensor
    into multiple smaller shards of approximately shard_size_bytes each.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[split_dataset_tensor] Loading dataset {input_file}...")
    data = torch.load(input_file, map_location="cpu")
    print(f"[split_dataset_tensor] Loaded tensor with shape {data.shape} and dtype {data.dtype}")

    element_size = data.element_size()  # size in bytes per element
    num_elements_per_shard = shard_size_bytes // element_size
    num_elements = data.numel()
    num_shards = (num_elements + num_elements_per_shard - 1) // num_elements_per_shard

    print(f"[split_dataset_tensor] Each shard ~{num_elements_per_shard} elements (~{shard_size_bytes/1e6:.1f} MB)")
    print(f"[split_dataset_tensor] Total elements: {num_elements}, will create {num_shards} shards")

    for shard_idx in range(num_shards):
        start_idx = shard_idx * num_elements_per_shard
        end_idx = min((shard_idx + 1) * num_elements_per_shard, num_elements)
        shard = data[start_idx:end_idx]

        shard_file = os.path.join(output_dir, f"shard_{shard_idx:03d}.bin")
        torch.save(shard, shard_file)
        print(f"[split_dataset_tensor] Saved shard {shard_idx} with {end_idx - start_idx} elements "
              f"(~{(end_idx - start_idx) * element_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    datasets = {
        "../data/babybabellm_all_torch.bin": "../data/shards/train",
        "../data/dev_babybabellm_torch.bin": "../data/shards/valid"
    }

    for infile, outdir in datasets.items():
        split_dataset_tensor(infile, outdir, shard_size_bytes=400_000_000)

