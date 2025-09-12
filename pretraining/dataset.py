# dataset.py
# coding=utf-8

import os
import pickle
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset


# ===== Masking and helper classes =====
class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens, counts=None):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)
        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)
        mask_ratios = span_random_numbers_1[indices]

        if counts is not None:
            counts = counts.float()
            counts[tokens < self.n_special_tokens] = float('-inf')
            counts_p = torch.nn.functional.softmax(counts, dim=0)
            mask_ratios = mask_ratios * counts_p

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        if random_mask.sum().item() > 0:
            replacement_tokens[random_mask] = torch.randint(
                low=self.n_special_tokens,
                high=self.vocab_size,
                size=[random_mask.sum().item()],
                dtype=torch.long
            )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class RandomIndex:
    def __init__(self, n_segments):
        self.n_segments = n_segments
        if n_segments > 0:
            self.indices = torch.randperm(n_segments)
        else:
            self.indices = torch.tensor([], dtype=torch.long)
        self.index = 0

    def get_random_index(self):
        if self.n_segments == 0:
            raise IndexError("RandomIndex has zero segments")
        if self.index >= self.n_segments:
            self.indices = torch.randperm(self.n_segments)
            self.index = 0
        index = int(self.indices[self.index].item())
        self.index += 1
        return index


# ===== shard loader & index builder (with caching) =====
def load_shard(shard_file):
    # keep original semantics; do not map to GPU
    return torch.load(shard_file, weights_only=False)


def _build_segment_index_for_shard(shard_file, seq_length):
    """Return list of (doc_idx, start, end) for a single shard (no shard_idx)."""
    segments = []
    documents = load_shard(shard_file)
    for doc_idx, doc in enumerate(documents):
        # Safely handle 0-d tensors (scalars)
        if isinstance(doc, torch.Tensor) and doc.dim() == 0:
            doc = doc.unsqueeze(0)
        # If doc is not a tensor but a list/other, try to convert or just skip empty
        if isinstance(doc, torch.Tensor):
            doc_len = len(doc)
        else:
            # fallback: try len()
            try:
                doc_len = len(doc)
            except Exception:
                continue
        for offset in range(0, doc_len, seq_length - 2):
            if doc_len > 1:
                start = offset
                end = min(offset + seq_length - 2, doc_len)
                segments.append((doc_idx, start, end))
    return segments


def build_or_load_indices(shard_dir, seq_length, cache_file=None, rank=None, world_size=None):
    """
    Build or load cached shard indices.
    Returns:
      shard_indices: list of tuples (shard_idx, doc_idx, start, end)
      shard_files: list of shard file paths (in same order as shard_idx)
    """
    if cache_file is None:
        # default cache name inside shard_dir
        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")

    # If a cache exists, attempt to load it
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            shard_indices = data["shard_indices"]
            shard_files = data["shard_files"]
            # If rank/world_size was used to subset, we assume cache was built with same partitioning.
            # Caller is responsible for using same rank/world_size as when cache was created.
            return shard_indices, shard_files
        except Exception:
            # corrupted cache -> rebuild
            print(f"Warning: failed to load cache {cache_file}, rebuilding indices")
    # Build indices and save
    shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")])
    if rank is not None and world_size is not None:
        shard_files = shard_files[rank::world_size]

    shard_indices = []
    # enumerate shards and build their per-shard indices
    for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="Building segment indices")):
        try:
            per_shard_segments = _build_segment_index_for_shard(shard_file, seq_length)
            for (doc_idx, start, end) in per_shard_segments:
                shard_indices.append((shard_idx, doc_idx, start, end))
        except Exception as e:
            print(f"Warning: failed processing shard {shard_file}: {e}")
            continue

    # Save cache atomically
    try:
        tmp_cache = cache_file + ".tmp"
        with open(tmp_cache, "wb") as f:
            pickle.dump({"shard_indices": shard_indices, "shard_files": shard_files}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_cache, cache_file)
    except Exception as e:
        print(f"Warning: could not write cache file {cache_file}: {e}")

    return shard_indices, shard_files


# ===== common helper: show_random_item =====
def show_random_item(self, tokenizer):
    if len(self) == 0:
        print("Dataset empty: no item to show.")
        return
    index = random.randint(0, len(self) - 1)
    input_ids, target_ids, attention_mask, real_mask_p = self[index]  # triggers lazy loading of the required shard
    print("Random item sample:")
    print("Input ids:", input_ids)
    print("Target ids:", target_ids)
    print("Attention mask shape:", attention_mask.shape)
    print("Mask ratio:", real_mask_p)


# ===== MaskedDataset (lazy, cached indices) =====
class MaskedDataset(Dataset):
    def __init__(self, shard_dir: str, tokenizer, args, seq_length, rank=None, world_size=None):
        """
        Same API as original MaskedDataset, but:
         - builds (shard_idx, doc_idx, start, end) index lazily and caches it to disk
         - loads shards one at a time in __getitem__
         - handles 0-d tensors
        """
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(
            args.n_special_tokens,
            args.mask_random_p,
            args.mask_keep_p,
            args.vocab_size,
            self.mask_index
        )

        # use a cache file inside shard_dir that depends on seq_length
        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, seq_length, cache_file, rank, world_size)

        self._loaded_shard = None
        self._loaded_shard_idx = None

        # per-segment counters (initialized to None to avoid allocating large tensors before needed)
        self.counts = [None] * len(self.shard_indices)
        self.mask_counts = [None] * len(self.shard_indices)
        self.random_index = RandomIndex(len(self.shard_indices))

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]
        if self._loaded_shard_idx != shard_idx:
            self._loaded_shard = load_shard(self.shard_files[shard_idx])
            self._loaded_shard_idx = shard_idx
        segment = self._loaded_shard[doc_idx][start:end]
        # ensure 1-d tensor
        if isinstance(segment, torch.Tensor) and segment.dim() == 0:
            segment = segment.unsqueeze(0)
        return segment.long()

    def __len__(self):
        return len(self.shard_indices)

    def __getitem__(self, index):
        tokens = self._load_segment(index)
        seq_length = min(self.seq_length, tokens.size(0))
        tokens = tokens[:seq_length].long()

        if self.counts[index] is None:
            self.counts[index] = torch.zeros_like(tokens)
        if self.mask_counts[index] is None:
            self.mask_counts[index] = torch.zeros_like(tokens)

        self.counts[index][:seq_length] += 1
        mask_ratios, replacement_tokens = self.masking_strategy(tokens, self.mask_counts[index][:seq_length])
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)
        self.mask_counts[index][:seq_length][target_ids != -100] += 1

        input_ids = torch.cat([torch.LongTensor([self.cls_index]), input_ids])
        target_ids = torch.cat([torch.LongTensor([-100]), target_ids])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        padding_length = self.seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.pad_index] * padding_length)])
            target_ids = torch.cat([target_ids, torch.LongTensor([-100] * padding_length)])
            attention_mask = torch.block_diag(attention_mask, torch.zeros(padding_length, padding_length, dtype=torch.bool))

        attention_mask = ~attention_mask
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def set_global_step(self, global_step):
        self.global_step = global_step

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_start + (self.args.mask_p_end - self.args.mask_p_start) * self.global_step / self.args.max_steps
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()
        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.sum() / mask_ratios.numel()
        return input_ids, target_ids, real_mask_p


# attach show_random_item (keeps compatibility with training script)
MaskedDataset.show_random_item = show_random_item


# ===== CausalDataset (lazy, cached indices) =====
class CausalDataset(Dataset):
    def __init__(self, shard_dir: str, tokenizer, args, seq_length, rank=None, world_size=None):
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, seq_length, cache_file, rank, world_size)

        self._loaded_shard = None
        self._loaded_shard_idx = None

        self.counts = [None] * len(self.shard_indices)
        self.random_index = RandomIndex(len(self.shard_indices))

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]
        if self._loaded_shard_idx != shard_idx:
            self._loaded_shard = load_shard(self.shard_files[shard_idx])
            self._loaded_shard_idx = shard_idx
        segment = self._loaded_shard[doc_idx][start:end]
        if isinstance(segment, torch.Tensor) and segment.dim() == 0:
            segment = segment.unsqueeze(0)
        return segment.long()

    def __len__(self):
        return len(self.shard_indices)

    def __getitem__(self, index):
        tokens = self._load_segment(index)
        seq_length = min(self.seq_length, tokens.size(0))

        if self.counts[index] is None:
            self.counts[index] = torch.zeros_like(tokens)
        self.counts[index][:seq_length] += 1

        input_ids = torch.cat([torch.LongTensor([self.cls_index]), tokens[:seq_length]])
        target_ids = torch.cat([torch.LongTensor([-100]), tokens[:seq_length]])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        padding_length = self.seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.pad_index] * padding_length)])
            target_ids = torch.cat([target_ids, torch.LongTensor([-100] * padding_length)])
            attention_mask = torch.block_diag(attention_mask, torch.zeros(padding_length, padding_length, dtype=torch.bool))

        # causal attention mask
        attention_mask = attention_mask.tril()
        attention_mask = ~attention_mask
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, torch.zeros([])

    def set_global_step(self, global_step):
        self.global_step = global_step


# attach show_random_item
CausalDataset.show_random_item = show_random_item


# ===== ValidationDataset (lazy + cached indices) =====
class ValidationDataset(Dataset):
    def __init__(self, shard_dir: str, tokenizer, args, rank=None, world_size=None, seed=42):
        # keep API identical: seq_length comes from args
        self.seq_length = args.seq_length
        self.n_special_tokens = args.n_special_tokens

        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.masking_strategy = SpanMaskingStrategy(
            args.n_special_tokens,
            args.mask_random_p,
            args.mask_keep_p,
            args.vocab_size,
            self.mask_index
        )

        cache_file = os.path.join(shard_dir, f"shard_indices_seq{self.seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, self.seq_length, cache_file, rank, world_size)

        self._loaded_shard = None
        self._loaded_shard_idx = None

        rng = random.Random(rank if rank is not None else seed)
        rng.shuffle(self.shard_indices)

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]
        if self._loaded_shard_idx != shard_idx:
            self._loaded_shard = load_shard(self.shard_files[shard_idx])
            self._loaded_shard_idx = shard_idx
        segment = self._loaded_shard[doc_idx][start:end]
        if isinstance(segment, torch.Tensor) and segment.dim() == 0:
            segment = segment.unsqueeze(0)
        return segment.long()

    def __len__(self):
        return len(self.shard_indices)

    def __getitem__(self, index):
        tokens = self._load_segment(index)
        seq_length = min(self.seq_length - 2, tokens.size(0))

        segment = torch.cat([torch.LongTensor([self.cls_index]), tokens[:seq_length]])
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)

        mask_ratios, replacement_tokens = self.masking_strategy(segment)
        input_ids, target_ids, real_mask_p = self.apply_mask(segment, mask_ratios, replacement_tokens)

        padding_length = self.seq_length - segment.size(0) + 1
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.pad_index] * padding_length)])
            target_ids = torch.cat([target_ids, torch.LongTensor([-100] * padding_length)])
            attention_mask = torch.block_diag(attention_mask, torch.zeros(padding_length, padding_length, dtype=torch.bool))

        attention_mask = ~attention_mask
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = 0.15
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + 0.5)), largest=False).values.max().item()
        mask = mask_ratios < mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.sum() / mask_ratios.numel()
        return input_ids, target_ids, real_mask_p


# Exported names: MaskedDataset, CausalDataset, ValidationDataset
__all__ = ["MaskedDataset", "CausalDataset", "ValidationDataset"]


