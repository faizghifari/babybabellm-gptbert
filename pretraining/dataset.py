# coding=utf-8
import os
import pickle
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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
        length = tokens.numel()
        if length == 0:
            return torch.tensor([], dtype=torch.float), tokens.clone()

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.long)
        cumsum = torch.cumsum(span_lengths, dim=0)
        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.long)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.long)
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
                size=(random_mask.sum().item(),),
                dtype=torch.long
            )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class RandomIndex:
    def __init__(self, n_segments):
        self.n_segments = n_segments
        self.indices = torch.randperm(n_segments) if n_segments > 0 else torch.tensor([], dtype=torch.long)
        self.index = 0

    def get_random_index(self):
        if self.n_segments == 0:
            raise IndexError("RandomIndex has zero segments")
        if self.index >= self.n_segments:
            self.indices = torch.randperm(self.n_segments)
            self.index = 0
        idx = int(self.indices[self.index].item())
        self.index += 1
        return idx


# ===== Shard loader & index builder =====
def load_shard(shard_file):
    if not os.path.exists(shard_file):
        raise FileNotFoundError(f"Shard file not found: {shard_file}")
    return torch.load(shard_file, weights_only=False)


def _build_segment_index_for_shard(shard_file, seq_length):
    segments = []
    documents = load_shard(shard_file)
    for doc_idx, doc in enumerate(documents):
        if isinstance(doc, torch.Tensor) and doc.dim() == 0:
            doc = doc.unsqueeze(0)
        if not isinstance(doc, torch.Tensor):
            try:
                doc = torch.tensor(doc, dtype=torch.long)
            except Exception:
                continue
        doc_len = doc.numel()
        if doc_len == 0:
            continue
        step = max(1, seq_length - 2)
        for offset in range(0, doc_len, step):
            start = offset
            end = min(offset + step, doc_len)
            segments.append((doc_idx, start, end))
    return segments


def build_or_load_indices(shard_dir, seq_length, cache_file=None, rank=None, world_size=None):
    if cache_file is None:
        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            shard_files = [os.path.join(shard_dir, os.path.basename(f)) for f in data["shard_files"]]
            return data["shard_indices"], shard_files
        except Exception:
            print(f"Warning: failed to load cache {cache_file}, rebuilding indices")

    shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")])
    if rank is not None and world_size is not None:
        shard_files = shard_files[rank::world_size]

    shard_indices = []
    for shard_idx, shard_file in enumerate(tqdm(shard_files, desc="Building segment indices")):
        try:
            segments = _build_segment_index_for_shard(shard_file, seq_length)
            for doc_idx, start, end in segments:
                shard_indices.append((shard_idx, doc_idx, start, end))
        except Exception as e:
            print(f"Warning: failed processing shard {shard_file}: {e}")
            continue

    if len(shard_indices) == 0:
        print(f"Warning: no segments found in {shard_dir}. Adding dummy shard and segment.")
        if len(shard_files) == 0:
            dummy_file = os.path.join(shard_dir, "dummy.bin")
            torch.save(torch.tensor([0], dtype=torch.long), dummy_file)
            shard_files.append(dummy_file)
        shard_indices.append((0, 0, 0, 1))

    try:
        tmp_cache = cache_file + ".tmp"
        with open(tmp_cache, "wb") as f:
            pickle.dump({"shard_indices": shard_indices, "shard_files": shard_files}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_cache, cache_file)
    except Exception as e:
        print(f"Warning: could not write cache file {cache_file}: {e}")

    return shard_indices, shard_files


# ===== Helper: random-sample sanity check =====
def show_random_item(self, tokenizer):
    if len(self) == 0:
        print("Dataset empty: no item to show.")
        return
    try:
        index = random.randint(0, len(self) - 1)
        input_ids, target_ids, attention_mask, real_mask_p = self[index]
        print("Random item sample:")
        print("Input ids:", input_ids[:10])
        print("Target ids:", target_ids[:10])
        print("Attention mask shape:", attention_mask.shape)
        print("Mask ratio:", real_mask_p)
    except Exception as e:
        print(f"Failed to show random item: {e}")


# ===== Base Dataset with shard preloading =====
class BaseDataset(Dataset):
    def __init__(self, shard_dir, seq_length, tokenizer, args, rank=None, world_size=None):
        self.seq_length = seq_length
        self.args = args
        self.rank = rank
        self.world_size = world_size

        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, seq_length, cache_file, rank, world_size)

        self._loaded_shard = None
        self._loaded_shard_idx = None
        self._loaded_shards = [None] * len(self.shard_files)
        self.counts = [None] * len(self.shard_indices)
        self.mask_counts = [None] * len(self.shard_indices)
        self.random_index = RandomIndex(len(self.shard_indices))

        # Try preloading all shards asynchronously
        self._preload_shards_async()

    def _preload_shards_async(self, max_workers=4):
        """Load all shards in parallel using threads with tqdm progress."""
        def load_shard_safe(idx):
            try:
                self._loaded_shards[idx] = load_shard(self.shard_files[idx])
            except Exception as e:
                print(f"Failed to load shard {self.shard_files[idx]}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(load_shard_safe, range(len(self.shard_files))),
                      total=len(self.shard_files),
                      desc="Preloading shards"))

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]

        # Use preloaded shard if available
        if self._loaded_shards[shard_idx] is not None:
            shard = self._loaded_shards[shard_idx]
        else:
            if self._loaded_shard_idx != shard_idx:
                shard = load_shard(self.shard_files[shard_idx])
                self._loaded_shard = shard
                self._loaded_shard_idx = shard_idx
            else:
                shard = self._loaded_shard

        doc = shard[doc_idx]

        # Ensure doc is tensor and at least 1D
        if not isinstance(doc, torch.Tensor):
            doc = torch.tensor(doc, dtype=torch.long)
        if doc.dim() == 0:
            doc = doc.unsqueeze(0)

        if start >= doc.numel():
            return torch.tensor([], dtype=torch.long)
        segment = doc[start:end].long()
        return segment


# ===== MaskedDataset =====
class MaskedDataset(BaseDataset):
    def __init__(self, shard_dir, tokenizer, args, seq_length, rank=None, world_size=None):
        super().__init__(shard_dir, seq_length, tokenizer, args, rank, world_size)
        self.n_special_tokens = args.n_special_tokens
        self.vocab_size = args.vocab_size
        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(
            self.n_special_tokens, args.mask_random_p, args.mask_keep_p, self.vocab_size, self.mask_index
        )

    def __len__(self):
        return len(self.shard_indices)

    def __getitem__(self, index):
        tokens = self._load_segment(index)
        if tokens.numel() == 0:
            tokens = torch.tensor([self.cls_index], dtype=torch.long)

        seq_len = min(self.seq_length, tokens.numel())
        tokens = tokens[:seq_len].clamp(0, self.vocab_size - 1)

        if self.counts[index] is None:
            self.counts[index] = torch.zeros_like(tokens)
        if self.mask_counts[index] is None:
            self.mask_counts[index] = torch.zeros_like(tokens)
        self.counts[index][:seq_len] += 1

        mask_ratios, replacement_tokens = self.masking_strategy(tokens, self.mask_counts[index][:seq_len])
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)
        self.mask_counts[index][:seq_len][target_ids != -100] += 1

        input_ids = torch.cat([torch.LongTensor([self.cls_index]), input_ids])
        target_ids = torch.cat([torch.LongTensor([-100]), target_ids])
        attention_mask = torch.ones(len(input_ids), len(input_ids), dtype=torch.bool)

        padding_length = self.seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.pad_index] * padding_length)])
            target_ids = torch.cat([target_ids, torch.LongTensor([-100] * padding_length)])
            pad_mask = torch.zeros(padding_length, len(input_ids), dtype=torch.bool)
            attention_mask = torch.block_diag(attention_mask, pad_mask)

        attention_mask = ~attention_mask
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = getattr(self, "global_step", 0)
        mask_threshold = torch.topk(mask_ratios, max(1, int(len(mask_ratios) * mask_p))).values.max().item()
        mask = mask_ratios <= mask_threshold
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.float().mean()
        return input_ids, target_ids, real_mask_p


# ===== CausalDataset =====
class CausalDataset(BaseDataset):
    def __init__(self, shard_dir, tokenizer, args, seq_length, rank=None, world_size=None):
        super().__init__(shard_dir, seq_length, tokenizer, args, rank, world_size)
        self.n_special_tokens = args.n_special_tokens
        self.vocab_size = args.vocab_size
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

    def __len__(self):
        return len(self.shard_indices)

    def __getitem__(self, index):
        tokens = self._load_segment(index)
        if tokens.numel() == 0:
            tokens = torch.tensor([self.cls_index], dtype=torch.long)

        seq_len = min(self.seq_length, tokens.numel())
        tokens = tokens[:seq_len].clamp(0, self.vocab_size - 1)

        if self.counts[index] is None:
            self.counts[index] = torch.zeros_like(tokens)
        self.counts[index][:seq_len] += 1

        input_ids = torch.cat([torch.LongTensor([self.cls_index]), tokens])
        target_ids = torch.cat([torch.LongTensor([-100]), tokens])
        attention_mask = torch.ones(len(input_ids), len(input_ids), dtype=torch.bool)

        padding_length = self.seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.pad_index] * padding_length)])
            target_ids = torch.cat([target_ids, torch.LongTensor([-100] * padding_length)])
            pad_mask = torch.zeros(padding_length, len(input_ids), dtype=torch.bool)
            attention_mask = torch.block_diag(attention_mask, pad_mask)

        attention_mask = attention_mask.tril()
        attention_mask = ~attention_mask
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, torch.zeros([])


# ===== ValidationDataset =====
class ValidationDataset(MaskedDataset):
    def __init__(self, shard_dir, tokenizer, args, rank=None, world_size=None, seed=42):
        super().__init__(shard_dir, tokenizer, args, args.seq_length, rank, world_size)
        rng = random.Random(rank if rank is not None else seed)
        rng.shuffle(self.shard_indices)


# ===== Attach helper =====
MaskedDataset.show_random_item = show_random_item
CausalDataset.show_random_item = show_random_item
ValidationDataset.show_random_item = show_random_item

__all__ = ["MaskedDataset", "CausalDataset", "ValidationDataset"]


