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
        if length == 0:
            return torch.tensor([]), tokens.clone()

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


# ===== shard loader & index builder =====
def load_shard(shard_file):
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
        doc_len = doc.size(0)
        if doc_len == 0:
            continue
        step = max(1, seq_length - 2)
        # ensure at least one segment per document
        for offset in range(0, max(1, doc_len), step):
            start = offset
            end = min(offset + step, doc_len)
            segments.append((doc_idx, start, end))
        if len(segments) == 0:
            segments.append((doc_idx, 0, doc_len))
    return segments


def build_or_load_indices(shard_dir, seq_length, cache_file=None, rank=None, world_size=None):
    if cache_file is None:
        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            return data["shard_indices"], data["shard_files"]
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
        print(f"Warning: no segments found in {shard_dir}, dataset will be empty!")

    try:
        tmp_cache = cache_file + ".tmp"
        with open(tmp_cache, "wb") as f:
            pickle.dump({"shard_indices": shard_indices, "shard_files": shard_files}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_cache, cache_file)
    except Exception as e:
        print(f"Warning: could not write cache file {cache_file}: {e}")

    return shard_indices, shard_files


# ===== helper: show_random_item =====
def show_random_item(self, tokenizer):
    if len(self) == 0:
        print("Dataset empty: no item to show.")
        return
    index = random.randint(0, len(self) - 1)
    input_ids, target_ids, attention_mask, real_mask_p = self[index]
    print("Random item sample:")
    print("Input ids:", input_ids)
    print("Target ids:", target_ids)
    print("Attention mask shape:", attention_mask.shape)
    print("Mask ratio:", real_mask_p)


# ===== MaskedDataset =====
class MaskedDataset(Dataset):
    def __init__(self, shard_dir, tokenizer, args, seq_length, rank=None, world_size=None):
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.masking_strategy = SpanMaskingStrategy(
            args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index
        )

        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, seq_length, cache_file, rank, world_size)

        if len(self.shard_indices) == 0:
            raise ValueError(f"No usable segments found in {shard_dir} with seq_length={seq_length}!")

        self._loaded_shard = None
        self._loaded_shard_idx = None
        self.counts = [None] * len(self.shard_indices)
        self.mask_counts = [None] * len(self.shard_indices)
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
        tokens = tokens[:seq_length]

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
        mask_p = self.args.mask_p_start + (self.args.mask_p_end - self.args.mask_p_start) * self.global_step / max(1, self.args.max_steps)
        topk = max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item()))
        mask_threshold = torch.topk(mask_ratios, topk, largest=False).values.max().item()
        mask = mask_ratios <= mask_threshold
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.sum() / mask_ratios.numel()
        return input_ids, target_ids, real_mask_p


MaskedDataset.show_random_item = show_random_item


# ===== CausalDataset =====
class CausalDataset(Dataset):
    def __init__(self, shard_dir, tokenizer, args, seq_length, rank=None, world_size=None):
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        cache_file = os.path.join(shard_dir, f"shard_indices_seq{seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, seq_length, cache_file, rank, world_size)

        if len(self.shard_indices) == 0:
            raise ValueError(f"No usable segments found in {shard_dir} with seq_length={seq_length}!")

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

        attention_mask = attention_mask.tril()
        attention_mask = ~attention_mask
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, torch.zeros([])

    def set_global_step(self, global_step):
        self.global_step = global_step


CausalDataset.show_random_item = show_random_item


# ===== ValidationDataset =====
class ValidationDataset(Dataset):
    def __init__(self, shard_dir, tokenizer, args, rank=None, world_size=None, seed=42):
        self.seq_length = args.seq_length
        self.n_special_tokens = args.n_special_tokens

        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.masking_strategy = SpanMaskingStrategy(
            args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index
        )

        cache_file = os.path.join(shard_dir, f"shard_indices_seq{self.seq_length}.pkl")
        self.shard_indices, self.shard_files = build_or_load_indices(shard_dir, self.seq_length, cache_file, rank, world_size)

        if len(self.shard_indices) == 0:
            raise ValueError(f"No usable segments found in {shard_dir} with seq_length={self.seq_length}!")

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
        topk = max(1, int(mask_ratios.size(0) * mask_p + 0.5))
        mask_threshold = torch.topk(mask_ratios, topk, largest=False).values.max().item()
        mask = mask_ratios < mask_threshold
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.sum() / mask_ratios.numel()
        return input_ids, target_ids, real_mask_p


ValidationDataset.show_random_item = show_random_item

__all__ = ["MaskedDataset", "CausalDataset", "ValidationDataset"]

