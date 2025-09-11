import os
import torch
import random
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
        self.indices = torch.randperm(n_segments)
        self.index = 0

    def get_random_index(self):
        if self.index >= self.n_segments:
            self.indices = torch.randperm(self.n_segments)
            self.index = 0
        index = self.indices[self.index]
        self.index += 1
        return index


# ===== Lazy shard loader =====
def load_shard(shard_file):
    return torch.load(shard_file, weights_only=False)


# ===== Common show_random_item() =====
def show_random_item(self, tokenizer):
    index = random.randint(0, len(self) - 1)
    input_ids, target_ids, attention_mask, real_mask_p = self[index]  # triggers lazy shard load
    print("Random item sample:")
    print("Input ids:", input_ids)
    print("Target ids:", target_ids)
    print("Attention mask shape:", attention_mask.shape)
    print("Mask ratio:", real_mask_p)


# ===== Masked Dataset =====
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
            args.n_special_tokens,
            args.mask_random_p,
            args.mask_keep_p,
            args.vocab_size,
            self.mask_index
        )

        self.shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")])
        if rank is not None and world_size is not None:
            self.shard_files = self.shard_files[rank::world_size]

        self.shard_indices = []
        self._loaded_shard = None
        self._loaded_shard_idx = None

        # Build segment index lazily
        for shard_idx, shard_file in enumerate(self.shard_files):
            documents = load_shard(shard_file)
            for doc_idx, doc in enumerate(documents):
                if doc.dim() == 0:
                    doc = doc.unsqueeze(0)
                for offset in range(0, len(doc), self.seq_length - 2):
                    if len(doc) > 1:
                        start = offset
                        end = min(offset + self.seq_length - 2, len(doc))
                        self.shard_indices.append((shard_idx, doc_idx, start, end))

        self.counts = [None] * len(self.shard_indices)
        self.mask_counts = [None] * len(self.shard_indices)
        self.random_index = RandomIndex(len(self.shard_indices))

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]
        if self._loaded_shard_idx != shard_idx:
            self._loaded_shard = load_shard(self.shard_files[shard_idx])
            self._loaded_shard_idx = shard_idx
        segment = self._loaded_shard[doc_idx][start:end].long()
        return segment

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


# Attach show_random_item
MaskedDataset.show_random_item = show_random_item


# ===== Causal Dataset =====
class CausalDataset(Dataset):
    def __init__(self, shard_dir, tokenizer, args, seq_length, rank=None, world_size=None):
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")])
        if rank is not None and world_size is not None:
            self.shard_files = self.shard_files[rank::world_size]

        self.shard_indices = []
        self._loaded_shard = None
        self._loaded_shard_idx = None

        for shard_idx, shard_file in enumerate(self.shard_files):
            documents = load_shard(shard_file)
            for doc_idx, doc in enumerate(documents):
                if doc.dim() == 0:
                    doc = doc.unsqueeze(0)
                for offset in range(0, len(doc), self.seq_length - 2):
                    if len(doc) > 1:
                        start = offset
                        end = min(offset + self.seq_length - 2, len(doc))
                        self.shard_indices.append((shard_idx, doc_idx, start, end))

        self.counts = [None] * len(self.shard_indices)
        self.random_index = RandomIndex(len(self.shard_indices))

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]
        if self._loaded_shard_idx != shard_idx:
            self._loaded_shard = load_shard(self.shard_files[shard_idx])
            self._loaded_shard_idx = shard_idx
        segment = self._loaded_shard[doc_idx][start:end].long()
        return segment

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


CausalDataset.show_random_item = show_random_item


# ===== Validation Dataset =====
class ValidationDataset(Dataset):
    def __init__(self, shard_dir, tokenizer, args, rank=None, world_size=None, seed=42):
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

        self.shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")])
        if rank is not None and world_size is not None:
            self.shard_files = self.shard_files[rank::world_size]

        self.shard_indices = []
        self._loaded_shard = None
        self._loaded_shard_idx = None

        for shard_idx, shard_file in enumerate(self.shard_files):
            documents = load_shard(shard_file)
            for doc_idx, doc in enumerate(documents):
                if doc.dim() == 0:
                    doc = doc.unsqueeze(0)
                for offset in range(0, len(doc), self.seq_length - 2):
                    if len(doc) > 1:
                        start = offset
                        end = min(offset + self.seq_length - 2, len(doc))
                        self.shard_indices.append((shard_idx, doc_idx, start, end))

        rng = random.Random(rank if rank is not None else seed)
        rng.shuffle(self.shard_indices)

    def _load_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_indices[index]
        if self._loaded_shard_idx != shard_idx:
            self._loaded_shard = load_shard(self.shard_files[shard_idx])
            self._loaded_shard_idx = shard_idx
        segment = self._loaded_shard[doc_idx][start:end].long()
        return segment

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

