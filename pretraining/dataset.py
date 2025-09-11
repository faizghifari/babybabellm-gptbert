import os
import torch
import random

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
        idx = self.indices[self.index]
        self.index += 1
        return idx


class LazyShardDataset(torch.utils.data.Dataset):
    """
    Base dataset for lazy shard loading.
    Supports masked and causal behavior with CLS, padding, and attention masks exactly like original datasets.
    """
    def __init__(self, shard_dir, seq_length, tokenizer, args, rank=None, world_size=None, masked=True, seed=42):
        self.seq_length = seq_length
        self.args = args
        self.masked = masked
        self.global_step = 0

        self.shard_files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".bin")])
        if rank is not None and world_size is not None:
            self.shard_files = self.shard_files[rank::world_size]

        self.tokenizer = tokenizer
        self.n_special_tokens = args.n_special_tokens
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")
        self.mask_index = tokenizer.token_to_id("<mask>") if masked else None

        if masked:
            self.masking_strategy = SpanMaskingStrategy(
                args.n_special_tokens,
                args.mask_random_p,
                args.mask_keep_p,
                args.vocab_size,
                self.mask_index
            )

        # Lazy index: store (shard_idx, doc_idx, start, end)
        self.shard_offsets = []
        self._build_index(seed if rank is None else rank)

        self._loaded_shard_idx = None
        self._loaded_segments = None

    def _build_index(self, seed):
        for shard_idx, shard_file in enumerate(self.shard_files):
            documents = torch.load(shard_file, weights_only=False)
            for doc_idx, doc in enumerate(documents):
                doc_len = len(doc)
                for offset in range(0, doc_len, self.seq_length - 2):
                    if doc_len > 1:
                        start = offset
                        end = min(offset + self.seq_length - 2, doc_len)
                        self.shard_offsets.append((shard_idx, doc_idx, start, end))
        if not self.masked:  # deterministic order for causal
            random.Random(seed).shuffle(self.shard_offsets)
        self.total_segments = len(self.shard_offsets)

    def _load_shard(self, shard_idx):
        if self._loaded_shard_idx == shard_idx:
            return
        self._loaded_segments = torch.load(self.shard_files[shard_idx], weights_only=False)
        self._loaded_shard_idx = shard_idx

    def set_global_step(self, global_step):
        self.global_step = global_step

    def _get_segment(self, index):
        shard_idx, doc_idx, start, end = self.shard_offsets[index]
        self._load_shard(shard_idx)
        segment = self._loaded_segments[doc_idx][start:end]
        return segment

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index):
        tokens = self._get_segment(index).long()
        seq_length = min(self.seq_length, tokens.size(0))

        if self.masked:
            # MaskedDataset behavior
            mask_ratios, replacement_tokens = self.masking_strategy(tokens)
            input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)
        else:
            # CausalDataset behavior
            input_ids = torch.cat([torch.LongTensor([self.cls_index]), tokens[:seq_length]])
            target_ids = torch.cat([torch.LongTensor([-100]), tokens[:seq_length]])
            real_mask_p = torch.zeros([])

        # Padding
        padding_length = self.seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.LongTensor([self.pad_index] * padding_length)])
            target_ids = torch.cat([target_ids, torch.LongTensor([-100] * padding_length)])

        # Attention mask
        attention_mask = torch.ones(seq_length + 1, seq_length + 1, dtype=torch.bool)
        if padding_length > 0:
            attention_mask = torch.block_diag(attention_mask, torch.zeros(padding_length, padding_length, dtype=torch.bool))

        if self.masked:
            attention_mask = ~attention_mask
        else:
            attention_mask = attention_mask.tril()
            attention_mask = ~attention_mask

        # Remove last row/column
        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        attention_mask = attention_mask[:-1, :-1]

        return input_ids, target_ids, attention_mask, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_start + (self.args.mask_p_end - self.args.mask_p_start) * self.global_step / self.args.max_steps
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()
        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)
        real_mask_p = mask.sum() / mask_ratios.numel()
        return input_ids, target_ids, real_mask_p


# ===== Convenience wrappers =====
def MaskedDatasetLazy(*args, **kwargs):
    return LazyShardDataset(*args, masked=True, **kwargs)

def CausalDatasetLazy(*args, **kwargs):
    return LazyShardDataset(*args, masked=False, **kwargs)

def ValidationDatasetLazy(shard_dir, tokenizer, args, rank=None, world_size=None, seed=42):
    """
    Validation dataset with lazy loading and deterministic shuffling.
    """
    return LazyShardDataset(shard_dir, seq_length=args.seq_length, tokenizer=tokenizer, args=args, rank=rank,
                            world_size=world_size, masked=True, seed=seed)


