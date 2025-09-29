#!/usr/bin/env python3
"""
Convert BabyBabyLLM GPT-BERT checkpoints into Hugging Face-compatible repos.

This script:
- Discovers checkpoints via CLI-controlled directory/glob/regex.
- Packages tokenizer + remote code that embeds the original training architecture.
- Saves weights directly from the training state_dict (no new/random layers).
- Uploads to the Hub using flexible repo naming templates, or saves locally.

It also supports rehosting existing Hub repos by injecting standardized remote code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False

from huggingface_hub import create_repo, upload_folder, list_models, snapshot_download
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


# -----------------------------
# Defaults and patterns
# -----------------------------
REPO_USERNAME_ENV_VAR = "HF_USERNAME"

CHECKPOINT_DIR = Path("babybabellm-gptbert/model_checkpoints")
CHECKPOINT_GLOB = "mono_*_small_1_2*.bin"
CHECKPOINT_PATTERN = re.compile(
    r"^mono_(?P<lang>[a-z]{3})_small_1_2(?P<ema>_ema)?\.bin$"
)


# -----------------------------
# Tokenizer
# -----------------------------
def build_tokenizer(lang: str, args) -> PreTrainedTokenizerFast:
    """Load a tokenizer controlled by CLI args.

    Precedence:
    1) --tokenizer-id (Hub repo id; supports {lang} template)
    2) --tokenizer-path (local dir or tokenizer.json; supports {lang} template)
    3) Fallback to common in-repo locations (also tries language-specific files)

    We do not add or alter tokens to preserve training compatibility.
    """
    # 1) Hub tokenizer by id
    if getattr(args, "tokenizer_id", None):
        source = args.tokenizer_id.format(lang=lang)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
            # annotate where it came from for README purposes
            setattr(tok, "_tokenizer_file", f"hub:{source}")
            return tok  # type: ignore[return-value]
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from Hub id '{source}': {e}")

    # 2) Local tokenizer path
    if getattr(args, "tokenizer_path", None):
        source = args.tokenizer_path.format(lang=lang)
        print(source)
        p = Path(source)
        # Try AutoTokenizer first (works with dirs that have tokenizer.json)
        if p.exists():
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True, use_fast=True)
                setattr(tok, "_tokenizer_file", str(p))
                return tok  # type: ignore[return-value]
            except Exception:
                # fall back to raw tokenizer.json
                pass
            if p.is_file() and p.name.endswith(".json"):
                core = Tokenizer.from_file(str(p))
                print(f"[tokenizer] Loaded raw tokenizer from {p}")
                special_map = {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "pad_token": "<pad>",
                    "mask_token": "<mask>",
                }
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, **special_map)
                for needed in ["bos_token", "eos_token", "pad_token", "mask_token", "unk_token"]:
                    if getattr(tokenizer, needed) is None:
                        raise ValueError(f"Required special token missing in base tokenizer: {needed}")
                tokenizer._tokenizer_file = str(p)
                return tokenizer
            # If directory but AutoTokenizer failed and no tokenizer.json file, we'll continue to fallback

    # 3) Fallback candidates (prefer project tokenizer, try language-specific then generic)
    candidates = [
        Path(f"babybabellm-gptbert/tokenizers/tokenizer_{lang}.json"),
        Path("babybabellm-gptbert/tokenizers/tokenizer.json"),
        Path(f"gpt-bert/tokenizers/tokenizer_{lang}.json"),
        Path("gpt-bert/tokenizers/tokenizer.json"),
        Path("tokenizer.json"),
    ]
    tok_file = None
    for c in candidates:
        if c.exists():
            tok_file = c
            break
    if tok_file is None:
        raise FileNotFoundError("Could not find tokenizer in --tokenizer-path or known locations.")
    core = Tokenizer.from_file(str(tok_file))
    special_map = {
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
    }
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, **special_map)
    for needed in ["bos_token", "eos_token", "pad_token", "mask_token", "unk_token"]:
        if getattr(tokenizer, needed) is None:
            raise ValueError(f"Required special token missing in base tokenizer: {needed}")
    tokenizer._tokenizer_file = str(tok_file)
    return tokenizer


def write_remote_code_files(target_dir: Path, causal: bool):
    """Package the original training architecture for remote inference.

    This embeds the exact model as trained (from gpt-bert/pretraining/model_extra.py)
    to ensure state dict keys match and no layers are randomly reinitialized.
    If `causal` is True, default AutoModel maps to the CausalLM wrapper; otherwise to MaskedLM.
    """
    config_txt = (
        "from transformers import PretrainedConfig\n\n"
        "class GPTBertConfig(PretrainedConfig):\n"
        "    model_type = 'gpt_bert'\n"
        "    def __init__(self, **kwargs):\n"
        "        self.attention_probs_dropout_prob = kwargs.pop('attention_probs_dropout_prob', 0.1)\n"
        "        self.hidden_dropout_prob = kwargs.pop('hidden_dropout_prob', 0.1)\n"
        "        self.hidden_size = kwargs.pop('hidden_size', 768)\n"
        "        self.intermediate_size = kwargs.pop('intermediate_size', 2560)\n"
        "        self.max_position_embeddings = kwargs.pop('max_position_embeddings', 512)\n"
        "        self.position_bucket_size = kwargs.pop('position_bucket_size', 32)\n"
        "        self.num_attention_heads = kwargs.pop('num_attention_heads', 12)\n"
        "        self.num_hidden_layers = kwargs.pop('num_hidden_layers', 12)\n"
        "        self.vocab_size = kwargs.pop('vocab_size', 16384)\n"
        "        self.layer_norm_eps = kwargs.pop('layer_norm_eps', 1e-5)\n"
        f"        self.auto_map = {{\n"
        f"            'AutoConfig': 'configuration_gpt_bert.GPTBertConfig',\n"
        f"            'AutoModel': 'modeling_gpt_bert.{ 'GPTBertForCausalLM' if causal else 'GPTBertForMaskedLM' }',\n"
        f"            'AutoModelForCausalLM': 'modeling_gpt_bert.GPTBertForCausalLM',\n"
        f"            'AutoModelForMaskedLM': 'modeling_gpt_bert.GPTBertForMaskedLM',\n"
        f"        }}\n"
        "        super().__init__(**kwargs)\n"
    )
    (target_dir / "configuration_gpt_bert.py").write_text(config_txt, encoding="utf-8")

    orig_path = Path("gpt-bert/pretraining/model_extra.py")
    try:
        orig_code = orig_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read original model code at {orig_path}: {e}")

    wrapper_code = textwrap.dedent(
        """
        from transformers import PreTrainedModel
        from transformers.modeling_outputs import MaskedLMOutput, CausalLMOutputWithCrossAttentions
        from .configuration_gpt_bert import GPTBertConfig
        import torch
        import torch.nn as nn


        def _normalize_mask_tensor(mask):
            if mask.dtype == torch.bool:
                if mask.numel() == 0:
                    return mask
                true_fraction = mask.float().mean().item()
                if true_fraction > 0.5:
                    mask = ~mask
            else:
                mask = mask <= 0
            return mask.to(torch.bool)


        def _ensure_valid_rows(mask):
            row_masked = mask.all(dim=-1)
            if row_masked.any():
                idx = row_masked.nonzero(as_tuple=False)
                mask[idx[:, 0], idx[:, 1], idx[:, 1]] = False
            return mask


        def _build_babylm_attention_mask(input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape[:2]
            device = input_ids.device
            if attention_mask is None:
                mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
            else:
                mask = attention_mask
                if mask.dim() == 0:
                    mask = mask.unsqueeze(0)
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                if mask.dim() == 2:
                    mask = _normalize_mask_tensor(mask)
                    mask = mask.unsqueeze(1) | mask.unsqueeze(2)
                elif mask.dim() == 3:
                    if mask.size(1) == 1 and mask.size(2) == seq_len:
                        mask = _normalize_mask_tensor(mask.squeeze(1))
                        mask = mask.unsqueeze(1) | mask.unsqueeze(2)
                    elif mask.size(1) == seq_len and mask.size(2) == 1:
                        mask = _normalize_mask_tensor(mask.squeeze(2))
                        mask = mask.unsqueeze(1) | mask.unsqueeze(2)
                    else:
                        mask = _normalize_mask_tensor(mask)
                elif mask.dim() == 4:
                    if mask.size(1) == 1:
                        mask = mask[:, 0]
                    else:
                        mask = mask.any(dim=1)
                    mask = _normalize_mask_tensor(mask)
                else:
                    raise ValueError("Unsupported attention_mask dimensions: {}".format(mask.dim()))
                mask = mask.to(device=device, dtype=torch.bool)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1) | mask.unsqueeze(2)
                if mask.dim() != 3:
                    raise ValueError("attention_mask must broadcast to a square matrix")
                if mask.size(0) == 1 and batch_size > 1:
                    mask = mask.expand(batch_size, -1, -1).clone()
                elif mask.size(0) != batch_size:
                    raise ValueError("attention_mask batch dimension {} does not match inputs {}".format(mask.size(0), batch_size))
                rows = min(mask.size(1), seq_len)
                cols = min(mask.size(2), seq_len)
                if mask.size(1) != seq_len or mask.size(2) != seq_len:
                    new_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool, device=device)
                    new_mask[:, :rows, :cols] = mask[:, :rows, :cols]
                    mask = new_mask
            mask = _ensure_valid_rows(mask)
            return mask.unsqueeze(1)


        class GPTBertForMaskedLM(PreTrainedModel):
            config_class = GPTBertConfig
            base_model_prefix = 'gpt_bert'

            def __init__(self, config: GPTBertConfig):
                super().__init__(config)
                self.model = Bert(config)

            def tie_weights(self):
                try:
                    self.model.classifier.nonlinearity[-1].weight = self.model.embedding.word_embedding.weight
                except Exception:
                    pass
                return super().tie_weights()

            def forward(self, input_ids, attention_mask=None, labels=None):
                mask_4d = _build_babylm_attention_mask(input_ids, attention_mask)
                static_embeddings, relative_embedding = self.model.embedding(input_ids)
                if static_embeddings.dim() == 3 and static_embeddings.shape[0] == input_ids.shape[0]:
                    static_embeddings = static_embeddings.transpose(0, 1)
                contextualized = self.model.transformer(static_embeddings, mask_4d, relative_embedding)
                hs = contextualized.transpose(0, 1)
                B, S, H = hs.shape
                flat = hs.reshape(B * S, H)
                logits_flat = self.model.classifier.nonlinearity(flat)
                vocab = logits_flat.size(-1)
                logits = logits_flat.view(B, S, vocab)
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(logits.view(-1, vocab), labels.view(-1))
                return MaskedLMOutput(loss=loss, logits=logits)


        class GPTBertForCausalLM(PreTrainedModel):
            config_class = GPTBertConfig
            base_model_prefix = 'gpt_bert'

            def __init__(self, config: GPTBertConfig):
                super().__init__(config)
                self.model = Bert(config)

            def prepare_inputs_for_generation(self, input_ids, **kwargs):
                return {'input_ids': input_ids, 'attention_mask': kwargs.get('attention_mask', None)}

            def forward(self, input_ids, attention_mask=None, labels=None):
                mask_4d = _build_babylm_attention_mask(input_ids, attention_mask)
                static_embeddings, relative_embedding = self.model.embedding(input_ids)
                if static_embeddings.dim() == 3 and static_embeddings.shape[0] == input_ids.shape[0]:
                    static_embeddings = static_embeddings.transpose(0, 1)
                contextualized = self.model.transformer(static_embeddings, mask_4d, relative_embedding)
                hs = contextualized.transpose(0, 1)
                B, S, H = hs.shape
                flat = hs.reshape(B * S, H)
                logits_flat = self.model.classifier.nonlinearity(flat)
                vocab = logits_flat.size(-1)
                logits = logits_flat.view(B, S, vocab)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)
        """
    )

    full_modeling = (
        "# Original training architecture (verbatim)\n" + orig_code + "\n\n# HF wrappers that preserve state dict keys and behavior\n" + wrapper_code
    )
    (target_dir / "modeling_gpt_bert.py").write_text(full_modeling, encoding="utf-8")


def _patch_rehost_config(config_path: Path, prefer_causal: bool):
    """Patch an existing config.json for rehosting.

    - Keep original model_type as-is.
    - Inject/overwrite auto_map preferring causal mapping when prefer_causal True.
    - Ensure architectures includes both GPTBertForCausalLM and GPTBertForMaskedLM (order causal first if prefer_causal).
    - Force use_return_dict = True for HF outputs.
    """
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read config for patch: {e}")
        return
    # Preserve model_type; just ensure it's there
    mt = data.get("model_type", "gpt_bert")
    data["model_type"] = mt  # no change
    # Architectures
    arch: List[str] = data.get("architectures", []) or []
    needed = ["GPTBertForCausalLM", "GPTBertForMaskedLM"]
    for n in needed:
        if n not in arch:
            arch.append(n)
    # Reorder if prefer_causal
    if prefer_causal:
        arch = [a for a in needed if a in arch] + [a for a in arch if a not in needed]
    data["architectures"] = arch
    # auto_map
    if prefer_causal:
        data["auto_map"] = {
            "AutoConfig": "configuration_gpt_bert.GPTBertConfig",
            "AutoModel": "modeling_gpt_bert.GPTBertForCausalLM",
            "AutoModelForCausalLM": "modeling_gpt_bert.GPTBertForCausalLM",
            "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBertForMaskedLM",
        }
    else:
        data["auto_map"] = {
            "AutoConfig": "configuration_gpt_bert.GPTBertConfig",
            "AutoModel": "modeling_gpt_bert.GPTBertForMaskedLM",
            "AutoModelForCausalLM": "modeling_gpt_bert.GPTBertForCausalLM",
            "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBertForMaskedLM",
        }
    config_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rehost_readme(out_dir: Path, src_repo: str, dst_repo: str, prefer_causal: bool):
    note = [
        f"# {dst_repo}",
        "", 
        f"Rehosted from `{src_repo}` with standardized remote code and auto_map.",
        "- Original `model_type` preserved.",
        f"- Default AutoModel mapping points to {'GPTBertForCausalLM' if prefer_causal else 'GPTBertForMaskedLM' }.",
        "- Added both causal & masked LM wrappers for evaluation.",
        "", 
        "Example:",
        "```python",
        f"from transformers import AutoTokenizer, AutoModel",
        f"m='{dst_repo}'",
        "tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True)",
        "model=AutoModel.from_pretrained(m, trust_remote_code=True)",
        "print(model(**tok('Hello world', return_tensors='pt')).logits.shape)",
        "```",
    ]
    if prefer_causal:
        note += [
            "Generation:",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            f"m='{dst_repo}'",
            "tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True)",
            "model=AutoModelForCausalLM.from_pretrained(m, trust_remote_code=True)",
            "print(tok.decode(model.generate(**tok('Hello', return_tensors='pt'), max_new_tokens=20)[0], skip_special_tokens=True))",
            "```",
        ]
    (out_dir / "README.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def format_repo_id(args, lang: str, variant: Optional[str] = None) -> str:
    """Resolve HF repo id from args and language using a flexible template.

    Available template fields:
        - username: Hub username/org
        - lang: language code or suffix fragment
        - causal_suffix: "-causal" when --causal set, else ""
        - variant: passed through (e.g., 'main' or 'ema') or ''
        - variant_suffix: "-ema" when variant == 'ema' and --separate-ema set, else ""
    """
    username = getattr(args, 'username', None)
    if not username:
        raise ValueError("Hub username is not set. Provide --username or set the HF_USERNAME environment variable.")
    causal_suffix = "-causal" if getattr(args, 'causal', False) else ""
    variant_val = variant or ""
    variant_suffix = "-ema" if (variant_val == "ema" and getattr(args, 'separate_ema', False)) else ""
    template = getattr(args, 'repo_template', None) or "{username}/babybabellm-gpt_bert-{lang}{variant_suffix}{causal_suffix}"
    return template.format(
        username=username,
        lang=lang,
        causal_suffix=causal_suffix,
        variant=variant_val,
        variant_suffix=variant_suffix,
    )


def rehost_repos(args):
    prefix = args.rehost_prefix.rstrip('/')
    print(f"[rehost] Listing models with prefix: {prefix}")
    # list_models returns generator; filter manually
    upstream = []
    for m in list_models(author=prefix.split('/')[0]):
        repo_id = m.modelId
        if repo_id.startswith(prefix):
            upstream.append(repo_id)
    if not upstream:
        print("[rehost] No upstream repos found for prefix")
        return
    print(f"[rehost] Found {len(upstream)} repos")
    # Derive suffix after the base repo prefix component for naming.
    # Example:
    #  prefix: suchirsalham/babybabellm-  src: suchirsalham/babybabellm-mono-deu -> suffix mono-deu -> lang deu
    #  prefix: suchirsalham/babybabellm-  src: suchirsalham/babybabellm-multi-all -> suffix multi-all -> lang multi-all
    #  prefix: suchirsalham/babybabellm-mono- (legacy) src: suchirsalham/babybabellm-mono-deu -> suffix deu
    prefix_repo_part = prefix.split('/', 1)[1]
    for src in sorted(upstream):
        src_name = src.split('/', 1)[1]
        if not src_name.startswith(prefix_repo_part):
            # Should not happen due to startswith earlier, but keep safe
            continue
        remainder = src_name[len(prefix_repo_part):]
        # Normalize: if prefix already ended with 'mono-' we get pure lang codes.
        lang_fragment = remainder
        if lang_fragment.startswith('mono-'):
            lang_fragment = lang_fragment[len('mono-'):]
        # Special multi-all normalization
        if lang_fragment in {'multi_all', 'multiall'}:
            lang_fragment = 'multi-all'
        if lang_fragment in {'multismall'}:
            lang_fragment = 'multi-small'    
        if lang_fragment == '' or lang_fragment == '-':
            print(f"[rehost] Skip (empty suffix): {src}")
            continue
        lang = lang_fragment.strip('-')
        # Use flexible repo template for destination repo naming
        dst_repo = format_repo_id(args, lang)
        print(f"[rehost] Processing {src} -> {dst_repo}")
        with tempfile.TemporaryDirectory() as td:
            local_src = Path(td) / "src"
            local_src.mkdir()
            try:
                snapshot_download(repo_id=src, local_dir=str(local_src), local_dir_use_symlinks=False)
            except Exception as e:
                print(f"[rehost] Failed snapshot {src}: {e}")
                continue
            # Ensure remote code present
            write_remote_code_files(local_src, causal=True)  # prefer causal mapping globally for rehost
            # Patch config
            cfg_path = local_src / "config.json"
            if not cfg_path.exists():
                # create minimal config if absent
                minimal = {
                    "model_type": "gpt_bert",
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "intermediate_size": 2560,
                    "vocab_size": 16384,
                }
                cfg_path.write_text(json.dumps(minimal, indent=2), encoding="utf-8")
            _patch_rehost_config(cfg_path, prefer_causal=True)
            # README
            _write_rehost_readme(local_src, src, dst_repo, prefer_causal=True)
            if args.push:
                try:
                    create_repo(dst_repo, exist_ok=True, private=False)
                    upload_folder(folder_path=str(local_src), repo_id=dst_repo, commit_message=f"Rehost from {src}")
                    print(f"[rehost] Uploaded {dst_repo}")
                except Exception as e:
                    print(f"[rehost] Upload failed for {dst_repo}: {e}")
            else:
                final_local = Path("converted") / dst_repo.split('/')[-1]
                if final_local.exists():
                    shutil.rmtree(final_local)
                shutil.copytree(local_src, final_local)
                print(f"[rehost] Saved locally at {final_local}")
    print("[rehost] Done.")



def derive_vocab_size_from_state(state_dict: Dict[str, torch.Tensor], default: int) -> int:
    for key in [
        "embedding.word_embedding.weight",
        "model.embedding.word_embedding.weight",
        "model.model.embedding.word_embedding.weight",
        "embedding.word_embeddings.weight",
    ]:
        t = state_dict.get(key)
        if t is not None and hasattr(t, "shape"):
            return int(t.shape[0])
    return int(default)


def write_model_card(
    out_dir: Path,
    repo_id: str,
    lang: str,
    config_dict: Dict[str, Any],
    tokenizer,
    default_variant: str,
    available_variants: List[str],
    raw_files: List[str],
    causal: bool,
):
    files_listing = []
    if (out_dir / "model.safetensors").exists():
        files_listing.append("- model.safetensors (alias of default variant)")
    if (out_dir / "model_main.safetensors").exists():
        files_listing.append("- model_main.safetensors")
    if (out_dir / "model_ema.safetensors").exists():
        files_listing.append("- model_ema.safetensors")
    if (out_dir / "pytorch_model.bin").exists():
        files_listing.append("- pytorch_model.bin (legacy PyTorch format)")
    if raw_files:
        for rf in raw_files:
            files_listing.append(f"- {rf} (raw training checkpoint)")
    files_section = "\n".join(files_listing) if files_listing else "(generated after conversion)"

    causal_section = "" if not causal else f"""\n### Causal LM Wrapper\nThis repo includes a lightweight GPTBertForCausalLM wrapper.\nGeneration example:\n```python\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\nmid='{repo_id}'\ntok=AutoTokenizer.from_pretrained(mid)\nmodel=AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)\nprint(tok.decode(model.generate(**tok('Hello', return_tensors='pt'), max_new_tokens=20)[0], skip_special_tokens=True))\n```\n"""
    # Minimal YAML frontmatter for Hub metadata validation
    header = "\n".join([
        "---",
        "library_name: transformers",
        f"pipeline_tag: {'text-generation' if causal else 'fill-mask'}",
        "tags: [gpt-bert, babylm, remote-code]",
        "license: other",
        "---",
        "",
    ])
    card = header + f"""# {repo_id}\n\nGPT-BERT style BabyBabyLLM model for language **{lang}**.\n\nThis repository may include both *main* and *EMA* variants.\n\n**Default variant exposed to generic loaders:** `{default_variant}`\n\n## Variants Available\n{', '.join(sorted(available_variants))}\n\n## Files\n{files_section}\n\n## Configuration\n```json\n{json.dumps(config_dict, indent=2)}\n```\nTokenizer file: `{Path(getattr(tokenizer, '_tokenizer_file', 'unknown')).name}`\n\n## Quick Usage\n```python\nfrom transformers import AutoTokenizer, AutoModelForMaskedLM\nmodel_id = '{repo_id}'\ntok = AutoTokenizer.from_pretrained(model_id)\nmodel = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)\nout = model(**tok('Hello world', return_tensors='pt'))\n```\n{causal_section}\n## Notes\n- Converted on {datetime.now(timezone.utc).isoformat()}\n- Weights are the exact trained parameters; no new layers were initialized.\n- Requires `trust_remote_code=True` due to custom architecture.\n"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")


def _save_single_variant(
    ckpt_path: Path,
    base_config_dict: Dict[str, Any],
    force_vocab_match: bool,
    lang: str,
    variant_tag: str,
    out_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    vocab_from_weights = derive_vocab_size_from_state(state_dict, base_config_dict.get("vocab_size", 0))
    if force_vocab_match and vocab_from_weights:
        config_dict = {**base_config_dict, "vocab_size": int(vocab_from_weights)}
    else:
        config_dict = dict(base_config_dict)

    # Harmonize parameter keys so they load into the HF wrapper's `model` module.
    prefixed_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = key
        if name.startswith("module."):
            name = name[len("module."):]
        if not name.startswith("model."):
            name = f"model.{name}"
        prefixed_state_dict[name] = value
    state_dict = prefixed_state_dict

    # Save weights directly from original state dict
    if HAS_SAFETENSORS:
        out_file = out_dir / ("model.safetensors" if not (out_dir / "model.safetensors").exists() and variant_tag else f"model_{variant_tag}.safetensors")
        try:
            # Clone any shared-storage tensors to avoid safetensors shared memory warning
            # (e.g., tied head/embedding weights). See: safetensors torch_shared_tensors docs.
            processed: Dict[str, torch.Tensor] = {}
            storage_map: Dict[int, str] = {}
            for k, v in state_dict.items():
                if not isinstance(v, torch.Tensor):
                    continue
                stor_id = v.storage().data_ptr() if v.storage() is not None else id(v)
                if stor_id in storage_map:
                    processed[k] = v.clone().contiguous()
                else:
                    storage_map[stor_id] = k
                    processed[k] = v.detach().contiguous()
            save_safetensors(processed, str(out_file))
        except Exception as e:
            print(f"[WARN] safetensors save failed ({e}); falling back to PyTorch bin for variant {variant_tag}")
            torch.save(state_dict, out_dir / (f"pytorch_model_{variant_tag}.bin"))
    else:
        torch.save(state_dict, out_dir / (f"pytorch_model_{variant_tag}.bin"))

    return config_dict, state_dict


def _write_primary_config(out_dir: Path, config_dict: Dict[str, Any], prefer_causal: bool = False):
    """Write a canonical config.json for the repo (after variants are saved)."""
    cfg = dict(config_dict)
    cfg.setdefault("model_type", "gpt_bert")
    # Ensure architectures and auto_map are present
    arch = cfg.get("architectures", []) or []
    for n in ["GPTBertForMaskedLM", "GPTBertForCausalLM"]:
        if n not in arch:
            arch.append(n)
    cfg["architectures"] = arch
    if prefer_causal:
        cfg["auto_map"] = {
            "AutoConfig": "configuration_gpt_bert.GPTBertConfig",
            "AutoModel": "modeling_gpt_bert.GPTBertForCausalLM",
            "AutoModelForCausalLM": "modeling_gpt_bert.GPTBertForCausalLM",
            "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBertForMaskedLM",
        }
    else:
        cfg["auto_map"] = {
            "AutoConfig": "configuration_gpt_bert.GPTBertConfig",
            "AutoModel": "modeling_gpt_bert.GPTBertForMaskedLM",
            "AutoModelForCausalLM": "modeling_gpt_bert.GPTBertForCausalLM",
            "AutoModelForMaskedLM": "modeling_gpt_bert.GPTBertForMaskedLM",
        }
    # Populate token ids if tokenizer available
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id", "sep_token_id", "cls_token_id"]:
            val = getattr(tok, attr, None)
            if val is not None:
                cfg[attr] = val
    except Exception as e:
        print(f"[WARN] Could not reload tokenizer for ids: {e}")
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_raw_checkpoints(out_dir: Path, current_ckpt: Path, include_raw: bool) -> List[str]:
    if not include_raw:
        return []
    copied: List[str] = []
    def _cp(src: Path):
        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied.append(dst.name)
    _cp(current_ckpt)
    # counterpart
    name = current_ckpt.name
    if name.endswith('_ema.bin'):
        base = name.replace('_ema.bin', '.bin')
        cand = current_ckpt.parent / base
        if cand.exists():
            _cp(cand)
    else:
        ema_name = name.replace('.bin', '_ema.bin')
        cand = current_ckpt.parent / ema_name
        if cand.exists():
            _cp(cand)
    return copied


def convert_language(
    lang: str,
    ckpt_main: Optional[Path],
    ckpt_ema: Optional[Path],
    args,
    base_config: Dict[str, Any],
):
    repo_id_base = format_repo_id(args, lang)
    variants_to_do: List[Tuple[str, Path]] = []
    if args.variant in ("main", "ema", "both") and ckpt_main is not None:
        variants_to_do.append(("main", ckpt_main))
    if args.variant in ("ema", "both") and ckpt_ema is not None:
        variants_to_do.append(("ema", ckpt_ema))
    if not variants_to_do:
        print(f"[!] No checkpoints found for language {lang} matching variant selection")
        return

    if args.separate_ema:
        # Process each variant independently into its own repo if ema
        for tag, ckpt in variants_to_do:
            # Build per-variant repo id when storing EMA separately
            repo_id = format_repo_id(args, lang, variant=tag)
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td) / repo_id.split('/')[-1]
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"[+] Converting {ckpt.name} -> {repo_id} (separate repo)")
                config_dict, state_dict = _save_single_variant(ckpt, base_config, args.force_vocab_match, lang, tag, out_dir)
                tokenizer = build_tokenizer(lang, args)
                tokenizer.save_pretrained(out_dir)
                (out_dir / "original_project_config.json").write_text(json.dumps(base_config, indent=2), encoding="utf-8")
                write_remote_code_files(out_dir, args.causal)
                raw_files = _copy_raw_checkpoints(out_dir, ckpt, args.include_raw)
                # Pointer files (default only if matches)
                if tag == args.default_variant:
                    # Create legacy pytorch_model.bin
                    torch.save(state_dict, out_dir / "pytorch_model.bin")
                # Always ensure model.safetensors points to chosen default variant when only one variant
                if tag != args.default_variant:
                    # create alias if default not present yet
                    if not (out_dir / "model.safetensors").exists():
                        shutil.copy2(out_dir / f"model_{tag}.safetensors", out_dir / "model.safetensors")
                write_model_card(out_dir, repo_id, lang, config_dict, tokenizer, args.default_variant, [tag], raw_files, args.causal)
                # Ensure canonical config.json present (save_pretrained temp copy discarded)
                _write_primary_config(out_dir, config_dict, prefer_causal=args.causal)
                if args.push:
                    create_repo(repo_id, exist_ok=True, private=False)
                    upload_folder(folder_path=str(out_dir), repo_id=repo_id, commit_message=f"Add {tag} weights for {lang}")
                else:
                    final_local = Path("converted") / out_dir.name
                    final_local.parent.mkdir(exist_ok=True)
                    if final_local.exists():
                        shutil.rmtree(final_local)
                    shutil.copytree(out_dir, final_local)
                    print(f"Saved locally at: {final_local}")
        return

    # Single repo with potentially both variants
    repo_id = repo_id_base
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / repo_id.split('/')[-1]
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = build_tokenizer(lang, args)
        tokenizer.save_pretrained(out_dir)
        (out_dir / "original_project_config.json").write_text(json.dumps(base_config, indent=2), encoding="utf-8")
        write_remote_code_files(out_dir, args.causal)
        raw_files_accum: List[str] = []
        default_state_dict: Optional[Dict[str, torch.Tensor]] = None
        final_config: Optional[Dict[str, Any]] = None
        available_variant_tags: List[str] = []
        for tag, ckpt in variants_to_do:
            print(f"[+] Converting {ckpt.name} ({tag}) -> {repo_id}")
            config_dict, state_dict = _save_single_variant(ckpt, base_config, args.force_vocab_match, lang, tag, out_dir)
            final_config = config_dict  # Same config for all
            available_variant_tags.append(tag)
            new_raw = _copy_raw_checkpoints(out_dir, ckpt, args.include_raw)
            raw_files_accum.extend(new_raw)
            if tag == args.default_variant:
                default_state_dict = state_dict
        # Ensure alias files for default variant
        if args.default_variant in available_variant_tags:
            src = out_dir / ("model_" + args.default_variant + ".safetensors")
            if src.exists():
                shutil.copy2(src, out_dir / "model.safetensors")
                if default_state_dict is not None:
                    torch.save(default_state_dict, out_dir / "pytorch_model.bin")
        else:
            first_tag = available_variant_tags[0]
            src = out_dir / ("model_" + first_tag + ".safetensors")
            if src.exists():
                shutil.copy2(src, out_dir / "model.safetensors")
        assert final_config is not None
        write_model_card(out_dir, repo_id, lang, final_config, tokenizer, args.default_variant, available_variant_tags, raw_files_accum, args.causal)
        _write_primary_config(out_dir, final_config, prefer_causal=args.causal)
        if args.push:
            create_repo(repo_id, exist_ok=True, private=False)
            upload_folder(folder_path=str(out_dir), repo_id=repo_id, commit_message=f"Add {' & '.join(available_variant_tags)} weights for {lang}")
        else:
            final_local = Path("converted") / out_dir.name
            final_local.parent.mkdir(exist_ok=True)
            if final_local.exists():
                shutil.rmtree(final_local)
            shutil.copytree(out_dir, final_local)
            print(f"Saved locally at: {final_local}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["main", "ema", "both"], default="main", help="Which checkpoints to convert.")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face Hub.")
    parser.add_argument("--separate-ema", action="store_true", help="Store EMA in a separate -ema repo id (only affects EMA variant).")
    parser.add_argument("--username", default=None, help="Hub username/org.")
    # New: Source checkpoint discovery controls
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR), help="Directory containing source checkpoints (default: babybabellm-gptbert/model_checkpoints)")
    parser.add_argument("--checkpoint-glob", default=CHECKPOINT_GLOB, help="Glob pattern to list candidate checkpoint files inside --checkpoint-dir (default: mono_*_small_1_2*.bin)")
    parser.add_argument("--checkpoint-regex", default=CHECKPOINT_PATTERN.pattern, help="Regex with named groups 'lang' and optional 'ema' to parse filenames (default matches mono_*_small_1_2*.bin)")
    parser.add_argument("--languages", nargs="*", default=None, help="Restrict to these 3-letter language codes (auto-detect if omitted).")
    parser.add_argument("--force-vocab-match", action="store_true", help="Override config vocab_size with size inferred from checkpoint weights.")
    parser.add_argument("--config-file", default=None, help="Explicit config JSON (overrides auto selection).")
    parser.add_argument("--include-raw", action="store_true", help="Copy original raw training checkpoint(s) into repo (current & counterpart if present).")
    parser.add_argument("--default-variant", choices=["ema", "main"], default="ema", help="Which variant to expose as default (model.safetensors & pytorch_model.bin).")
    parser.add_argument("--causal", action="store_true", help="Emit CausalLM wrapper and push to repo id with -causal suffix.")
    parser.add_argument("--rehost-prefix", default=None, help="Rehost existing Hub repos whose ids start with this prefix (e.g. suchirsalham/babybabellm-mono-). When set, skips local checkpoint conversion.")
    # New: Repo naming template
    parser.add_argument(
        "--repo-template",
        default=None,
        help=(
            "Custom HF repo id template using fields {username}, {lang}, {causal_suffix}, {variant}, {variant_suffix}. "
            "Example: '{username}/bbllm-{lang}{variant_suffix}{causal_suffix}'"
        ),
    )
    # New: Tokenizer controls
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help=(
            "Tokenizer Hub repo id (e.g., 'babylm-org/baby-tokenizer-{lang}'). "
            "If set, overrides local detection. Supports the {lang} placeholder."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Local path to tokenizer directory or tokenizer.json. Supports {lang} placeholder. "
            "Used when --tokenizer-id is not provided."
        ),
    )
    args = parser.parse_args()

    if not args.username:
        env_username = os.environ.get(REPO_USERNAME_ENV_VAR)
        if env_username:
            args.username = env_username
        else:
            parser.error("Hub username missing. Specify --username or set the HF_USERNAME environment variable.")

    # Rehosting mode: bypass local checkpoint conversion
    if args.rehost_prefix:
        rehost_repos(args)
        return

    # Load base config
    if args.config_file:
        with open(args.config_file) as f:
            base_config = json.load(f)
    else:
        # try common locations; fallback to a minimal config
        candidates = [
            Path("babybabellm-gptbert/configs/small.json"),
            Path("gpt-bert/configs/small.json"),
        ]
        cfg_path = next((p for p in candidates if p.exists()), None)
        if cfg_path is None:
            base_config = {
                "hidden_size": 768,
                "intermediate_size": 2560,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "layer_norm_eps": 1e-5,
                "max_position_embeddings": 512,
                "position_bucket_size": 32,
                "attention_probs_dropout_prob": 0.1,
                "hidden_dropout_prob": 0.1,
                "vocab_size": 16384,
            }
        else:
            base_config = json.loads(cfg_path.read_text(encoding="utf-8"))
    print("Base config:")
    print(json.dumps(base_config, indent=2))

    # Discover all checkpoints once (using CLI-provided dir/glob/regex)
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        print(f"[!] Checkpoint dir not found: {ckpt_dir}")
        return
    try:
        filename_regex = re.compile(args.checkpoint_regex)
    except re.error as e:
        print(f"[!] Invalid --checkpoint-regex: {e}")
        return
    all_ckpts = list(ckpt_dir.glob(args.checkpoint_glob))
    lang_map: Dict[str, Dict[str, Path]] = {}
    for ck in all_ckpts:
        m = filename_regex.match(ck.name)
        if not m:
            continue
        lang = m.groupdict().get('lang')
        if not lang:
            continue
        # EMA detection: if regex has 'ema' group use it, else fallback to filename contains '_ema'
        ema_group = m.groupdict().get('ema')
        is_ema = (ema_group is not None) and (ema_group != '')
        if not is_ema:
            is_ema = '_ema' in ck.name
        if args.languages and lang not in args.languages:
            continue
        entry = lang_map.setdefault(lang, {})
        entry['ema' if is_ema else 'main'] = ck

    if not lang_map:
        print("No matching checkpoints found.")
        return

    for lang, ckdict in sorted(lang_map.items()):
        convert_language(
            lang,
            ckdict.get('main'),
            ckdict.get('ema'),
            args,
            base_config,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
