# coding=utf-8

import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from socket import gethostname
from tokenizers import Tokenizer
from statistics import mean
import json
import math
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from lamb import Lamb
from model_extra import Bert
from utils import (
    cosine_schedule_with_warmup_cooldown,
    is_main_process,
    get_rank,
    seed_everything,
    get_world_size,
)
from dataset import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path",
        default="../data/MULTILINGUAL-SMALL/train",
        type=str,
        help="Path to the training shards directory.",
    )
    parser.add_argument(
        "--valid_path",
        default="../data/MULTILINGUAL-SMALL/valid",
        type=str,
        help="Path to the validation shards directory.",
    )
    parser.add_argument(
        "--name", default="gptbert_baby_multi", type=str, help="Name of the run."
    )
    parser.add_argument(
        "--config_file",
        default="../configs/base.json",
        type=str,
        help="The BERT model config",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        type=str,
        help="Path to an existing tokenizer.json (required; created by external script).",
    )
    parser.add_argument(
        "--output_dir",
        default="../model_checkpoints",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_filename",
        default=None,
        type=str,
        help="The checkpoint filename to resume training.",
    )
    parser.add_argument(
        "--optimizer", default="lamb", type=str, help="The optimizer to use."
    )
    parser.add_argument(
        "--hybrid_numerator",
        default=15,
        type=int,
        help="The numerator of the hybrid ratio.",
    )
    parser.add_argument(
        "--hybrid_denominator",
        default=16,
        type=int,
        help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).",
    )
    parser.add_argument(
        "--seq_length", default=128, type=int, help="Sequence length for training."
    )
    parser.add_argument(
        "--local_batch_size",
        default=128,
        type=int,
        help="Batch size for training per GPU.",
    )
    parser.add_argument(
        "--global_batch_size",
        default=32768,
        type=int,
        help="Total batch size for training per GPUs and per grad accumulation step.",
    )
    parser.add_argument(
        "--batch_reduction",
        default=4,
        type=int,
        help="The initial batch size reduction factor.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1.0e-2,
        type=float,
        help="The initial learning rate.",
    )
    parser.add_argument(
        "--max_steps",
        default=31_250 // 2,
        type=int,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--ema_decay",
        default=0.999,
        type=float,
        help="Exponential moving average decay.",
    )
    parser.add_argument(
        "--validate_every",
        default=1_000,
        type=int,
        help="Run validation after every X training shards.",
    )
    parser.add_argument(
        "--validation_steps", default=1, type=int, help="Number of validation steps."
    )
    parser.add_argument(
        "--log_stats_every", default=100, type=int, help="Log stats every X steps."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.016,
        type=float,
        help="Proportion of training for warmup.",
    )
    parser.add_argument(
        "--cooldown_proportion",
        default=0.016,
        type=float,
        help="Proportion of training for cooldown.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--save_every", type=int, default=1_000, help="save every X steps"
    )
    parser.add_argument(
        "--mask_p_start", default=0.3, type=float, help="Initial masking probability."
    )
    parser.add_argument(
        "--mask_p_end", default=0.15, type=float, help="Final masking probability."
    )
    parser.add_argument(
        "--mask_random_p",
        default=0.1,
        type=float,
        help="Probability of replacing the masked token with a random token.",
    )
    parser.add_argument(
        "--mask_keep_p",
        default=0.1,
        type=float,
        help="Probability of keeping the masked token.",
    )
    parser.add_argument(
        "--weight_decay", default=0.1, type=float, help="Weight decay."
    )
    parser.add_argument(
        "--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon."
    )
    parser.add_argument(
        "--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1."
    )
    parser.add_argument(
        "--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2."
    )
    parser.add_argument(
        "--max_gradient",
        default=2.0,
        type=float,
        help="Max value for gradient clipping.",
    )
    parser.add_argument(
        "--mixed_precision",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Mixed precision training.",
    )
    parser.add_argument(
        "--n_special_tokens", default=16, type=int, help="Number of special tokens."
    )
    parser.add_argument(
        "--z_loss_weight", default=1e-4, type=float, help="Weight for the z loss."
    )
    parser.add_argument(
        "--token_weighted_loss",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use token weighted loss.",
    )
    parser.add_argument(
        "--single_gpu_hybrid",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable hybrid (masked+causal) scheduling on a single GPU by time-slicing.",
    )
    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        help="Override W&B project.",
    )
    parser.add_argument(
        "--wandb_entity",
        default=None,
        type=str,
        help="Override W&B entity.",
    )
    parser.add_argument(
        "--wandb_disabled",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Disable all Weights & Biases logging.",
    )
    parser.add_argument(
        "--vocab_size",
        default=int(os.environ.get("VOCAB_SIZE", 32768)),
        type=int,
        help="Tokenizer vocab size to train if tokenizer_path not provided.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_path = os.path.join(args.output_dir, f"{args.name}.bin")

    return args


def _maybe_build_tokenizer(args):
    """Validate tokenizer_path instead of building it. The tokenizer must be created by an external script."""
    if args.tokenizer_path is None or len(args.tokenizer_path) == 0:
        raise FileNotFoundError(
            "--tokenizer_path is required. Build the tokenizer via scripts/run_train.sh or your own preprocessing and pass the path."
        )
    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file not found: {args.tokenizer_path}. Ensure the preprocessing script created it."
        )
    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    # Read distributed env from torchrun or SLURM (fallbacks provided)
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    args.gpus_per_node = int(
        os.environ.get(
            "SLURM_GPUS_ON_NODE",
            os.environ.get("LOCAL_WORLD_SIZE", str(torch.cuda.device_count())),
        )
    )
    physical_gpus = torch.cuda.device_count()
    if args.gpus_per_node != physical_gpus:
        if args.world_size > 1 and not args.single_gpu_hybrid:
            raise AssertionError(
                f"Allocated gpus_per_node={args.gpus_per_node} but detected {physical_gpus} physical GPUs."
            )
        if is_main_process():
            print(
                f"[Warn] SLURM_GPUS_ON_NODE={args.gpus_per_node} differs from physical {physical_gpus}; proceeding (single_gpu_hybrid={args.single_gpu_hybrid}).",
                flush=True,
            )
    print(
        f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {args.gpus_per_node} allocated GPUs per node.",
        flush=True,
    )

    # Allow single-GPU hybrid mode by time-slicing objectives (masked first, then causal)
    if not (args.world_size == 1 and args.single_gpu_hybrid):
        assert args.world_size % args.hybrid_denominator == 0

    if args.world_size == 1 and args.single_gpu_hybrid:
        args.dataset_type = "hybrid_single"
    elif args.rank * args.hybrid_denominator < args.hybrid_numerator * args.world_size:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"

    print(f"Dataset type: {args.dataset_type}", flush=True)

    seed_everything(args.seed + args.rank)

    torch.distributed.init_process_group(
        backend="nccl", rank=args.rank, world_size=args.world_size
    )
    if args.rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    args.local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            str(args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)),
        )
    )
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}")

    args.vocab_size = tokenizer.get_vocab_size()
    args.cumulative_tokens = getattr(args, "cumulative_tokens", 0)

    # Informative projections (parity with gpt-bert)
    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs", flush=True)
        naive_tokens = (
            args.max_steps * get_world_size() * args.local_batch_size * args.seq_length
        )
        print(
            f"[info] Naive token estimate (no accumulation, base seq_len): {naive_tokens:,}",
            flush=True,
        )
        curriculum_multiplier = 0.7 * 1 + 0.2 * 2 + 0.1 * 4  # ~1.5
        effective_tokens = int(
            args.max_steps * args.global_batch_size * args.seq_length * curriculum_multiplier
        )
        print(
            f"[info] Effective token projection (global_batch={args.global_batch_size}, seq schedule factor={curriculum_multiplier:.2f}): {effective_tokens:,}",
            flush=True,
        )
        args.projected_effective_tokens = effective_tokens

    # W&B setup
    if is_main_process() and not args.wandb_disabled:
        project = (
            args.wandb_project
            if args.wandb_project is not None
            else os.environ.get("WANDB_PROJECT", "gpt-bert")
        )
        entity = (
            args.wandb_entity
            if args.wandb_entity is not None
            else os.environ.get("WANDB_ENTITY")
        )
        import wandb
        wandb_kwargs = {"name": args.name, "project": project}
        if entity:
            wandb_kwargs["entity"] = entity
        wandb.init(**wandb_kwargs)
    elif is_main_process() and args.wandb_disabled:
        print("W&B logging disabled via --wandb_disabled", flush=True)


def load_config(args):
    with open(args.config_file, "r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        try:
            import wandb
            if not args.wandb_disabled:
                wandb.config.update(args)
                wandb.config.update({"n_params": n_params})
        except Exception:
            pass
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(args.device)

    no_decay = ["bias", "layer_norm"]
    decay_params = [
        (n, p)
        for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay)
    ]
    no_decay_params = [
        (n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
    ]
    optimizer_grouped_parameters = [
        {"params": [p for _, p in decay_params], "weight_decay": args.weight_decay},
        {"params": [p for _, p in no_decay_params], "weight_decay": 0.0},
    ]

    if args.optimizer in ("adam", "adamw"):
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    else:
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )

    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1,
    )

    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(args.device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )

    ema_model: nn.Module = copy.deepcopy(model.module)
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step, epoch = 0, 0
    if args.checkpoint_filename is not None and os.path.exists(args.checkpoint_filename):
        state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema_model.load_state_dict(state_dict["ema_model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        global_step = state_dict.get("global_step", 0)
        epoch = state_dict.get("epoch", 0)
        if "cumulative_tokens" in state_dict:
            args.cumulative_tokens = state_dict["cumulative_tokens"]
        return model, ema_model, optimizer, scheduler, global_step, epoch

    return model, ema_model, optimizer, scheduler, global_step, epoch


def get_batch(dataloader, device, global_step):
    dataloader._dataset.set_global_step(global_step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [
        t.pin_memory().to(device, non_blocking=True) for t in batch
    ]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p


def training_epoch(
    model,
    ema_model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    global_step,
    epoch,
    args,
):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    num_steps = min(
        len(train_dataloader), (args.max_steps - global_step) * args.accumulate_steps
    )

    # Determine schedule thresholds in global steps (for seq_length changes)
    th1 = int(0.7 * args.max_steps + 0.5)
    th2 = int(0.9 * args.max_steps + 0.5)
    def phase(step: int) -> int:
        if step >= th2:
            return 2
        if step >= th1:
            return 1
        return 0
    current_phase = phase(global_step)

    # Synchronize planned local loop length across ranks
    try:
        _ns = torch.tensor([num_steps], device=args.device, dtype=torch.long)
        torch.distributed.all_reduce(_ns, torch.distributed.ReduceOp.MIN)
        num_steps = int(_ns.item())
    except Exception:
        pass

    train_dataloader = iter(train_dataloader)
    total_loss = torch.tensor(0.0, device=args.device)
    total_accuracy = torch.tensor(0.0, device=args.device)
    total_z_loss = torch.tensor(0.0, device=args.device)
    total_mask_p = torch.tensor(0.0, device=args.device)
    total_grad_norm = torch.tensor(0.0, device=args.device)
    micro_in_accum = 0

    input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(
        train_dataloader, args.device, global_step
    )

    pbar = tqdm(total=args.max_steps, desc="Train steps", initial=global_step) if is_main_process() else None
    for local_step in range(num_steps):
        input_ids, attention_mask, target_ids, mask_p = (
            input_ids_,
            attention_mask_,
            target_ids_,
            mask_p_,
        )
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            with ModelLogger(enable=global_step % 100 == 0, module=model):
                loss, accuracy, z_loss, num_tokens = model(
                    input_ids, attention_mask, target_ids
                )

        if local_step < num_steps - 1:
            input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(
                train_dataloader, args.device, global_step
            )

        if args.token_weighted_loss:
            total_tokens = torch.tensor(
                num_tokens, device=args.device, dtype=torch.long
            )
            torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
            weight = args.world_size * num_tokens / total_tokens / args.accumulate_steps
        else:
            weight = 1.0 / args.accumulate_steps

        ((loss + args.z_loss_weight * z_loss) * weight).backward()
        total_loss += loss.detach() * weight
        total_accuracy += accuracy * weight
        total_z_loss += z_loss * weight
        total_mask_p += mask_p * weight

        micro_in_accum += 1
        if (local_step + 1) % args.accumulate_steps != 0:
            if pbar is not None:
                pbar.set_postfix({
                    "micro": f"{micro_in_accum}/{args.accumulate_steps}",
                    "seq": getattr(train_dataloader, 'dataset', getattr(train_dataloader, '_dataset', None)).seq_length
                })
            continue
        micro_in_accum = 0

        total_grad_norm += (
            nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight
        )
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            for param_q, param_k in zip(
                model.module.parameters(), ema_model.parameters()
            ):
                param_k.data.mul_(args.ema_decay).add_(
                    (1.0 - args.ema_decay) * param_q.detach().data
                )

            if args.dataset_type == "masked":
                total_mlm_loss = total_loss / (
                    args.hybrid_numerator / args.hybrid_denominator
                )
                total_clm_loss = torch.zeros_like(total_mlm_loss)
                total_mask_p = total_mask_p / (
                    args.hybrid_numerator / args.hybrid_denominator
                )
            else:
                total_clm_loss = total_loss / (
                    1 - args.hybrid_numerator / args.hybrid_denominator
                )
                total_mlm_loss = torch.zeros_like(total_clm_loss)
                total_mask_p = torch.zeros_like(total_mask_p)

            metrics = torch.stack(
                [
                    total_loss,
                    total_accuracy,
                    total_z_loss,
                    total_mask_p,
                    total_mlm_loss,
                    total_clm_loss,
                ]
            )
            torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
            (
                total_loss,
                total_accuracy,
                total_z_loss,
                total_mask_p,
                total_mlm_loss,
                total_clm_loss,
            ) = metrics.tolist()

        # Token accounting
        seq_len_log = getattr(train_dataloader, "dataset", None)
        if seq_len_log is None:
            seq_len_log = getattr(train_dataloader, "_dataset", None)
        seq_length_value = getattr(seq_len_log, "seq_length", args.seq_length)
        step_tokens = args.current_global_batch_size * seq_length_value
        args.cumulative_tokens += step_tokens

        if is_main_process() and not args.wandb_disabled:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": total_loss,
                "train/z_loss": total_z_loss,
                "train/perplexity": math.exp(total_loss),
                "train/accuracy": total_accuracy * 100.0,
                "train/mlm_loss": total_mlm_loss,
                "train/clm_loss": total_clm_loss,
                "stats/learning_rate": optimizer.param_groups[0]["lr"],
                "stats/grad_norm": total_grad_norm,
                "stats/seq_length": seq_length_value,
                "stats/global_batch_size": args.current_global_batch_size,
                "stats/local_batch_size": args.current_local_batch_size,
                "stats/accumulate_steps": args.accumulate_steps,
                "stats/mask_p": total_mask_p,
                "tokens/cumulative": args.cumulative_tokens,
                "tokens/projected_effective_total": getattr(args, "projected_effective_tokens", None),
                "tokens/progress_pct": (
                    100.0
                    * args.cumulative_tokens
                    / max(1, getattr(args, "projected_effective_tokens", args.max_steps))
                ),
            }, commit=False)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=args.device)
        total_accuracy = torch.tensor(0.0, device=args.device)
        total_z_loss = torch.tensor(0.0, device=args.device)
        total_mask_p = torch.tensor(0.0, device=args.device)
        total_grad_norm = torch.tensor(0.0, device=args.device)

        if global_step % args.save_every == 0:
            save(model, ema_model, optimizer, scheduler, global_step, epoch, args)

        if (global_step + 1) % args.validate_every == 0:
            validation_epoch(model, valid_dataloader, epoch, args)
            model.train()

        if is_main_process() and not args.wandb_disabled:
            import wandb
            wandb.log({"global_step": global_step}, commit=True)

        global_step += 1
        if pbar is not None:
            pbar.update(1)

        # If we crossed a schedule boundary, break to reload dataloaders
        new_phase = phase(global_step)
        if new_phase > current_phase:
            if pbar is not None:
                pbar.close()
            return global_step
        if global_step >= args.max_steps:
            if pbar is not None:
                pbar.close()
            return global_step
    if pbar is not None:
        pbar.close()
    return global_step


@torch.no_grad()
def validation_epoch(model, valid_dataloader, epoch, args, commit=False):
    model = model.eval()

    losses, accuracies = [], []
    valid_dataloader = iter(valid_dataloader)
    input_ids, attention_mask, target_ids, _ = get_batch(
        valid_dataloader, args.device, 0
    )
    for _ in tqdm(
        range(args.validation_steps),
        desc="Valid iteration",
        disable=not is_main_process(),
    ):
        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)

        total_tokens = torch.tensor(num_tokens, device=args.device, dtype=torch.long)
        torch.distributed.all_reduce(total_tokens, torch.distributed.ReduceOp.SUM)
        weight = args.world_size * num_tokens / total_tokens

        metrics = torch.stack([loss * weight, accuracy * weight])
        torch.distributed.all_reduce(metrics, torch.distributed.ReduceOp.AVG)
        loss, accuracy = metrics.tolist()

        losses.append(loss)
        accuracies.append(accuracy)

    if is_main_process() and not args.wandb_disabled:
        import wandb
        wandb.log({
            "epoch": epoch,
            "validation/loss": mean(losses),
            "validation/accuracy": mean(accuracies) * 100.0,
            "validation/perplexity": math.exp(mean(losses)),
        }, commit=commit)


def save(model, ema_model, optimizer, scheduler, global_step, epoch, args):
    if is_main_process():
        out_dir = os.path.dirname(args.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        torch.save(model_to_save.state_dict(), args.output_path)
        torch.save(ema_model.state_dict(), args.output_path.replace(".bin", "_ema.bin"))
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch + 1,
                "cumulative_tokens": getattr(args, "cumulative_tokens", 0),
            },
            args.output_path.replace(".bin", "_state_dict.bin"),
        )


def load_datasets(
    args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader
):
    train_seed = args.seed + get_rank() + epoch * get_world_size()

    if (global_step + 1) / args.max_steps >= 0.9:
        seq_length = args.seq_length * 4
        global_batch_size = args.global_batch_size // 4
    elif (global_step + 1) / args.max_steps >= 0.7:
        seq_length = args.seq_length * 2
        global_batch_size = args.global_batch_size // 2
    else:
        seq_length = args.seq_length
        global_batch_size = args.global_batch_size

    if train_dataloader is None or train_dataloader.dataset.seq_length != seq_length:
        if args.world_size == 1 and args.single_gpu_hybrid:
            masked_fraction = args.hybrid_numerator / args.hybrid_denominator
            progress = (global_step + 1) / args.max_steps
            use_masked = progress <= masked_fraction
            rank = 0
            world_size = 1
            if use_masked:
                train_data = MaskedDataset(
                    args.train_path, tokenizer, args, seq_length, rank, world_size
                )
                args.dataset_type = "masked"
            else:
                train_data = CausalDataset(
                    args.train_path, tokenizer, args, seq_length, rank, world_size
                )
                args.dataset_type = "causal"
        else:
            if args.dataset_type == "masked":
                rank = args.rank
                world_size = (
                    args.world_size * args.hybrid_numerator // args.hybrid_denominator
                )
                train_data = MaskedDataset(
                    args.train_path, tokenizer, args, seq_length, rank, world_size
                )
            else:
                rank = (
                    args.rank
                    - args.world_size * args.hybrid_numerator // args.hybrid_denominator
                )
                world_size = (
                    args.world_size
                    * (args.hybrid_denominator - args.hybrid_numerator)
                    // args.hybrid_denominator
                )
                train_data = CausalDataset(
                    args.train_path, tokenizer, args, seq_length, rank, world_size
                )

        if is_main_process():
            train_data.show_random_item(tokenizer)
    else:
        train_data = train_dataloader.dataset

    args.current_global_batch_size = int(
        global_batch_size / args.batch_reduction * (1 - global_step / args.max_steps)
        + global_batch_size * (global_step / args.max_steps)
        + 0.5
    )
    total_local_batch_size = int(args.current_global_batch_size / args.world_size + 0.5)
    args.accumulate_steps = int(
        math.ceil(total_local_batch_size / args.local_batch_size)
    )
    args.current_local_batch_size = max(1, total_local_batch_size // args.accumulate_steps)
    if args.max_steps <= 10 and args.accumulate_steps > 1:
        shrink = args.accumulate_steps
        args.accumulate_steps = 1
        args.current_local_batch_size = min(args.local_batch_size, total_local_batch_size)
        if is_main_process():
            print(f"[short-run] reduce accumulation {shrink} -> 1 (local_batch={args.current_local_batch_size})")

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=args.current_local_batch_size,
        num_workers=0,
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )

    if valid_dataloader is None:
        valid_data = ValidationDataset(args.valid_path, tokenizer, args)
        valid_dataloader = DataLoader(
            valid_data,
            shuffle=False,
            batch_size=args.local_batch_size,
            num_workers=0,
            generator=torch.Generator().manual_seed(42),
            drop_last=True,
            pin_memory=True,
        )

    try:
        torch.distributed.barrier()
    except Exception:
        pass

    return train_dataloader, valid_dataloader


if __name__ == "__main__":
    args = parse_arguments()

    # Ensure tokenizer path is provided and exists (no auto-building here)
    args = _maybe_build_tokenizer(args)
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    setup_training(args, tokenizer)
    model, ema_model, optimizer, scheduler, global_step, start_epoch = (
        prepare_model_and_optimizer(args)
    )

    train_dataloader, valid_dataloader = None, None

    for epoch in count(start=start_epoch):
        train_dataloader, valid_dataloader = load_datasets(
            args, tokenizer, epoch, global_step, train_dataloader, valid_dataloader
        )
        global_step = training_epoch(
            model,
            ema_model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            global_step,
            epoch,
            args,
        )
        if global_step >= args.max_steps:
            break

    save(model, ema_model, optimizer, scheduler, global_step, epoch, args)
    validation_epoch(model, valid_dataloader, epoch, args, commit=True)
