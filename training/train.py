from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from fynn.config import ExperimentConfig
from fynn.inference.export import write_score_bundle_paths
from fynn.inference.generation import add_track_to_midi, generate_from_scratch
from fynn.model import FynnTransformer
from fynn.score_utils import score_ticks_per_beat
from fynn.tokenization.unified import UnifiedTokenizer
from fynn.training.dataset import SequenceCollator, load_prepared_dataset
from fynn.training.tensorboard_logger import TensorBoardLogger
from fynn.utils import device_for_training, ensure_dir, save_json, save_yaml, set_seed, torch_dtype_from_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MIDI generation transformer.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def get_lr_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int, min_ratio: float) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_ratio + (1 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def make_autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch_dtype_from_name(mixed_precision)
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate(
    model: FynnTransformer,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if (labels == -100).all():
                continue
            with make_autocast_context(device, mixed_precision):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_val = outputs["loss"].item()
            if math.isfinite(loss_val):
                total_loss += loss_val
                total_batches += 1
    return {"loss": total_loss / max(1, total_batches)}


def save_checkpoint(
    path: Path,
    model: FynnTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "config": config.to_dict(),
        },
        path,
    )


def _param_norm(model: FynnTransformer) -> float:
    total = 0.0
    for parameter in model.parameters():
        total += parameter.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def _mode_breakdown(modes: torch.Tensor) -> tuple[float, float]:
    if modes.numel() == 0:
        return 0.0, 0.0
    scratch_ratio = (modes == 0).float().mean().item()
    add_track_ratio = (modes == 1).float().mean().item()
    return scratch_ratio, add_track_ratio


def _log_tensorboard_histograms(writer: TensorBoardLogger, model: FynnTransformer, global_step: int) -> None:
    histogram_targets: dict[str, torch.Tensor] = {
        "weights/embedding": model.embedding.weight,
        "weights/lm_head": model.lm_head.weight,
    }
    if model.blocks:
        histogram_targets["weights/block0_attn_q_proj"] = model.blocks[0].attn.q_proj.weight
        histogram_targets["weights/block0_ffn_gate"] = model.blocks[0].ffn.gate.weight
        histogram_targets["weights/block_last_attn_q_proj"] = model.blocks[-1].attn.q_proj.weight
        histogram_targets["weights/block_last_ffn_gate"] = model.blocks[-1].ffn.gate.weight

    for name, value in histogram_targets.items():
        writer.add_histogram(name, value.detach().float().cpu().numpy(), global_step)

    gradient_targets: dict[str, torch.Tensor] = {}
    if model.embedding.weight.grad is not None:
        gradient_targets["grads/embedding"] = model.embedding.weight.grad
    if model.blocks and model.blocks[0].attn.q_proj.weight.grad is not None:
        gradient_targets["grads/block0_attn_q_proj"] = model.blocks[0].attn.q_proj.weight.grad
    if model.blocks and model.blocks[-1].attn.q_proj.weight.grad is not None:
        gradient_targets["grads/block_last_attn_q_proj"] = model.blocks[-1].attn.q_proj.weight.grad

    for name, value in gradient_targets.items():
        writer.add_histogram(name, value.detach().float().cpu().numpy(), global_step)


def _update_train_progress(
    progress_bar,
    *,
    global_step: int,
    epoch_loss: float,
    epoch_valid_batches: int,
    lr: float,
    status: str = "",
) -> None:
    postfix: dict[str, str | int] = {
        "step": global_step,
        "loss": f"{epoch_loss / max(1, epoch_valid_batches):.4f}" if epoch_valid_batches > 0 else "n/a",
        "lr": f"{lr:.2e}",
    }
    if status:
        postfix["status"] = status
    progress_bar.set_postfix(postfix, refresh=False)
    progress_bar.update(1)


def _score_stats(score) -> dict:
    """Extract lightweight statistics from a generated Score."""
    num_tracks = len(score.tracks)
    num_notes = sum(len(t.notes) for t in score.tracks)
    programs = sorted({t.program for t in score.tracks if not t.is_drum})
    has_drums = any(t.is_drum for t in score.tracks)
    last_tick = max(
        (max((n.end for n in t.notes), default=0) for t in score.tracks if t.notes),
        default=0,
    )
    tpb = score_ticks_per_beat(score)
    duration_beats = round(last_tick / tpb, 2)
    return {
        "num_tracks": num_tracks,
        "num_notes": num_notes,
        "programs": programs,
        "has_drums": has_drums,
        "duration_beats": duration_beats,
    }


def _generate_epoch_samples(
    model: FynnTransformer,
    tokenizer: UnifiedTokenizer,
    config: ExperimentConfig,
    device: torch.device,
    run_dir: Path,
    epoch: int,
    writer: TensorBoardLogger | None,
) -> list[dict]:
    if not config.sampling.enabled:
        return []

    was_training = model.training
    epoch_dir = ensure_dir(run_dir / config.sampling.output_subdir / f"epoch_{epoch:03d}")
    summary: list[dict] = []

    scratch_success = 0
    scratch_errors = 0
    scratch_total_notes = 0
    scratch_total_tracks = 0

    for index, prompt in enumerate(config.sampling.scratch_prompts):
        sample_dir = ensure_dir(epoch_dir / f"scratch_{index:02d}")
        item: dict = {
            "mode": "scratch",
            "prompt": prompt,
            "output_dir": str(sample_dir),
        }
        try:
            score = generate_from_scratch(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_prompt_tokens=config.data.max_prompt_tokens,
                max_new_tokens=config.sampling.max_new_tokens,
                temperature=config.sampling.temperature,
                top_k=config.sampling.top_k,
                top_p=config.sampling.top_p,
            )
            stats = _score_stats(score)
            item.update(stats)
            item["files"] = write_score_bundle_paths(score, sample_dir).to_dict()
            scratch_success += 1
            scratch_total_notes += stats["num_notes"]
            scratch_total_tracks += stats["num_tracks"]
        except Exception as exc:
            item["error"] = f"{exc.__class__.__name__}: {exc}"
            scratch_errors += 1
        save_json(sample_dir / "meta.json", item)
        summary.append(item)

    add_track_success: bool | None = None
    add_track_notes = 0
    add_track_tracks = 0

    if config.sampling.add_track_existing_midi:
        sample_dir = ensure_dir(epoch_dir / "add_track")
        item = {
            "mode": "add_track",
            "prompt": config.sampling.add_track_prompt,
            "instrument": config.sampling.add_track_instrument,
            "existing_midi": config.sampling.add_track_existing_midi,
            "output_dir": str(sample_dir),
        }
        try:
            score = add_track_to_midi(
                model=model,
                tokenizer=tokenizer,
                existing_midi_path=config.sampling.add_track_existing_midi,
                instrument=config.sampling.add_track_instrument,
                extra_prompt=config.sampling.add_track_prompt,
                device=device,
                max_prompt_tokens=config.data.max_prompt_tokens,
                max_context_tokens=config.data.max_context_tokens,
                max_new_tokens=config.sampling.max_new_tokens,
                temperature=config.sampling.temperature,
                top_k=config.sampling.top_k,
                top_p=config.sampling.top_p,
            )
            stats = _score_stats(score)
            item.update(stats)
            item["files"] = write_score_bundle_paths(score, sample_dir).to_dict()
            add_track_success = True
            add_track_notes = stats["num_notes"]
            add_track_tracks = stats["num_tracks"]
        except Exception as exc:
            item["error"] = f"{exc.__class__.__name__}: {exc}"
            add_track_success = False
        save_json(sample_dir / "meta.json", item)
        summary.append(item)

    save_json(epoch_dir / "summary.json", summary)

    if writer is not None:
        writer.add_text(
            f"samples/epoch_{epoch:03d}",
            json.dumps(summary, indent=2),
            epoch,
        )
        scratch_total = scratch_success + scratch_errors
        writer.add_scalar("samples/scratch_success", scratch_success, epoch)
        writer.add_scalar("samples/scratch_errors", scratch_errors, epoch)
        if scratch_total > 0:
            writer.add_scalar("samples/scratch_error_rate", scratch_errors / scratch_total, epoch)
        writer.add_scalar("samples/scratch_total_notes", scratch_total_notes, epoch)
        writer.add_scalar("samples/scratch_total_tracks", scratch_total_tracks, epoch)
        if scratch_success > 0:
            writer.add_scalar(
                "samples/scratch_avg_notes_per_sample",
                scratch_total_notes / scratch_success,
                epoch,
            )
            writer.add_scalar(
                "samples/scratch_avg_tracks_per_sample",
                scratch_total_tracks / scratch_success,
                epoch,
            )
        if add_track_success is not None:
            writer.add_scalar("samples/add_track_success", int(add_track_success), epoch)
            writer.add_scalar("samples/add_track_total_notes", add_track_notes, epoch)
            writer.add_scalar("samples/add_track_total_tracks", add_track_tracks, epoch)

    print(
        f"sample_generations epoch={epoch} dir={epoch_dir} "
        f"scratch={scratch_success}/{scratch_success + scratch_errors} "
        f"notes={scratch_total_notes}"
        + (f" add_track={'ok' if add_track_success else 'fail'}" if add_track_success is not None else "")
    )
    if was_training:
        model.train()
    return summary


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    set_seed(config.training.seed)

    tokenizer_path = Path(config.data.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact not found: {tokenizer_path}. "
            "Run fynn.data.build_tokenizer before training."
        )
    tokenizer = UnifiedTokenizer.load(tokenizer_path)

    config.model.vocab_size = tokenizer.vocab_size
    config.model.pad_id = tokenizer.pad_id
    config.model.max_seq_len = config.data.max_seq_len

    train_dataset = load_prepared_dataset(
        config.data.train_dataset_path,
        shuffle=True,
        seed=config.training.seed,
        pack_sequences=config.data.pack_sequences,
        max_seq_len=config.data.max_seq_len,
    )
    val_dataset = load_prepared_dataset(
        config.data.val_dataset_path,
        shuffle=False,
        seed=config.training.seed,
    )
    collator = SequenceCollator(pad_id=tokenizer.pad_id, max_seq_len=config.data.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    device = device_for_training()
    model = FynnTransformer(config.model).to(device)
    if config.training.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile ...")
        model = torch.compile(model)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )

    total_updates_per_epoch = math.ceil(len(train_loader) / config.training.grad_accumulation)
    total_steps = max(1, config.training.epochs * total_updates_per_epoch)
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=config.training.lr_warmup_steps,
        total_steps=total_steps,
        min_ratio=config.training.lr_min_ratio,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.training.mixed_precision == "fp16")
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]

    run_dir = ensure_dir(config.data.output_dir)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    metrics_path = run_dir / "metrics.jsonl"
    save_yaml(run_dir / "resolved_config.yaml", config.to_dict())
    save_json(
        run_dir / "run_meta.json",
        {
            "device": str(device),
            "num_parameters": model.num_parameters,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_hash": tokenizer.payload_hash(),
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
    )
    writer: TensorBoardLogger | None = None
    if config.training.tensorboard:
        tensorboard_dir = ensure_dir(run_dir / config.training.tensorboard_subdir)
        writer = TensorBoardLogger(log_dir=tensorboard_dir, flush_secs=config.training.tensorboard_flush_secs)
        writer.add_text("model/repr", repr(model), 0)
        writer.add_text("run/config", json.dumps(config.to_dict(), indent=2), 0)
        writer.add_text(
            "run/tokenizer",
            f"vocab_size={tokenizer.vocab_size}\ntext_vocab={tokenizer.text_tokenizer.vocab_size}\nmidi_vocab={tokenizer.midi_tokenizer.vocab_size}",
            0,
        )
        writer.add_scalar("run/num_parameters", model.num_parameters, 0)
        writer.add_scalar("run/train_samples", len(train_dataset), 0)
        writer.add_scalar("run/val_samples", len(val_dataset), 0)
        writer.add_text("run/device", str(device), 0)
        writer.add_text(
            "model/graph_status",
            "Graph tab is not emitted by the low-level writer. Use model/repr text and parameter histograms instead.",
            0,
        )

    print("Building MIDI vocabulary introspection cache...")
    _t_cache = time.time()
    tokenizer.build_midi_introspection_cache()
    _vocab_stats = tokenizer.vocab_introspection_stats()
    print(
        f"Introspection cache built in {time.time() - _t_cache:.3f}s  "
        f"total_ids={_vocab_stats.get('total_ids', 0)}  "
        f"multi_token_ids={_vocab_stats.get('ids_multi_raw_token', 0)}  "
        f"mean_raw_per_id={_vocab_stats.get('mean_raw_tokens_per_id', 0.0):.2f}  "
        f"max_raw_per_id={_vocab_stats.get('max_raw_tokens_per_id', 0)}"
    )
    if writer is not None:
        writer.add_text("run/vocab_introspection_stats", json.dumps(_vocab_stats, indent=2), 0)
        for _stat_key, _stat_val in _vocab_stats.items():
            if isinstance(_stat_val, (int, float)):
                writer.add_scalar(f"run/vocab_{_stat_key}", _stat_val, 0)

    for epoch in range(start_epoch, config.training.epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        epoch_valid_batches = 0
        epoch_start = time.time()
        with tqdm(
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{config.training.epochs}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.0,
            miniters=1,
        ) as progress_bar:
            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                lr = optimizer.param_groups[0]["lr"]

                # Skip batches where truncation left zero loss targets (all-prefix sequences)
                if (labels == -100).all():
                    _update_train_progress(
                        progress_bar,
                        global_step=global_step,
                        epoch_loss=epoch_loss,
                        epoch_valid_batches=epoch_valid_batches,
                        lr=lr,
                        status="skip-empty",
                    )
                    continue

                with make_autocast_context(device, config.training.mixed_precision):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs["loss"] / config.training.grad_accumulation

                if not torch.isfinite(loss):
                    tqdm.write(
                        f"Warning: non-finite loss at epoch={epoch} batch={step}, "
                        "clearing gradients and skipping batch"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    _update_train_progress(
                        progress_bar,
                        global_step=global_step,
                        epoch_loss=epoch_loss,
                        epoch_valid_batches=epoch_valid_batches,
                        lr=lr,
                        status="skip-nan",
                    )
                    continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                epoch_loss += loss.item() * config.training.grad_accumulation
                epoch_valid_batches += 1

                if step % config.training.grad_accumulation == 0 or step == len(train_loader):
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    lr = optimizer.param_groups[0]["lr"]
                    mean_loss = epoch_loss / max(1, epoch_valid_batches)
                    non_pad_tokens = int(attention_mask.sum().item())
                    scratch_ratio, add_track_ratio = _mode_breakdown(batch["modes"])

                    if writer is not None:
                        writer.add_scalar("train/loss_step", loss.item() * config.training.grad_accumulation, global_step)
                        writer.add_scalar("train/loss_running", mean_loss, global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                        writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                        writer.add_scalar("train/non_pad_tokens", non_pad_tokens, global_step)
                        writer.add_scalar("train/batch_seq_len", int(input_ids.size(1)), global_step)
                        writer.add_scalar("train/mode_scratch_ratio", scratch_ratio, global_step)
                        writer.add_scalar("train/mode_add_track_ratio", add_track_ratio, global_step)
                        writer.add_scalar("train/param_norm", _param_norm(model), global_step)

                _update_train_progress(
                    progress_bar,
                    global_step=global_step,
                    epoch_loss=epoch_loss,
                    epoch_valid_batches=epoch_valid_batches,
                    lr=lr,
                )

        train_loss = epoch_loss / max(1, epoch_valid_batches)
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "epoch_time_sec": time.time() - epoch_start,
        }

        if (epoch + 1) % config.training.eval_interval == 0 and len(val_dataset) > 0:
            val_metrics = evaluate(model, val_loader, device, config.training.mixed_precision)
            metrics["val_loss"] = val_metrics["loss"]
            if writer is not None:
                writer.add_scalar("eval/val_loss", metrics["val_loss"], global_step)
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                if writer is not None:
                    writer.add_scalar("eval/best_val_loss", best_val_loss, global_step)
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    best_val_loss,
                    config,
                )
        else:
            metrics["val_loss"] = None

        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("train/epoch_time_sec", metrics["epoch_time_sec"], epoch)
            if config.training.tensorboard_histogram_interval > 0 and (epoch + 1) % config.training.tensorboard_histogram_interval == 0:
                _log_tensorboard_histograms(writer, model, global_step)

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics))

        _generate_epoch_samples(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            run_dir=run_dir,
            epoch=epoch,
            writer=writer,
        )

        if config.training.save_every_epoch:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                best_val_loss,
                config,
            )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            best_val_loss,
            config,
        )

    if args.resume and start_epoch >= config.training.epochs:
        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            start_epoch - 1,
            global_step,
            best_val_loss,
            config,
        )
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
