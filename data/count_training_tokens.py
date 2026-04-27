from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from fynn.config import ExperimentConfig
from fynn.data.manifest_utils import iter_manifest_records, normalize_manifest_inputs
from fynn.data.prepare_dataset import (
    PreparedDatasetShardWriter,
    _iter_parallel_results,
    _iter_serial_results,
)
from fynn.tokenization.unified import UnifiedTokenizer


@dataclass(slots=True)
class SplitTotals:
    samples: int = 0
    input_tokens: int = 0
    loss_tokens: int = 0
    scratch_samples: int = 0
    add_track_samples: int = 0
    scratch_input_tokens: int = 0
    add_track_input_tokens: int = 0
    max_input_len: int = 0

    def add_sample(self, sample: dict[str, Any]) -> None:
        input_len = int(sample["input_ids"].numel())
        loss_len = int((sample["labels"] != -100).sum().item())
        mode = str(sample["mode"])

        self.samples += 1
        self.input_tokens += input_len
        self.loss_tokens += loss_len
        self.max_input_len = max(self.max_input_len, input_len)

        if mode == "scratch":
            self.scratch_samples += 1
            self.scratch_input_tokens += input_len
        elif mode == "add_track":
            self.add_track_samples += 1
            self.add_track_input_tokens += input_len

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "input_tokens": self.input_tokens,
            "loss_tokens": self.loss_tokens,
            "scratch_samples": self.scratch_samples,
            "add_track_samples": self.add_track_samples,
            "scratch_input_tokens": self.scratch_input_tokens,
            "add_track_input_tokens": self.add_track_input_tokens,
            "max_input_len": self.max_input_len,
            "mean_input_len": round(self.input_tokens / max(1, self.samples), 3),
            "mean_loss_len": round(self.loss_tokens / max(1, self.samples), 3),
        }


@dataclass(slots=True)
class ManifestTokenCounter:
    train_ratio: float
    seed: int
    flush_every_samples: int
    buffer: list[dict[str, Any]] = field(default_factory=list)
    train: SplitTotals = field(default_factory=SplitTotals)
    val: SplitTotals = field(default_factory=SplitTotals)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def add_result(self, result: dict[str, Any]) -> None:
        self.buffer.extend(result["samples"])
        if len(self.buffer) >= self.flush_every_samples:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        self.rng.shuffle(self.buffer)
        split_index = int(len(self.buffer) * self.train_ratio)
        train_samples = self.buffer[:split_index]
        val_samples = self.buffer[split_index:]
        self.buffer = []

        for sample in train_samples:
            self.train.add_sample(sample)
        for sample in val_samples:
            self.val.add_sample(sample)

    def finalize(self) -> None:
        self.flush()

    def to_dict(self) -> dict[str, Any]:
        total = SplitTotals()
        total.samples = self.train.samples + self.val.samples
        total.input_tokens = self.train.input_tokens + self.val.input_tokens
        total.loss_tokens = self.train.loss_tokens + self.val.loss_tokens
        total.scratch_samples = self.train.scratch_samples + self.val.scratch_samples
        total.add_track_samples = self.train.add_track_samples + self.val.add_track_samples
        total.scratch_input_tokens = self.train.scratch_input_tokens + self.val.scratch_input_tokens
        total.add_track_input_tokens = self.train.add_track_input_tokens + self.val.add_track_input_tokens
        total.max_input_len = max(self.train.max_input_len, self.val.max_input_len)
        return {
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "total": total.to_dict(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count exact training tokens for this project either from a manifest or from prepared dataset shards."
    )
    parser.add_argument("--config", type=str, default="", help="Optional training YAML config for default paths/token limits.")
    parser.add_argument(
        "--manifest",
        type=str,
        action="append",
        default=None,
        help="Raw manifest file or directory for manifest mode. Repeat to combine multiple inputs.",
    )
    parser.add_argument(
        "--manifest-root",
        type=str,
        default="",
        help="Base directory for relative MIDI paths. Defaults to each manifest file's parent directory.",
    )
    parser.add_argument("--tokenizer-path", type=str, default="", help="Tokenizer artifact path for manifest mode.")
    parser.add_argument("--prepared-train", type=str, default="", help="Prepared train.pt path for prepared mode.")
    parser.add_argument("--prepared-val", type=str, default="", help="Prepared val.pt path for prepared mode.")
    parser.add_argument("--train-ratio", type=float, default=0.95, help="Must match prepare_dataset if counting from manifest.")
    parser.add_argument("--seed", type=int, default=42, help="Must match prepare_dataset if counting from manifest.")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument("--max-target-tokens", type=int, default=6400)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--flush-every-samples", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
    return parser.parse_args()


def _apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    config = ExperimentConfig.from_yaml(args.config)
    if not args.tokenizer_path:
        args.tokenizer_path = config.data.tokenizer_path
    if not args.prepared_train:
        args.prepared_train = config.data.train_dataset_path
    if not args.prepared_val:
        args.prepared_val = config.data.val_dataset_path
    if args.max_prompt_tokens == 192:
        args.max_prompt_tokens = config.data.max_prompt_tokens
    if args.max_context_tokens == 1536:
        args.max_context_tokens = config.data.max_context_tokens
    if args.max_target_tokens == 6400:
        args.max_target_tokens = config.data.max_target_tokens
    if args.max_seq_len == 8192:
        args.max_seq_len = config.data.max_seq_len
    return args


def _count_prepared_split(path: Path) -> SplitTotals:
    payload = torch.load(path, weights_only=False)
    if payload.get("format") != PreparedDatasetShardWriter.FORMAT:
        raise ValueError(f"Unsupported prepared dataset format in {path}")
    if int(payload.get("version", 0)) != PreparedDatasetShardWriter.VERSION:
        raise ValueError(
            f"Incompatible prepared dataset version in {path}: {payload.get('version')!r}. "
            f"Expected {PreparedDatasetShardWriter.VERSION}. Rebuild the prepared dataset."
        )
    totals = SplitTotals()
    for shard in payload.get("shards", []):
        shard_path = path.parent / shard["path"]
        samples = torch.load(shard_path, weights_only=False)
        for sample in samples:
            totals.add_sample(sample)
    return totals


def _count_from_prepared(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.prepared_train)
    if not train_path.exists():
        raise FileNotFoundError(f"Prepared train manifest not found: {train_path}")

    train = _count_prepared_split(train_path)
    val = SplitTotals()
    if args.prepared_val:
        val_path = Path(args.prepared_val)
        if not val_path.exists():
            raise FileNotFoundError(f"Prepared val manifest not found: {val_path}")
        val = _count_prepared_split(val_path)

    total = SplitTotals()
    total.samples = train.samples + val.samples
    total.input_tokens = train.input_tokens + val.input_tokens
    total.loss_tokens = train.loss_tokens + val.loss_tokens
    total.scratch_samples = train.scratch_samples + val.scratch_samples
    total.add_track_samples = train.add_track_samples + val.add_track_samples
    total.scratch_input_tokens = train.scratch_input_tokens + val.scratch_input_tokens
    total.add_track_input_tokens = train.add_track_input_tokens + val.add_track_input_tokens
    total.max_input_len = max(train.max_input_len, val.max_input_len)

    return {
        "mode": "prepared",
        "prepared_train": str(train_path),
        "prepared_val": str(args.prepared_val) if args.prepared_val else "",
        "train": train.to_dict(),
        "val": val.to_dict(),
        "total": total.to_dict(),
    }


def _count_from_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_inputs = normalize_manifest_inputs(args.manifest)
    for manifest_input in manifest_inputs:
        if not manifest_input.exists():
            raise FileNotFoundError(f"Manifest input not found: {manifest_input}")
    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found: {tokenizer_path}")

    manifest_root = Path(args.manifest_root) if args.manifest_root else None
    manifest = iter_manifest_records(manifest_inputs, max_items=args.max_examples)

    tokenizer = UnifiedTokenizer.load(tokenizer_path)
    counter = ManifestTokenCounter(
        train_ratio=args.train_ratio,
        seed=args.seed,
        flush_every_samples=max(1, args.flush_every_samples),
    )

    if args.num_workers > 0:
        results: Iterable[dict[str, Any]] = _iter_parallel_results(
            manifest=manifest,
            manifest_root=manifest_root,
            tokenizer_path=tokenizer_path,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
            include_source_midi_path=False,
        )
    else:
        results = _iter_serial_results(
            manifest=manifest,
            manifest_root=manifest_root,
            tokenizer=tokenizer,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
            max_seq_len=args.max_seq_len,
            include_source_midi_path=False,
        )

    for result in results:
        counter.add_result(result)
    counter.finalize()

    return {
        "mode": "manifest",
        "manifest": str(manifest_inputs[0]),
        "manifest_root": str(manifest_root) if manifest_root is not None else "",
        "tokenizer_path": str(tokenizer_path),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "flush_every_samples": args.flush_every_samples,
        "max_examples": args.max_examples,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_context_tokens": args.max_context_tokens,
        "max_target_tokens": args.max_target_tokens,
        "max_seq_len": args.max_seq_len,
        **(
            {"manifest_inputs": [str(path) for path in manifest_inputs]}
            if len(manifest_inputs) > 1
            else {}
        ),
        **counter.to_dict(),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = _apply_config_defaults(parse_args())
    manifest_mode = bool(args.manifest and args.tokenizer_path)
    prepared_mode = bool(args.prepared_train)

    if manifest_mode:
        summary = _count_from_manifest(args)
    elif prepared_mode:
        summary = _count_from_prepared(args)
    else:
        raise SystemExit(
            "Provide either --prepared-train [--prepared-val] or --manifest together with --tokenizer-path."
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(summary)


if __name__ == "__main__":
    main()
