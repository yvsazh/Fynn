from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import IterableDataset


class PreparedShardDataset(IterableDataset):
    FORMAT = "prepared_sequence_shards"
    VERSION = 2

    def __init__(self, manifest_path: str | Path, shuffle: bool, seed: int):
        self.manifest_path = Path(manifest_path)
        payload = torch.load(self.manifest_path, weights_only=False)
        if payload.get("format") != self.FORMAT:
            raise ValueError(f"Unsupported prepared dataset format in {self.manifest_path}")
        if int(payload.get("version", 0)) != self.VERSION:
            raise ValueError(
                f"Incompatible prepared dataset version in {self.manifest_path}: "
                f"{payload.get('version')!r}. Expected {self.VERSION}. Rebuild the prepared dataset."
            )
        if not payload.get("completed", False):
            raise ValueError(f"Prepared dataset is incomplete: {self.manifest_path}")
        self.shards = list(payload.get("shards", []))
        self.num_samples = int(payload.get("num_samples", 0))
        self.shuffle = shuffle
        self.seed = seed
        self._iteration = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the base shuffle seed for the given epoch.

        Call once per epoch before iterating so that resumed training uses the
        correct shard/sample order rather than restarting from 0.
        """
        self._iteration = epoch

    def __iter__(self):
        shard_entries = list(self.shards)
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1
        if self.shuffle:
            rng.shuffle(shard_entries)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard_entries = shard_entries[worker_info.id :: worker_info.num_workers]

        for shard in shard_entries:
            shard_path = self.manifest_path.parent / shard["path"]
            samples = torch.load(shard_path, weights_only=False)
            indices = list(range(len(samples)))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                yield samples[index]


class PackingIterableDataset(IterableDataset):
    """Greedy sequence packing: concatenates short samples into a single context window.

    Samples are concatenated until the next sample would exceed *max_seq_len*.
    When that happens the current buffer is emitted and a new buffer starts.

    This eliminates the extreme padding that occurs when a very short sample
    (e.g. 80 tokens) shares a batch row with a long sample (6000 tokens).

    Notes
    -----
    * Loss targets are preserved: each sample's labels tensor is concatenated
      as-is, so prefix positions remain -100 and only target positions carry
      real token ids.
    * A packed row may contain tokens from multiple samples.  Position i in a
      packed row CAN attend to all previous positions (global causal mask),
      meaning sample B's prefix sees sample A's tokens.  This is the standard
      "simple packing" approximation used by most LLM training frameworks and
      is acceptable because the loss is only computed on target positions.
    * The ``mode`` field is taken from the first sample in each pack (used only
      for logging in the training loop).
    * Samples that are individually longer than *max_seq_len* are passed through
      unchanged; the collator will truncate them as usual.
    """

    def __init__(self, base: IterableDataset, max_seq_len: int) -> None:
        self.base = base
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        # Upper bound: actual packed count ≤ base count
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self):
        buf_ids: list[int] = []
        buf_labels: list[int] = []
        buf_mode: str = "scratch"

        for sample in self.base:
            ids: list[int] = sample["input_ids"].tolist()
            labels: list[int] = sample["labels"].tolist()
            mode: str = sample["mode"]

            if not buf_ids:
                buf_ids = ids
                buf_labels = labels
                buf_mode = mode
                continue

            if len(buf_ids) + len(ids) <= self.max_seq_len:
                # Fits — append to current buffer
                buf_ids.extend(ids)
                buf_labels.extend(labels)
                # mode stays as the first sample's mode (logging only)
            else:
                # Emit current buffer and start a new one
                yield {
                    "input_ids": torch.tensor(buf_ids, dtype=torch.long),
                    "labels": torch.tensor(buf_labels, dtype=torch.long),
                    "mode": buf_mode,
                }
                buf_ids = ids
                buf_labels = labels
                buf_mode = mode

        if buf_ids:
            yield {
                "input_ids": torch.tensor(buf_ids, dtype=torch.long),
                "labels": torch.tensor(buf_labels, dtype=torch.long),
                "mode": buf_mode,
            }


def load_prepared_dataset(
    path: str | Path,
    *,
    shuffle: bool,
    seed: int,
    pack_sequences: bool = False,
    max_seq_len: int = 0,
) -> IterableDataset:
    ds: IterableDataset = PreparedShardDataset(path, shuffle=shuffle, seed=seed)
    if pack_sequences and max_seq_len > 0:
        ds = PackingIterableDataset(ds, max_seq_len=max_seq_len)
    return ds


@dataclass(slots=True)
class SequenceCollator:
    pad_id: int
    max_seq_len: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        clipped_inputs = [item["input_ids"][: self.max_seq_len] for item in batch]
        clipped_labels = [item["labels"][: self.max_seq_len] for item in batch]
        batch_max_len = max(int(item.numel()) for item in clipped_inputs)

        input_ids = torch.full((len(batch), batch_max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), batch_max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), batch_max_len), dtype=torch.long)
        modes = torch.zeros(len(batch), dtype=torch.long)

        for row, (item_inputs, item_labels, item) in enumerate(zip(clipped_inputs, clipped_labels, batch)):
            seq_len = int(item_inputs.numel())
            input_ids[row, :seq_len] = item_inputs.to(dtype=torch.long)
            labels[row, :seq_len] = item_labels.to(dtype=torch.long)
            attention_mask[row, :seq_len] = 1
            modes[row] = 0 if item["mode"] == "scratch" else 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "modes": modes,
        }
