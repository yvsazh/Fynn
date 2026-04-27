from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import symusic
import torch

from fynn.instruments import instrument_label
from fynn.data.manifest_utils import (
    ManifestRecord,
    count_manifest_records,
    load_manifest as _load_manifest,
    normalize_manifest_inputs,
    resolve_manifest_entry as _resolve_manifest_entry,
    resolve_manifest_record,
    iter_manifest_records,
)
from fynn.midi import read_midi
from fynn.score_utils import (
    score_anchor_ticks,
    score_end_tick,
    score_prompt_metadata,
    split_add_track_context_target,
    split_add_track_window,
    tokenizable_track_indices,
    track_anchor_ticks,
    window_score,
)
from fynn.tokenization.unified import UnifiedTokenizer
from fynn.utils import ensure_dir, iter_bounded_executor_map, save_json

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


_WORKER_TOKENIZER: UnifiedTokenizer | None = None
_WORKER_MANIFEST_ROOT_OVERRIDE: Path | None = None
_WORKER_SEED: int = 42
_WORKER_MAX_PROMPT_TOKENS: int | None = 192
_WORKER_MAX_CONTEXT_TOKENS: int | None = 1536
_WORKER_MAX_TARGET_TOKENS: int | None = 6400
_WORKER_MAX_SEQ_LEN: int = 8192
_WORKER_INCLUDE_SOURCE_MIDI_PATH: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached train/val samples from MIDI + caption pairs.")
    parser.add_argument(
        "--manifest",
        type=str,
        action="append",
        required=True,
        help="Raw manifest file or directory. Repeat to combine multiple inputs.",
    )
    parser.add_argument(
        "--manifest-root",
        type=str,
        default="",
        help="Base directory for relative MIDI paths. Defaults to each manifest file's parent directory.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save train.pt / val.pt / meta.json.")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Where to save or load tokenizer JSON.")
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=192)
    parser.add_argument("--max-context-tokens", type=int, default=1536)
    parser.add_argument("--max-target-tokens", type=int, default=6400)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Maximum decoder sequence length. Used to keep prepared samples within the train-time budget.",
    )
    parser.add_argument(
        "--flush-every-samples",
        type=int,
        default=4096,
        help="Flush prepared samples to disk every N samples so long runs make visible progress on disk.",
    )
    parser.add_argument(
        "--include-source-midi-path",
        action="store_true",
        help="Keep source_midi_path inside each prepared sample. Disabled by default to reduce dataset size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel worker processes for dataset preparation. Use 0 to stay single-process.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run. Reads meta.json from --output-dir to find how many "
            "manifest items were already processed and skips them. Existing shards are kept."
        ),
    )
    return parser.parse_args()


def load_manifest(path: str | Path) -> list[dict]:
    return _load_manifest(path)


def resolve_manifest_entry(
    item: dict[str, Any],
    manifest_root: Path | None = None,
    source_manifest_path: Path | None = None,
) -> tuple[str, str]:
    return _resolve_manifest_entry(item, manifest_root=manifest_root, source_manifest_path=source_manifest_path)


def _tokenizable_track_indices_for_tokenizer(tokenizer: UnifiedTokenizer, score: symusic.Score) -> list[int]:
    config = getattr(getattr(tokenizer, "config", None), "midi", None)
    if config is None:
        return [index for index, track in enumerate(score.tracks) if len(track.notes) > 0]
    return tokenizable_track_indices(
        score,
        use_notes=True,
        use_sustain_pedals=bool(config.use_sustain_pedals),
        use_pitch_bends=bool(config.use_pitch_bends),
        control_change_numbers=getattr(config, "control_change_numbers", ()),
    )


def _score_has_tokenizable_tracks(tokenizer: UnifiedTokenizer, score: symusic.Score) -> bool:
    return bool(_tokenizable_track_indices_for_tokenizer(tokenizer, score))


def _augment_caption_with_score_metadata(caption: str, score: symusic.Score) -> str:
    suffix = score_prompt_metadata(score)
    caption = caption.strip()
    if not suffix:
        return caption
    if not caption:
        return suffix
    return f"{caption} | {suffix}"


def _pick_window_start(
    *,
    note_starts: list[int],
    score_end_tick: int,
    window_ticks: int,
    ticks_per_beat: int,
    rng: random.Random,
) -> int:
    max_start = max(0, score_end_tick - window_ticks)
    if max_start == 0:
        return 0

    if note_starts:
        anchor = int(rng.choice(note_starts))
        lower = max(0, anchor - window_ticks + 1)
        upper = min(anchor, max_start)
        if upper >= lower:
            start_tick = rng.randint(lower, upper)
        else:
            start_tick = min(max_start, max(0, anchor - window_ticks // 2))
    else:
        start_tick = rng.randint(0, max_start)

    beat = max(1, ticks_per_beat)
    return max(0, min(max_start, (start_tick // beat) * beat))


def _fit_scratch_excerpt(
    *,
    tokenizer: UnifiedTokenizer,
    score: symusic.Score,
    rng: random.Random,
    target_budget: int,
) -> tuple[symusic.Score, list[int]]:
    full_tokens = tokenizer.encode_midi(score)
    full_len = len(full_tokens)
    if full_len <= target_budget:
        return score, full_tokens

    score_end = score_end_tick(score)
    note_starts = score_anchor_ticks(score)
    beat = max(1, int(score.tpq))
    window_ticks = max(beat, int(score_end * (target_budget / max(1, full_len)) * 0.9))
    window_ticks = max(beat, (window_ticks // beat) * beat)
    best_score = score
    best_tokens = full_tokens
    best_len = full_len

    for _ in range(12):
        start_tick = _pick_window_start(
            note_starts=note_starts,
            score_end_tick=score_end,
            window_ticks=window_ticks,
            ticks_per_beat=int(score.tpq),
            rng=rng,
        )
        excerpt = window_score(score, start_tick, min(score_end, start_tick + window_ticks))
        if not _score_has_tokenizable_tracks(tokenizer, excerpt):
            window_ticks = max(beat, window_ticks // 2)
            continue
        excerpt_tokens = tokenizer.encode_midi(excerpt)
        excerpt_len = len(excerpt_tokens)
        if excerpt_len < best_len:
            best_score = excerpt
            best_tokens = excerpt_tokens
            best_len = excerpt_len
        if excerpt_len <= target_budget:
            return excerpt, excerpt_tokens

        scaled_ticks = int(window_ticks * (target_budget / max(1, excerpt_len)) * 0.92)
        if scaled_ticks >= window_ticks:
            scaled_ticks = window_ticks - beat
        window_ticks = max(beat, scaled_ticks)

    return best_score, best_tokens


def _fit_add_track_excerpt(
    *,
    tokenizer: UnifiedTokenizer,
    score: symusic.Score,
    target_idx: int,
    rng: random.Random,
    max_context_tokens: int,
    max_target_tokens: int,
    max_total_tokens: int,
) -> tuple[symusic.Score, symusic.Score, list[int], list[int]] | None:
    full_context, full_target = split_add_track_context_target(score, target_idx)
    if not _score_has_tokenizable_tracks(tokenizer, full_context) or not _score_has_tokenizable_tracks(tokenizer, full_target):
        return None

    full_context_tokens = tokenizer.encode_midi(full_context)
    full_target_tokens = tokenizer.encode_midi(full_target)
    full_context_len = len(full_context_tokens)
    full_target_len = len(full_target_tokens)
    if (
        full_context_len <= max_context_tokens
        and full_target_len <= max_target_tokens
        and full_context_len + full_target_len <= max_total_tokens
    ):
        return full_context, full_target, full_context_tokens, full_target_tokens

    target_note_starts = track_anchor_ticks(score.tracks[target_idx])
    score_end = score_end_tick(score)
    beat = max(1, int(score.tpq))
    scale = min(
        1.0,
        max_context_tokens / max(1, full_context_len),
        max_target_tokens / max(1, full_target_len),
        max_total_tokens / max(1, full_context_len + full_target_len),
    )
    window_ticks = max(beat, int(score_end * scale * 0.9))
    window_ticks = max(beat, (window_ticks // beat) * beat)
    best_bundle: tuple[symusic.Score, symusic.Score, list[int], list[int]] | None = None
    best_overflow: tuple[int, int, int] | None = None

    for _ in range(16):
        start_tick = _pick_window_start(
            note_starts=target_note_starts,
            score_end_tick=score_end,
            window_ticks=window_ticks,
            ticks_per_beat=int(score.tpq),
            rng=rng,
        )
        context_excerpt, target_excerpt = split_add_track_window(
            score,
            target_idx,
            start_tick,
            min(score_end, start_tick + window_ticks),
        )
        if not _score_has_tokenizable_tracks(tokenizer, context_excerpt) or not _score_has_tokenizable_tracks(tokenizer, target_excerpt):
            continue

        context_tokens = tokenizer.encode_midi(context_excerpt)
        target_tokens = tokenizer.encode_midi(target_excerpt)
        context_len = len(context_tokens)
        target_len = len(target_tokens)
        total_len = context_len + target_len
        overflow = (
            max(0, context_len - max_context_tokens),
            max(0, target_len - max_target_tokens),
            max(0, total_len - max_total_tokens),
        )
        if best_overflow is None or overflow < best_overflow:
            best_bundle = (context_excerpt, target_excerpt, context_tokens, target_tokens)
            best_overflow = overflow
        if overflow == (0, 0, 0):
            return best_bundle

        scale_down = min(
            max_context_tokens / max(1, context_len),
            max_target_tokens / max(1, target_len),
            max_total_tokens / max(1, total_len),
        )
        scaled_ticks = int(window_ticks * scale_down * 0.92)
        if scaled_ticks >= window_ticks:
            scaled_ticks = window_ticks - beat
        window_ticks = max(beat, scaled_ticks)

    return best_bundle


class _FallbackProgressBar:
    def __init__(self, *, desc: str, total: int | None, unit: str) -> None:
        self.desc = desc
        self.total = total
        self.unit = unit
        self.n = 0
        self._postfix: dict[str, Any] = {}
        self._started_at = time.monotonic()
        self._last_print_at = 0.0

    def __enter__(self) -> "_FallbackProgressBar":
        self._emit(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._emit(force=True)
        return False

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        self._emit()

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        payload: dict[str, Any] = {}
        if ordered_dict is not None:
            payload.update(dict(ordered_dict))
        payload.update(kwargs)
        self._postfix = payload
        if refresh:
            self._emit(force=True)

    @staticmethod
    def write(message: str) -> None:
        print(message)

    def _emit(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_print_at < 2.0:
            return
        self._last_print_at = now
        elapsed = max(0.001, now - self._started_at)
        rate = self.n / elapsed
        if self.total:
            remaining = max(0, self.total - self.n)
            eta = remaining / max(rate, 1e-9)
            status = f"{self.n}/{self.total} {self.unit}"
            timing = f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        else:
            status = f"{self.n} {self.unit}"
            timing = f"elapsed={elapsed:.1f}s"
        postfix = " ".join(f"{key}={value}" for key, value in self._postfix.items())
        suffix = f" {postfix}" if postfix else ""
        print(f"{self.desc}: {status} {timing}{suffix}")


def open_progress_bar(*, desc: str, total: int | None, unit: str = "item") -> Any:
    if _tqdm is None:
        return _FallbackProgressBar(desc=desc, total=total, unit=unit)
    return _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.5,
        miniters=1,
    )


def progress_write(message: str) -> None:
    if _tqdm is not None:
        _tqdm.write(message)
    else:
        print(message)


def _init_prepare_worker(
    tokenizer_path: str,
    manifest_root_override: str,
    seed: int,
    max_prompt_tokens: int | None,
    max_context_tokens: int | None,
    max_target_tokens: int | None,
    max_seq_len: int,
    include_source_midi_path: bool,
) -> None:
    global _WORKER_TOKENIZER
    global _WORKER_MANIFEST_ROOT_OVERRIDE
    global _WORKER_SEED
    global _WORKER_MAX_PROMPT_TOKENS
    global _WORKER_MAX_CONTEXT_TOKENS
    global _WORKER_MAX_TARGET_TOKENS
    global _WORKER_MAX_SEQ_LEN
    global _WORKER_INCLUDE_SOURCE_MIDI_PATH

    _WORKER_TOKENIZER = UnifiedTokenizer.load(tokenizer_path)
    _WORKER_MANIFEST_ROOT_OVERRIDE = Path(manifest_root_override) if manifest_root_override else None
    _WORKER_SEED = seed
    _WORKER_MAX_PROMPT_TOKENS = max_prompt_tokens
    _WORKER_MAX_CONTEXT_TOKENS = max_context_tokens
    _WORKER_MAX_TARGET_TOKENS = max_target_tokens
    _WORKER_MAX_SEQ_LEN = max_seq_len
    _WORKER_INCLUDE_SOURCE_MIDI_PATH = include_source_midi_path


def _compact_sample(sample: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "mode": sample["mode"],
        "input_ids": torch.tensor(sample["input_ids"], dtype=torch.int32),
        "labels": torch.tensor(sample["labels"], dtype=torch.int32),
    }
    if "source_midi_path" in sample:
        compact["source_midi_path"] = sample["source_midi_path"]
    return compact


def _prepare_samples_for_score(
    *,
    tokenizer: UnifiedTokenizer,
    midi_path: str,
    caption: str,
    score: symusic.Score,
    sample_seed: int,
    max_prompt_tokens: int | None,
    max_context_tokens: int | None,
    max_target_tokens: int | None,
    max_seq_len: int,
    include_source_midi_path: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(sample_seed)

    caption = _augment_caption_with_score_metadata(caption, score)
    prompt_tokens = tokenizer.encode_text(caption, max_prompt_tokens)
    prompt_len = len(prompt_tokens)
    scratch_budget_with_eos = min(max_target_tokens or 10**9, max(1, max_seq_len - prompt_len - 5))
    scratch_budget_without_eos = min(max_target_tokens or 10**9, max(1, max_seq_len - prompt_len - 4))
    scratch_full_tokens = tokenizer.encode_midi(score)
    scratch_full_len = len(scratch_full_tokens)

    if scratch_full_len <= scratch_budget_with_eos:
        scratch_score = score
        scratch_tokens = scratch_full_tokens
        scratch_append_eos = True
    elif scratch_full_len <= scratch_budget_without_eos:
        scratch_score = score
        scratch_tokens = scratch_full_tokens
        scratch_append_eos = False
    else:
        scratch_score, scratch_tokens = _fit_scratch_excerpt(
            tokenizer=tokenizer,
            score=score,
            rng=rng,
            target_budget=scratch_budget_without_eos,
        )
        scratch_append_eos = False

    scratch = tokenizer.scratch_sequence_from_midi_tokens(
        prompt_tokens=prompt_tokens,
        midi_tokens=scratch_tokens,
        append_eos=scratch_append_eos,
    )
    if include_source_midi_path:
        scratch["source_midi_path"] = midi_path
    samples = [_compact_sample(scratch)]

    targetable_track_indices = _tokenizable_track_indices_for_tokenizer(tokenizer, score)
    if len(targetable_track_indices) >= 1 and len(score.tracks) >= 2:
        target_idx = int(rng.choice(targetable_track_indices))
        target_track = score.tracks[target_idx]
        prompt = tokenizer.build_add_track_prompt(
            instrument=instrument_label(int(target_track.program), bool(target_track.is_drum), name=str(target_track.name)),
            extra_prompt=caption,
        )
        add_prompt_tokens = tokenizer.encode_text(prompt, max_prompt_tokens)
        add_prompt_len = len(add_prompt_tokens)
        add_total_with_eos = max(1, max_seq_len - add_prompt_len - 8)
        add_total_without_eos = max(1, max_seq_len - add_prompt_len - 7)

        full_context, full_target = split_add_track_context_target(score, target_idx)
        if _score_has_tokenizable_tracks(tokenizer, full_context) and _score_has_tokenizable_tracks(tokenizer, full_target):
            full_context_tokens = tokenizer.encode_midi(full_context)
            full_target_tokens = tokenizer.encode_midi(full_target)
            full_context_len = len(full_context_tokens)
            full_target_len = len(full_target_tokens)

            if (
                full_context_len <= (max_context_tokens or 10**9)
                and full_target_len <= (max_target_tokens or 10**9)
                and full_context_len + full_target_len <= add_total_with_eos
            ):
                add_context = full_context
                add_target = full_target
                add_context_tokens = full_context_tokens
                add_target_tokens = full_target_tokens
                add_append_eos = True
            elif (
                full_context_len <= (max_context_tokens or 10**9)
                and full_target_len <= (max_target_tokens or 10**9)
                and full_context_len + full_target_len <= add_total_without_eos
            ):
                add_context = full_context
                add_target = full_target
                add_context_tokens = full_context_tokens
                add_target_tokens = full_target_tokens
                add_append_eos = False
            else:
                fitted = _fit_add_track_excerpt(
                    tokenizer=tokenizer,
                    score=score,
                    target_idx=target_idx,
                    rng=rng,
                    max_context_tokens=max_context_tokens or 10**9,
                    max_target_tokens=max_target_tokens or 10**9,
                    max_total_tokens=add_total_without_eos,
                )
                if fitted is not None:
                    add_context, add_target, add_context_tokens, add_target_tokens = fitted
                    add_append_eos = False
                else:
                    add_context = None
                    add_target = None
                    add_context_tokens = None
                    add_target_tokens = None
        else:
            add_context = None
            add_target = None
            add_context_tokens = None
            add_target_tokens = None

        add_fits_without_eos = (
            add_context is not None
            and add_target is not None
            and add_context_tokens is not None
            and add_target_tokens is not None
            and len(add_context_tokens) <= (max_context_tokens or 10**9)
            and len(add_target_tokens) <= (max_target_tokens or 10**9)
            and len(add_context_tokens) + len(add_target_tokens) <= add_total_without_eos
        )

        if add_fits_without_eos:
            add_track = tokenizer.add_track_sequence_from_midi_tokens(
                prompt_tokens=add_prompt_tokens,
                context_tokens=add_context_tokens,
                target_tokens=add_target_tokens,
                append_eos=add_append_eos,
            )
            if include_source_midi_path:
                add_track["source_midi_path"] = midi_path
            samples.append(_compact_sample(add_track))
    return samples


_WORKER_TASK_TIMEOUT_SECONDS = 2


def _prepare_manifest_item(task: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
    index, item, source_manifest_path = task
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("prepare_dataset worker tokenizer is not initialized")

    _use_alarm = hasattr(signal, "SIGALRM")
    if _use_alarm:
        def _alarm_handler(signum: int, frame: Any) -> None:
            raise TimeoutError(f"worker task timed out after {_WORKER_TASK_TIMEOUT_SECONDS}s")
        _old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(_WORKER_TASK_TIMEOUT_SECONDS)

    try:
        resolved = resolve_manifest_record(
            ManifestRecord(index=index, item=item, manifest_path=Path(source_manifest_path)),
            manifest_root=_WORKER_MANIFEST_ROOT_OVERRIDE,
        )
        midi_path = resolved.midi_path
        caption = resolved.caption
        score = read_midi(midi_path)
        if not _score_has_tokenizable_tracks(_WORKER_TOKENIZER, score):
            return {"samples": [], "skipped": 1, "error": None}
        return {
            "samples": _prepare_samples_for_score(
                tokenizer=_WORKER_TOKENIZER,
                midi_path=midi_path,
                caption=caption,
                score=score,
                sample_seed=_WORKER_SEED + index,
                max_prompt_tokens=_WORKER_MAX_PROMPT_TOKENS,
                max_context_tokens=_WORKER_MAX_CONTEXT_TOKENS,
                max_target_tokens=_WORKER_MAX_TARGET_TOKENS,
                max_seq_len=_WORKER_MAX_SEQ_LEN,
                include_source_midi_path=_WORKER_INCLUDE_SOURCE_MIDI_PATH,
            ),
            "skipped": 0,
            "error": None,
        }
    except Exception as exc:
        return {"samples": [], "skipped": 1, "error": str(exc)}
    finally:
        if _use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old_handler)


def _iter_serial_results(
    manifest: Iterable[ManifestRecord],
    manifest_root: Path | None,
    tokenizer: UnifiedTokenizer,
    seed: int,
    max_prompt_tokens: int,
    max_context_tokens: int,
    max_target_tokens: int,
    max_seq_len: int,
    include_source_midi_path: bool,
) -> Iterable[dict[str, Any]]:
    for record in manifest:
        try:
            resolved = resolve_manifest_record(record, manifest_root=manifest_root)
        except Exception as exc:
            yield {"samples": [], "skipped": 1, "error": str(exc)}
            continue
        midi_path = resolved.midi_path
        caption = resolved.caption
        try:
            score = read_midi(midi_path)
        except Exception as exc:
            yield {"samples": [], "skipped": 1, "error": f"{midi_path}: {exc}"}
            continue
        if not _score_has_tokenizable_tracks(tokenizer, score):
            yield {"samples": [], "skipped": 1, "error": None}
            continue
        yield {
            "samples": _prepare_samples_for_score(
                tokenizer=tokenizer,
                midi_path=midi_path,
                caption=caption,
                score=score,
                sample_seed=seed + record.index,
                max_prompt_tokens=max_prompt_tokens,
                max_context_tokens=max_context_tokens,
                max_target_tokens=max_target_tokens,
                max_seq_len=max_seq_len,
                include_source_midi_path=include_source_midi_path,
            ),
            "skipped": 0,
            "error": None,
        }


def _iter_parallel_results(
    manifest: Iterable[ManifestRecord],
    manifest_root: Path | None,
    tokenizer_path: Path,
    seed: int,
    max_prompt_tokens: int,
    max_context_tokens: int,
    max_target_tokens: int,
    max_seq_len: int,
    num_workers: int,
    include_source_midi_path: bool,
) -> Iterable[dict[str, Any]]:
    tasks = ((record.index, record.item, str(record.manifest_path)) for record in manifest)
    try:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_prepare_worker,
            initargs=(
                str(tokenizer_path),
                str(manifest_root) if manifest_root is not None else "",
                seed,
                max_prompt_tokens,
                max_context_tokens,
                max_target_tokens,
                max_seq_len,
                include_source_midi_path,
            ),
        ) as executor:
            yield from iter_bounded_executor_map(
                executor,
                _prepare_manifest_item,
                tasks,
                max_in_flight=max(1, num_workers * 2),
            )
    except (OSError, PermissionError) as exc:
        progress_write(f"Parallel dataset prep unavailable ({exc}); falling back to single-process mode.")
        tokenizer = UnifiedTokenizer.load(tokenizer_path)
        yield from _iter_serial_results(
            manifest=manifest,
            manifest_root=manifest_root,
            tokenizer=tokenizer,
            seed=seed,
            max_prompt_tokens=max_prompt_tokens,
            max_context_tokens=max_context_tokens,
            max_target_tokens=max_target_tokens,
            max_seq_len=max_seq_len,
            include_source_midi_path=include_source_midi_path,
        )


class PreparedDatasetShardWriter:
    FORMAT = "prepared_sequence_shards"
    VERSION = 2

    def __init__(
        self,
        *,
        output_dir: Path,
        train_ratio: float,
        seed: int,
        flush_every_samples: int,
        manifest_inputs: list[Path],
        manifest_root: Path | None,
        tokenizer_path: Path,
        tokenizer: UnifiedTokenizer,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.train_dir = ensure_dir(self.output_dir / "train_shards")
        self.val_dir = ensure_dir(self.output_dir / "val_shards")
        self.train_ratio = train_ratio
        self.flush_every_samples = max(1, flush_every_samples)
        self.manifest_inputs = list(manifest_inputs)
        self.manifest_path = self.manifest_inputs[0] if self.manifest_inputs else Path()
        self.manifest_root = manifest_root
        self.tokenizer_path = tokenizer_path
        self.tokenizer = tokenizer
        self.buffer: list[dict[str, Any]] = []

        if resume_state is not None:
            self.train_shards = resume_state["train_shards"]
            self.val_shards = resume_state["val_shards"]
            self.num_train = resume_state["num_train"]
            self.num_val = resume_state["num_val"]
            self.skipped = resume_state["skipped"]
            self.processed_manifest_items = resume_state["processed_manifest_items"]
            self.flushes = resume_state["flushes"]
            # Advance the RNG to match the state after all completed flushes
            self.rng = random.Random(seed)
            dummy = list(range(self.flush_every_samples))
            for _ in range(self.flushes):
                self.rng.shuffle(dummy)
        else:
            self.train_shards = []
            self.val_shards = []
            self.num_train = 0
            self.num_val = 0
            self.skipped = 0
            self.processed_manifest_items = 0
            self.flushes = 0
            self.rng = random.Random(seed)

        self.num_samples = self.num_train + self.num_val

    def add_result(self, result: dict[str, Any]) -> None:
        self.processed_manifest_items += 1
        self.skipped += int(result["skipped"])
        if result["error"]:
            progress_write(f"Skip manifest item: {result['error']}")
        self.buffer.extend(result["samples"])
        if len(self.buffer) >= self.flush_every_samples:
            self.flush()

    @property
    def buffered_samples(self) -> int:
        return len(self.buffer)

    @property
    def prepared_samples(self) -> int:
        return self.num_samples + self.buffered_samples

    def flush(self) -> None:
        if not self.buffer:
            return

        self.rng.shuffle(self.buffer)
        split_index = int(len(self.buffer) * self.train_ratio)
        train_samples = self.buffer[:split_index]
        val_samples = self.buffer[split_index:]
        self.buffer = []

        if train_samples:
            shard_path = self.train_dir / f"shard_{len(self.train_shards):05d}.pt"
            torch.save(train_samples, shard_path)
            self.train_shards.append({"path": str(shard_path.relative_to(self.output_dir)), "num_samples": len(train_samples)})
            self.num_train += len(train_samples)

        if val_samples:
            shard_path = self.val_dir / f"shard_{len(self.val_shards):05d}.pt"
            torch.save(val_samples, shard_path)
            self.val_shards.append({"path": str(shard_path.relative_to(self.output_dir)), "num_samples": len(val_samples)})
            self.num_val += len(val_samples)

        self.num_samples = self.num_train + self.num_val
        self.flushes += 1
        self.write_state(completed=False)
        progress_write(
            f"Flushed dataset shards: train={self.num_train} val={self.num_val} "
            f"processed_manifest_items={self.processed_manifest_items}"
        )

    def write_state(self, *, completed: bool) -> None:
        torch.save(self._split_manifest("train", self.train_shards, self.num_train, completed), self.output_dir / "train.pt")
        torch.save(self._split_manifest("val", self.val_shards, self.num_val, completed), self.output_dir / "val.pt")
        save_json(
            self.output_dir / "meta.json",
            {
                "format": self.FORMAT,
                "version": self.VERSION,
                "completed": completed,
                "processed_manifest_items": self.processed_manifest_items,
                "num_samples": self.num_samples,
                "num_train": self.num_train,
                "num_val": self.num_val,
                "skipped": self.skipped,
                "tokenizer_path": str(self.tokenizer_path),
                "tokenizer_hash": self.tokenizer.payload_hash(),
                "tokenizer_vocab_size": self.tokenizer.vocab_size,
                "manifest": str(self.manifest_path),
                "manifest_root": str(self.manifest_root) if self.manifest_root is not None else "",
                **(
                    {"manifest_inputs": [str(path) for path in self.manifest_inputs]}
                    if len(self.manifest_inputs) > 1
                    else {}
                ),
            },
        )

    def finalize(self) -> None:
        self.flush()
        self.write_state(completed=True)

    def _split_manifest(
        self,
        split: str,
        shards: list[dict[str, Any]],
        num_samples: int,
        completed: bool,
    ) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "split": split,
            "completed": completed,
            "num_samples": num_samples,
            "shards": shards,
        }


def _prepare_progress_postfix(writer: PreparedDatasetShardWriter) -> dict[str, Any]:
    return {
        "samples": writer.prepared_samples,
        "written": writer.num_samples,
        "skipped": writer.skipped,
        "buffered": writer.buffered_samples,
        "flushes": writer.flushes,
    }


def _load_resume_state(output_dir: Path) -> dict[str, Any]:
    meta_path = output_dir / "meta.json"
    train_pt_path = output_dir / "train.pt"
    val_pt_path = output_dir / "val.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Cannot resume: meta.json not found in {output_dir}")
    if not train_pt_path.exists():
        raise FileNotFoundError(f"Cannot resume: train.pt not found in {output_dir}")

    import json as _json
    meta = _json.loads(meta_path.read_text(encoding="utf-8"))

    if meta.get("completed"):
        raise SystemExit(f"Dataset in {output_dir} is already marked as completed. Nothing to resume.")

    train_manifest = torch.load(train_pt_path, weights_only=False)
    val_manifest = torch.load(val_pt_path, weights_only=False) if val_pt_path.exists() else {"shards": []}

    processed = int(meta.get("processed_manifest_items", 0))
    num_train = int(meta.get("num_train", 0))
    num_val = int(meta.get("num_val", 0))
    skipped = int(meta.get("skipped", 0))
    flushes = len(train_manifest.get("shards", []))

    print(
        f"Resuming from {output_dir}: "
        f"processed_manifest_items={processed}, "
        f"num_train={num_train}, num_val={num_val}, "
        f"skipped={skipped}, flushes={flushes}"
    )

    return {
        "train_shards": list(train_manifest.get("shards", [])),
        "val_shards": list(val_manifest.get("shards", [])),
        "num_train": num_train,
        "num_val": num_val,
        "skipped": skipped,
        "processed_manifest_items": processed,
        "flushes": flushes,
        "skip_items": processed,
    }


def main() -> None:
    args = parse_args()
    manifest_inputs = normalize_manifest_inputs(args.manifest)
    manifest_root = Path(args.manifest_root) if args.manifest_root else None

    tokenizer_path = Path(args.tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer artifact not found: {tokenizer_path}. "
            "Run fynn.data.build_tokenizer first."
        )
    tokenizer = UnifiedTokenizer.load(tokenizer_path)

    resume_state: dict[str, Any] | None = None
    skip_items = 0
    if args.resume:
        resume_state = _load_resume_state(Path(args.output_dir))
        skip_items = resume_state.pop("skip_items")

    total_items = count_manifest_records(manifest_inputs, max_items=args.max_examples)
    manifest = iter_manifest_records(manifest_inputs, max_items=args.max_examples)

    if skip_items > 0:
        import itertools as _itertools
        manifest = _itertools.islice(manifest, skip_items, None)
        total_items = max(0, total_items - skip_items)

    writer = PreparedDatasetShardWriter(
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        seed=args.seed,
        flush_every_samples=args.flush_every_samples,
        manifest_inputs=manifest_inputs,
        manifest_root=manifest_root,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        resume_state=resume_state,
    )

    if args.num_workers > 0:
        results = _iter_parallel_results(
            manifest=manifest,
            manifest_root=manifest_root,
            tokenizer_path=tokenizer_path,
            seed=args.seed,
            max_prompt_tokens=args.max_prompt_tokens,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
            max_seq_len=args.max_seq_len,
            num_workers=args.num_workers,
            include_source_midi_path=args.include_source_midi_path,
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
            include_source_midi_path=args.include_source_midi_path,
        )

    with open_progress_bar(desc="Preparing dataset", total=total_items, unit="file") as progress:
        for result in results:
            writer.add_result(result)
            progress.update(1)
            progress.set_postfix(_prepare_progress_postfix(writer), refresh=False)

        writer.finalize()
        progress.set_postfix(_prepare_progress_postfix(writer), refresh=False)

    print(
        f"Prepared {writer.num_samples} samples "
        f"(train={writer.num_train}, val={writer.num_val}, skipped={writer.skipped})"
    )


if __name__ == "__main__":
    main()
