from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from fynn.utils import resolve_path


_JSON_STREAM_CHUNK_SIZE = 1024 * 1024


@dataclass(slots=True, frozen=True)
class ManifestRecord:
    index: int
    item: dict[str, Any]
    manifest_path: Path


@dataclass(slots=True, frozen=True)
class ResolvedManifestRecord:
    index: int
    item: dict[str, Any]
    manifest_path: Path
    midi_path: str
    caption: str


def normalize_manifest_inputs(manifest_inputs: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(manifest_inputs, (str, Path)):
        return [Path(manifest_inputs)]
    return [Path(path) for path in manifest_inputs]


def expand_manifest_inputs(manifest_inputs: str | Path | Iterable[str | Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_input in normalize_manifest_inputs(manifest_inputs):
        path = raw_input.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Manifest input not found: {path}")
        if path.is_dir():
            files = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in {".jsonl", ".json"}
            )
            if not files:
                raise FileNotFoundError(f"No .jsonl or .json files found in manifest directory: {path}")
            expanded.extend(files)
        else:
            expanded.append(path)
    return expanded


def iter_manifest_file(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return

    for item in _iter_streaming_json_values(path):
        if isinstance(item, dict):
            yield item
        else:
            raise ValueError(f"Unsupported manifest item in {path}: {type(item)!r}")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_manifest_file(path))


def count_manifest_records(
    manifest_inputs: str | Path | Iterable[str | Path],
    *,
    max_items: int = 0,
) -> int:
    limit = max(0, int(max_items))
    total = 0
    for manifest_path in expand_manifest_inputs(manifest_inputs):
        file_total = _count_manifest_file(manifest_path)
        if limit > 0:
            remaining = limit - total
            total += min(file_total, max(0, remaining))
            if total >= limit:
                return limit
        else:
            total += file_total
    return total


def iter_manifest_records(
    manifest_inputs: str | Path | Iterable[str | Path],
    *,
    max_items: int = 0,
) -> Iterator[ManifestRecord]:
    limit = max(0, int(max_items))
    index = 0
    for manifest_path in expand_manifest_inputs(manifest_inputs):
        for item in iter_manifest_file(manifest_path):
            yield ManifestRecord(index=index, item=item, manifest_path=manifest_path)
            index += 1
            if limit > 0 and index >= limit:
                return


def resolve_manifest_record(
    record: ManifestRecord,
    *,
    manifest_root: Path | None = None,
) -> ResolvedManifestRecord:
    midi_path, caption = resolve_manifest_entry(
        record.item,
        manifest_root=manifest_root,
        source_manifest_path=record.manifest_path,
    )
    return ResolvedManifestRecord(
        index=record.index,
        item=record.item,
        manifest_path=record.manifest_path,
        midi_path=midi_path,
        caption=caption,
    )


def iter_resolved_manifest_records(
    manifest_inputs: str | Path | Iterable[str | Path],
    *,
    manifest_root: Path | None = None,
    max_items: int = 0,
) -> Iterator[ResolvedManifestRecord]:
    for record in iter_manifest_records(manifest_inputs, max_items=max_items):
        yield resolve_manifest_record(record, manifest_root=manifest_root)


def resolve_manifest_entry(
    item: dict[str, Any],
    manifest_root: Path | None = None,
    source_manifest_path: Path | None = None,
) -> tuple[str, str]:
    caption = str(item.get("caption", "")).strip()
    if not caption:
        raise ValueError("Manifest item has no caption")

    base_dir = manifest_root
    if base_dir is None and source_manifest_path is not None:
        base_dir = Path(source_manifest_path).parent

    def _resolve_candidate(raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve(strict=False)
        if base_dir is None:
            raise ValueError(
                "Relative manifest paths require source_manifest_path or --manifest-root; "
                "implicit current-working-directory resolution is not allowed."
            )
        return resolve_path(candidate, base_dir)

    if "midi_path" in item:
        raw_path = str(item["midi_path"]).strip()
    elif "location" in item:
        raw_path = str(item["location"]).strip()
    else:
        raise ValueError("Manifest item must contain either 'midi_path' or 'location'")
    if not raw_path:
        raise ValueError("Manifest item has an empty 'midi_path'/'location'")

    midi_path = _resolve_candidate(raw_path)

    return str(midi_path), caption


def _count_manifest_file(path: str | Path) -> int:
    return sum(1 for _ in iter_manifest_file(path))


def _iter_streaming_json_values(path: Path) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    mode: str | None = None
    buffer = ""
    reached_eof = False
    array_finished = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            if not reached_eof and len(buffer) < _JSON_STREAM_CHUNK_SIZE:
                chunk = handle.read(_JSON_STREAM_CHUNK_SIZE)
                if chunk:
                    buffer += chunk
                else:
                    reached_eof = True

            buffer = buffer.lstrip()
            if array_finished:
                if buffer:
                    raise ValueError(f"Unexpected trailing content after JSON array in {path}")
                if reached_eof:
                    return
                continue

            if not buffer:
                if reached_eof:
                    return
                continue

            if mode is None:
                if buffer[0] == "[":
                    mode = "array"
                    buffer = buffer[1:]
                    continue
                mode = "sequence"

            if mode == "array":
                buffer = buffer.lstrip()
                if not buffer:
                    if reached_eof:
                        raise ValueError(f"Unterminated JSON array in {path}")
                    continue
                if buffer[0] == "]":
                    buffer = buffer[1:]
                    array_finished = True
                    continue

            try:
                item, consumed = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if not reached_eof:
                    continue
                raise ValueError(f"Could not parse manifest JSON stream in {path}: {exc}") from exc

            yield item
            buffer = buffer[consumed:]

            if mode == "array":
                buffer = buffer.lstrip()
                if not buffer:
                    if reached_eof:
                        raise ValueError(f"Unterminated JSON array in {path}")
                    continue
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    buffer = buffer[1:]
                    array_finished = True
                    continue
                raise ValueError(f"Malformed JSON array manifest in {path}")
