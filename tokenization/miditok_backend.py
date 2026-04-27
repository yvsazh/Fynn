from __future__ import annotations

import json
import logging
import tempfile
from array import array
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import symusic

from fynn.score_utils import DEFAULT_CC_WHITELIST
from fynn.tokenization.expressive_remi import ExpressiveREMI

logger = logging.getLogger(__name__)


def _import_miditok():
    try:
        import miditok
    except ImportError as exc:
        raise RuntimeError("miditok is not installed") from exc
    return miditok


def _iter_normalized_midi_paths(midi_files: Iterable[str | Path]) -> Iterable[str]:
    for path in midi_files:
        yield str(Path(path).expanduser().resolve(strict=False))


class _DiskBackedPathSequence(Sequence[str]):
    def __init__(self, midi_files: Iterable[str | Path]) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._backing_path = Path(self._temp_dir.name) / "midi_paths.txt"
        self._offsets = array("Q")
        self._size = 0
        with self._backing_path.open("w", encoding="utf-8") as handle:
            for path in _iter_normalized_midi_paths(midi_files):
                self._offsets.append(handle.tell())
                handle.write(path)
                handle.write("\n")
                self._size += 1
        if self._size == 0:
            self.close()
            raise ValueError("No MIDI files were provided to fit the MidiTok tokenizer")

    def close(self) -> None:
        self._temp_dir.cleanup()

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._size))]
        if index < 0:
            index += self._size
        if index < 0 or index >= self._size:
            raise IndexError(index)
        with self._backing_path.open("r", encoding="utf-8") as handle:
            handle.seek(self._offsets[index])
            return handle.readline().rstrip("\n")

    def __iter__(self):
        with self._backing_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip("\n")


@dataclass(slots=True)
class MidiTokenizerConfig:
    backend: str = "miditok"
    tokenizer: str = "REMI"
    tokenizer_model: str = ""
    vocab_size: int = 30_000
    beat_res: tuple[tuple[int, int, int], ...] = (
        (0, 4, 8),
        (4, 12, 4),
    )
    num_velocities: int = 32
    num_tempos: int = 305
    tempo_range: tuple[int, int] = (16, 320)
    use_chords: bool = True
    use_rests: bool = True
    use_tempos: bool = True
    use_time_signatures: bool = True
    use_sustain_pedals: bool = True
    sustain_pedal_duration: bool = True
    use_pitch_bends: bool = True
    use_programs: bool = True
    one_token_stream_for_programs: bool = True
    program_changes: bool = True
    use_pitch_intervals: bool = False
    remove_duplicated_notes: bool = True
    max_bar_embedding: int | None = None
    control_change_numbers: tuple[int, ...] = ()
    control_change_num_bins: int = 32

    def __post_init__(self) -> None:
        self.beat_res = tuple(tuple(int(value) for value in item) for item in self.beat_res)
        self.tempo_range = tuple(int(value) for value in self.tempo_range)
        self.control_change_numbers = tuple(int(value) for value in self.control_change_numbers)
        self.control_change_num_bins = int(self.control_change_num_bins)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["beat_res"] = [list(item) for item in self.beat_res]
        payload["tempo_range"] = list(self.tempo_range)
        payload["control_change_numbers"] = list(self.control_change_numbers)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MidiTokenizerConfig":
        tokenizer_name = data.get("tokenizer", "REMI")
        return cls(
            backend=data.get("backend", "miditok"),
            tokenizer="REMI" if tokenizer_name == "ExpressiveREMI" else tokenizer_name,
            tokenizer_model=data.get("tokenizer_model", ""),
            vocab_size=int(data.get("vocab_size", 30_000)),
            beat_res=tuple(tuple(item) for item in data.get("beat_res", ((0, 4, 8), (4, 12, 4)))),
            num_velocities=int(data.get("num_velocities", 32)),
            num_tempos=int(data.get("num_tempos", 305)),
            tempo_range=tuple(data.get("tempo_range", (16, 320))),
            use_chords=bool(data.get("use_chords", True)),
            use_rests=bool(data.get("use_rests", True)),
            use_tempos=bool(data.get("use_tempos", True)),
            use_time_signatures=bool(data.get("use_time_signatures", True)),
            use_sustain_pedals=bool(data.get("use_sustain_pedals", True)),
            sustain_pedal_duration=bool(data.get("sustain_pedal_duration", True)),
            use_pitch_bends=bool(data.get("use_pitch_bends", True)),
            use_programs=bool(data.get("use_programs", True)),
            one_token_stream_for_programs=bool(data.get("one_token_stream_for_programs", True)),
            program_changes=bool(data.get("program_changes", True)),
            use_pitch_intervals=bool(data.get("use_pitch_intervals", False)),
            remove_duplicated_notes=bool(data.get("remove_duplicated_notes", True)),
            max_bar_embedding=data.get("max_bar_embedding"),
            control_change_numbers=tuple(data.get("control_change_numbers", ())),
            control_change_num_bins=int(data.get("control_change_num_bins", 32)),
        )


class MidiTokBackend:
    def __init__(self, config: MidiTokenizerConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self._piece_cache: dict[int, str] | None = None
        self._introspection_cache: dict[int, dict[str, Any]] | None = None

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("MidiTok tokenizer is not initialized")
        return len(self.tokenizer)

    @property
    def raw_event_graph(self) -> dict[str, set[str]]:
        tokenizer = self._require_tokenizer()
        graph = getattr(tokenizer, "tokens_types_graph", {})
        return {str(key): {str(value) for value in values if value != "PAD"} for key, values in graph.items()}

    def fit(self, midi_files: Iterable[str | Path]) -> None:
        tokenizer = self._build_tokenizer()
        model_kind = (self.config.tokenizer_model or "").strip().lower()
        if not model_kind or model_kind == "none":
            saw_any = False
            for _ in _iter_normalized_midi_paths(midi_files):
                saw_any = True
            if not saw_any:
                raise ValueError("No MIDI files were provided to fit the MidiTok tokenizer")
        else:
            files = _DiskBackedPathSequence(midi_files)
            try:
                tokenizer.train(
                    vocab_size=self.config.vocab_size,
                    model=self.config.tokenizer_model,
                    files_paths=files,
                )
            finally:
                files.close()
        self.tokenizer = tokenizer
        self._piece_cache = None
        self._introspection_cache = None

    def encode_path(self, path: str | Path, max_tokens: int | None = None) -> list[int]:
        tokenizer = self._require_tokenizer()
        encoded = tokenizer.encode(Path(path))
        ids = self._extract_ids(encoded)
        if max_tokens is not None and len(ids) > max_tokens:
            logger.warning("Truncating MIDI sequence from %d to %d tokens (%s)", len(ids), max_tokens, path)
            ids = ids[:max_tokens]
        return ids

    def encode_score(self, score: symusic.Score, max_tokens: int | None = None) -> list[int]:
        tokenizer = self._require_tokenizer()
        encoded = tokenizer.encode(score)
        ids = self._extract_ids(encoded)
        if max_tokens is not None and len(ids) > max_tokens:
            logger.warning("Truncating MIDI sequence from %d to %d tokens", len(ids), max_tokens)
            ids = ids[:max_tokens]
        return ids

    def decode_score(self, token_ids: list[int], ticks_per_beat: int = 480) -> symusic.Score:
        tokenizer = self._require_tokenizer()
        miditok = _import_miditok()
        ids = list(token_ids)
        are_ids_encoded = False
        if getattr(tokenizer, "is_trained", False):
            checker = getattr(tokenizer, "_are_ids_encoded", None)
            if callable(checker):
                are_ids_encoded = bool(checker(ids))
        tok_sequence = miditok.TokSequence(ids=ids, are_ids_encoded=are_ids_encoded)
        decoded = tokenizer.decode(tok_sequence)
        if int(getattr(decoded, "tpq", getattr(decoded, "ticks_per_quarter", ticks_per_beat))) != int(ticks_per_beat):
            decoded = decoded.resample(int(ticks_per_beat))
        return decoded

    def id_to_piece(self, token_id: int) -> str:
        return self._get_piece_cache().get(token_id, "")

    def id_to_raw_tokens(self, token_id: int) -> list[str]:
        piece = self.id_to_piece(token_id)
        if not piece:
            return []
        return piece.split(" ")

    def ids_to_raw_tokens(self, token_ids: list[int]) -> list[str]:
        result: list[str] = []
        for token_id in token_ids:
            result.extend(self.id_to_raw_tokens(token_id))
        return result

    @staticmethod
    def raw_token_event_type(token: str) -> str:
        if not token:
            return "Other"
        token_type = token.split("_", 1)[0]
        if token_type in {
            "Bar",
            "Position",
            "Pitch",
            "PitchDrum",
            "PitchIntervalTime",
            "PitchIntervalChord",
            "Velocity",
            "Duration",
            "Program",
            "Tempo",
            "TimeSig",
            "Chord",
            "Rest",
            "Pedal",
            "PedalOff",
            "PitchBend",
            "ControlChange",
            "CCValue",
        }:
            return token_type
        if token_type == "TimeSignature":
            return "TimeSig"
        return "Other"

    @staticmethod
    def raw_token_type(token: str) -> str:
        token_type = MidiTokBackend.raw_token_event_type(token)
        return {
            "Pitch": "pitch",
            "PitchDrum": "pitch",
            "PitchIntervalTime": "pitch",
            "PitchIntervalChord": "pitch",
            "Velocity": "velocity",
            "Duration": "duration",
            "Bar": "bar",
            "Position": "position",
            "Program": "program",
            "Tempo": "tempo",
            "TimeSig": "time_signature",
            "Chord": "chord",
            "Rest": "rest",
            "Pedal": "pedal",
            "PedalOff": "pedal",
            "PitchBend": "pitch_bend",
            "ControlChange": "control_change",
            "CCValue": "control_change_value",
        }.get(token_type, "other")

    def encoded_id_signature(self, token_id: int) -> dict[str, Any]:
        return self._get_introspection_cache().get(token_id, {})

    def build_vocab_introspection_cache(self) -> None:
        tokenizer = self._require_tokenizer()
        piece_cache: dict[int, str] = {}
        introspection_cache: dict[int, dict[str, Any]] = {}

        for piece, token_id in tokenizer.vocab.items():
            piece_cache[token_id] = piece
            raw_tokens = piece.split(" ")
            event_types = [self.raw_token_event_type(token) for token in raw_tokens]
            categories = [self.raw_token_type(token) for token in raw_tokens]
            category_set = set(categories)
            introspection_cache[token_id] = {
                "raw_tokens": raw_tokens,
                "event_types": event_types,
                "categories": categories,
                "first_raw_type": categories[0] if categories else "other",
                "last_raw_type": categories[-1] if categories else "other",
                "contains_program": "program" in category_set,
                "contains_pitch": "pitch" in category_set,
                "contains_velocity": "velocity" in category_set,
                "contains_duration": "duration" in category_set,
                "contains_bar": "bar" in category_set,
                "contains_position": "position" in category_set,
                "contains_time_signature": "time_signature" in category_set,
                "contains_tempo": "tempo" in category_set,
                "contains_chord": "chord" in category_set,
                "contains_rest": "rest" in category_set,
                "contains_pedal": "pedal" in category_set,
                "contains_pitch_bend": "pitch_bend" in category_set,
                "contains_control_change": "control_change" in category_set,
            }

        self._piece_cache = piece_cache
        self._introspection_cache = introspection_cache

    def vocab_introspection_stats(self) -> dict[str, Any]:
        cache = self._get_introspection_cache()
        if not cache:
            return {}

        total_raw = 0
        counts: dict[str, int] = {
            "ids_with_pitch": 0,
            "ids_with_bar": 0,
            "ids_with_tempo": 0,
            "ids_with_time_signature": 0,
            "ids_with_chord": 0,
            "ids_with_program": 0,
            "ids_with_rest": 0,
            "ids_with_pedal": 0,
            "ids_with_pitch_bend": 0,
            "ids_with_control_change": 0,
            "ids_single_raw_token": 0,
            "ids_multi_raw_token": 0,
            "max_raw_tokens_per_id": 0,
        }
        for signature in cache.values():
            raw_count = len(signature["raw_tokens"])
            total_raw += raw_count
            if raw_count == 1:
                counts["ids_single_raw_token"] += 1
            else:
                counts["ids_multi_raw_token"] += 1
            counts["max_raw_tokens_per_id"] = max(counts["max_raw_tokens_per_id"], raw_count)
            for key in (
                "pitch",
                "bar",
                "tempo",
                "time_signature",
                "chord",
                "program",
                "rest",
                "pedal",
                "pitch_bend",
                "control_change",
            ):
                if signature[f"contains_{key}"]:
                    counts[f"ids_with_{key}"] += 1

        return {
            "total_ids": len(cache),
            "mean_raw_tokens_per_id": round(total_raw / max(1, len(cache)), 3),
            **counts,
        }

    def _get_piece_cache(self) -> dict[int, str]:
        if self._piece_cache is None:
            self.build_vocab_introspection_cache()
        return self._piece_cache  # type: ignore[return-value]

    def _get_introspection_cache(self) -> dict[int, dict[str, Any]]:
        if self._introspection_cache is None:
            self.build_vocab_introspection_cache()
        return self._introspection_cache  # type: ignore[return-value]

    def to_serializable_dict(self) -> dict[str, Any]:
        tokenizer = self._require_tokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            tokenizer.save(temp_dir)
            params_payload = json.loads((temp_dir / "tokenizer.json").read_text(encoding="utf-8"))

        tokenizer_name = "ExpressiveREMI" if self.config.control_change_numbers else self.config.tokenizer
        return {
            "backend": "miditok",
            "config": self.config.to_dict(),
            "params": params_payload,
            "tokenizer": tokenizer_name,
            "tokenizer_model": self.config.tokenizer_model,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict[str, Any]) -> "MidiTokBackend":
        config = MidiTokenizerConfig.from_dict(payload.get("config", {}))
        if payload.get("tokenizer") == "ExpressiveREMI" and not config.control_change_numbers:
            config.control_change_numbers = tuple(DEFAULT_CC_WHITELIST)
        tokenizer = cls._load_tokenizer_from_params(config, payload["params"], payload.get("tokenizer"))
        return cls(config=config, tokenizer=tokenizer)

    def _build_tokenizer(self):
        miditok = _import_miditok()
        tokenizer_cls = self._resolve_tokenizer_class(miditok, self.config)
        tokenizer_config = miditok.TokenizerConfig(
            beat_res={(start, end): resolution for start, end, resolution in self.config.beat_res},
            num_velocities=self.config.num_velocities,
            num_tempos=self.config.num_tempos,
            tempo_range=self.config.tempo_range,
            use_chords=self.config.use_chords,
            use_rests=self.config.use_rests,
            use_tempos=self.config.use_tempos,
            use_time_signatures=self.config.use_time_signatures,
            use_sustain_pedals=self.config.use_sustain_pedals,
            sustain_pedal_duration=self.config.sustain_pedal_duration,
            use_pitch_bends=self.config.use_pitch_bends,
            use_programs=self.config.use_programs,
            one_token_stream_for_programs=self.config.one_token_stream_for_programs,
            program_changes=self.config.program_changes,
            use_pitch_intervals=self.config.use_pitch_intervals,
            remove_duplicated_notes=self.config.remove_duplicated_notes,
            special_tokens=[],
            control_change_numbers=list(self.config.control_change_numbers),
            control_change_num_bins=self.config.control_change_num_bins,
        )
        return tokenizer_cls(tokenizer_config, max_bar_embedding=self.config.max_bar_embedding)

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("MidiTok tokenizer is not fitted. Run build_tokenizer first.")
        return self.tokenizer

    @classmethod
    def _load_tokenizer_from_params(
        cls,
        config: MidiTokenizerConfig,
        params_payload: dict[str, Any],
        tokenizer_name: str | None = None,
    ):
        miditok = _import_miditok()
        tokenizer_cls = cls._resolve_tokenizer_class(miditok, config, tokenizer_name=tokenizer_name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            temp_path.write_text(json.dumps(params_payload), encoding="utf-8")
            return tokenizer_cls(params=temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _resolve_tokenizer_class(miditok, config: MidiTokenizerConfig, tokenizer_name: str | None = None):
        effective_name = tokenizer_name or ("ExpressiveREMI" if config.control_change_numbers else config.tokenizer)
        if effective_name == "ExpressiveREMI":
            return ExpressiveREMI
        return getattr(miditok, effective_name)

    @staticmethod
    def _extract_ids(encoded) -> list[int]:
        if encoded is None:
            return []
        if isinstance(encoded, list):
            if not encoded:
                return []
            if isinstance(encoded[0], int):
                return [int(token_id) for token_id in encoded]
            ids: list[int] = []
            for item in encoded:
                ids.extend(MidiTokBackend._extract_ids(item))
            return ids
        ids = getattr(encoded, "ids", None)
        if ids is not None:
            return [int(token_id) for token_id in ids]
        raise TypeError(f"Unsupported MidiTok encode output: {type(encoded)!r}")
