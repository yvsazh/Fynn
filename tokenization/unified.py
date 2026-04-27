from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import symusic

from fynn.instruments import instrument_label
from fynn.score_utils import split_add_track_context_target, tokenizable_track_indices
from fynn.tokenization.miditok_backend import MidiTokBackend, MidiTokenizerConfig
from fynn.tokenization.text import TextTokenizerConfig, build_text_tokenizer


@dataclass(slots=True)
class UnifiedTokenizerConfig:
    text: TextTokenizerConfig = field(default_factory=TextTokenizerConfig)
    midi: MidiTokenizerConfig = field(default_factory=MidiTokenizerConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text.to_dict(),
            "midi": self.midi.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedTokenizerConfig":
        return cls(
            text=TextTokenizerConfig(**data.get("text", {})),
            midi=MidiTokenizerConfig.from_dict(data.get("midi", {})),
        )


class UnifiedTokenizer:
    ARTIFACT_VERSION = 3
    SPECIAL_TOKENS = [
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<PROMPT>",
        "</PROMPT>",
        "<CONTEXT>",
        "</CONTEXT>",
        "<GEN_ALL>",
        "<ADD_TRACK>",
        "</ADD_TRACK>",
    ]

    def __init__(self, config: UnifiedTokenizerConfig, midi_backend: MidiTokBackend):
        self.config = config
        self.text_tokenizer = build_text_tokenizer(config.text)
        self.midi_tokenizer = midi_backend

        self.special_to_id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_special = {idx: token for token, idx in self.special_to_id.items()}
        self.pad_id = self.special_to_id["<PAD>"]
        self.bos_id = self.special_to_id["<BOS>"]
        self.eos_id = self.special_to_id["<EOS>"]
        self.text_offset = len(self.SPECIAL_TOKENS)
        self.midi_offset = self.text_offset + self.text_tokenizer.vocab_size
        self.vocab_size = self.midi_offset + self.midi_tokenizer.vocab_size
        self._generation_mask_ids_cache: list[int] | None = None

    @classmethod
    def fit_on_midi_files(
        cls,
        midi_files: Iterable[str | Path],
        config: UnifiedTokenizerConfig | None = None,
    ) -> "UnifiedTokenizer":
        effective_config = config or UnifiedTokenizerConfig()
        midi_backend = MidiTokBackend(effective_config.midi)
        midi_backend.fit(midi_files)
        return cls(effective_config, midi_backend)

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.ARTIFACT_VERSION,
            "special_tokens": list(self.SPECIAL_TOKENS),
            "config": self.config.to_dict(),
            "text": self.text_tokenizer.to_metadata(),
            "midi": self.midi_tokenizer.to_serializable_dict(),
            "offsets": {
                "special": 0,
                "text": self.text_offset,
                "midi": self.midi_offset,
            },
            "vocab_size": self.vocab_size,
        }

    def payload_hash(self) -> str:
        canonical = json.dumps(self.to_payload(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "UnifiedTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != cls.ARTIFACT_VERSION:
            raise ValueError(
                f"Incompatible tokenizer artifact version: {version!r}. "
                f"Expected {cls.ARTIFACT_VERSION}. Rebuild the tokenizer and datasets."
            )

        config = UnifiedTokenizerConfig.from_dict(payload.get("config", {}))
        midi_backend = MidiTokBackend.from_serializable_dict(payload["midi"])
        tokenizer = cls(config=config, midi_backend=midi_backend)

        offsets = payload.get("offsets", {})
        if offsets.get("text") != tokenizer.text_offset or offsets.get("midi") != tokenizer.midi_offset:
            raise ValueError("Tokenizer offsets in artifact do not match the loaded tokenizer state")
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("Tokenizer vocab_size in artifact does not match the loaded tokenizer state")
        return tokenizer

    def encode_text(self, text: str, max_tokens: int | None = None) -> list[int]:
        tokens = self.text_tokenizer.encode(text)
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        return [token + self.text_offset for token in tokens]

    def encode_midi(self, score_or_path: symusic.Score | str | Path, max_tokens: int | None = None) -> list[int]:
        if isinstance(score_or_path, (str, Path)):
            tokens = self.midi_tokenizer.encode_path(score_or_path, max_tokens=max_tokens)
        else:
            tokens = self.midi_tokenizer.encode_score(score_or_path, max_tokens=max_tokens)
        return [token + self.midi_offset for token in tokens]

    def decode_midi(self, token_ids: list[int], ticks_per_beat: int = 480) -> symusic.Score:
        midi_ids = [token - self.midi_offset for token in token_ids if self.midi_offset <= token < self.vocab_size]
        return self.midi_tokenizer.decode_score(midi_ids, ticks_per_beat=ticks_per_beat)

    def scratch_sequence(
        self,
        prompt: str,
        score_or_path: symusic.Score | str | Path,
        max_prompt_tokens: int | None = 192,
        max_target_tokens: int | None = None,
        append_eos: bool = True,
    ) -> dict[str, Any]:
        prompt_tokens = self.encode_text(prompt, max_prompt_tokens)
        midi_tokens = self.encode_midi(score_or_path, max_target_tokens)
        return self.scratch_sequence_from_midi_tokens(
            prompt_tokens=prompt_tokens,
            midi_tokens=midi_tokens,
            append_eos=append_eos,
        )

    def scratch_sequence_from_midi_tokens(
        self,
        *,
        prompt_tokens: list[int],
        midi_tokens: list[int],
        append_eos: bool = True,
    ) -> dict[str, Any]:
        prefix = [
            self.special_to_id["<PROMPT>"],
            *prompt_tokens,
            self.special_to_id["</PROMPT>"],
            self.special_to_id["<GEN_ALL>"],
            self.bos_id,
        ]
        full = prefix + midi_tokens
        if append_eos:
            full.append(self.eos_id)
        labels = full.copy()
        for index in range(len(prefix)):
            labels[index] = -100
        return {"mode": "scratch", "input_ids": full, "labels": labels}

    def add_track_sequence(
        self,
        prompt: str,
        context_score: symusic.Score,
        target_score: symusic.Score,
        max_prompt_tokens: int | None = 192,
        max_context_tokens: int | None = 1536,
        max_target_tokens: int | None = 6400,
        append_eos: bool = True,
    ) -> dict[str, Any]:
        if not context_score.tracks:
            raise ValueError("add_track_sequence requires at least one context track")
        prompt_tokens = self.encode_text(prompt, max_prompt_tokens)
        context_tokens = self.encode_midi(context_score, max_context_tokens)
        target_tokens = self.encode_midi(target_score, max_target_tokens)
        return self.add_track_sequence_from_midi_tokens(
            prompt_tokens=prompt_tokens,
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            append_eos=append_eos,
        )

    def add_track_sequence_from_midi_tokens(
        self,
        *,
        prompt_tokens: list[int],
        context_tokens: list[int],
        target_tokens: list[int],
        append_eos: bool = True,
    ) -> dict[str, Any]:
        prefix = [
            self.special_to_id["<PROMPT>"],
            *prompt_tokens,
            self.special_to_id["</PROMPT>"],
            self.special_to_id["<CONTEXT>"],
            *context_tokens,
            self.special_to_id["</CONTEXT>"],
            self.special_to_id["<ADD_TRACK>"],
            self.special_to_id["</ADD_TRACK>"],
            self.bos_id,
        ]
        full = prefix + target_tokens
        if append_eos:
            full.append(self.eos_id)
        labels = full.copy()
        for index in range(len(prefix)):
            labels[index] = -100
        return {"mode": "add_track", "input_ids": full, "labels": labels}

    def synthesize_add_track_sample(
        self,
        caption: str,
        score: symusic.Score,
        rng: random.Random,
        max_prompt_tokens: int | None = 192,
        max_context_tokens: int | None = 1536,
        max_target_tokens: int | None = 6400,
    ) -> dict[str, Any] | None:
        if len(score.tracks) < 2:
            return None

        targetable_track_indices = tokenizable_track_indices(
            score,
            use_notes=True,
            use_sustain_pedals=bool(self.config.midi.use_sustain_pedals),
            use_pitch_bends=bool(self.config.midi.use_pitch_bends),
            control_change_numbers=self.config.midi.control_change_numbers,
        )
        if not targetable_track_indices:
            return None

        target_idx = int(rng.choice(targetable_track_indices))
        context, target = split_add_track_context_target(score, target_idx)

        target_track = target.tracks[0]
        prompt = self.build_add_track_prompt(
            instrument=instrument_label(int(target_track.program), bool(target_track.is_drum), name=str(target_track.name)),
            extra_prompt=caption,
        )
        return self.add_track_sequence(
            prompt=prompt,
            context_score=context,
            target_score=target,
            max_prompt_tokens=max_prompt_tokens,
            max_context_tokens=max_context_tokens,
            max_target_tokens=max_target_tokens,
        )

    def build_add_track_prompt(self, instrument: str, extra_prompt: str = "") -> str:
        prompt = f"add track: {instrument.strip()}"
        if extra_prompt.strip():
            prompt = f"{prompt} | {extra_prompt.strip()}"
        return prompt

    def generation_mask_ids(self) -> list[int]:
        if self._generation_mask_ids_cache is None:
            self._generation_mask_ids_cache = list(range(self.midi_offset, self.vocab_size)) + [self.eos_id]
        return self._generation_mask_ids_cache

    # ------------------------------------------------------------------
    # Tokenizer introspection proxy API
    # ------------------------------------------------------------------

    def build_midi_introspection_cache(self) -> None:
        """
        Pre-build the MIDI vocabulary introspection cache.

        Call once after loading the tokenizer (e.g. at training startup).
        Subsequent calls to midi_raw_tokens_for_id / midi_encoded_signature
        return pre-computed data at dict-lookup speed.
        """
        self.midi_tokenizer.build_vocab_introspection_cache()
        self._generation_mask_ids_cache = None

    def vocab_introspection_stats(self) -> dict:
        """Return aggregate statistics about the MIDI vocabulary composition."""
        return self.midi_tokenizer.vocab_introspection_stats()

    def midi_raw_tokens_for_id(self, unified_id: int) -> list[str]:
        """
        Return raw REMI event token strings for a unified-space MIDI token ID.

        Returns an empty list for special or text-range IDs.
        """
        if not (self.midi_offset <= unified_id < self.vocab_size):
            return []
        return self.midi_tokenizer.id_to_raw_tokens(unified_id - self.midi_offset)

    def midi_raw_tokens_for_ids(self, unified_ids: list[int]) -> list[str]:
        """
        Flatten multiple unified-space MIDI token IDs to raw REMI token strings.

        Non-MIDI IDs (special / text range) are silently skipped.
        """
        midi_ids = [
            uid - self.midi_offset
            for uid in unified_ids
            if self.midi_offset <= uid < self.vocab_size
        ]
        return self.midi_tokenizer.ids_to_raw_tokens(midi_ids)

    def midi_encoded_signature(self, unified_id: int) -> dict:
        """
        Return the encoded-id signature dict for a unified-space MIDI token ID.

        Returns an empty dict for special or text-range IDs.
        """
        if not (self.midi_offset <= unified_id < self.vocab_size):
            return {}
        return self.midi_tokenizer.encoded_id_signature(unified_id - self.midi_offset)

    def midi_token_type(self, unified_id: int) -> str:
        """
        Return the coarse musical event type for a unified-space MIDI token ID.

        Returns ``"special"`` for special tokens, ``"text"`` for text tokens,
        and the raw REMI event type (e.g. ``"pitch"``, ``"bar"``) for MIDI tokens.
        """
        if unified_id < self.text_offset:
            return "special"
        if unified_id < self.midi_offset:
            return "text"
        raw_tokens = self.midi_tokenizer.id_to_raw_tokens(unified_id - self.midi_offset)
        if not raw_tokens:
            return "other"
        return self.midi_tokenizer.raw_token_type(raw_tokens[0])
