from __future__ import annotations

import argparse
from pathlib import Path

from fynn.config import ExperimentConfig
from fynn.data.manifest_utils import iter_manifest_records, normalize_manifest_inputs, resolve_manifest_record
from fynn.tokenization.unified import UnifiedTokenizer, UnifiedTokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and save a unified tokenizer on a MIDI corpus.")
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
    parser.add_argument("--output", type=str, required=True, help="Path to tokenizer JSON.")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config to source tokenizer defaults from.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for tokenizer fitting.")
    parser.add_argument("--text-kind", type=str, default="", choices=["", "byte", "tiktoken"])
    parser.add_argument("--text-encoding", type=str, default="")
    parser.add_argument("--midi-tokenizer", type=str, default="")
    parser.add_argument("--midi-model", type=str, default="")
    parser.add_argument("--midi-vocab-size", type=int, default=0)
    parser.add_argument("--num-velocities", type=int, default=0)
    parser.add_argument("--num-tempos", type=int, default=0)
    parser.add_argument("--tempo-min", type=int, default=0)
    parser.add_argument("--tempo-max", type=int, default=0)
    parser.add_argument("--use-chords", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-rests", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-tempos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-time-signatures", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-sustain-pedals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sustain-pedal-duration", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-pitch-bends", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-programs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--one-token-stream-for-programs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--program-changes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--control-change-numbers", type=str, default="")
    parser.add_argument("--control-change-num-bins", type=int, default=0)
    return parser.parse_args()


def resolve_tokenizer_config(args: argparse.Namespace) -> UnifiedTokenizerConfig:
    if args.config:
        config = ExperimentConfig.from_yaml(args.config).tokenizer
    else:
        config = UnifiedTokenizerConfig()

    if args.text_kind:
        config.text.kind = args.text_kind
    if args.text_encoding:
        config.text.tiktoken_encoding = args.text_encoding
    if args.midi_tokenizer:
        config.midi.tokenizer = args.midi_tokenizer
    if args.midi_model:
        config.midi.tokenizer_model = args.midi_model
    if args.midi_vocab_size > 0:
        config.midi.vocab_size = args.midi_vocab_size
    if args.num_velocities > 0:
        config.midi.num_velocities = args.num_velocities
    if args.num_tempos > 0:
        config.midi.num_tempos = args.num_tempos
    if args.tempo_min > 0:
        config.midi.tempo_range = (args.tempo_min, config.midi.tempo_range[1])
    if args.tempo_max > 0:
        config.midi.tempo_range = (config.midi.tempo_range[0], args.tempo_max)
    if args.use_chords is not None:
        config.midi.use_chords = args.use_chords
    if args.use_rests is not None:
        config.midi.use_rests = args.use_rests
    if args.use_tempos is not None:
        config.midi.use_tempos = args.use_tempos
    if args.use_time_signatures is not None:
        config.midi.use_time_signatures = args.use_time_signatures
    if args.use_sustain_pedals is not None:
        config.midi.use_sustain_pedals = args.use_sustain_pedals
    if args.sustain_pedal_duration is not None:
        config.midi.sustain_pedal_duration = args.sustain_pedal_duration
    if args.use_pitch_bends is not None:
        config.midi.use_pitch_bends = args.use_pitch_bends
    if args.use_programs is not None:
        config.midi.use_programs = args.use_programs
    if args.one_token_stream_for_programs is not None:
        config.midi.one_token_stream_for_programs = args.one_token_stream_for_programs
    if args.program_changes is not None:
        config.midi.program_changes = args.program_changes
    if args.control_change_numbers.strip():
        config.midi.control_change_numbers = tuple(
            int(value.strip())
            for value in args.control_change_numbers.split(",")
            if value.strip()
        )
    if args.control_change_num_bins > 0:
        config.midi.control_change_num_bins = args.control_change_num_bins
    return config


def main() -> None:
    args = parse_args()
    manifest_inputs = normalize_manifest_inputs(args.manifest)
    manifest_root = Path(args.manifest_root) if args.manifest_root else None

    class _MidiPathIterable:
        def __init__(self) -> None:
            self.midi_files = 0
            self.skipped = 0

        def __iter__(self):
            for record in iter_manifest_records(manifest_inputs, max_items=args.max_files):
                try:
                    resolved = resolve_manifest_record(record, manifest_root=manifest_root)
                except Exception as exc:
                    self.skipped += 1
                    print(f"Skip manifest item: {exc}")
                    continue
                self.midi_files += 1
                yield resolved.midi_path

    midi_files = _MidiPathIterable()

    tokenizer_config = resolve_tokenizer_config(args)
    tokenizer = UnifiedTokenizer.fit_on_midi_files(midi_files=midi_files, config=tokenizer_config)
    tokenizer.save(args.output)
    print(f"Saved tokenizer to {args.output}")
    print(
        " ".join(
            [
                f"midi_files={midi_files.midi_files}",
                f"skipped={midi_files.skipped}",
                f"vocab_size={tokenizer.vocab_size}",
                f"text_vocab={tokenizer.text_tokenizer.vocab_size}",
                f"midi_vocab={tokenizer.midi_tokenizer.vocab_size}",
                f"hash={tokenizer.payload_hash()}",
            ]
        )
    )


if __name__ == "__main__":
    main()
