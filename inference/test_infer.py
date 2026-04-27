from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fynn.config import ModelConfig
from fynn.inference.export import write_score_bundle_paths
from fynn.inference.generation import add_track_to_midi, generate_from_scratch
from fynn.model import FynnTransformer
from fynn.tokenization.unified import UnifiedTokenizer
from fynn.utils import device_for_training, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test inference for MIDI generation.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer JSON path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for exported .mid files.")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for scratch generation or extra conditioning.")
    parser.add_argument("--existing-midi", type=str, default="", help="Optional MIDI path for add-track mode.")
    parser.add_argument("--instrument", type=str, default="drums", help="Instrument name for add-track mode.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def load_model(checkpoint_path: str | Path, tokenizer: UnifiedTokenizer, device: torch.device) -> FynnTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_payload = checkpoint.get("config", {}).get("model", {})
    if model_payload:
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        unknown = set(model_payload) - valid_fields
        if unknown:
            import warnings
            warnings.warn(f"Ignoring unknown ModelConfig keys from checkpoint: {sorted(unknown)}")
        model_payload = {k: v for k, v in model_payload.items() if k in valid_fields}
    model_config = ModelConfig(**model_payload) if model_payload else ModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    model_config.pad_id = tokenizer.pad_id
    model = FynnTransformer(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = device_for_training()
    tokenizer = UnifiedTokenizer.load(args.tokenizer)
    model = load_model(args.checkpoint, tokenizer, device)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "device": device,
    }

    if args.existing_midi:
        score = add_track_to_midi(
            model=model,
            tokenizer=tokenizer,
            existing_midi_path=args.existing_midi,
            instrument=args.instrument,
            extra_prompt=args.prompt,
            **gen_kwargs,
        )
    else:
        if not args.prompt:
            raise ValueError("--prompt is required in scratch generation mode")
        score = generate_from_scratch(model=model, tokenizer=tokenizer, prompt=args.prompt, **gen_kwargs)

    output_dir = ensure_dir(args.output_dir)
    bundle = write_score_bundle_paths(score, output_dir).to_dict()
    summary_path = Path(output_dir) / "bundle.json"
    summary_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": bundle}, indent=2))


if __name__ == "__main__":
    main()
