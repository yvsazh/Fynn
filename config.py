from __future__ import annotations

import dataclasses
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fynn.tokenization.unified import UnifiedTokenizerConfig
from fynn.utils import load_yaml, resolve_path


def _warn_unknown_keys(cls: type, section: str, payload: dict) -> None:
    """Emit a warning for any YAML keys that don't match dataclass fields."""
    if not dataclasses.is_dataclass(cls):
        return
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        warnings.warn(
            f"Unknown keys in '{section}' config section: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}",
            stacklevel=3,
        )


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 0
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 16
    d_ffn: int = 1408
    max_seq_len: int = 8192
    dropout: float = 0.1
    num_kv_heads: int = 4
    rope_base: int = 10_000
    gradient_checkpointing: bool = True
    pad_id: int = 0


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 4
    grad_accumulation: int = 8
    lr: float = 3e-4
    lr_warmup_steps: int = 2_000
    lr_min_ratio: float = 0.1
    epochs: int = 40
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    num_workers: int = 0
    log_interval: int = 50
    eval_interval: int = 1
    save_every_epoch: bool = True
    seed: int = 42
    tensorboard: bool = True
    tensorboard_subdir: str = "tensorboard"
    tensorboard_flush_secs: int = 30
    tensorboard_histogram_interval: int = 1
    tensorboard_log_graph: bool = True
    compile_model: bool = False


@dataclass(slots=True)
class DataConfig:
    tokenizer_path: str = "artifacts/tokenizer/default.json"
    train_dataset_path: str = "artifacts/datasets/train.pt"
    val_dataset_path: str = "artifacts/datasets/val.pt"
    output_dir: str = "artifacts/run"
    max_prompt_tokens: int = 192
    max_context_tokens: int = 1536
    max_target_tokens: int = 6400
    max_seq_len: int = 8192
    pack_sequences: bool = True


@dataclass(slots=True)
class SampleGenerationConfig:
    enabled: bool = False
    output_subdir: str = "samples"
    scratch_prompts: list[str] = field(
        default_factory=lambda: [
            "cinematic piano motif with evolving strings",
            "tight electronic groove with warm bass",
        ]
    )
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_k: int = 128
    top_p: float = 0.95
    add_track_existing_midi: str = ""
    add_track_instrument: str = "drums"
    add_track_prompt: str = ""


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SampleGenerationConfig = field(default_factory=SampleGenerationConfig)
    tokenizer: UnifiedTokenizerConfig = field(default_factory=UnifiedTokenizerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        payload = load_yaml(config_path)

        known_sections = {"model", "training", "data", "sampling", "tokenizer"}
        unknown_sections = set(payload) - known_sections
        if unknown_sections:
            warnings.warn(f"Unknown top-level config sections: {sorted(unknown_sections)}", stacklevel=2)

        model_dict = payload.get("model", {})
        training_dict = payload.get("training", {})
        data_dict = payload.get("data", {})
        sampling_dict = payload.get("sampling", {})
        tokenizer_cfg = payload.get("tokenizer", {})

        _warn_unknown_keys(ModelConfig, "model", model_dict)
        _warn_unknown_keys(TrainingConfig, "training", training_dict)
        _warn_unknown_keys(DataConfig, "data", data_dict)
        _warn_unknown_keys(SampleGenerationConfig, "sampling", sampling_dict)

        config = cls(
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**training_dict),
            data=DataConfig(**data_dict),
            sampling=SampleGenerationConfig(**sampling_dict),
            tokenizer=UnifiedTokenizerConfig.from_dict(tokenizer_cfg),
        )
        base_dir = config_path.parent
        config.data.tokenizer_path = str(resolve_path(config.data.tokenizer_path, base_dir))
        config.data.train_dataset_path = str(resolve_path(config.data.train_dataset_path, base_dir))
        config.data.val_dataset_path = str(resolve_path(config.data.val_dataset_path, base_dir))
        config.data.output_dir = str(resolve_path(config.data.output_dir, base_dir))
        if config.sampling.add_track_existing_midi:
            config.sampling.add_track_existing_midi = str(
                resolve_path(config.sampling.add_track_existing_midi, base_dir)
            )
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
