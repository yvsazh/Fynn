from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TextTokenizerConfig:
    kind: str = "tiktoken"
    tiktoken_encoding: str = "cl100k_base"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ByteTextTokenizer:
    def __init__(self, config: TextTokenizerConfig):
        self.config = config
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(max(0, min(255, token)) for token in tokens).decode("utf-8", errors="ignore")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.config.kind,
            "vocab_size": self.vocab_size,
        }


class TiktokenTextTokenizer:
    def __init__(self, config: TextTokenizerConfig):
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("tiktoken is not installed") from exc

        self.config = config
        self.encoding = tiktoken.get_encoding(config.tiktoken_encoding)
        self.vocab_size = int(getattr(self.encoding, "n_vocab"))

    def encode(self, text: str) -> list[int]:
        return list(self.encoding.encode(text))

    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": self.config.kind,
            "tiktoken_encoding": self.config.tiktoken_encoding,
            "vocab_size": self.vocab_size,
        }


def build_text_tokenizer(config: TextTokenizerConfig):
    if config.kind == "byte":
        return ByteTextTokenizer(config)
    if config.kind == "tiktoken":
        try:
            return TiktokenTextTokenizer(config)
        except Exception as exc:
            warnings.warn(
                f"Falling back to byte text tokenizer because tiktoken encoding "
                f"{config.tiktoken_encoding!r} could not be loaded: {exc}",
                stacklevel=2,
            )
            config.kind = "byte"
            return ByteTextTokenizer(config)
    raise ValueError(f"Unsupported text tokenizer kind: {config.kind}")
