from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from fynn.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10_000, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get(self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        if cos.device != device or cos.dtype != dtype:
            cos = cos.to(device=device, dtype=dtype)
            sin = sin.to(device=device, dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GQAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_kv_heads: int, rope_base: int, max_seq_len: int, dropout: float):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if nhead % num_kv_heads != 0:
            raise ValueError("nhead must be divisible by num_kv_heads")
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // nhead
        self.groups = nhead // num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, channels = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_kv[0].size(2) if past_kv is not None else 0
        cos, sin = self.rope.get(seq_len, x.device, x.dtype, offset=past_len)
        q, k = apply_rotary(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        if self.groups > 1:
            kv_seq_len = k.size(2)
            k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(batch_size, self.nhead, kv_seq_len, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(batch_size, self.nhead, kv_seq_len, self.head_dim)

        attn_mask = None
        is_causal = False
        if attention_mask is not None:
            total_len = past_len + seq_len
            key_mask = attention_mask[:, None, None, :total_len].to(dtype=torch.bool, device=q.device)
            causal_mask = torch.ones((seq_len, total_len), dtype=torch.bool, device=q.device).tril(diagonal=past_len)
            attn_mask = key_mask & causal_mask[None, None, :, :]
        elif past_kv is None:
            is_causal = True

        dropout_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.out_proj(attn), new_kv


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ffn, bias=False)
        self.up = nn.Linear(d_model, d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQAttention(
            d_model=config.d_model,
            nhead=config.nhead,
            num_kv_heads=config.num_kv_heads,
            rope_base=config.rope_base,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFFN(config.d_model, config.d_ffn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv = self.attn(self.attn_norm(x), attention_mask=attention_mask, past_kv=past_kv)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_kv


class FynnTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if module is self.lm_head:
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | list]:
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        total_len = input_ids.size(1) + past_len
        if total_len > self.config.max_seq_len:
            raise ValueError(f"Total sequence length {total_len} exceeds model max_seq_len={self.config.max_seq_len}")

        x = self.embedding(input_ids)

        new_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values else None
            if self.training and self.config.gradient_checkpointing:
                if use_cache:
                    raise ValueError(
                        "use_cache=True is incompatible with gradient_checkpointing=True during "
                        "training. Call model.eval() before using the KV cache."
                    )
                x, _ = checkpoint(block, x, attention_mask, use_reentrant=False)
            else:
                x, new_kv = block(x, attention_mask=attention_mask, past_kv=layer_past)
                if use_cache:
                    new_key_values.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        result: dict[str, torch.Tensor | list] = {"logits": logits}
        if use_cache:
            result["past_key_values"] = new_key_values
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
        return result

    def estimate_mfu(
        self,
        batch_size: int,
        seq_len: int,
        step_time_sec: float,
        peak_hardware_flops: float = 312e12,
    ) -> float:
        """Estimate model FLOP utilisation (MFU).

        peak_hardware_flops: theoretical peak FP16/BF16 FLOPS for the hardware.
        Defaults to 312 TFLOPS (A100-80GB SXM). Common values:
          A100-80GB SXM : 312e12
          H100 SXM      : 989e12
          RTX 4090      : 82.6e12
          RTX 3090      : 35.6e12
        """
        flops = 6 * self.num_parameters * batch_size * seq_len
        achieved = flops / max(step_time_sec, 1e-6)
        return achieved / peak_hardware_flops
