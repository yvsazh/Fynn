from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import symusic
import torch

from fynn.instruments import apply_program_to_score_tracks, resolve_gm_instrument_slug
from fynn.midi import read_midi
from fynn.model import FynnTransformer
from fynn.score_utils import merge_scores, score_ticks_per_beat
from fynn.tokenization.unified import UnifiedTokenizer


def _filter_logits(logits: torch.Tensor, allowed_ids: list[int], temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    filtered = torch.full_like(logits, float("-inf"))
    allowed = torch.tensor(allowed_ids, device=logits.device)
    filtered.index_copy_(1, allowed, logits.index_select(1, allowed))
    filtered = filtered / max(temperature, 1e-5)

    if top_k > 0 and top_k < filtered.size(-1):
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1:]
        filtered = torch.where(filtered < threshold, torch.full_like(filtered, float("-inf")), filtered)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)

    return filtered


@torch.no_grad()
def sample_sequence(
    model: FynnTransformer,
    input_ids: list[int],
    tokenizer: UnifiedTokenizer,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
    top_k: int = 128,
    top_p: float = 0.95,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[int]:
    model.eval()
    tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated: list[int] = list(input_ids)
    allowed_ids = tokenizer.generation_mask_ids()
    if not allowed_ids:
        raise RuntimeError("Tokenizer generation mask is empty.")
    past_key_values = None

    for step_idx in range(max_new_tokens):
        if len(generated) >= model.config.max_seq_len:
            break

        result = model(tokens, past_key_values=past_key_values, use_cache=True)
        logits = result["logits"][:, -1, :]
        past_key_values = result["past_key_values"]

        filtered = _filter_logits(logits, allowed_ids, temperature, top_k, top_p)
        probs = torch.softmax(filtered, dim=-1)

        if torch.isnan(probs).any() or probs.sum().item() < 1e-8:
            allowed_t = torch.tensor(allowed_ids, device=logits.device)
            best_idx = logits[0, allowed_t].argmax()
            next_token = allowed_t[best_idx].unsqueeze(0).unsqueeze(0)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        next_id = next_token.item()
        generated.append(next_id)
        tokens = next_token

        if next_id == tokenizer.eos_id:
            break

        if progress_callback is not None:
            progress_callback(step_idx + 1, max_new_tokens)

    return generated


def _extract_generated_midi(sequence: list[int], tokenizer: UnifiedTokenizer) -> list[int]:
    generated = []
    started = False
    for token in sequence:
        if token == tokenizer.bos_id:
            started = True
            continue
        if not started:
            continue
        if token == tokenizer.eos_id:
            break
        if token >= tokenizer.midi_offset:
            generated.append(token)
    return generated


@torch.no_grad()
def generate_continuation_from_midi_tokens(
    model: FynnTransformer,
    tokenizer: UnifiedTokenizer,
    context_tokens: list[int],
    device: torch.device,
    prompt: str = "",
    max_prompt_tokens: int = 192,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
    top_k: int = 128,
    top_p: float = 0.95,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[int], list[int]]:
    prompt_tokens = tokenizer.encode_text(prompt, max_prompt_tokens)
    prefix = [
        tokenizer.special_to_id["<PROMPT>"],
        *prompt_tokens,
        tokenizer.special_to_id["</PROMPT>"],
        tokenizer.special_to_id["<GEN_ALL>"],
        tokenizer.bos_id,
        *context_tokens,
    ]
    full_sequence = sample_sequence(
        model=model,
        input_ids=prefix,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        progress_callback=progress_callback,
    )
    full_midi_tokens = _extract_generated_midi(full_sequence, tokenizer)
    if len(full_midi_tokens) < len(context_tokens):
        raise RuntimeError("Model continuation output was shorter than the priming MIDI context.")
    continuation_tokens = full_midi_tokens[len(context_tokens):]
    return full_midi_tokens, continuation_tokens


@torch.no_grad()
def generate_continuation_from_score(
    model: FynnTransformer,
    tokenizer: UnifiedTokenizer,
    context_score: symusic.Score,
    device: torch.device,
    prompt: str = "",
    max_prompt_tokens: int = 192,
    max_context_tokens: int | None = 1536,
    max_new_tokens: int = 512,
    temperature: float = 0.9,
    top_k: int = 128,
    top_p: float = 0.95,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[symusic.Score, symusic.Score]:
    context_tokens = tokenizer.encode_midi(context_score, max_tokens=max_context_tokens)
    full_tokens, continuation_tokens = generate_continuation_from_midi_tokens(
        model=model,
        tokenizer=tokenizer,
        context_tokens=context_tokens,
        device=device,
        prompt=prompt,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        progress_callback=progress_callback,
    )
    ticks_per_beat = score_ticks_per_beat(context_score)
    full_score = tokenizer.decode_midi(full_tokens, ticks_per_beat=ticks_per_beat)
    continuation_score = tokenizer.decode_midi(continuation_tokens, ticks_per_beat=ticks_per_beat)
    return full_score, continuation_score


@torch.no_grad()
def generate_from_scratch(
    model: FynnTransformer,
    tokenizer: UnifiedTokenizer,
    prompt: str,
    device: torch.device,
    max_prompt_tokens: int = 192,
    max_new_tokens: int = 1024,
    temperature: float = 0.9,
    top_k: int = 128,
    top_p: float = 0.95,
    progress_callback: Callable[[int, int], None] | None = None,
) -> symusic.Score:
    prefix = [
        tokenizer.special_to_id["<PROMPT>"],
        *tokenizer.encode_text(prompt, max_prompt_tokens),
        tokenizer.special_to_id["</PROMPT>"],
        tokenizer.special_to_id["<GEN_ALL>"],
        tokenizer.bos_id,
    ]
    full_sequence = sample_sequence(
        model=model,
        input_ids=prefix,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        progress_callback=progress_callback,
    )
    midi_tokens = _extract_generated_midi(full_sequence, tokenizer)
    return tokenizer.decode_midi(midi_tokens)


@torch.no_grad()
def add_track_to_score(
    model: FynnTransformer,
    tokenizer: UnifiedTokenizer,
    context_score: symusic.Score,
    instrument: str,
    extra_prompt: str,
    device: torch.device,
    max_prompt_tokens: int = 192,
    max_context_tokens: int = 1536,
    max_new_tokens: int = 1024,
    temperature: float = 0.9,
    top_k: int = 128,
    top_p: float = 0.95,
    progress_callback: Callable[[int, int], None] | None = None,
) -> symusic.Score:
    resolved = resolve_gm_instrument_slug(instrument)
    if resolved is not None:
        req_program, req_is_drum, canonical = resolved
        prompt = tokenizer.build_add_track_prompt(canonical, extra_prompt)
    else:
        req_program, req_is_drum = None, None
        prompt = tokenizer.build_add_track_prompt(instrument, extra_prompt)

    context_tokens = tokenizer.encode_midi(context_score, max_context_tokens)
    prefix = [
        tokenizer.special_to_id["<PROMPT>"],
        *tokenizer.encode_text(prompt, max_prompt_tokens),
        tokenizer.special_to_id["</PROMPT>"],
        tokenizer.special_to_id["<CONTEXT>"],
        *context_tokens,
        tokenizer.special_to_id["</CONTEXT>"],
        tokenizer.special_to_id["<ADD_TRACK>"],
        tokenizer.special_to_id["</ADD_TRACK>"],
        tokenizer.bos_id,
    ]
    full_sequence = sample_sequence(
        model=model,
        input_ids=prefix,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        progress_callback=progress_callback,
    )
    new_track_tokens = _extract_generated_midi(full_sequence, tokenizer)
    new_track_score = tokenizer.decode_midi(
        new_track_tokens,
        ticks_per_beat=score_ticks_per_beat(context_score),
    )

    if resolved is not None:
        apply_program_to_score_tracks(new_track_score, req_program, req_is_drum)

    return merge_scores(context_score, new_track_score)


def add_track_to_midi(
    model: FynnTransformer,
    tokenizer: UnifiedTokenizer,
    existing_midi_path: str | Path,
    instrument: str,
    extra_prompt: str,
    device: torch.device,
    progress_callback: Callable[[int, int], None] | None = None,
    **kwargs,
) -> symusic.Score:
    return add_track_to_score(
        model=model,
        tokenizer=tokenizer,
        context_score=read_midi(existing_midi_path),
        instrument=instrument,
        extra_prompt=extra_prompt,
        device=device,
        progress_callback=progress_callback,
        **kwargs,
    )
