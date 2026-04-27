from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import symusic


DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO_QPM = 120.0
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_CC_WHITELIST = (1, 7, 10, 11, 91, 93)


@dataclass(slots=True, frozen=True)
class ScoreEventCounts:
    tracks: int
    notes: int
    controls: int
    pedals: int
    pitch_bends: int
    lyrics: int
    markers: int
    tempos: int
    time_signatures: int
    key_signatures: int


def score_ticks_per_beat(score: symusic.Score) -> int:
    return int(getattr(score, "ticks_per_quarter", getattr(score, "tpq", DEFAULT_TICKS_PER_BEAT)) or DEFAULT_TICKS_PER_BEAT)


def copy_score(score: symusic.Score) -> symusic.Score:
    return score.copy()


def copy_score_structure(score: symusic.Score) -> symusic.Score:
    clone = symusic.Score(score_ticks_per_beat(score))
    for tempo in score.tempos:
        clone.tempos.append(tempo.copy())
    for time_sig in score.time_signatures:
        clone.time_signatures.append(time_sig.copy())
    for key_sig in getattr(score, "key_signatures", []):
        clone.key_signatures.append(key_sig.copy())
    for marker in getattr(score, "markers", []):
        clone.markers.append(marker.copy())
    return clone


def copy_track(track: symusic.Track) -> symusic.Track:
    clone = symusic.Track(name=str(track.name), program=int(track.program), is_drum=bool(track.is_drum))
    for note in track.notes:
        clone.notes.append(note.copy())
    for control in track.controls:
        clone.controls.append(control.copy())
    for pedal in track.pedals:
        clone.pedals.append(pedal.copy())
    for pitch_bend in track.pitch_bends:
        clone.pitch_bends.append(pitch_bend.copy())
    for lyric in track.lyrics:
        clone.lyrics.append(lyric.copy())
    return clone


def count_score_events(score: symusic.Score) -> ScoreEventCounts:
    return ScoreEventCounts(
        tracks=len(score.tracks),
        notes=sum(len(track.notes) for track in score.tracks),
        controls=sum(len(track.controls) for track in score.tracks),
        pedals=sum(len(track.pedals) for track in score.tracks),
        pitch_bends=sum(len(track.pitch_bends) for track in score.tracks),
        lyrics=sum(len(track.lyrics) for track in score.tracks),
        markers=len(getattr(score, "markers", [])),
        tempos=len(score.tempos),
        time_signatures=len(score.time_signatures),
        key_signatures=len(getattr(score, "key_signatures", [])),
    )


def score_end_tick(score: symusic.Score) -> int:
    end_tick = 0
    for tempo in score.tempos:
        end_tick = max(end_tick, int(tempo.time))
    for time_sig in score.time_signatures:
        end_tick = max(end_tick, int(time_sig.time))
    for key_sig in getattr(score, "key_signatures", []):
        end_tick = max(end_tick, int(key_sig.time))
    for marker in getattr(score, "markers", []):
        end_tick = max(end_tick, int(marker.time))
    for track in score.tracks:
        for note in track.notes:
            end_tick = max(end_tick, int(note.end))
        for control in track.controls:
            end_tick = max(end_tick, int(control.time))
        for pedal in track.pedals:
            end_tick = max(end_tick, int(pedal.end))
        for pitch_bend in track.pitch_bends:
            end_tick = max(end_tick, int(pitch_bend.time))
        for lyric in track.lyrics:
            end_tick = max(end_tick, int(lyric.time))
    return max(1, end_tick)


def tokenizable_track_indices(
    score: symusic.Score,
    *,
    use_notes: bool = True,
    use_sustain_pedals: bool = False,
    use_pitch_bends: bool = False,
    control_change_numbers: Iterable[int] = (),
) -> list[int]:
    allowed_controls = {int(number) for number in control_change_numbers}
    indices: list[int] = []
    for index, track in enumerate(score.tracks):
        if use_notes and len(track.notes) > 0:
            indices.append(index)
            continue
        if use_sustain_pedals and len(track.pedals) > 0:
            indices.append(index)
            continue
        if use_pitch_bends and len(track.pitch_bends) > 0:
            indices.append(index)
            continue
        if allowed_controls and any(int(control.number) in allowed_controls for control in track.controls):
            indices.append(index)
    return indices


def track_anchor_ticks(track: symusic.Track) -> list[int]:
    ticks = [int(note.time) for note in track.notes]
    ticks.extend(int(control.time) for control in track.controls)
    ticks.extend(int(pedal.time) for pedal in track.pedals)
    ticks.extend(int(pitch_bend.time) for pitch_bend in track.pitch_bends)
    return ticks


def score_anchor_ticks(score: symusic.Score) -> list[int]:
    ticks: list[int] = []
    for track in score.tracks:
        ticks.extend(track_anchor_ticks(track))
    return ticks


def merge_scores(base_score: symusic.Score, *overlay_scores: symusic.Score) -> symusic.Score:
    merged = copy_score_structure(base_score)
    for track in base_score.tracks:
        merged.tracks.append(copy_track(track))
    for overlay in overlay_scores:
        for track in overlay.tracks:
            merged.tracks.append(copy_track(track))
    merged.sort()
    return merged


def split_add_track_context_target(score: symusic.Score, target_idx: int) -> tuple[symusic.Score, symusic.Score]:
    if target_idx < 0 or target_idx >= len(score.tracks):
        raise IndexError(target_idx)
    context = copy_score_structure(score)
    target = copy_score_structure(score)
    for index, track in enumerate(score.tracks):
        cloned = copy_track(track)
        if index == target_idx:
            target.tracks.append(cloned)
        else:
            context.tracks.append(cloned)
    return context, target


def split_add_track_window(
    score: symusic.Score,
    target_idx: int,
    start_tick: int,
    end_tick: int,
) -> tuple[symusic.Score, symusic.Score]:
    context_full, target_full = split_add_track_context_target(score, target_idx)
    return window_score(context_full, start_tick, end_tick), window_score(target_full, start_tick, end_tick)


def score_marker_texts(score: symusic.Score, max_items: int = 4) -> list[str]:
    values: list[str] = []
    for marker in getattr(score, "markers", []):
        text = " ".join(str(marker.text).split())
        if not text or text in values:
            continue
        values.append(text)
        if len(values) >= max_items:
            break
    return values


def score_lyric_snippets(score: symusic.Score, max_items: int = 4) -> list[str]:
    values: list[str] = []
    for track in score.tracks:
        for lyric in track.lyrics:
            text = " ".join(str(lyric.text).split())
            if not text or text in values:
                continue
            values.append(text)
            if len(values) >= max_items:
                return values
    return values


def score_control_numbers(score: symusic.Score) -> list[int]:
    numbers = {int(control.number) for track in score.tracks for control in track.controls}
    return sorted(numbers)


def score_prompt_metadata(score: symusic.Score) -> str:
    parts: list[str] = []
    markers = score_marker_texts(score, max_items=3)
    if markers:
        parts.append(f"markers: {', '.join(markers)}")
    lyric_snippets = score_lyric_snippets(score, max_items=2)
    if lyric_snippets:
        parts.append("lyrics: present")
    control_numbers = score_control_numbers(score)
    if control_numbers:
        labels = ", ".join(f"cc{number}" for number in control_numbers[:6])
        suffix = " +" if len(control_numbers) > 6 else ""
        parts.append(f"controls: {labels}{suffix}")
    if any(len(track.pedals) > 0 for track in score.tracks):
        parts.append("pedals: present")
    if any(len(track.pitch_bends) > 0 for track in score.tracks):
        parts.append("pitch bends: present")
    return " | ".join(parts)


def _append_shifted_events(
    target: Any,
    events: Iterable[Any],
    *,
    start_tick: int,
    end_tick: int,
    shift: int,
) -> None:
    seen: set[tuple[int, tuple[Any, ...]]] = set()
    for event in events:
        event_time = int(event.time)
        if event_time < start_tick or event_time >= end_tick:
            continue
        copied = event.copy()
        copied.time = event_time + shift
        signature = _event_signature(copied)
        if signature in seen:
            continue
        target.append(copied)
        seen.add(signature)


def _copy_meta_window(
    score: symusic.Score,
    start_tick: int,
    end_tick: int,
) -> symusic.Score:
    windowed = symusic.Score(score_ticks_per_beat(score))

    active_tempo = _last_event_at_or_before(score.tempos, start_tick)
    if active_tempo is not None:
        windowed.tempos.append(symusic.Tempo(0, qpm=float(active_tempo.qpm)))
    else:
        windowed.tempos.append(symusic.Tempo(0, qpm=DEFAULT_TEMPO_QPM))
    _append_shifted_events(windowed.tempos, score.tempos, start_tick=start_tick + 1, end_tick=end_tick, shift=-start_tick)

    active_time_sig = _last_event_at_or_before(score.time_signatures, start_tick)
    if active_time_sig is not None:
        windowed.time_signatures.append(
            symusic.TimeSignature(0, int(active_time_sig.numerator), int(active_time_sig.denominator))
        )
    else:
        windowed.time_signatures.append(symusic.TimeSignature(0, *DEFAULT_TIME_SIGNATURE))
    _append_shifted_events(
        windowed.time_signatures,
        score.time_signatures,
        start_tick=start_tick + 1,
        end_tick=end_tick,
        shift=-start_tick,
    )

    active_key_sig = _last_event_at_or_before(getattr(score, "key_signatures", []), start_tick)
    if active_key_sig is not None:
        windowed.key_signatures.append(symusic.KeySignature(0, int(active_key_sig.key), int(active_key_sig.tonality)))
    _append_shifted_events(
        windowed.key_signatures,
        getattr(score, "key_signatures", []),
        start_tick=start_tick + 1,
        end_tick=end_tick,
        shift=-start_tick,
    )

    _append_shifted_events(
        windowed.markers,
        getattr(score, "markers", []),
        start_tick=start_tick,
        end_tick=end_tick,
        shift=-start_tick,
    )
    return windowed


def window_score(score: symusic.Score, start_tick: int, end_tick: int) -> symusic.Score:
    start_tick = max(0, int(start_tick))
    end_tick = max(start_tick + 1, int(end_tick))
    windowed = _copy_meta_window(score, start_tick, end_tick)
    preserve_marker_only_tracks = _score_has_markers_in_range(score, start_tick=start_tick, end_tick=end_tick)

    for track in score.tracks:
        clipped = symusic.Track(name=str(track.name), program=int(track.program), is_drum=bool(track.is_drum))

        for note in track.notes:
            note_start = int(note.time)
            note_end = int(note.end)
            if note_end <= start_tick or note_start >= end_tick:
                continue
            clipped_start = max(start_tick, note_start)
            clipped_end = min(end_tick, note_end)
            clipped.notes.append(
                symusic.Note(
                    clipped_start - start_tick,
                    max(1, clipped_end - clipped_start),
                    int(note.pitch),
                    int(note.velocity),
                )
            )

        _append_active_controls(clipped.controls, track.controls, start_tick=start_tick, end_tick=end_tick)

        for pedal in track.pedals:
            pedal_start = int(pedal.time)
            pedal_end = int(pedal.end)
            if pedal_end <= start_tick or pedal_start >= end_tick:
                continue
            clipped_start = max(start_tick, pedal_start)
            clipped_end = min(end_tick, pedal_end)
            clipped.pedals.append(symusic.Pedal(clipped_start - start_tick, max(1, clipped_end - clipped_start)))

        _append_active_scalar_events(
            clipped.pitch_bends,
            track.pitch_bends,
            start_tick=start_tick,
            end_tick=end_tick,
        )

        _append_shifted_events(
            clipped.lyrics,
            track.lyrics,
            start_tick=start_tick,
            end_tick=end_tick,
            shift=-start_tick,
        )

        if _track_has_any_events(clipped) or _should_preserve_marker_only_track(
            source_track=track,
            has_markers_in_window=preserve_marker_only_tracks,
        ):
            windowed.tracks.append(clipped)

    windowed.sort()
    return windowed


def _append_active_controls(
    target: Any,
    controls: Iterable[symusic.ControlChange],
    *,
    start_tick: int,
    end_tick: int,
) -> None:
    controls_by_number: dict[int, list[symusic.ControlChange]] = {}
    for control in controls:
        controls_by_number.setdefault(int(control.number), []).append(control)

    seen: set[tuple[int, tuple[Any, ...]]] = set()
    for number, items in controls_by_number.items():
        active = _last_event_at_or_before(items, start_tick)
        if active is not None:
            copied = symusic.ControlChange(0, number, int(active.value))
            signature = _event_signature(copied)
            if signature not in seen:
                target.append(copied)
                seen.add(signature)
        for control in items:
            control_time = int(control.time)
            if control_time <= start_tick or control_time >= end_tick:
                continue
            copied = symusic.ControlChange(control_time - start_tick, number, int(control.value))
            signature = _event_signature(copied)
            if signature not in seen:
                target.append(copied)
                seen.add(signature)


def _append_active_scalar_events(
    target: Any,
    events: Iterable[Any],
    *,
    start_tick: int,
    end_tick: int,
) -> None:
    seen: set[tuple[int, tuple[Any, ...]]] = set()
    active = _last_event_at_or_before(events, start_tick)
    if active is not None:
        copied = active.copy()
        copied.time = 0
        signature = _event_signature(copied)
        target.append(copied)
        seen.add(signature)
    for event in events:
        event_time = int(event.time)
        if event_time <= start_tick or event_time >= end_tick:
            continue
        copied = event.copy()
        copied.time = event_time - start_tick
        signature = _event_signature(copied)
        if signature in seen:
            continue
        target.append(copied)
        seen.add(signature)


def _last_event_at_or_before(events: Iterable[Any], tick: int) -> Any | None:
    active = None
    for event in events:
        if int(event.time) <= tick:
            active = event
        else:
            break
    return active


def _event_signature(event: Any) -> tuple[int, tuple[Any, ...]]:
    if hasattr(event, "number") and hasattr(event, "value"):
        return int(event.time), (int(event.number), int(event.value))
    if hasattr(event, "value"):
        return int(event.time), (int(event.value),)
    if hasattr(event, "qpm"):
        return int(event.time), (round(float(event.qpm), 6),)
    if hasattr(event, "numerator") and hasattr(event, "denominator"):
        return int(event.time), (int(event.numerator), int(event.denominator))
    if hasattr(event, "key") and hasattr(event, "tonality"):
        return int(event.time), (int(event.key), int(event.tonality))
    if hasattr(event, "text"):
        return int(event.time), (str(event.text),)
    return int(event.time), ()


def _track_has_any_events(track: symusic.Track) -> bool:
    return any(
        (
            len(track.notes) > 0,
            len(track.controls) > 0,
            len(track.pedals) > 0,
            len(track.pitch_bends) > 0,
            len(track.lyrics) > 0,
        )
    )


def _score_has_markers_in_range(score: symusic.Score, *, start_tick: int, end_tick: int) -> bool:
    return any(start_tick <= int(marker.time) < end_tick for marker in getattr(score, "markers", []))


def _should_preserve_marker_only_track(*, source_track: symusic.Track, has_markers_in_window: bool) -> bool:
    if not has_markers_in_window:
        return False
    return not _track_has_any_events(source_track)
