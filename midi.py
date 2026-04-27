from __future__ import annotations

from pathlib import Path

import symusic


_SUSTAIN_PEDAL_CC = 64
_SUSTAIN_ON_THRESHOLD = 64


def _copy_score_meta(score: symusic.Score) -> symusic.Score:
    clone = symusic.Score(int(getattr(score, "tpq", getattr(score, "ticks_per_quarter", 480)) or 480))
    for tempo in score.tempos:
        clone.tempos.append(tempo.copy())
    for time_signature in score.time_signatures:
        clone.time_signatures.append(time_signature.copy())
    for key_signature in getattr(score, "key_signatures", []):
        clone.key_signatures.append(key_signature.copy())
    for marker in getattr(score, "markers", []):
        clone.markers.append(marker.copy())
    return clone


def _normalize_track_from_loaded(track: symusic.Track) -> symusic.Track:
    clone = symusic.Track(name=str(track.name), program=int(track.program), is_drum=bool(track.is_drum))
    for note in track.notes:
        clone.notes.append(note.copy())

    pedal_start: int | None = None
    seen_pedals: set[tuple[int, int]] = set()

    def _append_pedal(start: int, end: int) -> None:
        signature = (int(start), int(end))
        if signature in seen_pedals:
            return
        clone.pedals.append(symusic.Pedal(int(start), max(1, int(end) - int(start))))
        seen_pedals.add(signature)

    for control in track.controls:
        number = int(control.number)
        value = int(control.value)
        time = int(control.time)
        if number != _SUSTAIN_PEDAL_CC:
            clone.controls.append(control.copy())
            continue
        if value >= _SUSTAIN_ON_THRESHOLD:
            if pedal_start is None:
                pedal_start = time
            continue
        if pedal_start is not None and time >= pedal_start:
            _append_pedal(pedal_start, time)
            pedal_start = None
    if pedal_start is not None:
        _append_pedal(pedal_start, pedal_start + 1)

    for pedal in track.pedals:
        _append_pedal(int(pedal.time), int(pedal.end))
    for pitch_bend in track.pitch_bends:
        clone.pitch_bends.append(pitch_bend.copy())
    for lyric in track.lyrics:
        clone.lyrics.append(lyric.copy())
    return clone


def _prepare_track_for_dump(track: symusic.Track) -> symusic.Track:
    clone = symusic.Track(name=str(track.name), program=int(track.program), is_drum=bool(track.is_drum))
    for note in track.notes:
        clone.notes.append(note.copy())
    for control in track.controls:
        if int(control.number) == _SUSTAIN_PEDAL_CC:
            continue
        clone.controls.append(control.copy())
    for pedal in track.pedals:
        clone.controls.append(symusic.ControlChange(int(pedal.time), _SUSTAIN_PEDAL_CC, 127))
        clone.controls.append(symusic.ControlChange(int(pedal.end), _SUSTAIN_PEDAL_CC, 0))
    for pitch_bend in track.pitch_bends:
        clone.pitch_bends.append(pitch_bend.copy())
    for lyric in track.lyrics:
        clone.lyrics.append(lyric.copy())
    return clone


def _normalize_loaded_score(score: symusic.Score) -> symusic.Score:
    normalized = _copy_score_meta(score)
    for track in score.tracks:
        normalized.tracks.append(_normalize_track_from_loaded(track))
    normalized.sort()
    return normalized


def _prepare_score_for_dump(score: symusic.Score) -> symusic.Score:
    prepared = _copy_score_meta(score)
    for track in score.tracks:
        prepared.tracks.append(_prepare_track_for_dump(track))
    prepared.sort()
    return prepared


def read_midi(path: str | Path) -> symusic.Score:
    return _normalize_loaded_score(symusic.Score(str(Path(path).expanduser())))


def write_midi(score: symusic.Score, path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_score_for_dump(score).dump_midi(output_path)


__all__ = ["read_midi", "write_midi"]
