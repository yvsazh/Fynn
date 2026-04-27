from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import symusic

from fynn.instruments import instrument_label, instrument_slug
from fynn.score_utils import copy_score_structure, copy_track


@dataclass(slots=True, frozen=True)
class ScoreBundlePaths:
    output_dir: Path
    full_path: Path
    track_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, str]:
        bundle = {"full": str(self.full_path)}
        for index, track_path in enumerate(self.track_paths):
            bundle[f"track_{index:02d}"] = str(track_path)
        return bundle


def write_score_bundle_paths(score: symusic.Score, output_dir: str | Path) -> ScoreBundlePaths:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "full_composition.mid"
    score.dump_midi(full_path)
    track_paths: list[Path] = []

    for index, track in enumerate(score.tracks):
        single = copy_score_structure(score)
        cloned = copy_track(track)
        cloned.name = instrument_label(int(track.program), bool(track.is_drum), name=str(track.name))
        single.tracks.append(cloned)
        track_name = instrument_slug(int(track.program), bool(track.is_drum), name=str(track.name))
        track_path = output_dir / f"track_{index:02d}_{track_name}.mid"
        single.dump_midi(track_path)
        track_paths.append(track_path)

    return ScoreBundlePaths(
        output_dir=output_dir,
        full_path=full_path,
        track_paths=tuple(track_paths),
    )


def export_score_bundle(score: symusic.Score, output_dir: str | Path) -> symusic.Score:
    write_score_bundle_paths(score, output_dir)
    return score


__all__ = ["ScoreBundlePaths", "export_score_bundle", "write_score_bundle_paths"]
