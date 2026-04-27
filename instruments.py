from __future__ import annotations

import re

import symusic


GM_PROGRAM_NAMES = {
    0: "acoustic_grand_piano",
    1: "bright_acoustic_piano",
    2: "electric_grand_piano",
    3: "honky_tonk_piano",
    4: "electric_piano_1",
    5: "electric_piano_2",
    6: "harpsichord",
    7: "clavinet",
    8: "celesta",
    9: "glockenspiel",
    10: "music_box",
    11: "vibraphone",
    12: "marimba",
    13: "xylophone",
    14: "tubular_bells",
    15: "dulcimer",
    16: "drawbar_organ",
    17: "percussive_organ",
    18: "rock_organ",
    19: "church_organ",
    20: "reed_organ",
    21: "accordion",
    22: "harmonica",
    23: "tango_accordion",
    24: "guitar_nylon",
    25: "guitar_steel",
    26: "guitar_jazz",
    27: "guitar_clean",
    28: "guitar_muted",
    29: "overdriven_guitar",
    30: "distortion_guitar",
    31: "guitar_harmonics",
    32: "acoustic_bass",
    33: "electric_bass_finger",
    34: "electric_bass_pick",
    35: "fretless_bass",
    36: "slap_bass_1",
    37: "slap_bass_2",
    38: "synth_bass_1",
    39: "synth_bass_2",
    40: "violin",
    41: "viola",
    42: "cello",
    43: "contrabass",
    44: "tremolo_strings",
    45: "pizzicato_strings",
    46: "orchestral_harp",
    47: "timpani",
    48: "string_ensemble_1",
    49: "string_ensemble_2",
    50: "synth_strings_1",
    51: "synth_strings_2",
    52: "choir_aahs",
    53: "voice_oohs",
    54: "synth_choir",
    55: "orchestra_hit",
    56: "trumpet",
    57: "trombone",
    58: "tuba",
    59: "muted_trumpet",
    60: "french_horn",
    61: "brass_section",
    62: "synth_brass_1",
    63: "synth_brass_2",
    64: "soprano_sax",
    65: "alto_sax",
    66: "tenor_sax",
    67: "baritone_sax",
    68: "oboe",
    69: "english_horn",
    70: "bassoon",
    71: "clarinet",
    72: "piccolo",
    73: "flute",
    74: "recorder",
    75: "pan_flute",
    76: "blown_bottle",
    77: "shakuhachi",
    78: "whistle",
    79: "ocarina",
    80: "lead_square",
    81: "lead_sawtooth",
    82: "lead_calliope",
    83: "lead_chiff",
    84: "lead_charang",
    85: "lead_voice",
    86: "lead_fifths",
    87: "lead_bass_lead",
    88: "pad_new_age",
    89: "pad_warm",
    90: "pad_polysynth",
    91: "pad_choir",
    92: "pad_bowed",
    93: "pad_metallic",
    94: "pad_halo",
    95: "pad_sweep",
    96: "fx_rain",
    97: "fx_soundtrack",
    98: "fx_crystal",
    99: "fx_atmosphere",
    100: "fx_brightness",
    101: "fx_goblins",
    102: "fx_echoes",
    103: "fx_sci_fi",
    104: "sitar",
    105: "banjo",
    106: "shamisen",
    107: "koto",
    108: "kalimba",
    109: "bagpipe",
    110: "fiddle",
    111: "shanai",
    112: "tinkle_bell",
    113: "agogo",
    114: "steel_drums",
    115: "woodblock",
    116: "taiko_drum",
    117: "melodic_tom",
    118: "synth_drum",
    119: "reverse_cymbal",
    120: "guitar_fret_noise",
    121: "breath_noise",
    122: "seashore",
    123: "bird_tweet",
    124: "telephone_ring",
    125: "helicopter",
    126: "applause",
    127: "gunshot",
}


def instrument_label(program: int, is_drum: bool, name: str = "") -> str:
    if name.strip():
        return name.strip()
    if is_drum:
        return "drums"
    return GM_PROGRAM_NAMES.get(program, f"program_{program}")


def instrument_slug(program: int, is_drum: bool, name: str = "") -> str:
    raw = instrument_label(program, is_drum, name=name).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return slug or ("drums" if is_drum else f"program_{program}")


def resolve_gm_instrument_slug(slug: str) -> tuple[int, bool, str] | None:
    """Повертає (program, is_drum, канонічний рядок для промпта) або None, якщо не розпізнано.

    Канонічний рядок збігається з тим, що дає ``instrument_label`` без імені доріжки
    (значення з ``GM_PROGRAM_NAMES`` або ``drums``), щоб промпт збігався з тренуванням.
    """
    raw = (slug or "").strip()
    if not raw:
        return None
    if raw.lower() == "drums":
        return (0, True, "drums")
    key = raw.lower()
    for program, name in GM_PROGRAM_NAMES.items():
        if name.lower() == key:
            return (program, False, name)
    return None


def apply_program_to_score_tracks(score: symusic.Score, program: int, is_drum: bool) -> None:
    """Виставляє GM program / drum на всі доріжки (наприклад після add-track decode)."""
    label = instrument_label(program, is_drum, name="")
    for track in score.tracks:
        track.program = program
        track.is_drum = is_drum
        track.name = label
