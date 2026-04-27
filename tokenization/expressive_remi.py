from __future__ import annotations

from typing import TYPE_CHECKING

from miditok.classes import Event, TokSequence, TokenizerConfig
from miditok.constants import MIDI_INSTRUMENTS, TIME_SIGNATURE
from miditok.tokenizations.remi import REMI
from miditok.utils import compute_ticks_per_bar
from symusic import ControlChange, Note, Pedal, PitchBend, Score, Tempo, TimeSignature, Track

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np


class ExpressiveREMI(REMI):
    def _tweak_config_before_creating_voc(self) -> None:
        super()._tweak_config_before_creating_voc()
        self.config.additional_params.setdefault("control_change_numbers", [])
        self.config.additional_params.setdefault("control_change_num_bins", 32)

    @property
    def control_change_numbers(self) -> tuple[int, ...]:
        return tuple(int(number) for number in self.config.additional_params.get("control_change_numbers", []))

    @property
    def control_change_num_bins(self) -> int:
        return max(2, int(self.config.additional_params.get("control_change_num_bins", 32)))

    def _quantize_control_value(self, value: int) -> int:
        return max(0, min(self.control_change_num_bins - 1, round(int(value) * (self.control_change_num_bins - 1) / 127)))

    def _dequantize_control_value(self, bin_index: int) -> int:
        return max(0, min(127, round(int(bin_index) * 127 / max(1, self.control_change_num_bins - 1))))

    def _create_track_events(
        self,
        track: Track,
        ticks_per_beat: "np.ndarray",
        time_division: int,
        ticks_bars: Sequence[int],
        ticks_beats: Sequence[int],
        attribute_controls_indexes: Mapping[int, Sequence[int] | bool] | None = None,
    ) -> list[Event]:
        events = super()._create_track_events(
            track,
            ticks_per_beat,
            time_division,
            ticks_bars,
            ticks_beats,
            attribute_controls_indexes=attribute_controls_indexes,
        )
        if not self.control_change_numbers:
            return events

        allowed_numbers = set(self.control_change_numbers)
        program = track.program if not track.is_drum else -1
        for control in track.controls:
            if int(control.number) not in allowed_numbers:
                continue
            if self.config.use_programs:
                events.append(Event("Program", program, control.time, program, "ProgramControlChange"))
            events.append(Event("ControlChange", int(control.number), control.time, program))
            events.append(
                Event(
                    "CCValue",
                    self._quantize_control_value(int(control.value)),
                    control.time,
                    program,
                )
            )
        return events

    def _create_base_vocabulary(self) -> list[str]:
        vocab = super()._create_base_vocabulary()
        if self.control_change_numbers:
            vocab += [f"ControlChange_{number}" for number in self.control_change_numbers]
            vocab += [f"CCValue_{index}" for index in range(self.control_change_num_bins)]
        return vocab

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        graph = super()._create_token_types_graph()
        if not self.control_change_numbers:
            return graph

        first_note_token_type = "PitchDrum" if self.config.use_pitchdrum_tokens else "Pitch"
        graph.setdefault("ControlChange", set()).add("CCValue")
        graph["CCValue"] = {first_note_token_type, "Position", "Bar"}

        if self.config.use_programs:
            graph.setdefault("Program", set()).add("ControlChange")
            graph["CCValue"].add("Program")

        for token_type in ("Position", "Rest", "Tempo", "TimeSig", "Chord", "PitchBend", "Pedal", "PedalOff"):
            if token_type in graph:
                graph[token_type].add("ControlChange")
                graph["CCValue"].add(token_type)

        if "Duration" in graph:
            graph["Duration"].add("ControlChange")
            graph["CCValue"].add("Duration")

        if self.config.program_changes:
            graph["ControlChange"].add("Program")

        return graph

    def _tokens_to_score(
        self,
        tokens: TokSequence | list[TokSequence],
        programs: list[tuple[int, bool]] | None = None,
    ) -> Score:
        if self.config.one_token_stream_for_programs:
            tokens = [tokens]
        for index, tokens_i in enumerate(tokens):
            tokens[index] = tokens_i.tokens
        score = Score(self.time_division)
        dur_offset = 2 if self.config.use_velocities else 1

        tracks: dict[int, Track] = {}
        tempo_changes: list[Tempo] = []
        time_signature_changes: list[TimeSignature] = []

        def check_inst(program: int) -> None:
            if program not in tracks:
                tracks[program] = Track(
                    program=0 if program == -1 else program,
                    is_drum=program == -1,
                    name="Drums" if program == -1 else MIDI_INSTRUMENTS[program]["name"],
                )

        def is_track_empty(track: Track) -> bool:
            return len(track.notes) == len(track.controls) == len(track.pitch_bends) == 0

        current_track = None
        for sequence_index, sequence in enumerate(tokens):
            if sequence_index == 0:
                if self.config.use_time_signatures:
                    for token in sequence:
                        tok_type, tok_val = token.split("_", 1)
                        if tok_type == "TimeSig":
                            time_signature_changes.append(TimeSignature(0, *self._parse_token_time_signature(tok_val)))
                            break
                        if tok_type in {
                            "Pitch",
                            "PitchDrum",
                            "Velocity",
                            "Duration",
                            "PitchBend",
                            "Pedal",
                            "ControlChange",
                        }:
                            break
                if len(time_signature_changes) == 0:
                    time_signature_changes.append(TimeSignature(0, *TIME_SIGNATURE))

            current_time_sig = time_signature_changes[-1]
            ticks_per_bar = compute_ticks_per_bar(current_time_sig, score.ticks_per_quarter)
            ticks_per_beat = self._tpb_per_ts[current_time_sig.denominator]
            ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)

            current_tick = tick_at_last_ts_change = tick_at_current_bar = 0
            current_bar = -1
            bar_at_last_ts_change = 0
            current_program = 0
            previous_note_end = 0
            previous_pitch_onset = dict.fromkeys(self.config.programs, -128)
            previous_pitch_chord = dict.fromkeys(self.config.programs, -128)
            active_pedals: dict[int, int] = {}
            pending_control_number: int | None = None

            if not self.config.one_token_stream_for_programs:
                is_drum = False
                if programs is not None:
                    current_program, is_drum = programs[sequence_index]
                elif self.config.use_programs:
                    for token in sequence:
                        tok_type, tok_val = token.split("_", 1)
                        if tok_type.startswith("Program"):
                            current_program = int(tok_val)
                            if current_program == -1:
                                is_drum, current_program = True, 0
                            break
                current_track = Track(
                    program=current_program,
                    is_drum=is_drum,
                    name="Drums" if current_program == -1 else MIDI_INSTRUMENTS[current_program]["name"],
                )
            current_track_use_duration = current_program in self.config.use_note_duration_programs

            for token_index, token in enumerate(sequence):
                tok_type, tok_val = token.split("_", 1)
                if token == "Bar_None":
                    current_bar += 1
                    if current_bar > 0:
                        current_tick = tick_at_current_bar + ticks_per_bar
                    tick_at_current_bar = current_tick
                    pending_control_number = None
                elif tok_type == "Rest":
                    current_tick = max(previous_note_end, current_tick)
                    current_tick += self._tpb_rests_to_ticks[ticks_per_beat][tok_val]
                    real_current_bar = bar_at_last_ts_change + self._units_between(tick_at_last_ts_change, current_tick, ticks_per_bar)
                    if real_current_bar > current_bar:
                        if current_bar == -1:
                            current_bar = 0
                        tick_at_current_bar += (real_current_bar - current_bar) * ticks_per_bar
                        current_bar = real_current_bar
                    pending_control_number = None
                elif tok_type == "Position":
                    if current_bar == -1:
                        current_bar = 0
                    current_tick = tick_at_current_bar + int(tok_val) * ticks_per_pos
                    pending_control_number = None
                elif tok_type in {"Pitch", "PitchDrum", "PitchIntervalTime", "PitchIntervalChord"}:
                    if tok_type in {"Pitch", "PitchDrum"}:
                        pitch = int(tok_val)
                    elif tok_type == "PitchIntervalTime":
                        pitch = previous_pitch_onset[current_program] + int(tok_val)
                    else:
                        pitch = previous_pitch_chord[current_program] + int(tok_val)
                    if not self.config.pitch_range[0] <= pitch <= self.config.pitch_range[1]:
                        pending_control_number = None
                        continue

                    if tok_type != "PitchIntervalChord":
                        previous_pitch_onset[current_program] = pitch
                    previous_pitch_chord[current_program] = pitch

                    try:
                        if self.config.use_velocities:
                            vel_type, vel = sequence[token_index + 1].split("_", 1)
                        else:
                            vel_type, vel = "Velocity", 100
                        if current_track_use_duration:
                            dur_type, dur = sequence[token_index + dur_offset].split("_", 1)
                        else:
                            dur_type = "Duration"
                            dur = int(self.config.default_note_duration * ticks_per_beat)
                        if vel_type == "Velocity" and dur_type == "Duration":
                            if isinstance(dur, str):
                                dur = self._tpb_tokens_to_ticks[ticks_per_beat][dur]
                            new_note = Note(current_tick, dur, pitch, int(vel))
                            if self.config.one_token_stream_for_programs:
                                check_inst(current_program)
                                tracks[current_program].notes.append(new_note)
                            else:
                                current_track.notes.append(new_note)
                            previous_note_end = max(previous_note_end, current_tick + dur)
                    except IndexError:
                        pass
                    pending_control_number = None
                elif tok_type == "Program":
                    current_program = int(tok_val)
                    current_track_use_duration = current_program in self.config.use_note_duration_programs
                    if not self.config.one_token_stream_for_programs and self.config.program_changes:
                        if current_program != -1:
                            current_track.program = current_program
                        else:
                            current_track.program = 0
                            current_track.is_drum = True
                elif tok_type == "Tempo":
                    if sequence_index == 0:
                        tempo_changes.append(Tempo(current_tick, float(tok_val)))
                    previous_note_end = max(previous_note_end, current_tick)
                    pending_control_number = None
                elif tok_type == "TimeSig":
                    num, den = self._parse_token_time_signature(tok_val)
                    if num != current_time_sig.numerator or den != current_time_sig.denominator:
                        current_time_sig = TimeSignature(current_tick, num, den)
                        if sequence_index == 0:
                            time_signature_changes.append(current_time_sig)
                        tick_at_last_ts_change = tick_at_current_bar
                        bar_at_last_ts_change = current_bar
                        ticks_per_bar = compute_ticks_per_bar(current_time_sig, score.ticks_per_quarter)
                        ticks_per_beat = self._tpb_per_ts[den]
                        ticks_per_pos = self._compute_ticks_per_pos(ticks_per_beat)
                    pending_control_number = None
                elif tok_type == "Pedal":
                    pedal_program = int(tok_val) if self.config.use_programs else current_program
                    if self.config.sustain_pedal_duration and token_index + 1 < len(sequence):
                        next_type, next_val = sequence[token_index + 1].split("_", 1)
                        if next_type == "Duration":
                            duration = self._tpb_tokens_to_ticks[ticks_per_beat][next_val]
                            new_pedal = Pedal(current_tick, duration)
                            if self.config.one_token_stream_for_programs:
                                check_inst(pedal_program)
                                tracks[pedal_program].pedals.append(new_pedal)
                            else:
                                current_track.pedals.append(new_pedal)
                    elif pedal_program not in active_pedals:
                        active_pedals[pedal_program] = current_tick
                    pending_control_number = None
                elif tok_type == "PedalOff":
                    pedal_program = int(tok_val) if self.config.use_programs else current_program
                    if pedal_program in active_pedals:
                        new_pedal = Pedal(active_pedals[pedal_program], current_tick - active_pedals[pedal_program])
                        if self.config.one_token_stream_for_programs:
                            check_inst(pedal_program)
                            tracks[pedal_program].pedals.append(new_pedal)
                        else:
                            current_track.pedals.append(new_pedal)
                        del active_pedals[pedal_program]
                    pending_control_number = None
                elif tok_type == "PitchBend":
                    new_pitch_bend = PitchBend(current_tick, int(tok_val))
                    if self.config.one_token_stream_for_programs:
                        check_inst(current_program)
                        tracks[current_program].pitch_bends.append(new_pitch_bend)
                    else:
                        current_track.pitch_bends.append(new_pitch_bend)
                    pending_control_number = None
                elif tok_type == "ControlChange":
                    pending_control_number = int(tok_val)
                elif tok_type == "CCValue":
                    if pending_control_number is not None:
                        new_control = ControlChange(current_tick, pending_control_number, self._dequantize_control_value(int(tok_val)))
                        if self.config.one_token_stream_for_programs:
                            check_inst(current_program)
                            tracks[current_program].controls.append(new_control)
                        else:
                            current_track.controls.append(new_control)
                    pending_control_number = None

            if not self.config.one_token_stream_for_programs and not is_track_empty(current_track):
                score.tracks.append(current_track)

        if self.config.one_token_stream_for_programs:
            score.tracks = list(tracks.values())
        score.tempos = tempo_changes
        score.time_signatures = time_signature_changes
        return score
