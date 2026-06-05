"""
MIDI 文件解析模块。

使用 pretty_midi 加载 MIDI 文件，提取每个非打击乐轨道的
program 编号、音符列表和音域信息。
"""

import os
from typing import List, Dict, Optional


def parse_midi(midi_path: str) -> List[Dict]:
    """
    解析 MIDI 文件，返回每个非打击乐轨道的摘要。

    Args:
        midi_path: MIDI 文件路径（.mid）。

    Returns:
        轨道信息列表，每个元素包含:
        {
            "track_index": int,
            "midi_program": int,
            "midi_instrument_name": str,
            "note_range": [min_pitch, max_pitch],
            "notes": [pitch1, pitch2, ...],
            "raw_notes": [{"pitch", "velocity", "start", "end"}, ...],
        }
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("请先安装 pretty_midi: pip install pretty_midi")

    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"MIDI 文件不存在: {midi_path}")

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    tracks = []

    for idx, instrument in enumerate(midi_data.instruments):
        # 跳过打击乐轨道（channel 9，即 drum 轨道）
        if instrument.is_drum:
            # 收集打击乐音符用于统计（可输出警告）
            continue

        notes = instrument.notes
        if not notes:
            continue  # 跳过空轨道

        # 确定 program 编号
        midi_program = instrument.program
        if midi_program is None:
            midi_program = 0

        # 提取所有音符的 pitch（转为 Python int，避免 numpy 类型不可 JSON 序列化）
        pitches = sorted([int(note.pitch) for note in notes])
        if not pitches:
            continue

        # 提取完整音符数据（用于 FluidSynth 真实音色渲染）
        raw_notes = [
            {
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "start": float(note.start),
                "end": float(note.end),
            }
            for note in notes
        ]

        min_pitch = int(min(pitches))
        max_pitch = int(max(pitches))

        # 获取 GM 乐器名称
        try:
            midi_instrument_name = pretty_midi.program_to_instrument_name(midi_program)
        except Exception:
            midi_instrument_name = f"Program {midi_program}"

        track_info = {
            "track_index": idx,
            "midi_program": int(midi_program) if midi_program is not None else 0,
            "midi_instrument_name": midi_instrument_name,
            "note_range": [min_pitch, max_pitch],
            "notes": pitches,
            "raw_notes": raw_notes,
        }
        tracks.append(track_info)

    return tracks


def get_note_count_in_range(notes: List[int], low: int, high: int) -> int:
    """
    计算音符列表中落在指定音域 [low, high] 内的音符数量。

    Args:
        notes: 音符 pitch 列表。
        low: 音域下限（包含）。
        high: 音域上限（包含）。

    Returns:
        落在区间内的音符数。
    """
    return sum(1 for p in notes if low <= p <= high)


def get_notes_by_octave(
    notes: List[int],
    group_size: int = 12,
) -> Dict[int, List[int]]:
    """
    按八度（或自定义区间大小）将音符分组。

    Args:
        notes: 音符 pitch 列表。
        group_size: 每组半音数（默认 12 = 一个八度）。

    Returns:
        {octave_number: [pitch1, pitch2, ...]}
        只包含有音符的组，按 octave_number 升序排列。
        例如 group_size=12 时，pitch 60-71 属于 octave 5（MIDI 60//12=5）。
    """
    groups: Dict[int, List[int]] = {}
    for pitch in notes:
        group_key = pitch // group_size
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(pitch)
    # 排序每个组内的音符
    for g in groups:
        groups[g] = sorted(set(groups[g]))
    return groups
