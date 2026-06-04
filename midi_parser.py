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

        # 提取所有音符的 pitch
        pitches = sorted([note.pitch for note in notes])
        if not pitches:
            continue

        min_pitch = min(pitches)
        max_pitch = max(pitches)

        # 获取 GM 乐器名称
        try:
            midi_instrument_name = pretty_midi.program_to_instrument_name(midi_program)
        except Exception:
            midi_instrument_name = f"Program {midi_program}"

        track_info = {
            "track_index": idx,
            "midi_program": midi_program,
            "midi_instrument_name": midi_instrument_name,
            "note_range": [min_pitch, max_pitch],
            "notes": pitches,
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
