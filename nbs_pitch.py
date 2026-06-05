"""
NBS (Note Block Studio) 音高引擎。

封装 NBS 的完整音高模型：
- MIDI ↔ NBS key 精确互转（考虑 instrument_key）
- 乐器 instrument_key 映射表
- 音域自动计算
- NBS 八度命名（以 F# 为边界）

核心公式（来自 NBS 格式规范）:
    actual_pitch_cents = (note_key + instrument_key - 45) × 100 + note_pitch
    → 等效 MIDI: midi_note = note_key + instrument_key - 24

校准基准: F#4 = MIDI 66 = NBS key 45, harp 的 instrument_key = 45
    验证: key=45 的 harp 发音 = 45 + 45 - 24 = 66 = F#4 ✓

关键结论:
- 所有乐器在 NBS 中共享相同的 key 键盘 [32, 56]（F#3~F#5）
- instrument_key 决定同一键位在不同乐器上的实际发音八度
- NBS 导入 MIDI 不做自动移调（已验证），音高直接映射
"""

from typing import Dict, Tuple

# ============================================================
# 基础常量
# ============================================================

NBS_FSHARP4_KEY = 45          # NBS key 系统中 F#4 的键位编号
MIDI_FSHARP4 = 66             # F#4 对应的 MIDI 编号

# 公式推导:
#   NBS 规范: pitch = (note_key + instrument_key - 45) * 100 + note_pitch
#   转为 MIDI 半音: midi = (pitch / 100) + 此时忽略 note_pitch（微调分量）
#   → midi = note_key + instrument_key - 45 + ...
#   锚点校准: harp(key=45) 在 note_key=45 时应发出 F#4(MIDI 66)
#   → 66 = 45 + 45 - 45 + offset → offset = 21
#   → 最终公式: midi_note = note_key + instrument_key - 24
#   其中 24 = 45 - 21（即 NBS_FSHARP4_KEY - offset）

# MC 音符盒可用音域 (NBS key 32~56, 即 F#3~F#5, 共 25 个半音)
# 所有乐器在 NBS 中共享同一键盘，通过 instrument_key 在不同八度发音
MC_NBS_KEY_MIN = 32
MC_NBS_KEY_MAX = 56


# ============================================================
# Instrument Key 映射表
#
# instrument_key 决定乐器发音的八度位置。
# 所有乐器在 NBS 中使用相同的 key 区间 [32,56]，
# 通过不同的 instrument_key 值在不同八度实际发音。
#
# harp = 45（基准，与 F#4 对齐）
# 低八度组: 45 - 12 = 33, 45 - 24 = 21
# 高八度组: 45 + 12 = 57, 45 + 24 = 69
# ============================================================

INSTRUMENT_KEY_MAP: Dict[str, int] = {
    # 基准组 (instrument_key = 45)
    "harp":              45,
    "banjo":             45,
    "bit":               45,
    "pling":             45,
    "iron_xylophone":    45,
    "trumpet":           45,
    "trumpet_exposed":   45,
    # 低 1 八度 (45 - 12 = 33)
    "trumpet_weathered": 33,
    "trumpet_oxidized":  33,
    "guitar":            33,
    # 低 2 八度 (45 - 24 = 21)
    "bass":              21,
    "didgeridoo":        21,
    # 高 1 八度 (45 + 12 = 57)
    "flute":             57,
    "cow_bell":          57,
    # 高 2 八度 (45 + 24 = 69)
    "bell":              69,
    "icechime":          69,
    "xylobone":          69,
}


# ============================================================
# 核心转换函数
# ============================================================

def get_instrument_key(instrument_id: str) -> int:
    """获取乐器的 NBS instrument_key。未知乐器默认返回 45。"""
    return INSTRUMENT_KEY_MAP.get(instrument_id, 45)


def nbs_key_to_midi(nbs_key: int, instrument_id: str) -> float:
    """
    NBS key → 实际发音 MIDI 音高。

    midi = nbs_key + instrument_key - 24
    """
    return float(nbs_key + get_instrument_key(instrument_id) - 24)


def midi_to_nbs_key(midi_note: float, instrument_id: str) -> float:
    """
    MIDI 音高 → 指定乐器上对应的 NBS key。

    nbs_key = midi_note - instrument_key + 24
    """
    return float(midi_note - get_instrument_key(instrument_id) + 24)


def get_instrument_range(instrument_id: str) -> Tuple[int, int]:
    """
    乐器实际发音的 MIDI 音域。

    由 NBS key 区间 [32, 56] 通过 instrument_key 映射计算。
    """
    ik = get_instrument_key(instrument_id)
    return (MC_NBS_KEY_MIN + ik - 24, MC_NBS_KEY_MAX + ik - 24)


def get_all_instrument_ranges() -> Dict[str, Tuple[int, int]]:
    """所有乐器的实际发音音域。"""
    return {iid: get_instrument_range(iid) for iid in INSTRUMENT_KEY_MAP}


def get_nbs_key_range_for_midi(
    min_pitch: int,
    max_pitch: int,
    instrument_id: str,
) -> Tuple[float, float]:
    """
    目标 MIDI 区间在指定乐器上对应的 NBS key 区间。
    """
    return (
        midi_to_nbs_key(min_pitch, instrument_id),
        midi_to_nbs_key(max_pitch, instrument_id),
    )


def get_instrument_midpoint(instrument_id: str) -> int:
    """乐器中间音 MIDI 编号。"""
    lo, hi = get_instrument_range(instrument_id)
    return (lo + hi) // 2


def get_octave_offset(instrument_id: str) -> int:
    """
    乐器相对于竖琴的八度偏移量。
    +1 = 比竖琴高一个八度，-1 = 比竖琴低一个八度，0 = 同八度。
    """
    return (get_instrument_key(instrument_id) - NBS_FSHARP4_KEY) // 12


# ============================================================
# NBS F# 八度命名
# ============================================================

def nbs_key_to_fsharp(nbs_key: float) -> str:
    """
    NBS key → F# 八度表示（如 F#3, F#4）。

    NBS 八度以 F# 为边界：F#N 起始于 key = 6 + (N-1) × 12。
    key 32→F#3, key 45→F#4, key 56→F#5。
    """
    octave = int((nbs_key - 6) // 12) + 1
    return f"F#{octave}"


def midi_range_to_nbs_fsharp(
    min_pitch: int,
    max_pitch: int,
    instrument_id: str = "harp",
) -> str:
    """
    MIDI 音高范围 → NBS F# 八度区间表示。

    如 MIDI 60~71 (harp) → "F#3~F#4"
    """
    nbs_lo, nbs_hi = get_nbs_key_range_for_midi(min_pitch, max_pitch, instrument_id)
    return f"{nbs_key_to_fsharp(nbs_lo)}~{nbs_key_to_fsharp(nbs_hi)}"
