"""
Minecraft 原版音符盒乐器数据表。

仅包含可发音高的乐器，打击乐器（snare, basedrum, hat）不参与音高匹配。

基准点：以竖琴 (harp) 的中间音 F4 (MIDI 65) 作为参考原点。
每件乐器的中间音相对该原点的偏移量即为 transpose。
"""

from typing import Dict, List, Optional

# 竖琴基准中间音（samples/harp.ogg 音高为 F4）
HARP_MIDPOINT = 65  # F4

# 乐器定义:
# - instrument_id: 唯一标识
# - name: 中文名称
# - transpose: 半音偏移量（该乐器中间音相对于竖琴中间音 F4 的偏移）
# - actual_low: 实际最低音（MIDI 编号）
# - actual_high: 实际最高音（MIDI 编号）
#
# 各乐器音域以中间音为中心，上下各跨 12 半音（一个八度），
# 范围跨度 = 24 半音（两个八度）。
# 竖琴中间音 = F4 (MIDI 65) → 音域 F3~F5 (53~77)
INSTRUMENT_DEFS: List[Dict] = [
    {"instrument_id": "harp",              "name": "竖琴/钢琴",      "transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "banjo",             "name": "班卓琴",         "transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "bit",               "name": "芯片（方波）",    "transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "pling",             "name": "扣弦（电钢琴）",  "transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "iron_xylophone",    "name": "铁木琴（颤片琴）","transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "trumpet",           "name": "小号（普通）",    "transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "trumpet_exposed",   "name": "小号（斑驳）",    "transpose": 0,   "actual_low": 53, "actual_high": 77},
    {"instrument_id": "trumpet_weathered", "name": "小号（锈蚀）",    "transpose": -12, "actual_low": 41, "actual_high": 65},
    {"instrument_id": "trumpet_oxidized",  "name": "小号（氧化）",    "transpose": -12, "actual_low": 41, "actual_high": 65},
    {"instrument_id": "guitar",            "name": "吉他",           "transpose": -12, "actual_low": 41, "actual_high": 65},
    {"instrument_id": "bass",              "name": "贝斯",           "transpose": -24, "actual_low": 29, "actual_high": 53},
    {"instrument_id": "didgeridoo",        "name": "迪吉里杜管",     "transpose": -24, "actual_low": 29, "actual_high": 53},
    {"instrument_id": "flute",             "name": "长笛",           "transpose": 12,  "actual_low": 65, "actual_high": 89},
    {"instrument_id": "cow_bell",          "name": "牛铃",           "transpose": 12,  "actual_low": 65, "actual_high": 89},
    {"instrument_id": "bell",              "name": "铃铛（钟琴）",    "transpose": 24,  "actual_low": 77, "actual_high": 101},
    {"instrument_id": "icechime",          "name": "管钟",           "transpose": 24,  "actual_low": 77, "actual_high": 101},
    {"instrument_id": "xylobone",          "name": "木琴",           "transpose": 24,  "actual_low": 77, "actual_high": 101},
]

# 打击乐器 ID 列表（不参与音高匹配）
PERCUSSION_IDS: List[str] = ["snare", "basedrum", "hat"]


def get_instrument(instrument_id: str) -> Optional[Dict]:
    """根据 instrument_id 获取乐器定义。"""
    for inst in INSTRUMENT_DEFS:
        if inst["instrument_id"] == instrument_id:
            return inst.copy()
    return None


def get_all_instruments() -> List[Dict]:
    """获取所有非打击乐器定义列表（深拷贝）。"""
    return [{**inst} for inst in INSTRUMENT_DEFS]


def get_instrument_ids() -> List[str]:
    """获取所有乐器 ID 列表。"""
    return [inst["instrument_id"] for inst in INSTRUMENT_DEFS]


def get_instrument_range(instrument_id: str) -> Optional[tuple]:
    """
    返回乐器的实际音域 (low_midi, high_midi)。
    None 表示未知乐器。
    """
    inst = get_instrument(instrument_id)
    if inst is None:
        return None
    return (inst["actual_low"], inst["actual_high"])


def get_midpoint(instrument_id: str) -> Optional[int]:
    """返回乐器的中间音 MIDI 编号。"""
    rng = get_instrument_range(instrument_id)
    if rng is None:
        return None
    return (rng[0] + rng[1]) // 2


def get_transpose(instrument_id: str) -> Optional[int]:
    """
    返回乐器中间音相对于竖琴中间音 (F4) 的半音偏移量。

    该值即乐器数据的 transpose 字段，由中间音差值计算：
    transpose = instrument_midpoint - HARP_MIDPOINT (65)
    """
    mid = get_midpoint(instrument_id)
    if mid is None:
        return None
    return mid - HARP_MIDPOINT
