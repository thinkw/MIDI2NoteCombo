"""
Minecraft 原版音符盒乐器数据表。

仅包含可发音高的乐器，打击乐器（snare, basedrum, hat）不参与音高匹配。
"""

from typing import Dict, List, Optional

# 乐器定义:
# - instrument_id: 唯一标识
# - name: 中文名称
# - transpose: 半音偏移量（相对于标准音域 F#3~F#5, MIDI 54~78）
# - actual_low: 实际最低音（MIDI 编号）
# - actual_high: 实际最高音（MIDI 编号）
INSTRUMENT_DEFS: List[Dict] = [
    {"instrument_id": "harp",              "name": "竖琴/钢琴",      "transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "banjo",             "name": "班卓琴",         "transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "bit",               "name": "芯片（方波）",    "transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "pling",             "name": "扣弦（电钢琴）",  "transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "iron_xylophone",    "name": "铁木琴（颤片琴）","transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "trumpet",           "name": "小号（普通）",    "transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "trumpet_exposed",   "name": "小号（斑驳）",    "transpose": 0,   "actual_low": 54, "actual_high": 78},
    {"instrument_id": "trumpet_weathered", "name": "小号（锈蚀）",    "transpose": -12, "actual_low": 42, "actual_high": 66},
    {"instrument_id": "trumpet_oxidized",  "name": "小号（氧化）",    "transpose": -12, "actual_low": 42, "actual_high": 66},
    {"instrument_id": "guitar",            "name": "吉他",           "transpose": -12, "actual_low": 42, "actual_high": 66},
    {"instrument_id": "bass",              "name": "贝斯",           "transpose": -24, "actual_low": 30, "actual_high": 54},
    {"instrument_id": "didgeridoo",        "name": "迪吉里杜管",     "transpose": -24, "actual_low": 30, "actual_high": 54},
    {"instrument_id": "flute",             "name": "长笛",           "transpose": 12,  "actual_low": 66, "actual_high": 90},
    {"instrument_id": "cow_bell",          "name": "牛铃",           "transpose": 12,  "actual_low": 66, "actual_high": 90},
    {"instrument_id": "bell",              "name": "铃铛（钟琴）",    "transpose": 24,  "actual_low": 78, "actual_high": 102},
    {"instrument_id": "icechime",          "name": "管钟",           "transpose": 24,  "actual_low": 78, "actual_high": 102},
    {"instrument_id": "xylobone",          "name": "木琴",           "transpose": 24,  "actual_low": 78, "actual_high": 102},
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


def get_transpose(instrument_id: str) -> Optional[int]:
    """返回乐器的 transposition 半音数。"""
    inst = get_instrument(instrument_id)
    if inst is None:
        return None
    return inst["transpose"]
