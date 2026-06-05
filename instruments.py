"""
Minecraft 原版音符盒乐器数据表。

仅保留乐器 ID 与中文名称。音域、中间音、transpose 等计算
全部委托给 nbs_pitch.py 的 NBS 公式。
"""

from typing import Dict, List, Optional

# 打击乐器（不参与音高匹配）
PERCUSSION_IDS: List[str] = ["snare", "basedrum", "hat"]

# 乐器定义（精简版：仅 ID + 名称）
INSTRUMENT_DEFS: List[Dict] = [
    {"instrument_id": "harp",              "name": "竖琴/钢琴"},
    {"instrument_id": "banjo",             "name": "班卓琴"},
    {"instrument_id": "bit",               "name": "芯片（方波）"},
    {"instrument_id": "pling",             "name": "扣弦（电钢琴）"},
    {"instrument_id": "iron_xylophone",    "name": "铁木琴（颤片琴）"},
    {"instrument_id": "trumpet",           "name": "小号（普通）"},
    {"instrument_id": "trumpet_exposed",   "name": "小号（斑驳）"},
    {"instrument_id": "trumpet_weathered", "name": "小号（锈蚀）"},
    {"instrument_id": "trumpet_oxidized",  "name": "小号（氧化）"},
    {"instrument_id": "guitar",            "name": "吉他"},
    {"instrument_id": "bass",              "name": "贝斯"},
    {"instrument_id": "didgeridoo",        "name": "迪吉里杜管"},
    {"instrument_id": "flute",             "name": "长笛"},
    {"instrument_id": "cow_bell",          "name": "牛铃"},
    {"instrument_id": "bell",              "name": "铃铛（钟琴）"},
    {"instrument_id": "icechime",          "name": "管钟"},
    {"instrument_id": "xylobone",          "name": "木琴"},
]

# 向后兼容常量
HARP_MIDPOINT = 65  # F4 (legacy, 仅用于旧代码迁移)


def get_instrument(instrument_id: str) -> Optional[Dict]:
    """根据 instrument_id 获取乐器定义。"""
    for inst in INSTRUMENT_DEFS:
        if inst["instrument_id"] == instrument_id:
            return inst.copy()
    return None


def get_all_instruments() -> List[Dict]:
    """获取所有非打击乐器定义列表。"""
    return [{**inst} for inst in INSTRUMENT_DEFS]


def get_instrument_ids() -> List[str]:
    """获取所有乐器 ID 列表。"""
    return [inst["instrument_id"] for inst in INSTRUMENT_DEFS]


def get_instrument_range(instrument_id: str) -> Optional[tuple]:
    """
    返回乐器的实际音域 (low_midi, high_midi)。
    由 nbs_pitch.get_instrument_range() 计算。
    """
    from nbs_pitch import get_instrument_range as _nbs_get_range
    try:
        return _nbs_get_range(instrument_id)
    except KeyError:
        return None


def get_midpoint(instrument_id: str) -> Optional[int]:
    """
    返回乐器中间音 MIDI 编号。
    由 nbs_pitch.get_instrument_midpoint() 计算。
    """
    from nbs_pitch import get_instrument_midpoint as _nbs_get_mid
    try:
        return _nbs_get_mid(instrument_id)
    except KeyError:
        return None


def get_transpose(instrument_id: str) -> Optional[int]:
    """
    [legacy] 返回 instrument_key - 45（等价于旧版 transpose 字段）。
    """
    from nbs_pitch import get_instrument_key
    try:
        return get_instrument_key(instrument_id) - 45
    except KeyError:
        return None
