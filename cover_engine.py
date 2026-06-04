"""
区间覆盖枚举引擎。

给定目标音域 [target_low, target_high]，枚举所有可行的
MC 乐器组合（1~4 个乐器），使得组合音域的并集覆盖目标音域。
"""

import itertools
from typing import List, Tuple, Optional

from instruments import get_all_instruments, get_instrument_range

# 最大组合中包含的乐器数量
MAX_COMBO_SIZE = 4


def get_covering_combinations(
    target_low: int,
    target_high: int,
    max_instruments: int = MAX_COMBO_SIZE,
) -> List[List[Tuple[str, int, int]]]:
    """
    枚举所有能够覆盖目标音域的 MC 乐器组合。

    Args:
        target_low: 目标最低音（MIDI 编号）。
        target_high: 目标最高音（MIDI 编号）。
        max_instruments: 组合中最多包含的乐器数（默认 4）。

    Returns:
        可行组合的列表。每个组合是一个列表，元素为
        (instrument_id, actual_low, actual_high)。
        按组合中乐器数量升序排列（优先推荐更少乐器的组合）。
    """
    if target_low > target_high:
        return []

    instruments = get_all_instruments()
    # 筛选与目标区间有交集的乐器（且至少有一个音域）
    candidates = []
    for inst in instruments:
        low = inst["actual_low"]
        high = inst["actual_high"]
        if high < target_low or low > target_high:
            continue  # 无交集
        candidates.append((inst["instrument_id"], low, high))

    if not candidates:
        return []

    result = []
    for k in range(1, min(max_instruments, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, k):
            if _covers(combo, target_low, target_high):
                result.append(list(combo))
        # 如果较小组合已找到解，仍继续枚举更大组合（因为可能有更好的音色匹配）
    # 按乐器数量升序排列
    result.sort(key=lambda c: len(c))
    return result


def _covers(
    combo: Tuple[Tuple[str, int, int], ...],
    target_low: int,
    target_high: int,
) -> bool:
    """
    检查一组乐器的音域并集是否完全覆盖目标音域 [target_low, target_high]。
    区间覆盖策略：将每个乐器的音域视为闭区间，合并后检查覆盖。
    """
    # 按 low 排序
    intervals = sorted([(low, high) for _, low, high in combo], key=lambda x: x[0])

    covered_end = target_low - 1  # 当前已覆盖的右端点
    for low, high in intervals:
        if low > covered_end + 1:
            # 存在间隙
            return False
        covered_end = max(covered_end, high)
        if covered_end >= target_high:
            return True
    return covered_end >= target_high


def get_note_distribution(
    notes: List[int],
    combo: List[Tuple[str, int, int]],
) -> List[Tuple[str, int, int, float]]:
    """
    根据音符 pitch 分布，将音符分配到组合内各乐器的音域区间。

    Args:
        notes: 音符 pitch 列表。
        combo: 乐器组合，每个元素为 (inst_id, low, high)。

    Returns:
        [(inst_id, assigned_low, assigned_high, weight), ...]
        weight = 该乐器负责的音符数 / 总音符数
    """
    total_notes = len(notes)
    if total_notes == 0:
        return [(inst_id, low, high, 0.0) for inst_id, low, high in combo]

    result = []
    for inst_id, low, high in combo:
        count = sum(1 for p in notes if low <= p <= high)
        weight = count / total_notes
        result.append((inst_id, low, high, weight))

    return result


def check_uncovered_notes(
    target_low: int,
    target_high: int,
    all_instrument_ranges: List[Tuple[int, int]],
) -> List[int]:
    """
    检查目标音域中是否某些音高无法被任何单个乐器覆盖。

    Args:
        target_low, target_high: 目标音域。
        all_instrument_ranges: 所有乐器的音域列表 [(low, high), ...]。

    Returns:
        无法被覆盖的音高列表。
    """
    uncovered = []
    for pitch in range(target_low, target_high + 1):
        covered = any(low <= pitch <= high for low, high in all_instrument_ranges)
        if not covered:
            uncovered.append(pitch)
    return uncovered
