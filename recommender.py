"""
推荐算法模块。

综合音域覆盖和音色相似度，为每个 MIDI 轨道推荐最优的
Minecraft 音符盒乐器组合。
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from instruments import get_all_instruments, get_instrument_range
from cover_engine import get_covering_combinations, get_note_distribution, MAX_COMBO_SIZE
from midi_parser import get_note_count_in_range


def load_similarity(db_dir: str = "db") -> Dict[str, Dict[str, float]]:
    """
    加载 GM-MC 相似度表。

    Args:
        db_dir: 数据库目录。

    Returns:
        similarity[gm_program_str][mc_instrument_id] = float
    """
    path = os.path.join(db_dir, "similarity.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"相似度表不存在: {path}\n"
            "请先运行: python build_gm_similarity.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recommend_for_track(
    track: Dict,
    similarity: Dict[str, Dict[str, float]],
    max_instruments: int = MAX_COMBO_SIZE,
) -> Dict:
    """
    为单个 MIDI 轨道推荐最优乐器组合。

    Args:
        track: midi_parser.parse_midi 返回的轨道信息。
        similarity: GM-MC 相似度表。
        max_instruments: 组合中最多乐器数。

    Returns:
        {
            "track_index": int,
            "midi_program": int,
            "midi_instrument_name": str,
            "note_range": [min, max],
            "recommended_combination": [
                {"instrument": str, "note_range_start": int, "note_range_end": int, "transpose": int},
                ...
            ],
            "uncovered_notes": [],
        }
    """
    track_index = track["track_index"]
    midi_program = track["midi_program"]
    midi_instrument_name = track["midi_instrument_name"]
    note_range = track["note_range"]
    notes = track.get("notes", [])
    target_low, target_high = note_range

    result = {
        "track_index": track_index,
        "midi_program": midi_program,
        "midi_instrument_name": midi_instrument_name,
        "note_range": note_range,
        "recommended_combination": [],
        "uncovered_notes": [],
    }

    # 获取该 GM program 的相似度向量
    gm_sims = similarity.get(str(midi_program), {})
    if not gm_sims:
        # 使用默认相似度（所有乐器均匀低分）
        from instruments import get_instrument_ids
        mc_ids = get_instrument_ids()
        gm_sims = {inst_id: 0.1 for inst_id in mc_ids}

    # Step 1: 获取可行组合
    combos = get_covering_combinations(target_low, target_high, max_instruments)

    if not combos:
        # Step 2: 音域不足，尝试八度平移建议
        result["uncovered_notes"] = _octave_shift_recommendations(
            notes, target_low, target_high
        )
        # 如果八度平移后仍有无法覆盖的，尝试以最大可用组合近似
        return result

    # Step 3: 计算每个组合的加权相似度得分
    best_combo = None
    best_score = -1.0

    for combo in combos:
        # 计算音符分布权重
        distribution = get_note_distribution(notes, combo)
        score = 0.0
        for inst_id, low, high, weight in distribution:
            inst_sim = gm_sims.get(inst_id, 0.0)
            score += weight * inst_sim

        if score > best_score:
            best_score = score
            best_combo = combo

    if best_combo is None:
        return result

    # Step 4: 构建推荐结果
    distribution = get_note_distribution(notes, best_combo)
    from instruments import get_transpose

    for inst_id, low, high, weight in distribution:
        transpose = get_transpose(inst_id) or 0
        result["recommended_combination"].append({
            "instrument": inst_id,
            "note_range_start": low,
            "note_range_end": high,
            "transpose": transpose,
        })

    return result


def _octave_shift_recommendations(
    notes: List[int],
    target_low: int,
    target_high: int,
) -> List[Dict]:
    """
    对超出 MC 乐器音域的音符给出八度平移建议。

    Returns:
        [{"pitch": int, "shift_advice": str}, ...]
    """
    from instruments import get_instrument_range

    instruments = get_all_instruments()
    # 所有乐器的最小/最大音域
    all_mins = [inst["actual_low"] for inst in instruments]
    all_maxs = [inst["actual_high"] for inst in instruments]
    global_min = min(all_mins) if all_mins else 30
    global_max = max(all_maxs) if all_maxs else 102

    # 找出每个音符最近的可覆盖音域
    uncovered = []
    for pitch in sorted(set(notes)):
        if global_min <= pitch <= global_max:
            continue  # 已在全局可覆盖范围内

        advice_parts = []
        if pitch < global_min:
            # 尝试上移八度
            shifted = pitch
            octave = 0
            while shifted < global_min and octave < 3:
                shifted += 12
                octave += 1
            if global_min <= shifted <= global_max:
                # 找到最近的乐器
                best_inst = _find_closest_instrument(shifted)
                advice_parts.append(f"上移 {octave} 个八度到 {shifted} ({best_inst})")
            else:
                advice_parts.append("无法通过八度平移覆盖（音高过低）")
        elif pitch > global_max:
            # 尝试下移八度
            shifted = pitch
            octave = 0
            while shifted > global_max and octave < 3:
                shifted -= 12
                octave += 1
            if global_min <= shifted <= global_max:
                best_inst = _find_closest_instrument(shifted)
                advice_parts.append(f"下移 {octave} 个八度到 {shifted} ({best_inst})")
            else:
                advice_parts.append("无法通过八度平移覆盖（音高过高）")

        uncovered.append({
            "pitch": pitch,
            "shift_advice": "; ".join(advice_parts),
        })

    return uncovered


def _find_closest_instrument(pitch: int) -> str:
    """找到能覆盖指定 pitch 的乐器中音域最匹配的。"""
    from instruments import get_all_instruments
    instruments = get_all_instruments()
    best = None
    best_dist = float("inf")
    for inst in instruments:
        low = inst["actual_low"]
        high = inst["actual_high"]
        if low <= pitch <= high:
            # 距离中心越近越好
            center = (low + high) / 2
            dist = abs(pitch - center)
            if dist < best_dist:
                best_dist = dist
                best = inst["instrument_id"]
    return best or "未知"
