"""
推荐算法模块（v2）。

按八度/自定义区间分组，使用 NNLS 混合优化器为每个音区独立推荐
Minecraft 音符盒乐器组合，支持多乐器音色混合。
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

from instruments import get_all_instruments, get_instrument_range, get_instrument_ids
from midi_parser import get_notes_by_octave
from mix_optimizer import (
    filter_candidates_by_range,
    find_best_mix,
    normalize_vector,
)


# YAMNet embedding dimension
EMBEDDING_DIM = 1024


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


def load_mc_vectors(db_dir: str = "db") -> Tuple[np.ndarray, Dict[str, int]]:
    """
    从 FAISS 索引和元数据中加载 MC 乐器向量。

    Args:
        db_dir: 数据库目录。

    Returns:
        (mc_vectors, id_to_index)
        - mc_vectors: numpy 数组 (n_instruments, 1024)
        - id_to_index: dict {instrument_id: vector_index}
    """
    import faiss

    metadata_path = os.path.join(db_dir, "mc_metadata.json")
    index_path = os.path.join(db_dir, "mc_vectors.faiss")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"MC 元数据不存在: {metadata_path}\n"
            "请先运行: python build_mc_db.py"
        )
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"MC 向量索引不存在: {index_path}\n"
            "请先运行: python build_mc_db.py"
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    index = faiss.read_index(index_path)
    # 从 FAISS 索引提取向量矩阵（兼容多个版本）
    dim = index.d
    n = index.ntotal
    try:
        # faiss < 1.9: xb 属性
        vectors = faiss.rev_swig_ptr(index.xb, n * dim).reshape(n, dim).copy()
    except AttributeError:
        try:
            # faiss >= 1.9: codes 属性 + swig_ptr
            vectors = faiss.rev_swig_ptr(
                faiss.cast_integer_to_float_ptr(index.codes),
                n * dim,
            ).reshape(n, dim).copy()
        except Exception:
            try:
                # faiss >= 1.13: vector_to_array
                vectors = faiss.vector_to_array(index.codes).reshape(n, dim).copy()
            except Exception:
                # 最终降级：逐个重建
                vectors = np.array([index.reconstruct(i) for i in range(n)], dtype=np.float32)

    id_to_idx = {entry["instrument_id"]: entry["vector_index"] for entry in metadata}

    return vectors, id_to_idx


def _build_target_vector(
    gm_program: int,
    similarity: Dict[str, Dict[str, float]],
    mc_vectors: np.ndarray,
    id_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    构建 GM 乐器的伪目标向量。

    在 Fallback 模式下（无实际 GM 向量），用相似度加权 MC 向量
    合成伪目标向量：v_target = normalize(Σ sim_i × v_i)

    Args:
        gm_program: GM 乐器 program 编号（0-127）。
        similarity: 相似度表。
        mc_vectors: MC 向量矩阵 (n, 1024)。
        id_to_idx: instrument_id -> vector_index 映射。

    Returns:
        归一化的 (1024,) 目标向量。
    """
    sims = similarity.get(str(gm_program), {})
    all_ids = get_instrument_ids()

    weighted_sum = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    for inst_id in all_ids:
        if inst_id in id_to_idx and inst_id in sims:
            weight = sims[inst_id]
            vec = mc_vectors[id_to_idx[inst_id]]
            weighted_sum += weight * vec

    return normalize_vector(weighted_sum)


def _build_mc_range_map() -> Dict[str, Tuple[int, int]]:
    """构建 instrument_id -> (low, high) 音域映射。"""
    instruments = get_all_instruments()
    return {inst["instrument_id"]: (inst["actual_low"], inst["actual_high"]) for inst in instruments}


def _build_mc_vector_map(
    mc_vectors: np.ndarray,
    id_to_idx: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """构建 instrument_id -> normalized_vector 映射。"""
    return {
        inst_id: normalize_vector(mc_vectors[idx])
        for inst_id, idx in id_to_idx.items()
    }


def _try_split_range(
    min_pitch: int,
    max_pitch: int,
    mc_ranges: Dict[str, Tuple[int, int]],
    split_size: int = 6,
) -> List[Tuple[int, int]]:
    """
    当无乐器能覆盖整个音区时，尝试拆分为更小的子区间。

    Args:
        min_pitch: 最低音。
        max_pitch: 最高音。
        mc_ranges: 乐器音域表。
        split_size: 子区间大小（半音数，默认 6）。

    Returns:
        子区间列表 [(low, high), ...]，每个子区间至少有一个乐器能覆盖。
        若拆分后仍无乐器覆盖，跳过该子区间。
    """
    sub_ranges = []
    current = min_pitch
    while current <= max_pitch:
        sub_end = min(current + split_size - 1, max_pitch)
        # 检查是否有乐器能覆盖该子区间
        candidates = filter_candidates_by_range(current, sub_end, mc_ranges)
        if candidates:
            sub_ranges.append((current, sub_end))
            current = sub_end + 1
        else:
            # 尝试逐半音收缩
            found = False
            for end in range(sub_end, current - 1, -1):
                if filter_candidates_by_range(current, end, mc_ranges):
                    sub_ranges.append((current, end))
                    current = end + 1
                    found = True
                    break
            if not found:
                # 无法覆盖，跳过当前半音
                print(f"[警告] 音高 {current} 无法被任何乐器覆盖，已跳过。")
                current += 1
    return sub_ranges


def recommend_for_track(
    track: Dict,
    similarity: Dict[str, Dict[str, float]],
    mc_vectors: np.ndarray,
    id_to_idx: Dict[str, int],
    group_size: int = 12,
    max_instruments: int = 3,
) -> Dict:
    """
    为单个 MIDI 轨道按音区分区推荐最优乐器混合组合。

    Args:
        track: midi_parser.parse_midi 返回的轨道信息。
        similarity: GM-MC 相似度表。
        mc_vectors: MC 向量矩阵 (n, 1024)。
        id_to_idx: instrument_id -> vector_index 映射。
        group_size: 分组大小（半音数，默认 12 = 一个八度）。
        max_instruments: 每个音区最多使用的乐器数。

    Returns:
        {
            "track_index": int,
            "midi_program": int,
            "midi_instrument_name": str,
            "octave_recommendations": [
                {
                    "octave": int,
                    "note_range": [min_pitch, max_pitch],
                    "instruments": [{"instrument": str, "weight": float}, ...],
                    "similarity": float,
                },
                ...
            ],
            "uncovered_notes": [],
        }
    """
    track_index = track["track_index"]
    midi_program = track["midi_program"]
    midi_instrument_name = track["midi_instrument_name"]
    notes = track.get("notes", [])

    result = {
        "track_index": track_index,
        "midi_program": midi_program,
        "midi_instrument_name": midi_instrument_name,
        "octave_recommendations": [],
        "uncovered_notes": [],
    }

    if not notes:
        return result

    # 按八度/自定义区间分组
    octave_groups = get_notes_by_octave(notes, group_size=group_size)

    # 构建 MC 数据
    mc_ranges = _build_mc_range_map()
    mc_vector_map = _build_mc_vector_map(mc_vectors, id_to_idx)

    # 构建目标向量
    target_vec = _build_target_vector(midi_program, similarity, mc_vectors, id_to_idx)

    # 对每个八度组独立推荐
    for octave_key in sorted(octave_groups.keys()):
        pitches = octave_groups[octave_key]
        min_pitch = min(pitches)
        max_pitch = max(pitches)

        # 筛选能完全覆盖该音区的候选乐器
        candidate_ids = filter_candidates_by_range(min_pitch, max_pitch, mc_ranges)

        if not candidate_ids:
            # 尝试拆分子区间
            print(f"[警告] 轨道 {track_index} 八度 {octave_key} "
                  f"(MIDI {min_pitch}~{max_pitch}) 无乐器可完整覆盖，尝试拆分子区间。")
            sub_ranges = _try_split_range(min_pitch, max_pitch, mc_ranges)
            for sub_low, sub_high in sub_ranges:
                sub_candidates = filter_candidates_by_range(sub_low, sub_high, mc_ranges)
                if sub_candidates:
                    cand_vectors = {cid: mc_vector_map[cid] for cid in sub_candidates}
                    best_mix, sim = find_best_mix(target_vec, cand_vectors, max_instruments)
                    result["octave_recommendations"].append({
                        "octave": octave_key,
                        "note_range": [sub_low, sub_high],
                        "instruments": [{"instrument": inst, "weight": w} for inst, w in best_mix],
                        "similarity": sim,
                    })
            continue

        # 构建候选乐器向量
        cand_vectors = {cid: mc_vector_map[cid] for cid in candidate_ids}

        # NNLS 混合优化
        best_mix, sim = find_best_mix(target_vec, cand_vectors, max_instruments)

        result["octave_recommendations"].append({
            "octave": octave_key,
            "note_range": [min_pitch, max_pitch],
            "instruments": [{"instrument": inst, "weight": w} for inst, w in best_mix],
            "similarity": sim,
        })

    return result
