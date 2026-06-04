"""
多乐器混合优化模块。

使用非负最小二乘法（NNLS）求解最优乐器混合权重，
以逼近目标 GM 音色。
"""

import numpy as np
from scipy.optimize import nnls
from typing import Dict, List, Tuple

from utils import normalize_vector


def filter_candidates_by_range(
    min_pitch: int,
    max_pitch: int,
    mc_ranges: Dict[str, Tuple[int, int]],
    nbs_offset: int = 0,
) -> List[str]:
    """
    筛选能完全覆盖指定音区的 MC 乐器。

    NBS 导入 MIDI 时会将音高整体下移（通过 --nbs_offset 配置），
    匹配时需先将 MIDI 音高映射到 NBS 位置后再与乐器音域比较。

    Args:
        min_pitch: 音区最低音（MIDI 编号）。
        max_pitch: 音区最高音（MIDI 编号）。
        mc_ranges: {instrument_id: (low, high)} 乐器音域表。
        nbs_offset: NBS 导入 MIDI 时的半音偏移量（正数=下移，0=无偏移）。

    Returns:
        候选乐器 ID 列表。
    """
    # 映射到 NBS 位置：NBS 导入时会将 MIDI 音高整体偏移
    nbs_min = min_pitch - nbs_offset
    nbs_max = max_pitch - nbs_offset

    candidates = []
    for inst_id, (low, high) in mc_ranges.items():
        if low <= nbs_min and high >= nbs_max:
            candidates.append(inst_id)
    return candidates


def find_best_mix(
    target_vec: np.ndarray,
    instrument_vectors: Dict[str, np.ndarray],
    max_instruments: int = 3,
) -> Tuple[List[Tuple[str, float]], float]:
    """
    使用 NNLS 求解最优乐器混合权重，最大化与目标向量的余弦相似度。

    Args:
        target_vec: (E,) 归一化目标向量。
        instrument_vectors: {inst_id: (E,) 归一化向量}。
        max_instruments: 最多使用的乐器数（默认 3）。

    Returns:
        ([(inst_id, weight), ...], similarity)
        - 乐器按权重降序排列，weights 和为 1，保留两位小数
        - similarity ∈ [0, 1]，保留两位小数
    """
    if not instrument_vectors:
        return [], 0.0

    ids = list(instrument_vectors.keys())
    # 构建矩阵 A，列为候选乐器向量
    A = np.column_stack([instrument_vectors[i] for i in ids])

    # 非负最小二乘
    weights, _ = nnls(A, target_vec)

    if np.sum(weights) < 1e-10:
        return [], 0.0

    # 选择权重最大的 max_instruments 个
    idx = np.argsort(weights)[::-1][:max_instruments]
    idx = [i for i in idx if weights[i] > 1e-3]
    if not idx:
        return [], 0.0

    # 重新归一化选中乐器的权重
    selected_weights = weights[idx] / np.sum(weights[idx])

    # 只用选中的乐器计算混合向量和相似度
    A_selected = A[:, idx]
    mixed_vec = A_selected @ selected_weights
    mixed_vec_norm = normalize_vector(mixed_vec)

    sim = float(np.dot(mixed_vec_norm, target_vec))
    sim = round(max(0.0, min(1.0, sim)), 2)

    selected = [(ids[i], round(float(selected_weights[j]), 2)) for j, i in enumerate(idx)]

    # 修正四舍五入导致的权重和不为 1.0 的情况
    total = sum(w for _, w in selected)
    if abs(total - 1.0) > 1e-6 and len(selected) > 0:
        # 调整权重最大的乐器使总和为 1.0
        selected.sort(key=lambda x: x[1])
        selected[-1] = (selected[-1][0], round(selected[-1][1] + 1.0 - total, 2))
        selected.sort(key=lambda x: -x[1])

    return selected, sim
