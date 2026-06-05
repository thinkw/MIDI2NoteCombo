"""
推荐算法模块（v2）。

按八度/自定义区间分组，使用 NNLS 混合优化器为每个音区独立推荐
Minecraft 音符盒乐器组合，支持多乐器音色混合。

支持两种目标音色模式：
- 默认模式：基于 GM 相似度加权 MC 向量构建全局伪目标向量
- --accurate 模式：对每个音区渲染合成音频，提取真实目标向量
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional

from instruments import get_all_instruments, get_instrument_ids
from midi_parser import get_notes_by_octave
from mix_optimizer import (
    filter_candidates_by_range,
    find_best_mix,
    normalize_vector,
)
from nbs_pitch import (
    get_all_instrument_ranges,
    get_nbs_key_range_for_midi,
    nbs_key_to_fsharp,
)


# YAMNet embedding dimension
EMBEDDING_DIM = 1024

# 缓存 accurate 模式失败状态，避免每个音区重复输出错误
_accurate_failed_warned = False

# 缓存 MC 音域映射（模块级别，避免每次调用重建）
_mc_range_cache: Optional[Dict[str, Tuple[int, int]]] = None
# 缓存 MC 向量映射
_mc_vector_cache: Optional[Tuple[np.ndarray, Dict[str, int], Dict[str, np.ndarray]]] = None


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
    """构建 instrument_id -> (low, high) 音域映射（由 nbs_pitch 计算，带缓存）。"""
    global _mc_range_cache
    if _mc_range_cache is None:
        _mc_range_cache = get_all_instrument_ranges()
    return _mc_range_cache


def _build_mc_vector_map(
    mc_vectors: np.ndarray,
    id_to_idx: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """构建 instrument_id -> normalized_vector 映射（带缓存）。"""
    global _mc_vector_cache

    # 以 mc_vectors 的 id() 和 id_to_idx 的键集作为缓存 key
    cache_key = (id(mc_vectors), tuple(sorted(id_to_idx.keys())))
    if _mc_vector_cache is not None:
        cached_key, cached_map = _mc_vector_cache
        if cached_key == cache_key:
            return cached_map

    vec_map = {
        inst_id: normalize_vector(mc_vectors[idx])
        for inst_id, idx in id_to_idx.items()
    }
    _mc_vector_cache = (cache_key, vec_map)
    return vec_map


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


def _render_octave_audio(
    pitches: List[int],
    program: int,
    fs: int = 16000,
    note_duration: float = 0.3,
    max_duration: float = 5.0,
    max_notes: int = 24,
) -> Optional[np.ndarray]:
    """
    将音区内的音符渲染为音频波形（正弦波合成）。

    Args:
        pitches: 该音区内的音符 pitch 列表。
        program: GM 乐器 program 编号。
        fs: 目标采样率（默认 16000）。
        note_duration: 每个音符持续时间（秒）。
        max_duration: 最大音频总时长（秒）。
        max_notes: 最多合成音符数（避免过长）。

    Returns:
        单声道波形 (n_samples,) 或 None。
    """
    import pretty_midi

    try:
        unique_pitches = sorted(set(pitches))[:max_notes]

        # 按时间线排列音符，间隙 0.05 秒
        gap = 0.05
        total_per_note = note_duration + gap

        mid = pretty_midi.PrettyMIDI(initial_tempo=120)
        instr = pretty_midi.Instrument(program=program, is_drum=False)

        for i, pitch in enumerate(unique_pitches):
            start = i * total_per_note
            if start + note_duration > max_duration:
                break
            note = pretty_midi.Note(
                velocity=100, pitch=pitch,
                start=start, end=start + note_duration,
            )
            instr.notes.append(note)

        if not instr.notes:
            # 无有效音符，至少放一个
            note = pretty_midi.Note(
                velocity=100, pitch=unique_pitches[0],
                start=0, end=note_duration,
            )
            instr.notes.append(note)

        mid.instruments.append(instr)

        # 正弦波合成
        audio = mid.synthesize(fs=fs)

        # pretty_midi 可能返回 (n,) 或 (n, 2)，统一转为 mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        # 确保音频至少 0.96 秒（YAMNet 最小窗口）
        min_len = int(0.96 * fs)
        if len(audio) < min_len:
            audio = np.pad(audio, (0, min_len - len(audio)), mode='constant')

        return audio

    except Exception as e:
        print(f"[警告] 音频渲染失败 (program={program}): {e}")
        return None


def _render_fluidsynth_audio(
    notes: List[Dict],
    program: int,
    soundfont_path: str,
    fs: int = 16000,
    max_duration: float = 30.0,
) -> Optional[np.ndarray]:
    """
    使用 FluidSynth + SoundFont 将 MIDI 音符渲染为真实 GM 音色音频。

    Args:
        notes: raw_notes 列表，每个元素为 {"pitch", "velocity", "start", "end"}。
        program: GM 乐器 program 编号。
        soundfont_path: SoundFont 文件路径。
        fs: 目标采样率（默认 16000）。
        max_duration: 最大渲染时长（秒），防止超长音轨。

    Returns:
        单声道波形 (n_samples,) 或 None。
    """
    try:
        import fluidsynth
    except ImportError:
        print("[信息] fluidsynth 未安装，无法使用 FluidSynth 渲染。"
              "请执行: pip install pyfluidsynth")
        return None

    if not notes:
        return None

    if not os.path.isfile(soundfont_path):
        print(f"[警告] SoundFont 文件不存在: {soundfont_path}")
        return None

    synth = fluidsynth.Synth(gain=1.0)
    # 不调用 synth.start() - 离屏渲染不需要音频驱动/MIDI输入设备
    sf_id = synth.sfload(soundfont_path)
    synth.program_select(0, sf_id, 0, program)

    max_end = max(n["end"] for n in notes)
    total_duration = min(max_end + 0.5, max_duration)

    synth_sr = 44100
    buf_size = 1024
    total_buf = int(total_duration * synth_sr / buf_size)

    # 构建事件队列：按时间排序所有 note-on / note-off
    events = []
    for n in notes:
        events.append((n["start"], "on", n["pitch"], n["velocity"]))
        events.append((n["end"], "off", n["pitch"], 0))
    events.sort()

    audio_chunks = []
    event_idx = 0

    for frame in range(total_buf):
        t = frame * buf_size / synth_sr
        while event_idx < len(events) and events[event_idx][0] <= t:
            _, etype, pitch, vel = events[event_idx]
            if etype == "on":
                synth.noteon(0, pitch, vel)
            else:
                synth.noteoff(0, pitch)
            event_idx += 1

        buf = synth.get_samples(buf_size)
        # buf 形状: (2 * buf_size,) (左右声道交替)
        left_channel = buf[::2]
        audio_chunks.extend(left_channel)

    synth.delete()

    audio = np.array(audio_chunks, dtype=np.float32)

    # 重采样到目标采样率（YAMNet 需要 16000Hz）
    try:
        from scipy import signal
        num_samples = int(len(audio) * fs / synth_sr)
        if num_samples > 1:
            audio = signal.resample(audio, num_samples).astype(np.float32)
        else:
            return None
    except ImportError:
        print("[信息] scipy 未安装，无法重采样。请执行: pip install scipy")
        synth.delete()
        return None

    # 确保音频至少 0.96 秒（YAMNet 最小窗口）
    min_len = int(0.96 * fs)
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)), mode='constant')

    return audio


def _build_accurate_target_vector(
    pitches: List[int],
    program: int,
    raw_notes: Optional[List[Dict]] = None,
    soundfont_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    通过渲染合成 + YAMNet 提取，构建精确的目标音色向量。

    优先使用 FluidSynth + SoundFont 渲染真实 GM 音色（最精确），
    SoundFont 不可用时回退到正弦波合成。

    Args:
        pitches: 该音区内的音符 pitch 列表。
        program: GM 乐器 program 编号。
        raw_notes: 原始音符数据（含 pitch/velocity/start/end），用于 FluidSynth 渲染。
        soundfont_path: SoundFont 文件路径，如指定则优先用 FluidSynth 渲染。

    Returns:
        归一化的 (1024,) 目标向量，或 None（渲染/提取失败时）。
    """
    global _accurate_failed_warned

    if _accurate_failed_warned:
        return None

    from utils import extract_audio_embedding

    # 优先：FluidSynth + SoundFont 真实 GM 音色渲染
    if soundfont_path and raw_notes:
        audio = _render_fluidsynth_audio(raw_notes, program, soundfont_path)
        if audio is not None:
            try:
                vector = extract_audio_embedding(audio)
                if vector is not None:
                    return normalize_vector(vector)
            except Exception:
                pass
        print("[信息] FluidSynth 渲染不可用，回退到正弦波合成。")

    # 回退：正弦波合成
    audio = _render_octave_audio(pitches, program)
    if audio is None:
        return None

    try:
        vector = extract_audio_embedding(audio)
    except Exception:
        vector = None

    if vector is None:
        _accurate_failed_warned = True
        print("[警告] YAMNet 模型不可用（请安装 tensorflow 和 tensorflow-hub），"
              "--accurate 模式自动回退到默认模式。")
        return None

    return normalize_vector(vector)


def recommend_for_track(
    track: Dict,
    similarity: Dict[str, Dict[str, float]],
    mc_vectors: np.ndarray,
    id_to_idx: Dict[str, int],
    group_size: int = 12,
    max_instruments: int = 3,
    accurate: bool = False,
    soundfont_path: Optional[str] = None,
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
        accurate: 是否启用渲染合成模式（为每个音区独立渲染目标向量）。
        soundfont_path: SoundFont 文件路径（如 FluidR3_GM.sf2）。
            指定后 accurate 模式优先用 FluidSynth 渲染真实 GM 音色。

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
    raw_notes_free = track.get("raw_notes", [])
    raw_notes = track.get("raw_notes", [])

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

    # 默认模式：构建全局伪目标向量（作为 accurate 模式的 fallback）
    global_target_vec = _build_target_vector(midi_program, similarity, mc_vectors, id_to_idx)

    # 对每个八度组独立推荐
    for octave_key in sorted(octave_groups.keys()):
        pitches = octave_groups[octave_key]
        min_pitch = min(pitches)
        max_pitch = max(pitches)

        # 确定该音区的目标向量
        target_vec = global_target_vec  # 默认使用全局向量

        if accurate and not _accurate_failed_warned:
            # accurate 模式：为该音区渲染真实目标向量
            # 筛选该音区内的原始音符（保留时间和力度信息），用于 FluidSynth 渲染
            group_raw_notes = [
                n for n in raw_notes
                if min_pitch <= n["pitch"] <= max_pitch
            ] if raw_notes else None
            accurate_vec = _build_accurate_target_vector(
                pitches, midi_program,
                raw_notes=group_raw_notes,
                soundfont_path=soundfont_path,
            )
            if accurate_vec is not None:
                target_vec = accurate_vec
            else:
                print(f"[信息] 轨道 {track_index}: 渲染合成不可用，回退到默认模式。")

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


def recommend_for_track_free(
    track: Dict,
    similarity: Dict[str, Dict[str, float]],
    mc_vectors: np.ndarray,
    id_to_idx: Dict[str, int],
    max_instruments: int = 3,
    num_recommendations: int = 3,
    accurate: bool = True,
    soundfont_path: Optional[str] = None,
) -> Dict:
    """
    忽略 MC 音域限制，为整个轨道推荐 top-N 乐器混合组合。

    与 recommend_for_track 的关键区别：
    - 不做音域筛选，全部 17 种非打击乐器都是候选
    - 不按八度拆分，整个轨道为一个整体
    - 返回 num_recommendations 组最佳混合（通过迭代排除已用乐器产生多样性）
    - 每组的 NBS key 可能超出 [32,56]（通过 nbs_outside_mc 字段标注）

    Args:
        track: midi_parser.parse_midi 返回的轨道信息。
        similarity: GM-MC 相似度表。
        mc_vectors: MC 向量矩阵 (n, 1024)。
        id_to_idx: instrument_id -> vector_index 映射。
        max_instruments: 每组最多使用的乐器数。
        num_recommendations: 返回的推荐组数（默认 3）。
        accurate: 是否启用渲染合成模式。
        soundfont_path: SoundFont 文件路径（如 FluidR3_GM.sf2）。
            指定后 accurate 模式优先用 FluidSynth 渲染真实 GM 音色。

    Returns:
        {
            "track_index": int,
            "midi_program": int,
            "midi_instrument_name": str,
            "note_range": [min_pitch, max_pitch],
            "recommendations": [
                {
                    "rank": 1,
                    "instruments": [{"instrument": str, "weight": float, "nbs_range": str, "outside_mc": bool}, ...],
                    "similarity": float,
                },
                ...
            ],
        }
    """
    track_index = track["track_index"]
    midi_program = track["midi_program"]
    midi_instrument_name = track["midi_instrument_name"]
    notes = track.get("notes", [])
    raw_notes_free = track.get("raw_notes", [])

    if not notes:
        return {
            "track_index": track_index,
            "midi_program": midi_program,
            "midi_instrument_name": midi_instrument_name,
            "note_range": [0, 0],
            "recommendations": [],
        }

    pitches = [int(n) for n in notes]
    low_midi = min(pitches)
    high_midi = max(pitches)

    result = {
        "track_index": track_index,
        "midi_program": midi_program,
        "midi_instrument_name": midi_instrument_name,
        "note_range": [low_midi, high_midi],
        "recommendations": [],
    }

    # 构建 MC 数据
    mc_vector_map = _build_mc_vector_map(mc_vectors, id_to_idx)

    # 构建目标向量
    target_vec = _build_target_vector(midi_program, similarity, mc_vectors, id_to_idx)
    if accurate and not _accurate_failed_warned:
        accurate_vec = _build_accurate_target_vector(
            pitches, midi_program,
            raw_notes=raw_notes_free,
            soundfont_path=soundfont_path,
        )
        if accurate_vec is not None:
            target_vec = accurate_vec

    # 候选池 = 全部非打击乐器
    all_ids = get_instrument_ids()
    remaining_ids = set(all_ids)

    for rank in range(1, num_recommendations + 1):
        # 从当前候选池构建向量映射
        cand_vectors = {cid: mc_vector_map[cid] for cid in remaining_ids if cid in mc_vector_map}
        if not cand_vectors:
            break

        best_mix, sim = find_best_mix(target_vec, cand_vectors, max_instruments)
        if not best_mix:
            break

        # 构建该推荐组的详细信息
        instruments_detail = []
        for inst_id, weight in best_mix:
            nbs_lo, nbs_hi = get_nbs_key_range_for_midi(low_midi, high_midi, inst_id)
            nbs_range_str = f"{nbs_key_to_fsharp(nbs_lo)}~{nbs_key_to_fsharp(nbs_hi)}"
            outside_mc = (nbs_lo < 32 or nbs_hi > 56)
            instruments_detail.append({
                "instrument": inst_id,
                "weight": weight,
                "nbs_range": nbs_range_str,
                "outside_mc": outside_mc,
            })

        result["recommendations"].append({
            "rank": rank,
            "instruments": instruments_detail,
            "similarity": sim,
        })

        # 移除权重最高的乐器以产生多样性
        if best_mix:
            top_inst = best_mix[0][0]
            remaining_ids.discard(top_inst)

    return result
