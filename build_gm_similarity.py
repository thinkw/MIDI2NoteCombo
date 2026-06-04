"""
GM 乐器相似度表离线构建工具。

计算 128 个 General MIDI 乐器与 MC 音符盒乐器之间的音色相似度。
优先尝试使用 FluidSynth 合成 GM 音色，若不可用则回退到手工映射表。
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, Optional

from instruments import get_all_instruments, get_instrument_ids
from utils import extract_vggish_embedding, load_audio, ensure_dir


# ============================================================
# Fallback: 手工近似映射表
# 基于乐器族系和听觉相似度粗略估计，相似度值在 [0, 1] 之间
# ============================================================
FALLBACK_SIMILARITY: Dict[int, Dict[str, float]] = {}

def _init_fallback():
    """初始化手工相似度映射表。"""
    global FALLBACK_SIMILARITY

    # 辅助函数: 为某个 GM program 设置对指定 MC 乐器的相似度
    def set_sim(gm_prog: int, mapping: Dict[str, float]):
        FALLBACK_SIMILARITY[gm_prog] = mapping

    # Piano family (0-7)
    set_sim(0, {"harp": 1.0, "piano": 1.0, "pling": 0.5, "iron_xylophone": 0.4, "banjo": 0.3})
    set_sim(1, {"harp": 1.0, "piano": 1.0, "pling": 0.4, "iron_xylophone": 0.3, "banjo": 0.3})
    set_sim(2, {"harp": 0.9, "piano": 1.0, "pling": 0.5, "iron_xylophone": 0.4, "banjo": 0.3})
    set_sim(3, {"harp": 0.9, "piano": 1.0, "pling": 0.4, "iron_xylophone": 0.3, "banjo": 0.2})
    set_sim(4, {"harp": 0.7, "piano": 0.8, "pling": 0.7, "iron_xylophone": 0.5, "bit": 0.5})
    set_sim(5, {"harp": 0.7, "piano": 0.8, "pling": 0.7, "iron_xylophone": 0.5, "bit": 0.5})
    set_sim(6, {"harp": 0.8, "piano": 0.8, "pling": 0.5, "iron_xylophone": 0.4})
    set_sim(7, {"harp": 0.7, "piano": 0.7, "pling": 0.5, "iron_xylophone": 0.4})

    # Chromatic Percussion (8-15)
    set_sim(8, {"iron_xylophone": 0.9, "xylobone": 0.8, "icechime": 0.5})
    set_sim(9, {"iron_xylophone": 0.8, "xylobone": 0.7, "icechime": 0.5})
    set_sim(10, {"iron_xylophone": 0.8, "xylobone": 0.7, "icechime": 0.5})
    set_sim(11, {"xylobone": 1.0, "iron_xylophone": 0.9, "icechime": 0.5})
    set_sim(12, {"xylobone": 0.8, "iron_xylophone": 0.7, "icechime": 0.6})
    set_sim(13, {"xylobone": 0.9, "iron_xylophone": 0.6, "icechime": 0.5})
    set_sim(14, {"icechime": 1.0, "bell": 0.8, "cow_bell": 0.6})
    set_sim(15, {"icechime": 0.9, "bell": 0.7, "cow_bell": 0.5})

    # Organ (16-23)
    set_sim(16, {"harp": 0.5, "piano": 0.5, "flute": 0.3})
    set_sim(17, {"harp": 0.5, "piano": 0.4})
    set_sim(18, {"harp": 0.4, "bit": 0.4})
    set_sim(19, {"harp": 0.5, "piano": 0.4, "flute": 0.3})
    set_sim(20, {"harp": 0.4, "bit": 0.5})
    set_sim(21, {"harp": 0.4, "bit": 0.5})
    set_sim(22, {"flute": 0.4})
    set_sim(23, {"flute": 0.4, "harp": 0.3})

    # Guitar (24-31)
    set_sim(24, {"guitar": 0.9, "banjo": 0.8, "harp": 0.6})
    set_sim(25, {"guitar": 1.0, "banjo": 0.8, "harp": 0.5})
    set_sim(26, {"guitar": 0.9, "banjo": 0.7, "harp": 0.5})
    set_sim(27, {"guitar": 0.8, "banjo": 0.7, "harp": 0.4})
    set_sim(28, {"guitar": 0.7, "banjo": 0.6, "harp": 0.3})
    set_sim(29, {"guitar": 0.7, "banjo": 0.6, "harp": 0.3})
    set_sim(30, {"guitar": 0.7, "harp": 0.4})
    set_sim(31, {"guitar": 0.7, "harp": 0.4})

    # Bass (32-39)
    set_sim(32, {"bass": 0.9, "guitar": 0.4})
    set_sim(33, {"bass": 0.9, "guitar": 0.4})
    set_sim(34, {"bass": 0.8, "guitar": 0.4})
    set_sim(35, {"bass": 0.8, "guitar": 0.3})
    set_sim(36, {"bass": 0.7, "guitar": 0.3})
    set_sim(37, {"bass": 0.7, "guitar": 0.3})
    set_sim(38, {"bass": 0.8, "guitar": 0.3})
    set_sim(39, {"bass": 0.8, "guitar": 0.3})

    # Strings (40-47) & Ensemble (48-55)
    set_sim(40, {"harp": 0.7, "flute": 0.5, "piano": 0.5})
    set_sim(41, {"harp": 0.7, "flute": 0.5, "piano": 0.5})
    set_sim(42, {"harp": 0.6, "flute": 0.5, "piano": 0.5})
    set_sim(43, {"harp": 0.6, "flute": 0.5, "piano": 0.4})
    set_sim(44, {"harp": 0.6, "flute": 0.5, "piano": 0.5})
    set_sim(45, {"harp": 0.5, "flute": 0.4, "piano": 0.5})
    set_sim(46, {"harp": 0.8, "banjo": 0.5, "piano": 0.6})
    set_sim(47, {"harp": 0.7, "banjo": 0.5, "piano": 0.6})
    set_sim(48, {"harp": 0.6, "flute": 0.5})
    set_sim(49, {"harp": 0.6, "flute": 0.5})
    set_sim(50, {"flute": 0.5, "harp": 0.4})
    set_sim(51, {"flute": 0.5, "harp": 0.4})
    set_sim(52, {"flute": 0.5, "harp": 0.4})
    set_sim(53, {"flute": 0.5, "harp": 0.4})
    set_sim(54, {"flute": 0.4})
    set_sim(55, {"flute": 0.4})

    # Brass (56-63)
    set_sim(56, {"trumpet": 0.9, "trumpet_exposed": 0.8,
                 "trumpet_weathered": 0.6, "trumpet_oxidized": 0.6})
    set_sim(57, {"trumpet": 0.9, "trumpet_exposed": 0.8,
                 "trumpet_weathered": 0.6, "trumpet_oxidized": 0.6})
    set_sim(58, {"trumpet": 0.8, "trumpet_exposed": 0.8,
                 "trumpet_weathered": 0.5, "trumpet_oxidized": 0.5})
    set_sim(59, {"trumpet": 0.8, "trumpet_exposed": 0.7,
                 "trumpet_weathered": 0.5})
    set_sim(60, {"trumpet": 0.7, "trumpet_exposed": 0.7,
                 "trumpet_weathered": 0.5})
    set_sim(61, {"trumpet": 0.7, "trumpet_exposed": 0.6})
    set_sim(62, {"trumpet": 0.7, "trumpet_exposed": 0.6})
    set_sim(63, {"trumpet": 0.7, "trumpet_exposed": 0.6})

    # Reed (64-71)
    set_sim(64, {"flute": 0.5, "harp": 0.3})
    set_sim(65, {"flute": 0.5, "harp": 0.3})
    set_sim(66, {"flute": 0.5, "harp": 0.3})
    set_sim(67, {"flute": 0.5, "harp": 0.3})
    set_sim(68, {"flute": 0.6, "harp": 0.3})
    set_sim(69, {"flute": 0.6, "harp": 0.3})
    set_sim(70, {"flute": 0.5, "harp": 0.3})
    set_sim(71, {"flute": 0.6, "harp": 0.3})

    # Pipe (72-79)
    set_sim(72, {"flute": 0.9, "harp": 0.3})
    set_sim(73, {"flute": 0.9, "harp": 0.3})
    set_sim(74, {"flute": 0.8, "harp": 0.3})
    set_sim(75, {"flute": 0.8, "harp": 0.3})
    set_sim(76, {"flute": 0.7, "harp": 0.3})
    set_sim(77, {"flute": 0.7, "harp": 0.3})
    set_sim(78, {"flute": 0.6, "didgeridoo": 0.7, "bass": 0.5})
    set_sim(79, {"flute": 0.6, "didgeridoo": 0.7, "bass": 0.5})

    # Synth Lead (80-87)
    set_sim(80, {"bit": 0.9, "harp": 0.4})
    set_sim(81, {"bit": 0.9, "harp": 0.4})
    set_sim(82, {"bit": 0.7, "flute": 0.4})
    set_sim(83, {"bit": 0.7, "flute": 0.4})
    set_sim(84, {"bit": 0.6, "flute": 0.4})
    set_sim(85, {"harp": 0.5, "banjo": 0.4, "bit": 0.4})
    set_sim(86, {"bit": 0.7, "harp": 0.4})
    set_sim(87, {"bit": 0.7, "harp": 0.4})

    # Synth Pad (88-95)
    set_sim(88, {"harp": 0.3, "bit": 0.4})
    set_sim(89, {"harp": 0.3, "bit": 0.4})
    set_sim(90, {"harp": 0.3, "bit": 0.4})
    set_sim(91, {"harp": 0.3, "bit": 0.4})
    set_sim(92, {"bit": 0.4})
    set_sim(93, {"bit": 0.4})
    set_sim(94, {"bit": 0.4})
    set_sim(95, {"bit": 0.4})

    # Synth SFX (96-103)
    set_sim(96, {"bit": 0.5})
    set_sim(97, {"bit": 0.5})
    set_sim(98, {"bit": 0.5})
    set_sim(99, {"bit": 0.5})
    set_sim(100, {"bit": 0.5})
    set_sim(101, {"bit": 0.5})
    set_sim(102, {"bit": 0.5})
    set_sim(103, {"bit": 0.5})

    # Ethnic (104-111)
    set_sim(104, {"banjo": 1.0, "guitar": 0.7, "harp": 0.5})
    set_sim(105, {"banjo": 0.9, "guitar": 0.6, "harp": 0.5})
    set_sim(106, {"banjo": 0.8, "guitar": 0.5, "harp": 0.4})
    set_sim(107, {"banjo": 0.7, "guitar": 0.5, "harp": 0.4})
    set_sim(108, {"flute": 0.7, "harp": 0.3})
    set_sim(109, {"flute": 0.8, "harp": 0.3})
    set_sim(110, {"flute": 0.7, "harp": 0.3})
    set_sim(111, {"flute": 0.7, "harp": 0.3})

    # Percussive (112-119)
    set_sim(112, {"xylobone": 0.9, "bell": 0.7, "iron_xylophone": 0.6})
    set_sim(113, {"bell": 0.9, "cow_bell": 0.8, "icechime": 0.8, "xylobone": 0.6})
    set_sim(114, {"xylobone": 0.8, "bell": 0.7})
    set_sim(115, {"xylobone": 0.9, "bell": 0.7})
    set_sim(116, {"icechime": 0.8, "bell": 0.7, "cow_bell": 0.6})
    set_sim(117, {"icechime": 0.8, "bell": 0.7, "cow_bell": 0.6})
    set_sim(118, {"cow_bell": 0.7, "bell": 0.6})
    set_sim(119, {"cow_bell": 0.7, "bell": 0.6})

    # Sound effects (120-127)
    # 这些在 Minecraft 中没有很好的对应，给予较低的均匀分布相似度
    for i in range(120, 128):
        mc_ids = get_instrument_ids()
        fallback = {inst_id: 0.15 for inst_id in mc_ids}
        # 稍微偏向 bit（方波）和 bell
        if "bit" in fallback:
            fallback["bit"] = 0.25
        if "bell" in fallback:
            fallback["bell"] = 0.25
        FALLBACK_SIMILARITY[i] = fallback

    # 为所有未设置的 GM program 提供默认值
    mc_ids = get_instrument_ids()
    for i in range(128):
        if i not in FALLBACK_SIMILARITY:
            FALLBACK_SIMILARITY[i] = {inst_id: 0.1 for inst_id in mc_ids}


# 初始化
_init_fallback()


def _generate_sine_tone(midi_note: int, duration: float = 1.0, sr: int = 16000) -> np.ndarray:
    """
    生成指定 MIDI 音高的正弦波（简易合成），用于 FuildSynth 不可用时的 fallback。

    注意: 这是极简版本，仅作音高参考；实际音色对比仍依赖手工映射表。
    """
    freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)


def try_fluidsynth(samples_dir: str, db_dir: str) -> bool:
    """
    尝试使用 FluidSynth 合成 GM 乐器音色并计算相似度。
    需要安装 fluidsynth 和 soundfont 文件。

    Returns:
        True 如果成功生成了相似度表。
    """
    import subprocess
    import tempfile

    # 检查 fluidsynth 是否可用
    try:
        import fluidsynth
    except ImportError:
        print("[信息] fluidsynth 未安装，使用手工映射表。")
        return False

    # 尝试查找 SoundFont
    soundfont_candidates = [
        "GeneralUser_GS.sf2",
        "FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/TimGM6mb.sf2",
    ]
    soundfont = None
    for sf_path in soundfont_candidates:
        if os.path.isfile(sf_path):
            soundfont = sf_path
            break

    if soundfont is None:
        print("[信息] 未找到 SoundFont 文件，使用手工映射表。")
        return False

    print(f"[信息] 使用 SoundFont: {soundfont}")

    # 加载 MC 乐器向量
    mc_metadata_path = os.path.join(db_dir, "mc_metadata.json")
    mc_index_path = os.path.join(db_dir, "mc_vectors.faiss")

    if not os.path.isfile(mc_metadata_path):
        print("[错误] 请先运行 build_mc_db.py 构建 MC 乐器向量库。")
        return False

    try:
        import faiss
    except ImportError:
        print("[信息] faiss 未安装，使用手工映射表。")
        return False

    with open(mc_metadata_path, "r", encoding="utf-8") as f:
        mc_metadata = json.load(f)
    mc_index = faiss.read_index(mc_index_path)

    # 重建 MC 向量矩阵（从 faiss 索引提取）
    mc_vectors = faiss.rev_swig_ptr(mc_index.xb, mc_index.ntotal * 128).reshape(mc_index.ntotal, 128).copy()

    # 合成 128 个 GM 乐器并提取向量
    fs = fluidsynth.Synth()
    fs.start()
    sf_id = fs.sfload(soundfont)
    fs.program_select(0, sf_id, 0, 0)

    similarity = {}
    for gm_prog in range(128):
        fs.program_change(0, gm_prog)
        # 合成中央 C (MIDI 60)
        # 使用 fluidsynth 的 noteon/noteoff
        fs.noteon(0, 60, 100)
        samples = []
        sr = 44100
        for _ in range(int(1.0 * sr / 1024)):
            buf = fs.get_samples(1024)
            samples.extend(buf[::2])  # left channel only
        fs.noteoff(0, 60)

        audio = np.array(samples, dtype=np.float32)
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)

        vector = extract_vggish_embedding(audio)
        if vector is None:
            continue

        # 归一化
        vector = vector / (np.linalg.norm(vector) or 1.0)

        sims = {}
        for mc_entry in mc_metadata:
            mc_id = mc_entry["instrument_id"]
            idx = mc_entry["vector_index"]
            mc_vec = mc_vectors[idx]
            sim = np.dot(vector, mc_vec)
            sims[mc_id] = float(max(0.0, sim))

        similarity[str(gm_prog)] = sims

    fs.delete()

    # 确保 key 是字符串（JSON 需要）
    similarity_json = {str(k): v for k, v in similarity.items()}
    return save_result(similarity_json, db_dir)


def build_with_fallback(db_dir: str) -> bool:
    """使用手工映射表生成相似度文件。"""
    import json

    # FALLBACK_SIMILARITY 的 key 是 int，需要转为 string
    similarity = {}
    for gm_prog in range(128):
        prog_str = str(gm_prog)
        if gm_prog in FALLBACK_SIMILARITY:
            similarity[prog_str] = FALLBACK_SIMILARITY[gm_prog]
        else:
            mc_ids = get_instrument_ids()
            similarity[prog_str] = {inst_id: 0.1 for inst_id in mc_ids}

    return save_result(similarity, db_dir)


def save_result(similarity: Dict[str, Dict[str, float]], db_dir: str) -> bool:
    """保存相似度表到 JSON 文件。"""
    ensure_dir(db_dir)
    output_path = os.path.join(db_dir, "similarity.json")

    # 嵌套字典格式化
    for gm_prog, sims in similarity.items():
        similarity[gm_prog] = dict(sorted(sims.items(), key=lambda x: -x[1]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(similarity, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 相似度表已保存到: {output_path}")
    print(f"  - GM 乐器数: {len(similarity)}")
    mc_count = len(next(iter(similarity.values()), {}))
    print(f"  - MC 乐器数（每个GM）: {mc_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description="构建 GM-MC 乐器相似度表")
    parser.add_argument("--samples_dir", default="samples",
                        help="MC 采样目录（FluidSynth 模式需要，默认: samples）")
    parser.add_argument("--db_dir", default="db",
                        help="数据库目录（默认: db）")
    parser.add_argument("--use_fluidsynth", action="store_true",
                        help="尝试使用 FluidSynth 生成相似度（需安装 fluidsynth 和 SoundFont）")
    parser.add_argument("--fallback", action="store_true",
                        help="直接使用手工映射表")
    args = parser.parse_args()

    # 默认使用 fallback，除非明确指定 FluidSynth 模式
    if args.use_fluidsynth:
        success = try_fluidsynth(
            samples_dir=args.samples_dir,
            db_dir=args.db_dir,
        )
        if not success:
            print("[信息] FluidSynth 模式失败，回退到手工映射表...")
            success = build_with_fallback(db_dir=args.db_dir)
    else:
        success = build_with_fallback(db_dir=args.db_dir)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
