#!/usr/bin/env python3
"""
MIDI2NoteCombo - Minecraft 原版音符盒乐器组合推荐工具

输入 MIDI 文件，输出每个轨道应使用的音符盒乐器组合，
以在 Note Block Studio 中最大限度还原原曲的音域和音色。

用法:
    # 先离线构建数据库
    python build_mc_db.py
    python build_gm_similarity.py

    # 再运行推荐
    python midi2notecombo.py --midi input.mid --output result.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, TextIO

# 减少 TensorFlow 日志输出
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from midi_parser import parse_midi
from recommender import load_similarity, load_mc_vectors, recommend_for_track, recommend_for_track_free
from utils import ensure_dir
from nbs_pitch import (
    get_instrument_key,
    get_octave_offset,
    get_nbs_key_range_for_midi,
    midi_range_to_nbs_fsharp,
    nbs_key_to_fsharp,
)


class Tee:
    """同时输出到控制台和日志文件。"""

    def __init__(self, file: TextIO):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def check_databases(db_dir: str) -> bool:
    """检查必需的数据库文件是否存在。"""
    required = [
        "mc_vectors.faiss",
        "mc_metadata.json",
        "similarity.json",
    ]
    missing = []
    for fname in required:
        path = os.path.join(db_dir, fname)
        if not os.path.isfile(path):
            missing.append(fname)

    if missing:
        print("[错误] 缺少以下数据库文件:")
        for fname in missing:
            print(f"  - {os.path.join(db_dir, fname)}")
        print()
        print("请先运行以下命令构建数据库:")
        print("  python build_mc_db.py")
        print("  python build_gm_similarity.py")
        return False
    return True


def print_summary(results: List[Dict], verbose: bool = False) -> None:
    """
    在控制台打印人类可读的摘要。

    所有推荐乐器的 NBS key 区间均在 MC 可播放范围 F#3~F#5 内。
    不同 instrument_key 的乐器通过八度偏移将同一段 MIDI 音高
    映射到 NBS key 32~56 的不同子区间，确保所有推荐实际可播放。
    """
    print("\n" + "=" * 60)
    print("  MIDI2NoteCombo —— 音符盒乐器组合推荐结果")
    print("=" * 60)

    if not results:
        print("  (无结果)")
        return

    for track_result in results:
        ti = track_result["track_index"]
        prog = track_result["midi_program"]
        name = track_result["midi_instrument_name"]
        octave_recs = track_result.get("octave_recommendations", [])
        uncovered = track_result.get("uncovered_notes", [])

        print(f"\n--- 轨道 {ti}: {name} (Program {prog}) ---")

        if octave_recs:
            print(f"  共 {len(octave_recs)} 个音区推荐:")
            for rec in octave_recs:
                octave = rec["octave"]
                note_range = rec["note_range"]
                instruments = rec["instruments"]
                sim = rec["similarity"]

                # 以 harp 为参照的 NBS 区间，用于判断是否需要低/高音域乐器
                nbs_range = midi_range_to_nbs_fsharp(
                    note_range[0], note_range[1], "harp"
                )
                # 若 harp 参照区间超出 MC F#3~F#5，说明必须用偏移乐器"托举"
                harp_in_mc = (
                    "超出MC音域需偏移乐器覆盖"
                    if note_range[0] < 53 or note_range[1] > 77
                    else "标准音域乐器可覆盖"
                )
                print(f"\n  [八度 {octave}] MIDI {note_range[0]}~{note_range[1]} "
                      f"(harp参照: {nbs_range}, {harp_in_mc}):")
                print(f"     匹配度: {sim}")

                if instruments:
                    print(f"     推荐乐器 ({len(instruments)} 个):")
                    for item in instruments:
                        inst_id = item["instrument"]
                        weight = item["weight"]

                        # 乐器 NBS 信息
                        inst_key = get_instrument_key(inst_id)
                        oct_off = get_octave_offset(inst_id)

                        if oct_off < 0:
                            tone_info = f"key={inst_key} (低{abs(oct_off)}八度)"
                        elif oct_off > 0:
                            tone_info = f"key={inst_key} (高{oct_off}八度)"
                        else:
                            tone_info = f"key={inst_key} (同八度)"

                        # 目标 MIDI 区间在此乐器上的 NBS key 范围
                        # 不变式: 因 filter_candidates_by_range 要求完整覆盖 MC 音域，
                        #   通过筛选的乐器其 NBS key 必定在 [32, 56] (F#3~F#5) 内
                        nbs_lo, nbs_hi = get_nbs_key_range_for_midi(
                            note_range[0], note_range[1], inst_id
                        )
                        nbs_key_info = f"NBS key [{nbs_key_to_fsharp(nbs_lo)}~{nbs_key_to_fsharp(nbs_hi)}]"
                        # 安全检查：若超出 MC 音域则输出警告
                        if nbs_lo < 32 or nbs_hi > 56:
                            nbs_key_info += f" [警告: 超出MC可播放音域F#3~F#5!]"

                        print(f"       - {inst_id:20s}  权重: {weight:.2f}  "
                              f"{tone_info}  {nbs_key_info}")
                else:
                    print(f"     推荐乐器: 无可行组合")
        else:
            print("  无音区推荐")

        if uncovered:
            print(f"\n  [警告] 无法覆盖的音符 ({len(uncovered)} 个):")
            for unc in uncovered:
                if isinstance(unc, dict):
                    print(f"    - MIDI {unc['pitch']}: {unc.get('shift_advice', '无法覆盖')}")
                else:
                    print(f"    - MIDI {unc}")

    print("\n" + "=" * 60)


def print_summary_free(results: List[Dict], verbose: bool = False) -> None:
    """
    自由推荐模式的打印函数：每个轨道 3 组推荐以 Markdown 表格输出。

    输出格式：
        # MIDI2NoteCombo —— 自由推荐模式

        ## 轨道 N: Instrument (Program X), MIDI Lo~Hi

        | 推荐 #1 (相似度: 0.987) | 推荐 #2 (相似度: 0.954) | 推荐 #3 (相似度: 0.921) |
        |:---|:---|:---|
        | harp (0.60) [F#3~F#5] | pling (0.55) [F#3~F#5] | bit (0.50) [F#3~F#5] |
        | bass (0.40) [F#5~F#7] [!] | guitar (0.45) [F#2~F#4] [!] | flute (0.50) [F#3~F#5] |

    [!] 表示 NBS key 超出 MC F#3~F#5 音域。
    """
    print()
    print("# MIDI2NoteCombo —— 自由推荐模式（忽略 MC 音域限制）\n")

    if not results:
        print("_(无结果)_")
        return

    for track_result in results:
        ti = track_result["track_index"]
        prog = track_result["midi_program"]
        name = track_result["midi_instrument_name"]
        note_range = track_result.get("note_range", [0, 0])
        recs = track_result.get("recommendations", [])

        print(f"## 轨道 {ti}: {name} (Program {prog}), MIDI {note_range[0]}~{note_range[1]}\n")

        if not recs:
            print("_(无推荐)_\n")
            continue

        # 构建表格头
        headers = [f"推荐 #{r['rank']} (相似度: {r['similarity']:.3f})" for r in recs]
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "|" + "|".join(":---" for _ in headers) + "|"

        print(header_line)
        print(sep_line)

        # 每列乐器行
        inst_columns = []
        for rec in recs:
            col = []
            for item in rec.get("instruments", []):
                nbs_range = item.get("nbs_range", "?")
                flag = " [!]" if item.get("outside_mc") else ""
                col.append(f"{item['instrument']} ({item['weight']:.2f}) [{nbs_range}]{flag}")
            inst_columns.append(col)

        max_rows = max(len(c) for c in inst_columns) if inst_columns else 0
        for row_idx in range(max_rows):
            cells = []
            for col in inst_columns:
                cells.append(col[row_idx] if row_idx < len(col) else "")
            print("| " + " | ".join(cells) + " |")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="MIDI2NoteCombo - Minecraft 音符盒乐器组合推荐工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python midi2notecombo.py --midi song.mid
  python midi2notecombo.py --midi song.mid --output result.json --verbose
  python midi2notecombo.py --midi song.mid --db_dir ./db --output out.json
  python midi2notecombo.py --midi song.mid --group_by octave --max_instruments 3
  python midi2notecombo.py --midi song.mid --group_by custom --group_size 6
        """,
    )
    parser.add_argument("--midi", required=True,
                        help="输入的 MIDI 文件路径（.mid）")
    parser.add_argument("--output", default="result.json",
                        help="输出的 JSON 文件路径（默认: result.json）")
    parser.add_argument("--db_dir", default="db",
                        help="数据库目录（默认: db）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="打印详细信息")
    parser.add_argument("--group_by", choices=["octave", "custom"], default="octave",
                        help="音区划分方式：octave=按八度分组, custom=按自定义区间大小分组（默认: octave）")
    parser.add_argument("--group_size", type=int, default=12,
                        help="自定义区间大小（--group_by custom 时生效，默认: 12）")
    parser.add_argument("--max_instruments", type=int, default=3,
                        help="每个音区最多使用的乐器数量（默认: 3）")
    parser.add_argument("--accurate", action="store_true",
                        help="启用渲染合成方式获取目标向量（对每个音区独立渲染音频并提取音色向量，更精确但较慢）")
    parser.add_argument("--sf2", default=None,
                        help="SoundFont 文件路径（如 FluidR3_GM.sf2）。"
                             "指定后 --accurate 模式优先用 FluidSynth 渲染真实 GM 音色，"
                             "精确度显著提升。会自动启用 --accurate 模式。")
    parser.add_argument("--log", default=None,
                        help="控制台输出日志文件路径（可选，如 --log output.log）")
    parser.add_argument("--no_mc_limit", action="store_true",
                        help="忽略 MC F#3~F#5 音域限制，自由推荐（自动开启 --verbose --accurate 并保存日志）")
    args = parser.parse_args()

    # --no_mc_limit 模式：自动开启 verbose + accurate + 日志
    if args.no_mc_limit:
        args.verbose = True
        args.accurate = True
        if not args.log:
            basename = os.path.splitext(os.path.basename(args.midi))[0]
            args.log = f"no_mc_limit_{basename}.log"
        if args.output == "result.json":
            args.output = "result_no_mc_limit.json"

    # --sf2 模式：自动启用 accurate，优先用 FluidSynth 渲染真实 GM 音色
    if args.sf2:
        args.accurate = True
        if not os.path.isfile(args.sf2):
            print(f"[警告] SoundFont 文件不存在: {args.sf2}，将回退到正弦波合成。")
            args.sf2 = None
        else:
            print(f"[信息] 使用 SoundFont: {args.sf2}（FluidSynth 真实音色渲染）")

    # 如果指定了日志文件，同时输出到控制台和文件
    _log_tee: Optional[Tee] = None
    if args.log:
        ensure_dir(os.path.dirname(args.log) or ".")
        _log_tee = Tee(open(args.log, "w", encoding="utf-8"))
        sys.stdout = _log_tee  # type: ignore[assignment]

    try:
        # 检查输入文件
        if not os.path.isfile(args.midi):
            print(f"[错误] MIDI 文件不存在: {args.midi}")
            sys.exit(1)

        # 检查数据库
        if not check_databases(args.db_dir):
            sys.exit(1)

        # 确定分组大小
        group_size = args.group_size if args.group_by == "custom" else 12

        print(f"[信息] 解析 MIDI 文件: {args.midi}")
        tracks = parse_midi(args.midi)

        if not tracks:
            print("[警告] MIDI 文件中没有非打击乐轨道，输出空结果。")
            results = []
        else:
            print(f"[信息] 找到 {len(tracks)} 个非打击乐轨道")
            if args.verbose:
                for t in tracks:
                    print(f"  - 轨道 {t['track_index']}: {t['midi_instrument_name']} "
                          f"(Program {t['midi_program']}), "
                          f"音域 [{t['note_range'][0]}, {t['note_range'][1]}], "
                          f"音符数 {len(t['notes'])}")

            # 加载相似度表
            similarity = load_similarity(args.db_dir)

            # 加载 MC 向量
            mc_vectors, id_to_idx = load_mc_vectors(args.db_dir)

            # 对每个轨道执行推荐
            results = []
            for track in tracks:
                if args.verbose:
                    print(f"[信息] 正在为轨道 {track['track_index']} 推荐乐器...")
                if args.no_mc_limit:
                    track_result = recommend_for_track_free(
                        track, similarity, mc_vectors, id_to_idx,
                        max_instruments=args.max_instruments,
                        num_recommendations=3,
                        accurate=args.accurate,
                        soundfont_path=args.sf2,
                    )
                else:
                    track_result = recommend_for_track(
                        track, similarity, mc_vectors, id_to_idx,
                        group_size=group_size,
                        max_instruments=args.max_instruments,
                        accurate=args.accurate,
                        soundfont_path=args.sf2,
                    )
                results.append(track_result)

        # 构建输出 JSON
        output_data = {
            "input_file": os.path.basename(args.midi),
            "tracks": results,
        }

        # 保存 JSON
        ensure_dir(os.path.dirname(args.output) or ".")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n[完成] 结果已保存到: {args.output}")

        # 打印摘要
        if args.no_mc_limit:
            print_summary_free(results, verbose=args.verbose)
        else:
            print_summary(results, verbose=args.verbose)

    finally:
        if _log_tee:
            print(f"\n[信息] 控制台日志已保存到: {args.log}")
            sys.stdout = _log_tee.stdout
            _log_tee.close()


if __name__ == "__main__":
    main()
