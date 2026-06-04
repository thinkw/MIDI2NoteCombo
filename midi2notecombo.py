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
from typing import Dict, List

from instruments import get_instrument_ids, get_all_instruments
from midi_parser import parse_midi
from recommender import load_similarity, recommend_for_track
from utils import ensure_dir


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
    """在控制台打印人类可读的摘要。"""
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
        note_range = track_result["note_range"]
        combo = track_result["recommended_combination"]
        uncovered = track_result["uncovered_notes"]

        print(f"\n--- 轨道 {ti}: {name} (Program {prog}) ---")
        print(f"  原始音域: MIDI {note_range[0]} ~ {note_range[1]} "
              f"({_midi_to_note_name(note_range[0])} ~ {_midi_to_note_name(note_range[1])})")

        if combo:
            print(f"  推荐组合 ({len(combo)} 个乐器):")
            for item in combo:
                inst_id = item["instrument"]
                rng = (item["note_range_start"], item["note_range_end"])
                tr = item.get("transpose", 0)
                print(f"    • {inst_id:20s} 负责音域: MIDI {rng[0]:3d} ~ {rng[1]:3d} "
                      f"({_midi_to_note_name(rng[0])} ~ {_midi_to_note_name(rng[1])})")
        else:
            print("  推荐组合: 无可行组合")

        if uncovered:
            print(f"  无法覆盖的音符 ({len(uncovered)} 个):")
            for unc in uncovered:
                if isinstance(unc, dict):
                    print(f"    • MIDI {unc['pitch']}: {unc.get('shift_advice', '无法覆盖')}")
                else:
                    print(f"    • MIDI {unc}")

    print("\n" + "=" * 60)


def _midi_to_note_name(midi: int) -> str:
    """将 MIDI 编号转为音名（如 60 -> C4）。"""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi // 12) - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"


def main():
    parser = argparse.ArgumentParser(
        description="MIDI2NoteCombo - Minecraft 音符盒乐器组合推荐工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python midi2notecombo.py --midi song.mid
  python midi2notecombo.py --midi song.mid --output result.json --verbose
  python midi2notecombo.py --midi song.mid --db_dir ./db --output out.json
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
    args = parser.parse_args()

    # 检查输入文件
    if not os.path.isfile(args.midi):
        print(f"[错误] MIDI 文件不存在: {args.midi}")
        sys.exit(1)

    # 检查数据库
    if not check_databases(args.db_dir):
        sys.exit(1)

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

        # 对每个轨道执行推荐
        results = []
        for track in tracks:
            if args.verbose:
                print(f"[信息] 正在为轨道 {track['track_index']} 推荐乐器...")
            track_result = recommend_for_track(track, similarity)
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
    print_summary(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
