"""
MC 乐器向量库离线构建工具。

从 samples/ 目录读取每种 MC 乐器的 .ogg 基础采样文件，
使用 YAMNet 模型提取 1024 维向量，存入 FAISS 索引和 JSON 元数据。
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Optional

from instruments import get_all_instruments, get_instrument_ids
from utils import load_audio, extract_vggish_embedding, ensure_dir


def build_mc_vector_db(samples_dir: str = "samples", db_dir: str = "db") -> bool:
    """
    从采样目录构建 MC 乐器向量数据库。

    Args:
        samples_dir: 包含 .ogg 文件的目录路径。
        db_dir: 输出数据库目录路径。

    Returns:
        是否构建成功。
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("请先安装 faiss-cpu: pip install faiss-cpu")

    if not os.path.isdir(samples_dir):
        print(f"[错误] 采样目录不存在: {samples_dir}")
        print("请创建 samples/ 目录并放入 MC 乐器的 .ogg 采样文件。")
        return False

    instruments = get_all_instruments()
    instrument_ids = get_instrument_ids()

    vectors = []
    metadata: List[Dict] = []
    missing_count = 0

    for inst in instruments:
        inst_id = inst["instrument_id"]
        # 支持 .ogg 和 .wav 格式
        ogg_path = os.path.join(samples_dir, f"{inst_id}.ogg")
        wav_path = os.path.join(samples_dir, f"{inst_id}.wav")

        audio_path = None
        if os.path.isfile(ogg_path):
            audio_path = ogg_path
        elif os.path.isfile(wav_path):
            audio_path = wav_path
        else:
            print(f"[警告] 缺少 {inst_id} 的采样文件（期望 {ogg_path} 或 {wav_path}），跳过。")
            missing_count += 1
            continue

        print(f"[信息] 处理 {inst_id} ({inst['name']}) ...")
        audio = load_audio(audio_path, sr=16000)
        if audio is None:
            print(f"[警告] 无法加载 {inst_id} 的音频，跳过。")
            missing_count += 1
            continue

        embedding = extract_vggish_embedding(audio)
        if embedding is None:
            print(f"[警告] 无法提取 {inst_id} 的向量，跳过。")
            missing_count += 1
            continue

        vectors.append(embedding)
        metadata.append({
            "instrument_id": inst_id,
            "name": inst["name"],
            "vector_index": len(vectors) - 1,
        })

    if not vectors:
        print("[错误] 没有成功提取任何乐器的向量。请检查采样文件。")
        return False

    vectors_np = np.array(vectors, dtype=np.float32)

    # 确保向量已 L2 归一化
    norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # 避免除零
    vectors_np = vectors_np / norms

    # 使用 IndexFlatIP（内积 = 归一化后的余弦相似度）
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_np)

    # 保存
    ensure_dir(db_dir)

    index_path = os.path.join(db_dir, "mc_vectors.faiss")
    metadata_path = os.path.join(db_dir, "mc_metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 向量库构建完毕:")
    print(f"  - 索引文件: {index_path}")
    print(f"  - 元数据文件: {metadata_path}")
    print(f"  - 已入库乐器: {len(vectors)} 个")
    if missing_count > 0:
        print(f"  - 缺失/跳过: {missing_count} 个")

    return True


def main():
    parser = argparse.ArgumentParser(description="构建 MC 乐器音频 Embedding 向量库")
    parser.add_argument("--samples_dir", default="samples",
                        help="包含 .ogg 采样文件的目录（默认: samples）")
    parser.add_argument("--db_dir", default="db",
                        help="输出数据库目录（默认: db）")
    args = parser.parse_args()

    success = build_mc_vector_db(
        samples_dir=args.samples_dir,
        db_dir=args.db_dir,
    )
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
