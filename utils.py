"""
辅助函数：音频加载、音频 Embedding 向量提取（YAMNet）、余弦相似度等。
"""

import os
import numpy as np
from typing import List, Tuple, Optional

AUDIO_SAMPLE_RATE = 16000
# YAMNet 输出 embedding 维度为 1024
EMBEDDING_DIM = 1024

# 向后兼容别名
VGGISH_SAMPLE_RATE = AUDIO_SAMPLE_RATE
VGGISH_EMBEDDING_DIM = EMBEDDING_DIM

# 缓存已加载的 YAMNet 模型
_yamnet_model = None


def load_audio(file_path: str, sr: int = AUDIO_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    使用 librosa 加载音频文件，返回单声道波形。

    Args:
        file_path: 音频文件路径（.ogg, .wav, .mp3 等）。
        sr: 目标采样率，默认 16000。

    Returns:
        numpy 数组 (n_samples,) 或 None（加载失败时）。
    """
    try:
        import librosa
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio
    except Exception as e:
        print(f"[错误] 无法加载音频文件 {file_path}: {e}")
        return None


def _load_yamnet_model():
    """
    延迟加载 YAMNet 模型（通过 tensorflow-hub）。
    模型 URL: https://tfhub.dev/google/yamnet/1
    YAMNet 是 VGGish 的继任者，TF2 原生支持，输出 1024 维 embedding。
    """
    global _yamnet_model
    if _yamnet_model is not None:
        return _yamnet_model

    import tensorflow_hub as hub

    print("[信息] 正在加载 YAMNet 模型（首次加载可能需要下载）...")
    _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    print("[信息] YAMNet 模型加载完成。")
    return _yamnet_model


def extract_vggish_embedding(audio: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    从音频波形中提取音频 Embedding 向量（对所有帧取平均）。

    使用 YAMNet 模型（TF2 兼容，替代不兼容 TF2 的 VGGish v1）。

    Args:
        audio: 单声道波形 (n_samples,)。
        sr: 采样率。

    Returns:
        Embedding 归一化向量 (1024,) 或 None。
    """
    if audio is None or len(audio) == 0:
        return None

    try:
        import tensorflow as tf

        model = _load_yamnet_model()
        audio = audio.astype(np.float32)

        # 确保音频至少有 ~0.96 秒（YAMNet 内部 0.96s 窗口），不足则 tile 填充
        min_samples = int(0.96 * sr)
        if len(audio) < min_samples:
            repeats = int(np.ceil(min_samples / len(audio)))
            audio = np.tile(audio, repeats)[:min_samples]

        audio_tensor = tf.convert_to_tensor(audio)

        # YAMNet 返回 (scores, embeddings, log_mel_spectrogram)
        scores, embeddings, spectrogram = model(audio_tensor)

        if embeddings is None:
            return None

        embeddings_np = embeddings.numpy()

        # embeddings shape: (num_frames, 1024) 或 (1024,)
        if embeddings_np.ndim == 1:
            vector = embeddings_np
        else:
            vector = embeddings_np.mean(axis=0)

        # L2 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)
    except Exception as e:
        print(f"[错误] Embedding 向量提取失败: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。
    输入应为已 L2 归一化的向量（内积即余弦相似度）。
    """
    return float(np.dot(a, b))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    L2 归一化向量。

    Args:
        v: 输入向量。

    Returns:
        归一化后的向量（副本）。
    """
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v.copy()


def ensure_dir(dir_path: str) -> None:
    """确保目录存在，不存在则创建。"""
    os.makedirs(dir_path, exist_ok=True)
