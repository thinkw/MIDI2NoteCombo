"""
辅助函数：音频加载、VGGish 向量提取、余弦相似度等。
"""

import os
import numpy as np
from typing import List, Tuple, Optional

VGGISH_SAMPLE_RATE = 16000
VGGISH_EMBEDDING_DIM = 128

# 缓存已加载的 VGGish 模型
_vggish_model = None


def load_audio(file_path: str, sr: int = VGGISH_SAMPLE_RATE) -> Optional[np.ndarray]:
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


def _load_vggish_model():
    """
    延迟加载 VGGish 模型（通过 tensorflow-hub）。
    模型 URL: https://tfhub.dev/google/vggish/1
    """
    global _vggish_model
    if _vggish_model is not None:
        return _vggish_model

    import tensorflow_hub as hub

    print("[信息] 正在加载 VGGish 模型（首次加载可能需要下载）...")
    _vggish_model = hub.load("https://tfhub.dev/google/vggish/1")
    print("[信息] VGGish 模型加载完成。")
    return _vggish_model


def extract_vggish_embedding(audio: np.ndarray, sr: int = VGGISH_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    从音频波形中提取 VGGish 128 维向量（对所有帧取平均）。

    Args:
        audio: 单声道波形 (n_samples,)。
        sr: 采样率。

    Returns:
        128 维归一化向量 (128,) 或 None。
    """
    if audio is None or len(audio) == 0:
        return None

    try:
        model = _load_vggish_model()
        # VGGish 期望 float32, shape (None,)
        audio = audio.astype(np.float32)

        # 确保音频至少有 0.96 秒（VGGish 最小帧长），不足则填充
        min_samples = int(0.96 * sr)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')

        embeddings = model(audio)
        # embeddings shape: (num_frames, 128)
        if embeddings is None:
            return None

        embeddings_np = embeddings.numpy()
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
        print(f"[错误] VGGish 向量提取失败: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度。
    输入应为已 L2 归一化的向量（内积即余弦相似度）。
    """
    return float(np.dot(a, b))


def ensure_dir(dir_path: str) -> None:
    """确保目录存在，不存在则创建。"""
    os.makedirs(dir_path, exist_ok=True)
