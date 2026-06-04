# MIDI2NoteCombo

Minecraft 原版音符盒乐器组合推荐工具。

输入 MIDI 文件，输出每个 MIDI 轨道应使用的 Minecraft 原版音符盒乐器组合（支持多乐器分音区协作），以在 Note Block Studio 中最大限度还原原曲的音域和音色。

## 核心特性

- 基于 VGGish 深度学习模型提取音色向量
- 使用 FAISS 进行高效的余弦相似度检索
- 区间覆盖枚举算法，自动寻找最优乐器组合
- 支持音域不足时的八度平移建议
- 纯本地运行，无需数据库服务器

## 依赖安装

### Python 版本

需要 Python 3.11+

### 安装依赖

```bash
pip install pretty_midi librosa numpy faiss-cpu tensorflow tensorflow-hub
```

> **注意**: `tensorflow` 和 `tensorflow-hub` 在某些 Python 版本下可能需要特定安装方式。如果遇到兼容性问题，请参考 [TensorFlow 官方安装指南](https://www.tensorflow.org/install)。

可选依赖（用于 FluidSynth 模式生成相似度表）:

```bash
pip install fluidsynth
```

同时需要安装 SoundFont 文件（如 `GeneralUser_GS.sf2` 或 `FluidR3_GM.sf2`）。

### 音频处理依赖

处理 `.ogg` 文件需要系统安装 ffmpeg:

- **macOS**: `brew install ffmpeg`
- **Linux**: `apt install ffmpeg` 或 `yum install ffmpeg`
- **Windows**: 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载并添加到 PATH

或者使用 pip 安装:

```bash
pip install pydub
```

## 项目结构

```
MIDI2NoteCombo/
├── midi2notecombo.py          # 主入口（命令行工具）
├── build_mc_db.py             # 构建 MC 乐器向量库
├── build_gm_similarity.py     # 构建 GM 相似度矩阵
├── midi_parser.py             # MIDI 解析模块
├── cover_engine.py            # 区间覆盖枚举
├── recommender.py             # 推荐算法
├── instruments.py             # 乐器数据表
├── utils.py                   # 辅助函数
├── db/                        # 向量库和相似度数据
│   ├── mc_vectors.faiss
│   ├── mc_metadata.json
│   └── similarity.json
├── samples/                   # MC 乐器 .ogg 采样文件（用户提供）
└── README.md
```

## 使用方法

### 第一步：准备采样文件

在 `samples/` 目录下放置 Minecraft 原版音符盒乐器的 `.ogg` 采样文件。

需要以下文件（对应乐器 ID）:

| 文件 | 乐器 | 文件 | 乐器 |
|------|------|------|------|
| harp.ogg | 竖琴/钢琴 | trumpet_exposed.ogg | 小号（斑驳） |
| piano.ogg | 钢琴 | trumpet_weathered.ogg | 小号（锈蚀） |
| banjo.ogg | 班卓琴 | trumpet_oxidized.ogg | 小号（氧化） |
| bit.ogg | 芯片（方波） | guitar.ogg | 吉他 |
| pling.ogg | 扣弦（电钢琴） | bass.ogg | 贝斯 |
| iron_xylophone.ogg | 铁木琴 | didgeridoo.ogg | 迪吉里杜管 |
| trumpet.ogg | 小号（普通） | flute.ogg | 长笛 |
| cow_bell.ogg | 牛铃 | bell.ogg | 铃铛/钟琴 |
| icechime.ogg | 管钟 | xylobone.ogg | 木琴 |

### 第二步：离线构建数据库

```bash
# 构建 MC 乐器向量库
python build_mc_db.py

# 构建 GM 相似度表（使用手工映射表）
python build_gm_similarity.py

# 如果安装并配置了 FluidSynth + SoundFont，可以尝试：
python build_gm_similarity.py --use_fluidsynth
```

### 第三步：运行推荐

```bash
# 基本用法
python midi2notecombo.py --midi song.mid

# 指定输出文件
python midi2notecombo.py --midi song.mid --output result.json

# 详细模式
python midi2notecombo.py --midi song.mid --verbose

# 指定数据库目录
python midi2notecombo.py --midi song.mid --db_dir ./my_db
```

## 输出格式

```json
{
  "input_file": "song.mid",
  "tracks": [
    {
      "track_index": 0,
      "midi_program": 0,
      "midi_instrument_name": "Acoustic Grand Piano",
      "note_range": [36, 84],
      "recommended_combination": [
        {
          "instrument": "bass",
          "note_range_start": 36,
          "note_range_end": 54,
          "transpose": -24
        },
        {
          "instrument": "harp",
          "note_range_start": 55,
          "note_range_end": 78,
          "transpose": 0
        },
        {
          "instrument": "flute",
          "note_range_start": 79,
          "note_range_end": 84,
          "transpose": 12
        }
      ],
      "uncovered_notes": []
    }
  ]
}
```

## MC 乐器音域参考

| 乐器 | transpose | 实际音域 (MIDI) | 音域 |
|------|-----------|-----------------|------|
| harp / piano / banjo / bit / pling / iron_xylophone / trumpet / trumpet_exposed | 0 | 54~78 | F#3~F#5 |
| trumpet_weathered / trumpet_oxidized / guitar | -12 | 42~66 | F#2~F#4 |
| bass / didgeridoo | -24 | 30~54 | F#1~F#3 |
| flute / cow_bell | +12 | 66~90 | F#4~F#6 |
| bell / icechime / xylobone | +24 | 78~102 | F#5~F#7 |

## 算法说明

1. **MIDI 解析**: 提取每个非打击乐轨道的音域范围和音符分布
2. **区间覆盖枚举**: 使用 `itertools.combinations` 枚举 1~4 个乐器的组合，检查音域完整覆盖
3. **音色相似度加权**: 根据 GM 乐器与 MC 乐器的音色相似度矩阵，以及音符在各音域区间的分布权重，加权计算组合得分
4. **八度平移**: 当目标音域超出 MC 乐器能力时，提供八度平移建议

## License

MIT
