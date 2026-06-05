# MIDI2NoteCombo

Minecraft 原版音符盒乐器组合推荐工具。

输入 MIDI 文件，输出每个 MIDI 轨道应使用的 Minecraft 原版音符盒乐器组合（支持多乐器分音区协作），以在 Note Block Studio 中最大限度还原原曲的音域和音色。

## 核心特性

- 基于 YAMNet 深度学习模型提取 1024 维音色向量
- 使用 FAISS 进行高效的余弦相似度检索
- **按八度/自定义区间分组**，每个音区独立推荐最优乐器组合
- **多乐器音色混合**：使用非负最小二乘法（NNLS）求解混合权重，精确逼近目标音色
- 区间覆盖枚举算法，自动为每个音区筛选候选乐器
- 支持超音域音符的警告输出与八度平移建议
- 纯本地运行，无需数据库服务器

## 依赖安装

### Python 版本

需要 Python 3.11+

### 安装依赖

```bash
pip install pretty_midi librosa numpy faiss-cpu tensorflow tensorflow-hub scipy
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
├── midi_parser.py             # MIDI 解析模块（含八度分组）
├── mix_optimizer.py           # NNLS 混合优化器
├── recommender.py             # 推荐算法（v2：按音区混合）
├── instruments.py             # 乐器数据表
├── utils.py                   # 辅助函数（音频加载、向量归一化等）
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
| banjo.ogg | 班卓琴 | trumpet_weathered.ogg | 小号（锈蚀） |
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

# 按八度分组推荐（默认）
python midi2notecombo.py --midi song.mid --group_by octave --max_instruments 3

# 按自定义区间分组（每 6 个半音一组）
python midi2notecombo.py --midi song.mid --group_by custom --group_size 6

# 限制每个音区最多 2 个乐器
python midi2notecombo.py --midi song.mid --max_instruments 2

# 启用精确渲染模式（对每个音区独立合成音频 + 提取真实目标向量）
python midi2notecombo.py --midi song.mid --accurate

# 将控制台输出同步保存到日志文件
python midi2notecombo.py --midi song.mid --log output.log
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--midi` | 输入 MIDI 文件路径（必填） | — |
| `--output` | 输出 JSON 文件路径 | `result.json` |
| `--db_dir` | 数据库目录 | `db` |
| `--verbose`, `-v` | 打印详细信息 | 关闭 |
| `--group_by` | 音区划分方式：`octave`（按八度）或 `custom`（自定义区间） | `octave` |
| `--group_size` | 自定义区间大小（半音数，`--group_by custom` 时生效） | `12` |
| `--max_instruments` | 每个音区最多使用的乐器数量 | `3` |
| `--accurate` | 启用渲染合成方式获取目标向量（对每个音区独立渲染音频并提取音色向量，更精确但较慢，需要 `tensorflow` + `tensorflow-hub`） | 关闭 |
| `--log` | 将控制台输出同步保存到文件（如 `--log output.log`） | 仅控制台输出 |

## 输出格式

每个轨道按八度/音区独立推荐乐器组合，包含混合权重和匹配度：

```json
{
  "input_file": "song.mid",
  "tracks": [
    {
      "track_index": 0,
      "midi_program": 0,
      "midi_instrument_name": "Acoustic Grand Piano",
      "octave_recommendations": [
        {
          "octave": 3,
          "note_range": [36, 47],
          "instruments": [
            {"instrument": "bass", "weight": 0.96},
            {"instrument": "didgeridoo", "weight": 0.04}
          ],
          "similarity": 0.95
        },
        {
          "octave": 4,
          "note_range": [48, 59],
          "instruments": [
            {"instrument": "guitar", "weight": 0.64},
            {"instrument": "trumpet_weathered", "weight": 0.25},
            {"instrument": "trumpet_oxidized", "weight": 0.11}
          ],
          "similarity": 0.56
        },
        {
          "octave": 5,
          "note_range": [60, 71],
          "instruments": [
            {"instrument": "harp", "weight": 0.53},
            {"instrument": "pling", "weight": 0.26},
            {"instrument": "iron_xylophone", "weight": 0.21}
          ],
          "similarity": 0.99
        }
      ],
      "uncovered_notes": []
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `octave` | 八度编号（或自定义分组编号） |
| `note_range` | 该音区的实际音符范围 `[min_pitch, max_pitch]` |
| `instruments` | 推荐乐器列表，按权重降序排列 |
| `instruments[].instrument` | 乐器 ID |
| `instruments[].weight` | 混合权重（0~1，本音区内权重和为 1） |
| `similarity` | 混合音色与目标音色的匹配度（0~1，越高越匹配） |
| `uncovered_notes` | 未被任何 MC 乐器覆盖的音符列表 |

### 控制台输出

控制台会打印每条推荐乐器的 **NBS key 音号区间**，直接对应 Note Block Studio 中的按键位置：

```
--- 轨道 0: Acoustic Grand Piano (Program 0) ---
  共 2 个音区推荐:

  [八度 0] MIDI 60~71 (NBS 基准: F#4~F#5):
     匹配度: 0.987
     推荐乐器 (3 个):
       - harp                 权重: 0.95  key=45 (同八度)  NBS key [F#4~F#5]
       - pling                权重: 0.82  key=45 (同八度)  NBS key [F#4~F#5]
       - guitar               权重: 0.71  key=33 (低1八度)  NBS key [F#5~F#6]
```

- **NBS key [F#4~F#5]** — 在 NBS 编辑器中将音符放在 F#4 到 F#5 之间即可发出目标音高
- **key=值 (同/低/高八度)** — 乐器的 `instrument_key` 参数，反映相对于竖琴(harp)的八度偏移

## NBS 音高模型

### 核心原理

Note Block Studio 中每个音符的**实际发音音高**由两个参数共同决定：

| 参数 | 含义 | 示例 |
|:---|:---|:---|
| `note_key` | NBS 编辑器中音符的**按键位置**（0~239，所有乐器共用同一键盘） | 在 F#4 位置放一个音符 → key = 45 |
| `instrument_key` | 每种乐器固有的**音高偏移量**，决定该乐器在相同键位上的实际发音八度 | harp = 45, bass = 21, flute = 57 |

**核心公式**（来自 NBS 官方格式规范）：

```
实际音高 (cents) = (note_key + instrument_key - 45) × 100 + note_pitch

等效 MIDI 音高: midi_note = note_key + instrument_key - 24
```

**校准基准**：选择竖琴 (harp, instrument_key=45) 在 note_key=45 (F#4) 时，实际发音为 MIDI 66 = F#4，作为整个系统的锚点。

### instrument_key 的作用

`instrument_key` 本质上是一个**八度平移器**：
- 竖琴组 (key=45)：在 NBS 键位 F#3~F#5 上以**原八度**发音
- 贝斯组 (key=21)：同样的键位，实际发音**低 2 个八度**
- 长笛组 (key=57)：同样的键位，实际发音**高 1 个八度**

这使得所有乐器共享同一套 NBS 键盘（F#3~F#5，即 key 32~56），却能在不同八度发出声音。

### NBS 八度命名

NBS 以 **F# 为八度边界**（与 Minecraft 音符盒的 25 音阶对应）：
- F#3 对应 key = 30
- F#4 对应 key = 42
- F#5 对应 key = 54
- 通用公式：**F#N 起始于 key = 6 + (N − 1) × 12**

### 乐器音域参考

所有乐器在 NBS 中均使用 key 区间 **[32, 56]**（即 F#3~F#5），通过不同的 `instrument_key` 在不同八度实际发音：

| 乐器组 | instrument_key | NBS key 区间 | 实际发音音域 | 相对竖琴 |
|:---|:---:|:---|:---|:---:|
| harp / banjo / bit / pling / iron_xylophone / trumpet / trumpet_exposed | 45 | 32~56 | F#3~F#5 | 同八度 |
| trumpet_weathered / trumpet_oxidized / guitar | 33 | 32~56 | F#2~F#4 | 低 1 八度 |
| bass / didgeridoo | 21 | 32~56 | F#1~F#3 | 低 2 八度 |
| flute / cow_bell | 57 | 32~56 | F#4~F#6 | 高 1 八度 |
| bell / icechime / xylobone | 69 | 32~56 | F#5~F#7 | 高 2 八度 |

> **使用方式**：控制台输出会直接给出 `NBS key [F#x~F#y]`，将这些键位上的音符放到对应乐器轨道即可。无需手动换算 instrument_key——NBS 会根据乐器自动处理。

## 算法说明

1. **MIDI 解析**: 提取每个非打击乐轨道的音域范围、音符分布和 program 编号
2. **八度分组**: 将轨道内所有音符按八度（或自定义区间大小）自动分组，每个组作为一个独立音区
3. **候选乐器筛选**: 对每个音区，筛选实际音域完全覆盖该音区的 MC 乐器
4. **目标音色构建**: 根据 MIDI track 的 GM program，通过相似度加权 MC 向量构建伪目标音色向量
5. **NNLS 混合优化**: 使用非负最小二乘法求解最优乐器混合权重，最大化混合音色与目标音色的余弦相似度
6. **超音域处理**: 当音区无法被任何乐器覆盖时，自动拆分为更小子区间递归处理；仍无法覆盖则输出警告并跳过

## License

MIT
