

# 项目：MIDI2NoteCombo —— Minecraft 原版音符盒乐器组合推荐工具

## 项目名称
MIDI2NoteCombo

## 目标
开发一个 Python 命令行工具，输入 MIDI 文件，输出每个 MIDI 轨道应使用的 Minecraft 原版音符盒乐器组合（支持多乐器分音区协作），以在 Note Block Studio 中最大限度还原原曲的音域和音色。

## 技术栈（已确认，按此实现）
- Python 3.14
- `pretty_midi`：MIDI 文件解析
- `librosa`：音频加载与预处理
- `tensorflow-hub`：加载 YAMNet 模型提取音频 embedding 向量（1024 维）
- `faiss-cpu`：向量索引与余弦相似度检索
- `numpy`：数值计算
- 不使用数据库服务端，全部本地文件存储（JSON + FAISS 索引）

## 核心约束
- **仅使用原版音符盒乐器**（见下方乐器数据表），不依赖模组。
- **每种乐器只有一个基础采样**，所有音高通过实时变调产生。
- 乐器的实际音域 = 标准音域 (F#3~F#5, MIDI 54~78) + 偏移（半音数）。
- **输出可以是单一乐器或多个乐器的组合**，每个乐器负责一段连续音区。

## 乐器数据表（完整）
以下为 Minecraft 原版所有可发音高的乐器（打击乐除外）：

| instrument_id     | 显示名称 | transpose (半音) | 实际低音 (MIDI) | 实际高音 (MIDI) |
|-------------------|----------|------------------|----------------|----------------|
| harp              | 竖琴/钢琴 | 0 | 54 | 78 |
| banjo             | 班卓琴 | 0 | 54 | 78 |
| bit               | “芯片”（方波） | 0 | 54 | 78 |
| pling             | “扣弦”（电钢琴） | 0 | 54 | 78 |
| iron_xylophone    | “铁木琴”（颤片琴） | 0 | 54 | 78 |
| trumpet           | 小号（普通） | 0 | 54 | 78 |
| trumpet_exposed   | 小号（斑驳） | 0 | 54 | 78 |
| trumpet_weathered | 小号（锈蚀） | -12 | 42 | 66 |
| trumpet_oxidized  | 小号（氧化） | -12 | 42 | 66 |
| guitar            | 吉他 | -12 | 42 | 66 |
| bass              | 贝斯 | -24 | 30 | 54 |
| didgeridoo        | 迪吉里杜管 | -24 | 30 | 54 |
| flute             | 长笛 | +12 | 66 | 90 |
| cow_bell          | 牛铃 | +12 | 66 | 90 |
| bell              | 铃铛（钟琴） | +24 | 78 | 102 |
| icechime          | 管钟 | +24 | 78 | 102 |
| xylobone          | 木琴 | +24 | 78 | 102 |

> 打击乐器（snare, basedrum, hat）不参与音高匹配，输出时忽略。

## 输入输出规格
- **输入**：MIDI 文件路径（`.mid`）
- **输出**：JSON 文件，格式如下：
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
        {"instrument": "bass", "note_range_start": 36, "note_range_end": 54, "transpose": -24},
        {"instrument": "harp", "note_range_start": 55, "note_range_end": 78, "transpose": 0},
        {"instrument": "flute", "note_range_start": 79, "note_range_end": 84, "transpose": 12}
      ],
      "uncovered_notes": []
    }
  ]
}
```
- 同时在控制台打印人类可读的摘要。

## 实现模块

### 1. 离线构建：MC 乐器向量库 (`build_mc_db.py`)
- 从指定目录读取每个乐器的 `.ogg` 基础采样文件（例如 `harp.ogg`）。需要你新建一个文件夹专门用来放置.ogg
- 使用 `librosa.load(sr=16000, mono=True)` 加载音频。
- 加载 YAMNet 模型（通过 `tensorflow-hub` 的 `https://tfhub.dev/google/yamnet/1`）。
- 将每个音频片段转换为 1024 维 embedding 向量，存入 FAISS 索引（IndexFlatIP），同时保存 `instrument_id -> vector` 的映射到 JSON。
- 索引和映射文件保存到 `db/` 目录。

### 2. 离线构建：GM 乐器相似度表 (`build_gm_similarity.py`)
- 需要一个通用 MIDI 音源（推荐使用 FluidSynth + GeneralUser GS SoundFont，或直接使用公开的 GM 乐器采样库）。
- 提取 128 个 GM 乐器的参考音频：中央 C（MIDI 60）持续 1 秒。
- 使用同样的 YAMNet 模型，得到每个 GM 乐器的 1024 维 embedding 向量。
- 计算每个 GM 向量与所有 MC 乐器向量的余弦相似度（`cosine_similarity = dot(a,b) / (norm(a)*norm(b))`）。
- 保存为 `similarity.json`，结构：`similarity[gm_program][mc_instrument_id] = float`。
- 若无法获取真实 GM 音源，则提供一个**手工近似映射表**作为 fallback（例如钢琴→harp，贝斯→bass，吉他→guitar，长笛→flute，小号→trumpet 等）。

### 3. MIDI 解析与轨道分析 (`midi_parser.py`)
- 使用 `pretty_midi` 加载 MIDI。
- 遍历每个非打击乐轨道（`channel != 9`）。
- 提取该轨道的 program 编号（如果没有 program，则使用 0）。
- 收集所有音符的 `pitch`，计算 `min_note` 和 `max_note`。
- 返回轨道列表：`[{"index": i, "program": p, "min_note": mn, "max_note": mx, "notes": [pitch1, pitch2,...]}]`

### 4. 区间覆盖枚举 (`cover_engine.py`)
- 输入：目标音域 `[target_low, target_high]`（MIDI 编号）。
- 根据乐器数据表，计算每个乐器的实际音域 `[low, high]`。
- 筛选出与目标区间有交集的乐器。
- 使用 `itertools.combinations` 枚举 1~4 个乐器的所有组合。
- 对于每个组合，检查这些乐器音域区间的并集是否完全覆盖 `[target_low, target_high]`（允许超出）。
- 返回所有可行组合的列表，每个组合包含 `[(inst_id, low, high), ...]`。

### 5. 音色相似度加权选择 (`recommender.py`)
- 加载 `similarity.json`。
- 对于 MIDI 轨道（已知 program, target_low, target_high, 音符列表）：
  - 调用 `cover_engine` 得到所有可行组合。
  - 若无可行组合，则输出警告并尝试扩展（八度平移），见步骤 6。
  - 对于每个可行组合：
    - 计算该组合内各乐器负责的音符数量：根据音符列表的 pitch 分布，分配到组合内每个乐器的音域区间中。
    - 权重 = 该乐器负责的音符数 / 总音符数。
    - 组合相似度 = sum(权重_i × similarity[program][inst_i])。
  - 选择相似度最高的组合。
- 输出组合及每个乐器应负责的音域区间（连续，例如 `[55,78]`）。

### 6. 音域不足与八度平移处理
- 若目标音域无法被任何组合覆盖（例如最低音 < 30 或最高音 > 102）：
  - 找出超出范围的音符。
  - 尝试对每个超出音符进行八度平移（±12, ±24），直到落入某个乐器的音域内。
  - 记录平移建议，输出到 `uncovered_notes` 字段（例如 `{"pitch": 24, "shift_advice": "shift up 2 octaves to 48 (bass)"}`）。
  - 如果平移后仍无法覆盖，则标记为不可覆盖。

### 7. 主程序 (`midi2notecombo.py`)
- 命令行接口：
  ```bash
  python midi2notecombo.py --midi input.mid --output result.json
  ```
- 可选参数：`--db_dir`（指定数据库目录），`--verbose` 打印详细信息。
- 流程：
  1. 加载 FAISS 索引和相似度表（如果不存在则提示先运行 `build_mc_db.py` 和 `build_gm_similarity.py`）。
  2. 解析 MIDI。
  3. 对每个轨道执行推荐。
  4. 生成 JSON 输出并打印摘要。

## 代码文件结构
```
MIDI2NoteCombo/
├── midi2notecombo.py          # 主入口
├── build_mc_db.py             # 构建 MC 乐器向量库
├── build_gm_similarity.py     # 构建 GM 相似度矩阵
├── midi_parser.py             # MIDI 解析模块
├── cover_engine.py            # 区间覆盖枚举
├── recommender.py             # 推荐算法
├── instruments.py             # 乐器数据表（transpose 等）
├── utils.py                   # 辅助函数（音频加载、YAMNet embedding 提取）
├── db/                        # 存放向量库和相似度 JSON
│   ├── mc_vectors.faiss
│   ├── mc_metadata.json
│   └── similarity.json
├── samples/                   # 存放 MC 乐器采样 .ogg 文件（用户需提供）
└── README.md
```

## 错误处理与健壮性
- 若缺少 MC 采样文件，打印明确提示并退出。
- 若 MIDI 文件无任何非打击乐轨道，输出空结果。
- 若某个轨道的音域完全无法覆盖，标记并给出建议。

## 依赖安装说明
在 README 中给出：
```bash
pip install pretty_midi librosa numpy faiss-cpu tensorflow tensorflow-hub scipy
```
（如果需要处理 .ogg，可能还需要 `pydub` 或 `ffmpeg`）

## 最终期望输出
一份完整的、可运行的 Python 项目代码，包含上述所有模块，以及必要的注释和文档。代码应当能够正确地：
- 离线构建 MC 乐器向量库（从用户提供的 `.ogg` 文件）。
- 离线构建 GM 相似度表（使用 FluidSynth 或提供手工映射 fallback）。
- 在线推荐并输出 JSON。

请生成完整代码。






🚀 增量更新：多乐器混合匹配 + 按八度分区推荐
🎯 新增需求
对于单个 MIDI 轨道，根据音符实际分布的八度（例如 C4–B4、C5–B5）自动划分为若干连续音区（默认按八度分组，也可按自定义区间）。

对每个音区独立推荐一组 Minecraft 乐器，允许多个乐器同时演奏该音区内的相同音符（即音色混合），以实现对原 MIDI 音色更精确的逼近。

输出每个音区的推荐乐器列表（含混合权重） 以及混合音色与目标音色的匹配度（0~1 之间，越高越好）。

🧠 核心算法升级
1. 多乐器混合模型
每个 Minecraft 乐器 i 有一个 1024 维 embedding 向量 v_i（由 YAMNet 提取并归一化）。

一组乐器以权重 w_i 混合后的音色向量为：v_mix = Σ(w_i * v_i), 其中 w_i ≥ 0, Σ w_i = 1。

对于给定的目标音色向量 v_target，寻找最优权重 w 使得 cos_sim(v_mix, v_target) 最大化。

求解方法：非负最小二乘法（NNLS），最小化 ||V * w - v_target||^2，其中 V 的列是候选乐器的向量。

2. 音区划分策略
将轨道内所有音符按 pitch // 12 分组（八度组），每个组作为一个独立音区。

若某八度内无音符则跳过。

也可允许用户通过参数 --group_size 自定义区间大小（例如 12 个半音为一个区间）。

3. 音区目标音色向量的获取
方法 A（轻量）：使用该 MIDI 轨道对应的 GM 乐器（由 program 决定）的标准向量（即离线构建的 GM 参考向量）。所有音符共用同一向量。

方法 B（精确）：对该音区内所有音符进行合成渲染（使用 FluidSynth + SoundFont 生成短音频），再用 YAMNet 提取整体 embedding 向量。此方法更准确但计算稍重，建议作为可选模式。

> **注意**：方法 B 为可选增强功能，暂不在本版本实现，预留 `--accurate` 参数接口。

为平衡效率与效果，推荐方法 A 作为默认，并保留方法 B 作为 --accurate 选项（未来实现）。

4. 候选乐器筛选（音域约束）
每个 Minecraft 乐器有固定的实际音域 [low, high]（见原乐器数据表）。

对于音区 [min_pitch, max_pitch]，只允许那些 实际音域完全覆盖该音区 的乐器参与混合。

若没有乐器能完整覆盖整个音区，则拆分该音区为更小的子区间（例如按半音或用户指定的粒度），递归处理。

5. 匹配度计算
求解最优混合权重后，计算混合向量 v_mix_opt = V * w_opt。

匹配度 similarity = cos_sim(v_mix_opt, v_target)，输出保留两位小数。

📦 新增模块与文件变更
文件	变更类型	说明
mix_optimizer.py	新增	实现 NNLS 混合权重求解、候选乐器筛选
midi_parser.py	修改	增加 get_notes_by_octave() 函数，返回按八度分组的音符
recommender.py	重写	改为按八度调用混合优化器，替换原来的单一组合推荐
cover_engine.py	保留	供旧版逻辑使用；混合优化器内置独立 filter_candidates_by_range()，不再调用 cover_engine
utils.py	修改	增加 normalize_vector()、cosine_similarity()
输出 JSON 格式	重构	见下方新格式
📤 更新后的输出 JSON 格式
json
{
  "input_file": "song.mid",
  "tracks": [
    {
      "track_index": 0,
      "midi_program": 0,
      "midi_instrument_name": "Acoustic Grand Piano",
      "octave_recommendations": [
        {
          "octave": 4,
          "note_range": [60, 71],
          "instruments": [
            {"instrument": "harp", "weight": 0.65},
            {"instrument": "flute", "weight": 0.35}
          ],
          "similarity": 0.92
        },
        {
          "octave": 5,
          "note_range": [72, 83],
          "instruments": [
            {"instrument": "bell", "weight": 0.7},
            {"instrument": "icechime", "weight": 0.3}
          ],
          "similarity": 0.87
        },
        {
          "octave": 3,
          "note_range": [36, 47],
          "instruments": [
            {"instrument": "bass", "weight": 1.0}
          ],
          "similarity": 0.95
        }
      ],
      "uncovered_notes": []
    }
  ]
}
说明：每个音区独立推荐，instruments 列表中的权重和为 1。similarity 越高表示混合音色与原轨道该音区越匹配。

🔧 实现细节：mix_optimizer.py
python
import numpy as np
from scipy.optimize import nnls

def find_best_mix(target_vec, instrument_vectors, max_instruments=3):
    """
    target_vec: (1024,) 归一化目标向量
    instrument_vectors: dict {inst_id: (1024,) 归一化向量}
    max_instruments: 最多使用的乐器数量
    返回: ([(inst_id, weight), ...], similarity)
    """
    ids = list(instrument_vectors.keys())
    # 构建矩阵 A，列为候选乐器向量
    A = np.column_stack([instrument_vectors[i] for i in ids])
    # 非负最小二乘
    weights, _ = nnls(A, target_vec)
    if np.sum(weights) == 0:
        return [], 0.0
    
    # 选择权重最大的 max_instruments 个
    idx = np.argsort(weights)[::-1][:max_instruments]
    idx = [i for i in idx if weights[i] > 1e-3]
    if not idx:
        return [], 0.0
    
    # 重新归一化选中乐器的权重
    selected_weights = weights[idx] / np.sum(weights[idx])
    selected = [(ids[i], float(selected_weights[j])) for j, i in enumerate(idx)]
    
    # 只用选中的乐器计算混合向量和相似度
    A_selected = A[:, idx]
    mixed_vec = A_selected @ selected_weights
    sim = np.dot(mixed_vec, target_vec) / (np.linalg.norm(mixed_vec) * np.linalg.norm(target_vec))
    return selected, float(sim)
🧩 集成到 recommender.py 的逻辑流程
python
def recommend_track(track, mc_vectors, mc_ranges, gm_vectors):
    # track 包含 program, notes 列表
    octave_groups = get_notes_by_octave(track['notes'])   # {octave: [pitches]}
    recommendations = []
    for octave, pitches in octave_groups.items():
        min_pitch, max_pitch = min(pitches), max(pitches)
        # 筛选能覆盖该音区的乐器
        candidates = {}
        for inst_id, (low, high) in mc_ranges.items():
            if low <= min_pitch and high >= max_pitch:
                candidates[inst_id] = mc_vectors[inst_id]
        if not candidates:
            # 无乐器覆盖，尝试拆分子区间（略）
            continue
        # 获取目标向量（默认使用 GM 标准向量）
        target_vec = gm_vectors.get(track['program'], gm_vectors[0])
        best_mix, sim = find_best_mix(target_vec, candidates)
        recommendations.append({
            "octave": octave,
            "note_range": [min_pitch, max_pitch],
            "instruments": [{"instrument": inst, "weight": w} for inst, w in best_mix],
            "similarity": sim
        })
    return recommendations
⚙️ 命令行参数新增
bash
python midi2notecombo.py --midi input.mid --output out.json --group_by octave
 可选参数:
   --group_by {octave,custom}  音区划分方式（默认: octave）
   --group_size 12             自定义区间大小（--group_by custom 时生效）
   --max_instruments 3         每个音区最多使用的乐器数量（默认: 3）
   --accurate                  启用渲染合成方式获取目标向量（更精确但稍慢，未来实现）
✅ 验证点
确保 scipy.optimize.nnls 可用（已在 pyproject.toml 中声明依赖，pip install 时会自动安装）。

当某八度内音符超出所有乐器的音域时，应输出警告并跳过该八度（或递归拆分）。

混合结果权重建议保留两位小数，匹配度保留两位小数。

对于单一乐器即可完美匹配的情况，推荐列表中只有一个乐器且权重为 1.0。

