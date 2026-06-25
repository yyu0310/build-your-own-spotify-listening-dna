# build-your-own-spotify-listening-dna

Spotify 多年来记录你听过的每一首歌。问题是，他们用这些数据推荐音乐的依据，是「其他人喜欢什么」，不是「你自己真的有反应的是什么」。

这个 pipeline 拿到你的原始聆听记录之后，下载每一首歌的完整音频，在本机跑声学分析。最终你会得到一个个人数据库，里面有每首歌 20 个音频特征，加上你真实的行为信号：你听到多少比例才跳走、你重复听了几次、你在一天的哪个时段放它。从这里跑的六支分析脚本，出来的图表和推荐完全建立在你自己的行为上。没有协同过滤，没有「和你相似的人」。

---

## 跑完你手上会有什么

- **`essentia_features.db`**：一个 SQLite 数据库，涵盖你整个聆听历史每首歌的 20 个音频特征。Energy、valence、danceability、6 个情绪轴、400 类 Discogs 曲风、arousal、tempo、key。

- **`engagement.csv`**：每首歌的参与度分数，结合完听率、跳过率、近期权重计算。这是用原始行为数据能算出来的、最接近「你到底有多喜欢这首歌」的数字。

- **`Output/` 里五张图：**

  - `taste_evolution.png`：从你开始用 Spotify 那年画到现在的折线图，显示你的聆听偏好在 5 个声学维度上怎么移动。如果 energy 分数在五年内持续往下，代表你越来越喜欢安静的音乐。某一年 valence 突然下沉，那段时期就会在图上浮出来。

  - `cluster_scatter.png`：你整个曲库的 PCA 散点图。每个点是一首歌，根据声学相似度定位。点的大小代表 engagement 分数，所以你最常完整听完的歌会在各群集中央大大一个点。通常会出现 4–6 个群，对应不同的情绪：一群是开车高能的歌，一群是深夜纯器乐，一群是你某段时期深陷的曲风。

  - `av_map.png`：x 轴是 valence（开心 vs. 忧郁），y 轴是 arousal（平静 vs. 激昂）。你的整个曲库以点云形式分布在这个情绪坐标系上。大点是你真的有在听的歌。如果你的大点聚在右上角，你偏好开心又有能量的音乐；聚在左下角，代表你倾向内省和低能量的音乐。

  - `context_timeofday.png`：把你的聆听分成四个时段（深夜、早晨、下午、晚间），显示每个时段的平均完听率。可以看出你半夜放的东西你是不是真的有在听，还是只是背景声。

  - `context_weekday.png`：同样的完听率分析，改以星期几为单位。

- **推荐工具**：输入任何一首你曲库里的歌，取回 10 首声学相似的曲目，排序依据是音频相似度乘上你过去对那类歌的参与度。结果是你早就拥有但可能忘了的歌，因为它们和你一直在完整听完的歌音乐上很像。

---

## 三层架构，各层做什么

### 第一层：采集

`spotify_collector.py` 通过 launchd（macOS）或 cron（Linux）每小时执行一次，调用 Spotify 的「最近播放」API，把最多 50 首写进 SQLite。

历史数据的部分，Spotify 提供一份完整的延伸串流历史下载（GDPR 数据导出，通常涵盖你创建账号以来的 5–15 年）。收到那些 JSON 文件（Spotify 需要 5–30 天发送）之后，一次性导入数据库，pipeline 就有你完整的历史可以处理。

### 第二层：下载与分析

`audio_pipeline.py` 取出每首尚未分析的曲目，在 YouTube 上找音频。搜索顺序：

1. **spotDL**：使用 Spotify 内置的 YouTube Music 对照表搜索，命中率高
2. **yt-dlp + 艺人别名**：spotDL 失败时，改用 `"艺人名 歌名"` 搜索，并套用 `artist_aliases.json` 把 Spotify 的罗马拼音名称转换成 YouTube 实际使用的名称（例如 "Jay Chou" → "周杰伦"，"Stefanie Sun" → "孙燕姿"）

下载以 20 条线程并行跑。下完的文件进 `audio_analyzer.py`，依序跑 Essentia TensorFlow 模型。Essentia 本身已吃满所有 CPU 核心，同时跑多个分析器不会加速，只会多占内存。作者的 MacBook Air M3 16GB RAM 上每首约 5–10 秒。

30 MB 大小守卫拦截搜索错误的结果；单首歌的 AAC 或 Opus 通常 3–10 MB，超过代表抓到了合辑或错误版本。

Apple M 系列吞吐量：约 300–600 首 / 小时。

### 第三层：Phase 2 分析

累积 1,000+ 首分析结果之后，Phase 2 脚本把音频特征和串流历史 JSON 里的行为数据结合起来。

`phase2_engagement.py` 是其他脚本的地基。对每首歌，它读取所有历史文件里的播放事件，计算：

- **加权播放次数**：依近期加权（2026 年 = 1.0，每往前一年 -0.12，下限 0.1），所以你最近重新发现的歌比你 2018 年常放的歌影响力更大
- **完听率**：平均播放毫秒数除以歌曲时长，时长从 Kaggle 参考数据集取
- **跳过率**：你主动按下一首的次数占比

Engagement 分数 = `log1p(加权播放次数) × 完听率 × (1 - 跳过率)`

聚类分析和情绪地图都以这个分数加权，所以视觉结果反映的是你真实的偏好，不只是原始播放次数。

---

## 每首歌提取的音频特征

| 类型 | 字段 |
|---|---|
| 与 Spotify 兼容 | `tempo`、`key`、`mode`、`loudness`、`energy`、`danceability`、`acousticness`、`instrumentalness`、`valence` |
| Essentia 独有 | `arousal`、`tempo_confidence`、`key_strength`、`genre_discogs`（400 分类）、`genre_rosamerica` |
| 情绪轴 | `mood_happy`、`mood_sad`、`mood_relaxed`、`mood_aggressive`、`mood_party`、`mood_electronic` |

---

## 快速开始

完整的逐步设置在 [AGENTS.md](AGENTS.md)，涵盖从申请数据下载到跑完 Phase 2 的每个步骤。精简版：

```bash
# 1. 申请 Spotify 延伸播放记录
#    Spotify > 设置 > 账号 > 隐私设置 > 下载你的数据
#    勾选「延伸播放串流记录」（需 5–30 天）

# 2. 安装依赖（两个虚拟环境）
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy spotipy gspread google-auth python-dotenv requests scikit-learn pandas matplotlib

python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp

# 3. 配置凭证
mkdir -p ~/.config/spotify-dna
cp .env.example ~/.config/spotify-dna/.env
# 填入 SPOTIFY_CLIENT_ID 和 SPOTIFY_CLIENT_SECRET

# 4. 跑一次 collector 完成 Spotify OAuth 授权
.venv/bin/python3 spotify_collector.py

# 5. 导入串流历史 JSON（见 AGENTS.md 指令）后启动 pipeline
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &
tail -f pipeline.log   # 实时追踪进度

# 停止
pkill -f audio_pipeline.py
```

**指令说明：**

- `caffeinate -i`：防止 Mac 睡眠，让管道可以持续长时间跑
- `--workers 20`：同时开 20 条下载线程，每条分别在 YouTube 搜索并下载一首歌
- `> pipeline.log 2>&1 &`：后台执行，输出转存进 `pipeline.log`，terminal 不会卡住

**管道内部流程（每首歌）：**

1. spotDL 搜索 YouTube Music
2. 失败的话，yt-dlp 搜索并套用艺人名称转换
3. 下载音频（AAC / Opus，3–10 MB）
4. Essentia TensorFlow 本机分析，提取 20 个特征（约 5–10 秒 / 首）
5. 写入 `essentia_features.db`，删除暂存音频

管道**断点续跑**，停掉后重新执行从上次中断处继续，不重复分析已完成的曲目。

预期吞吐量（Apple M 系列）：300–600 首 / 小时。一万首约 20–30 小时，开着去睡觉即可。

---

## 跑 Phase 2

```bash
.venv/bin/pip install scikit-learn pandas matplotlib

.venv/bin/python3 tools/phase2_engagement.py    # 必须先跑，产生 engagement.csv
.venv/bin/python3 tools/phase2_evolution.py     # taste_evolution.png
.venv/bin/python3 tools/phase2_cluster.py       # cluster_scatter.png（需要 engagement.csv）
.venv/bin/python3 tools/phase2_av_map.py        # av_map.png（需要 clusters.csv + engagement.csv）
.venv/bin/python3 tools/phase2_context.py       # context_timeofday.png、context_weekday.png
.venv/bin/python3 tools/phase2_recommend.py "歌名 艺人"

.venv/bin/python3 tools/phase2_fingerprint.py  # 打印音乐 DNA 摘要
```

---

## Spotify 艺人名对照 YouTube 搜索词

`artist_aliases.json` 记录 Spotify 罗马拼音名称对应的 YouTube 频道名或搜索词。Spotify 对许多东亚艺人统一显示英文拼音，但 YouTube 频道用的是本名，yt-dlp 用错名称搜索就找不到。这个表格就是两者之间的桥。

**欢迎提交 PR 补充更多艺人，尤其是韩国、中国大陆、越南的艺人。**

| Spotify | YouTube |
|---|---|
| Jay Chou | 周杰伦 |
| Stefanie Sun | 孙燕姿 |
| Enno Cheng | 郑宜农 Enno Cheng |
| Jonathan Lee | 李宗盛 |
| JJ Lin | 林俊杰 |
| Jolin Tsai | JOLIN 蔡依林 |
| Tanya Chua | 蔡健雅 |
| A-Mei Chang | 张惠妹 |
| Rainie Yang | 杨丞琳 |
| Princess Ai | 戴爱玲 |
| Jess Lee | 李佳薇 Jess Lee 官方音乐频道 |
| ABAO阿爆 | ABAO阿爆_阿仍仍 |
| Joanna Wang | Joanna Wang 王若琳 |
| Wayne's so Sad | 伤心欲绝Wayneʻs So Sad |
| GBOYSWAG | 鼓鼓 吕思纬 GBOYSWAG |
| Elephant Gym | 大象体操Elephant Gym |
| Flesh Juicer | Flesh Juicer 血肉果汁机 |
| Major in Body Bear 体熊专科 | major in bodybear |
| Tizzy Bac | tizzybacvideo |
| 当代电影大师 | 当代电影大师Modern Cinema Master |
| Kairi Yagi | 八木海莉 |
| 八木海莉⚡️電音遊戯 | 八木海莉 |
| Satoko Shibata | 柴田聡子 \| Satoko Shibata |

---

## 注意事项

- **Spotify Audio Features API 已停用**：Spotify 自 2024 年 11 月起对 Developer 模式应用返回 403。本项目以 Kaggle/HuggingFace 公开数据集加上自行 Essentia 计算绕过。

- **valence 分数有西方偏差**：emomusic 模型以 Million Song Dataset（主要是西方流行和摇滚）训练。台湾和日本的高能量曲目有时 valence 偏低，但听起来并不负面。这些曲风用 `mood_aggressive` 和 `mood_party` 比 `valence` 可靠。

- **只跑一个分析器**：Essentia TensorFlow 已用尽所有 CPU 核心的 intra-op 并行度。同时跑两个分析器不会提升吞吐量，只会减少每个进程的可用内存。Pipeline 强制一次只跑一个。

- **部分曲目在 YouTube 找不到**：现场录音、地区独家、Spotify 限定内容不会有对应的 YouTube 上传。这些会以 `download_not_found` 留在进度数据库。解法是自己找一个备用 URL，加进 `track_url_overrides.json`。自动化这一步需要另一套音乐搜索 API，有自己的限流和比对问题，所以刻意保留为手动。

- **2025–2026 的新歌**：Kaggle 和 HuggingFace 数据集有截止日期，非常新的曲目在两个参考库都可能找不到。Pipeline 仍会尝试从 YouTube 下载分析，但搜索比对错误的概率比旧歌高。
