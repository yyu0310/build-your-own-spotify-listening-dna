# build-your-own-spotify-listening-dna

用自己的聆听行为，打造你的音乐 DNA。

本项目下载你的 Spotify 播放记录，从 YouTube 抓取完整音频，以 Essentia TensorFlow 模型在本机分析 20 个音频特征，最后通过 6 支 Phase 2 分析脚本输出你专属的音乐人格图谱。

**与流媒体服务推荐的差异**：这里用的是你实际的聆听行为（跳过率、完听率、听歌时段），不是协同过滤（"与你相似的人也在听"）。

---

## 功能

### 采集层

| 脚本 | 功能 |
|---|---|
| `spotify_collector.py` | 每小时从 Spotify API 抓取最近播放，存入 SQLite |
| `audio_pipeline.py` | 并行下载音频（spotDL / yt-dlp），以 Essentia 分析，写入 DB |
| `sheet_backfill_from_db.py` | 将分析结果回填至 Google Sheets（可选） |

### 提取的音频特征

| 类型 | 字段 |
|---|---|
| 与 Spotify 兼容 | `tempo`、`key`、`mode`、`loudness`、`energy`、`danceability`、`acousticness`、`instrumentalness`、`valence` |
| Essentia 独有 | `arousal`、`tempo_confidence`、`key_strength`、`genre_discogs`（400 分类）、`genre_rosamerica` |
| 情绪轴 | `mood_happy`、`mood_sad`、`mood_relaxed`、`mood_aggressive`、`mood_party`、`mood_electronic` |

---

## 系统架构

```
macOS launchd（每小时）
  → spotify_collector.py
      → Spotify API（最近播放）
      → SQLite DBs

caffeinate（长时间运行）
  → audio_pipeline.py --workers 20
      → spotDL / yt-dlp（ThreadPoolExecutor 并行下载）
      → audio_analyzer.py（Essentia，单线程推理）
      → essentia_features.db
```

---

## Phase 2 分析套件

Pipeline 分析完 1,000+ 首后，执行 Phase 2 进行深度分析。

```bash
.venv/bin/pip install scikit-learn pandas matplotlib

# 按顺序执行：
.venv/bin/python3 tools/phase2_engagement.py    # 基础：计算每首歌的 engagement 分数
.venv/bin/python3 tools/phase2_evolution.py     # 口味演化折线图（2017→2026）
.venv/bin/python3 tools/phase2_cluster.py       # K-Means 聚类（以 engagement 加权）
.venv/bin/python3 tools/phase2_av_map.py        # Arousal-Valence 情绪地图
.venv/bin/python3 tools/phase2_context.py       # 时段聆听模式分析
.venv/bin/python3 tools/phase2_recommend.py "歌名 艺人"  # 从自己的曲库推荐
.venv/bin/python3 tools/phase2_fingerprint.py  # 打印音乐 DNA 摘要
```

---

## 快速开始

### 1. 申请 Spotify 扩展播放记录

前往 Spotify → 设置 → 帐号 → 隐私设置 → 下载你的数据。  
勾选"扩展流媒体记录"（非普通帐号数据）。  
Spotify 以 email 发送，通常需要 5–30 天。  
解压后将所有 `Streaming_History_Audio_*.json` 放入 `Database/Spotify Extended Streaming History/`。

### 2. 创建 Spotify Developer App

前往 https://developer.spotify.com/dashboard 创建应用。  
Redirect URI 设为：`http://127.0.0.1:8888/callback`。  
复制 Client ID 和 Client Secret。

### 3. 安装依赖（需两个虚拟环境）

```bash
# Essentia 分析环境
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy spotipy gspread google-auth python-dotenv requests scikit-learn pandas matplotlib

# 音频下载环境（spotDL 与 Essentia 有依赖冲突）
python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp

# 系统层 yt-dlp（备用）
brew install yt-dlp  # macOS
```

### 4. 配置凭证

```bash
mkdir -p ~/.config/spotify-dna
cp .env.example ~/.config/spotify-dna/.env
# 编辑 ~/.config/spotify-dna/.env，填入 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
```

### 5. 运行 Pipeline

```bash
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &
tail -f pipeline.log

# 停止
pkill -f audio_pipeline.py
```

---

## 注意事项

- **valence 偏差**：emomusic 模型以西方音乐（MSD）训练，台湾/日本金属/电子乐可能分数偏低，建议同时看 `mood_aggressive` + `mood_party`。
- **单一分析器性能上限**：Essentia TF 已占满所有 CPU 核心，多开分析器不会加速，只会增加内存负担。
- **Spotify Audio Features API 返回 403**：Spotify 自 2024 年 11 月起对 Developer 模式应用关闭此 API，本项目以 Kaggle/HuggingFace 公开数据集 + 本地 Essentia 计算绕过此限制。

详见完整英文文档：[README.md](README.md) 和 [AGENTS.md](AGENTS.md)。
