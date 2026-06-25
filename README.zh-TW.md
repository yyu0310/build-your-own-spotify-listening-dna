# build-your-own-spotify-listening-dna

用自己的聆聽行為，打造你的音樂 DNA。

本專案下載你的 Spotify 播放記錄，從 YouTube 抓取完整音訊，以 Essentia TensorFlow 模型在本機分析 20 個音頻特徵，最後透過 6 支 Phase 2 分析腳本輸出你專屬的音樂人格圖譜。

**和串流服務推薦的差異**：這裡用的是你實際的聆聽行為（跳過率、完聽率、聽歌時段），不是協作過濾（「和你相似的人也在聽」）。

---

## 功能

### 抓取層

| 腳本 | 功能 |
|---|---|
| `spotify_collector.py` | 每小時從 Spotify API 抓最近播放，存入 SQLite |
| `audio_pipeline.py` | 並行下載音訊（spotDL / yt-dlp），以 Essentia 分析，寫入 DB |
| `sheet_backfill_from_db.py` | 把分析結果回填至 Google Sheets（可選） |

### 提取的音頻特徵

| 類型 | 欄位 |
|---|---|
| 與 Spotify 相容 | `tempo`、`key`、`mode`、`loudness`、`energy`、`danceability`、`acousticness`、`instrumentalness`、`valence` |
| Essentia 獨有 | `arousal`、`tempo_confidence`、`key_strength`、`genre_discogs`（400 分類）、`genre_rosamerica` |
| 情緒軸 | `mood_happy`、`mood_sad`、`mood_relaxed`、`mood_aggressive`、`mood_party`、`mood_electronic` |

---

## 系統架構

```
macOS launchd（每小時）
  → spotify_collector.py
      → Spotify API（最近播放）
      → SQLite DBs

caffeinate（長時間運行）
  → audio_pipeline.py --workers 20
      → spotDL / yt-dlp（ThreadPoolExecutor 並行下載）
      → audio_analyzer.py（Essentia，單執行緒推論）
      → essentia_features.db
```

### 資料庫架構

三個獨立 SQLite DB（分庫可避免授權問題）：

| DB | 來源 | 說明 |
|---|---|---|
| `essentia_features.db` | 自行運算 | 主要輸出，可自由開源 |
| `kaggle_features.db` | Kaggle 公開資料集 | 參考用，可選 |
| `huggingface_features.db` | HuggingFace 資料集 | 參考用，可選 |

---

## Phase 2 分析套件

Pipeline 分析完 1,000+ 首後，執行 Phase 2 進行深度分析。

```bash
.venv/bin/pip install scikit-learn pandas matplotlib

# 依序執行：
.venv/bin/python3 tools/phase2_engagement.py    # 地基：算每首歌的 engagement 分數
.venv/bin/python3 tools/phase2_evolution.py     # 口味演化折線圖（2017→2026）
.venv/bin/python3 tools/phase2_cluster.py       # K-Means 聚類（以 engagement 加權）
.venv/bin/python3 tools/phase2_av_map.py        # Arousal-Valence 情緒地圖
.venv/bin/python3 tools/phase2_context.py       # 時段聆聽模式分析
.venv/bin/python3 tools/phase2_recommend.py "歌名 藝人"  # 從自己的曲庫推薦
.venv/bin/python3 tools/phase2_fingerprint.py  # 印出音樂 DNA 摘要
```

| 腳本 | 輸出 | 說明 |
|---|---|---|
| `phase2_engagement.py` | `engagement.csv` | 完聽率 × (1-跳過率) × 近期權重 |
| `phase2_evolution.py` | `taste_evolution.png` | 歷年 5 特徵趨勢圖 |
| `phase2_cluster.py` | `cluster_scatter.png` | PCA 散點，點大小 = 你有多喜歡它 |
| `phase2_av_map.py` | `av_map.png` | 情緒地圖：平靜↔激昂 × 負面↔正面 |
| `phase2_context.py` | `context_*.png` | 你在哪個時段最享受哪種音樂 |
| `phase2_recommend.py` | 終端輸出 | 「你喜歡 X，可能忘了 Y」 |

---

## 快速開始

### 1. 申請 Spotify 延伸播放記錄

前往 Spotify → 設定 → 帳號 → 隱私設定 → 下載你的資料。  
勾選「延伸播放串流記錄」（非一般帳號資料）。  
Spotify 以 email 寄送，通常需要 5–30 天。  
解壓縮後將所有 `Streaming_History_Audio_*.json` 放入 `Database/Spotify Extended Streaming History/`。

### 2. 建立 Spotify Developer App

前往 https://developer.spotify.com/dashboard 建立應用程式。  
Redirect URI 設為：`http://127.0.0.1:8888/callback`。  
複製 Client ID 和 Client Secret。

### 3. 安裝相依套件（需兩個虛擬環境）

```bash
# Essentia 分析環境
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy spotipy gspread google-auth python-dotenv requests scikit-learn pandas matplotlib

# 音訊下載環境（spotDL 與 Essentia 有相依衝突）
python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp

# 系統層 yt-dlp（備援用）
brew install yt-dlp
```

### 4. 設定憑證

```bash
mkdir -p ~/.config/spotify-dna
cp .env.example ~/.config/spotify-dna/.env
# 編輯 ~/.config/spotify-dna/.env，填入 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET
```

### 5. 跑 Pipeline

```bash
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &
tail -f pipeline.log

# 停止
pkill -f audio_pipeline.py
```

---

## Spotify 藝人名 → YouTube 搜尋名稱對照

`artist_aliases.json` 記錄 Spotify 羅馬拼音名稱對應的 YouTube 頻道或搜尋詞。許多華語/日語藝人在 Spotify 上使用英文拼音，但 YouTube 頻道使用本名，導致 yt-dlp 搜尋失敗，此表格即為解法。

**歡迎提交 PR 補充更多藝人！**

---

## 常見問題

**pipeline 跑完有很多 `download_not_found`**：先確認 `artist_aliases.json` 有沒有漏掉藝人。若真的不在 YouTube 上，可手動在 `track_url_overrides.json` 加入對應 URL。

**Essentia model 找不到**：`models/*.json` 只是後設資料，需另外下載 `.pb` 模型檔案，詳見 [AGENTS.md](AGENTS.md) Step 4。

**Spotify Audio Features API 回傳 403**：Spotify 自 2024 年 11 月起對 Development Mode 應用程式關閉此 API，本專案以 Kaggle/HuggingFace 公開資料集 + 自行 Essentia 計算繞過此限制。

---

## 注意事項

- **valence 偏差**：emomusic 模型以西方音樂（MSD）訓練，台灣/日本金屬/電子樂可能分數偏低。建議同時看 `mood_aggressive` + `mood_party`。
- **單一分析器效能上限**：Essentia TF 已佔滿所有 CPU 核心，多開分析器不會加速，只會增加記憶體負擔。
- **部分曲目無 YouTube 版本**：現場錄音、平台獨家等內容永遠無法下載，需手動加 override。
