# build-your-own-spotify-listening-dna

Spotify 多年來記錄你聽過的每一首歌。問題是，他們用這些資料推薦音樂的依據，是「其他人喜歡什麼」，不是「你自己真的有反應的是什麼」。

這個 pipeline 拿到你的原始聆聽記錄之後，下載每一首歌的完整音訊，在本機跑聲學分析。最終你會得到一個個人資料庫，裡面有每首歌 20 個音頻特徵，加上你真實的行為訊號：你聽到多少比例才跳走、你重複聽了幾次、你在一天的哪個時段放它。從這裡跑的六支分析腳本，出來的圖表和推薦完全建立在你自己的行為上。沒有協作過濾，沒有「和你相似的人」。

---

## 跑完你手上會有什麼

- **`essentia_features.db`**：一個 SQLite 資料庫，涵蓋你整個聆聽歷史每首歌的 20 個音頻特徵。Energy、valence、danceability、6 個情緒軸、400 類 Discogs 曲風、arousal、tempo、key。

- **`engagement.csv`**：每首歌的參與度分數，結合完聽率、跳過率、近期權重計算。這是用原始行為資料能算出來的、最接近「你到底有多喜歡這首歌」的數字。

- **`Output/` 裡五張圖：**

  - `taste_evolution.png`：從你開始用 Spotify 那年畫到現在的折線圖，顯示你的聆聽偏好在 5 個聲學維度上怎麼移動。如果 energy 分數在五年內持續往下，代表你越來越喜歡安靜的音樂。某一年 valence 突然下沉，那段時期就會在圖上浮出來。

  - `cluster_scatter.png`：你整個曲庫的 PCA 散點圖。每個點是一首歌，根據聲學相似度定位。點的大小代表 engagement 分數，所以你最常完整聽完的歌會在各群集中央大大一個點。通常會出現 4–6 個群，對應不同的情緒：一群是開車高能的歌，一群是深夜純器樂，一群是你某段時期深陷的曲風。

  - `av_map.png`：x 軸是 valence（開心 vs. 憂鬱），y 軸是 arousal（平靜 vs. 激昂）。你的整個曲庫以點雲形式分布在這個情緒座標系上。大點是你真的有在聽的歌。如果你的大點聚在右上角，你偏好開心又有能量的音樂；聚在左下角，代表你傾向內省和低能量的音樂。

  - `context_timeofday.png`：把你的聆聽分成四個時段（深夜、早晨、下午、晚間），顯示每個時段的平均完聽率。可以看出你半夜放的東西你是不是真的有在聽，還是只是背景聲。

  - `context_weekday.png`：同樣的完聽率分析，改以星期幾為單位。

- **推薦工具**：輸入任何一首你曲庫裡的歌，取回 10 首聲學相似的曲目，排序依據是音頻相似度乘上你過去對那類歌的參與度。結果是你早就擁有但可能忘了的歌，因為它們和你一直在完整聽完的歌音樂上很像。

---

## 三層架構，各層做什麼

### 第一層：蒐集

`spotify_collector.py` 透過 launchd（macOS）或 cron（Linux）每小時執行一次，呼叫 Spotify 的「最近播放」API，把最多 50 首寫進 SQLite。

歷史資料的部分，Spotify 提供一份完整的延伸串流歷史下載（GDPR 資料匯出，通常涵蓋你建立帳號以來的 5–15 年）。收到那些 JSON 檔（Spotify 需要 5–30 天寄送）之後，一次性灌進資料庫，pipeline 就有你完整的歷史可以處理。

### 第二層：下載與分析

`audio_pipeline.py` 取出每首尚未分析的曲目，在 YouTube 上找音訊。搜尋順序：

1. **spotDL**：使用 Spotify 內建的 YouTube Music 對照表搜尋，命中率高
2. **yt-dlp + 藝人別名**：spotDL 失敗時，改用 `"藝人名 歌名"` 搜尋，並套用 `artist_aliases.json` 把 Spotify 的羅馬拼音名稱轉換成 YouTube 實際使用的名稱（例如 "Jay Chou" → "周杰倫"，"Stefanie Sun" → "孫燕姿"）

下載以 20 條執行緒並行跑。下完的檔案進 `audio_analyzer.py`，依序跑 Essentia TensorFlow 模型。Essentia 本身已吃滿所有 CPU 核心，同時跑多個分析器不會加速，只會多占記憶體。作者的 MacBook Air M3 16GB RAM 上每首約 5–10 秒。

30 MB 大小守衛攔截搜尋錯誤的結果，單首歌的 AAC 或 Opus 通常 3–10 MB，超過代表抓到了合輯或錯誤版本。

Apple M 系列吞吐量：約 300–600 首 / 小時。

### 第三層：Phase 2 分析

累積 1,000+ 首分析結果之後，Phase 2 腳本把音頻特徵和串流歷史 JSON 裡的行為資料結合起來。

`phase2_engagement.py` 是其他腳本的地基。對每首歌，它讀取所有歷史檔案裡的播放事件，計算：

- **加權播放次數**：依近期加權（2026 年 = 1.0，每往前一年 -0.12，下限 0.1），所以你最近重新發現的歌比你 2018 年常放的歌影響力更大
- **完聽率**：平均播放毫秒數除以歌曲時長，時長從 Kaggle 參考資料集取
- **跳過率**：你主動按下一首的次數佔比

Engagement 分數 = `log1p(加權播放次數) × 完聽率 × (1 - 跳過率)`

聚類分析和情緒地圖都以這個分數加權，所以視覺結果反映的是你真實的偏好，不只是原始播放次數。

---

## 每首歌提取的音頻特徵

| 類型 | 欄位 |
|---|---|
| 與 Spotify 相容 | `tempo`、`key`、`mode`、`loudness`、`energy`、`danceability`、`acousticness`、`instrumentalness`、`valence` |
| Essentia 獨有 | `arousal`、`tempo_confidence`、`key_strength`、`genre_discogs`（400 分類）、`genre_rosamerica` |
| 情緒軸 | `mood_happy`、`mood_sad`、`mood_relaxed`、`mood_aggressive`、`mood_party`、`mood_electronic` |

---

## 快速開始

完整的逐步設定在 [AGENTS.md](AGENTS.md)，涵蓋從申請資料下載到跑完 Phase 2 的每個步驟。精簡版：

```bash
# 1. 申請 Spotify 延伸播放記錄
#    Spotify > 設定 > 帳號 > 隱私設定 > 下載你的資料
#    勾選「延伸播放串流記錄」（需 5–30 天）

# 2. 安裝相依套件（兩個虛擬環境）
python3 -m venv .venv
.venv/bin/pip install essentia-tensorflow numpy spotipy gspread google-auth python-dotenv requests scikit-learn pandas matplotlib

python3 -m venv .venv-spotdl
.venv-spotdl/bin/pip install spotdl yt-dlp

# 3. 設定憑證
mkdir -p ~/.config/spotify-dna
cp .env.example ~/.config/spotify-dna/.env
# 填入 SPOTIFY_CLIENT_ID 和 SPOTIFY_CLIENT_SECRET

# 4. 跑一次 collector 完成 Spotify OAuth 授權
.venv/bin/python3 spotify_collector.py

# 5. 灌入串流歷史 JSON（見 AGENTS.md 指令）後啟動 pipeline
caffeinate -i .venv/bin/python3 audio_pipeline.py --workers 20 > pipeline.log 2>&1 &
tail -f pipeline.log   # 即時追蹤進度

# 停止
pkill -f audio_pipeline.py
```

**指令說明：**

- `caffeinate -i`：防止 Mac 睡眠，讓管道可以持續長時間跑
- `--workers 20`：同時開 20 條下載執行緒，每條分別在 YouTube 搜尋並下載一首歌
- `> pipeline.log 2>&1 &`：背景執行，輸出轉存進 `pipeline.log`，terminal 不會卡住

**管道內部流程（每首歌）：**

1. spotDL 搜尋 YouTube Music
2. 失敗的話，yt-dlp 搜尋並套用藝人名稱轉換
3. 下載音訊（AAC / Opus，3–10 MB）
4. Essentia TensorFlow 本機分析，提取 20 個特徵（約 5–10 秒 / 首）
5. 寫入 `essentia_features.db`，刪除暫存音訊

管道**斷點續跑**，停掉後重新執行從上次中斷處繼續，不重複分析已完成的曲目。

預期吞吐量（Apple M 系列）：300–600 首 / 小時。一萬首約 20–30 小時，開著去睡覺即可。

---

## 跑 Phase 2

```bash
.venv/bin/pip install scikit-learn pandas matplotlib

.venv/bin/python3 tools/phase2_engagement.py    # 必須先跑，產生 engagement.csv
.venv/bin/python3 tools/phase2_evolution.py     # taste_evolution.png
.venv/bin/python3 tools/phase2_cluster.py       # cluster_scatter.png（需要 engagement.csv）
.venv/bin/python3 tools/phase2_av_map.py        # av_map.png（需要 clusters.csv + engagement.csv）
.venv/bin/python3 tools/phase2_context.py       # context_timeofday.png、context_weekday.png
.venv/bin/python3 tools/phase2_recommend.py "歌名 藝人"

.venv/bin/python3 tools/phase2_fingerprint.py  # 印出音樂 DNA 摘要
```

---

## Spotify 藝人名對照 YouTube 搜尋詞

`artist_aliases.json` 記錄 Spotify 羅馬拼音名稱對應的 YouTube 頻道名或搜尋詞。Spotify 對許多東亞藝人統一顯示英文拼音，但 YouTube 頻道用的是本名，yt-dlp 用錯名稱搜尋就找不到。這個表格就是兩者之間的橋。

**歡迎提交 PR 補充更多藝人，尤其是韓國、中國大陸、越南的藝人。**

| Spotify | YouTube |
|---|---|
| Jay Chou | 周杰倫 |
| Stefanie Sun | 孫燕姿 |
| Enno Cheng | 鄭宜農 Enno Cheng |
| Jonathan Lee | 李宗盛 |
| JJ Lin | 林俊傑 |
| Jolin Tsai | JOLIN 蔡依林 |
| Tanya Chua | 蔡健雅 |
| A-Mei Chang | 張惠妹 |
| Rainie Yang | 楊丞琳 |
| Princess Ai | 戴愛玲 |
| Jess Lee | 李佳薇 Jess Lee 官方音樂頻道 |
| ABAO阿爆 | ABAO阿爆_阿仍仍 |
| Joanna Wang | Joanna Wang 王若琳 |
| Wayne's so Sad | 傷心欲絕Wayneʻs So Sad |
| GBOYSWAG | 鼓鼓 呂思緯 GBOYSWAG |
| Elephant Gym | 大象體操Elephant Gym |
| Flesh Juicer | Flesh Juicer 血肉果汁機 |
| Major in Body Bear 體熊專科 | major in bodybear |
| Tizzy Bac | tizzybacvideo |
| 當代電影大師 | 當代電影大師Modern Cinema Master |
| Kairi Yagi | 八木海莉 |
| 八木海莉⚡️電音遊戯 | 八木海莉 |
| Satoko Shibata | 柴田聡子 \| Satoko Shibata |

---

## 注意事項

- **Spotify Audio Features API 已停用**：Spotify 自 2024 年 11 月起對 Developer 模式應用回傳 403。本專案以 Kaggle/HuggingFace 公開資料集加上自行 Essentia 計算繞過。

- **valence 分數有西方偏差**：emomusic 模型以 Million Song Dataset（主要是西方流行和搖滾）訓練。台灣和日本的高能量曲目有時 valence 偏低，但聽起來並不負面。這些曲風用 `mood_aggressive` 和 `mood_party` 比 `valence` 可靠。

- **只跑一個分析器**：Essentia TensorFlow 已用盡所有 CPU 核心的 intra-op 平行度。同時跑兩個分析器不會提升吞吐量，只會減少每個進程的可用記憶體。Pipeline 強制一次只跑一個。

- **部分曲目在 YouTube 找不到**：現場錄音、地區獨家、Spotify 限定內容不會有對應的 YouTube 上傳。這些會以 `download_not_found` 留在進度資料庫。解法是自己找一個備用 URL，加進 `track_url_overrides.json`。自動化這一步需要另一套音樂搜尋 API，有自己的限流和比對問題，所以刻意保留為手動。

- **2025–2026 的新歌**：Kaggle 和 HuggingFace 資料集有截止日期，非常新的曲目在兩個參考庫都可能找不到。Pipeline 仍會嘗試從 YouTube 下載分析，但搜尋比對錯誤的機率比舊歌高。
