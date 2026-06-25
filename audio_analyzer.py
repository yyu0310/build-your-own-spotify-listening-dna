#!/usr/bin/env python3
"""
audio_analyzer.py — 用 Essentia（native + TensorFlow 預訓練模型）分析單一音檔，
回傳一份完整的 audio features dict。

設計重點：
- 模型只在 Analyzer 建構時載入一次，之後對每首歌重複呼叫（17k 首不可能每首重載）。
- 全部走 essentia.standard，不 import essentia.tensorflow（arm64 該子模組有 bug，
  但真正的 TensorflowPredict* 演算法都在 standard 裡，繞過即可）。
- 模型的 output 節點名與 class 順序一律從同名 .json 動態讀取，不寫死。

可單獨執行做測試：
    python3 audio_analyzer.py <音檔路徑>
"""

import os
import json
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import essentia.standard as es

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# musicnn 嵌入模型的輸出層（200 維），所有 musicnn 分類頭都吃這個
MUSICNN_EMB = ("msd-musicnn-1", "model/dense/BiasAdd")
# discogs-effnet 嵌入模型，genre_discogs400 吃這個
EFFNET_EMB = ("discogs-effnet-bs64-1", "PartitionedCall:1")

# musicnn 分類頭：名稱 -> (.pb 檔名, 取哪個 class 當「正向分數」)
# 正向 class 用 json 的 classes 清單對應到我們要的語意欄位
MUSICNN_HEADS = {
    "danceability":       ("danceability-msd-musicnn-1", "danceable"),
    "acousticness":       ("mood_acoustic-msd-musicnn-1", "acoustic"),
    "instrumentalness":   ("voice_instrumental-msd-musicnn-1", "instrumental"),
    "mood_happy":         ("mood_happy-msd-musicnn-1", "happy"),
    "mood_sad":           ("mood_sad-msd-musicnn-1", "sad"),
    "mood_relaxed":       ("mood_relaxed-msd-musicnn-1", "relaxed"),
    "mood_aggressive":    ("mood_aggressive-msd-musicnn-1", "aggressive"),
    "mood_party":         ("mood_party-msd-musicnn-1", "party"),
    "mood_electronic":    ("mood_electronic-msd-musicnn-1", "electronic"),
}


def _model_path(name):
    return os.path.join(MODELS_DIR, name + ".pb")


def _load_classes(name):
    with open(os.path.join(MODELS_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f).get("classes", [])


class Analyzer:
    """載入一次、重複使用。對每首歌呼叫 analyze(path) -> dict。"""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self._log("載入模型中（只做一次）…")

        # --- 嵌入模型 ---
        emb_name, emb_out = MUSICNN_EMB
        self.musicnn_emb = es.TensorflowPredictMusiCNN(
            graphFilename=_model_path(emb_name), output=emb_out)

        eff_name, eff_out = EFFNET_EMB
        self.effnet_emb = es.TensorflowPredictEffnetDiscogs(
            graphFilename=_model_path(eff_name), output=eff_out)

        # --- musicnn 分類頭 ---
        self.heads = {}        # 欄位名 -> (predictor, 正向 class index)
        for field, (mname, pos_class) in MUSICNN_HEADS.items():
            classes = _load_classes(mname)
            idx = classes.index(pos_class)
            pred = es.TensorflowPredict2D(
                graphFilename=_model_path(mname), output="model/Softmax")
            self.heads[field] = (pred, idx)

        # --- emomusic：回歸，輸出 [valence, arousal]，範圍 1~9 ---
        self.emomusic = es.TensorflowPredict2D(
            graphFilename=_model_path("emomusic-msd-musicnn-2"),
            output="model/Identity")

        # --- genre：rosamerica（musicnn，8 粗類）+ discogs400（effnet，400 細類） ---
        self.rosa_classes = _load_classes("genre_rosamerica-msd-musicnn-1")
        self.genre_rosa = es.TensorflowPredict2D(
            graphFilename=_model_path("genre_rosamerica-msd-musicnn-1"),
            output="model/Softmax")

        self.discogs_classes = _load_classes("genre_discogs400-discogs-effnet-1")
        self.genre_discogs = es.TensorflowPredict2D(
            graphFilename=_model_path("genre_discogs400-discogs-effnet-1"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0")

        # rosamerica 的 8 個代碼對應到可讀標籤
        self.rosa_label = {
            "cla": "classical", "dan": "dance", "hip": "hiphop", "jaz": "jazz",
            "pop": "pop", "rhy": "rhythm_blues", "roc": "rock", "spe": "speech",
        }
        self._log("模型載入完成。")

    def _log(self, msg):
        if self.verbose:
            print(f"[analyzer] {msg}", flush=True)

    def analyze(self, filepath):
        """分析單一音檔，回傳 features dict。失敗會拋例外，由呼叫端處理。"""
        # 16k 單聲道：所有 TF 模型的標準輸入
        audio16 = es.MonoLoader(filename=filepath, sampleRate=16000,
                                resampleQuality=4)()
        # 44.1k：native 節奏/調性分析品質較好
        audio44 = es.MonoLoader(filename=filepath, sampleRate=44100)()

        out = {}

        # ---------- native 特徵 ----------
        bpm, _, beats_conf, _, _ = es.RhythmExtractor2013(method="multifeature")(audio44)
        out["tempo"] = round(float(bpm), 2)
        out["tempo_confidence"] = round(float(beats_conf), 3)

        key, scale, key_strength = es.KeyExtractor()(audio44)
        out["key"] = self._key_to_num(key)          # 0=C … 11=B
        out["mode"] = 1 if scale == "major" else 0   # 對齊 Spotify：1=大調 0=小調
        out["key_strength"] = round(float(key_strength), 3)

        rms = float(np.sqrt(np.mean(audio44 ** 2)))
        loudness_db = 20.0 * np.log10(rms + 1e-9)
        out["loudness"] = round(loudness_db, 2)
        # energy：Spotify 沒有對應算法，用 loudness 正規化到 0~1 當近似（標示為近似）
        out["energy"] = round(float(np.clip((loudness_db + 60.0) / 60.0, 0.0, 1.0)), 3)

        # ---------- TF：musicnn 嵌入 + 各分類頭 ----------
        emb = self.musicnn_emb(audio16)             # (frames, 200)

        for field, (pred, idx) in self.heads.items():
            probs = np.mean(pred(emb), axis=0)
            out[field] = round(float(probs[idx]), 3)

        # emomusic：valence / arousal（1~9 → 0~1）
        va = np.mean(self.emomusic(emb), axis=0)
        out["valence"] = round(float(np.clip((va[0] - 1.0) / 8.0, 0.0, 1.0)), 3)
        out["arousal"] = round(float(np.clip((va[1] - 1.0) / 8.0, 0.0, 1.0)), 3)

        # genre rosamerica（粗）
        rosa = np.mean(self.genre_rosa(emb), axis=0)
        rosa_code = self.rosa_classes[int(np.argmax(rosa))]
        out["genre_rosamerica"] = self.rosa_label.get(rosa_code, rosa_code)

        # ---------- TF：discogs-effnet 嵌入 + genre_discogs400（細） ----------
        emb_eff = self.effnet_emb(audio16)          # (frames, 1280)
        disco = np.mean(self.genre_discogs(emb_eff), axis=0)
        out["genre_discogs"] = self.discogs_classes[int(np.argmax(disco))]

        return out

    @staticmethod
    def _key_to_num(key):
        order = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        flat = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
        key = flat.get(key, key)
        return order.index(key) if key in order else None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法：python3 audio_analyzer.py <音檔路徑>")
        sys.exit(1)
    az = Analyzer()
    import time
    t0 = time.time()
    result = az.analyze(sys.argv[1])
    print(f"\n分析耗時 {time.time() - t0:.1f}s")
    for k, v in result.items():
        print(f"  {k:18} {v}")
