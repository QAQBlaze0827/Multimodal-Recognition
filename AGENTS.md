# Multimodal-Recognition

## 專案概述
多模態情緒辨識系統，即時分析 Webcam 臉部表情 + 麥克風語調，輸出 4 類情緒（neutral, happy, sad, anger）。支援 PC (Windows) 與 Raspberry Pi 5 部署。

## 已完成
- 完整推論管線：src/video/、src/audio/、src/fusion/、src/output/
- config.yaml 驅動設定 + src/config.py 預設值深層合併
- 人臉偵測雙後端：MediaPipe → Haar cascade 自動降級
- 視訊 heuristic fallback（smile detection + contrast）
- 音訊 heuristic fallback（energy-based）
- ONNX Runtime 統一 helper：create_ort_session() 在 shared_types.py
  - ARM (aarch64/armv7l) → XNNPACKExecutionProvider
  - 其他 → CPUExecutionProvider
- 相依清理：移除 faster-whisper、RPi 需求移除 mediapipe
- WSL 腳本：setup_wsl.sh、run_vision.sh、run_smoke_test.sh
- 訓練腳本全數完成：download_fer2013.py、download_audio_datasets.py、train_video_mini_xception.py、train_audio_tiny_cnn.py
- 環境建置：.venv（Python 3.11），訓練/執行依賴已安裝
- FER2013 資料集下載 + 訓練：mini_xception_fp32.onnx（28,709 張圖片，30 epochs）
- RAVDESS + TESS 音訊訓練：tiny_cnn_audio_fp32.onnx（4,240 樣本，80 epochs），**val_acc 76.1%**
- 音訊精度改善（Phase 1+2）：peak normalization、data augmentation（noise/volume/SpecAugment）、ReduceLROnPlateau、temporal smoothing（alpha=0.7）
- **CREMA-D 測試判定為雜訊**：含入後 val_acc 從 76% 暴跌至 53%，已排除
- **Phase 3 測試判定無效**：N_MFCC=26+delta、Conv1D 64→64、Dense 128、Dropout 0.5+L2 等變動均導致準確度下降
- 驗證通過：模型載入 + dummy inference 正常
- **Web 前後端 + SQLite 資料庫**：
  - `backend/`：FastAPI 伺服器（REST API + WebSocket + 靜態檔案服務）
  - `frontend/`：暗色系 SPA（Live/Replay/History/Analytics 四頁，RWD 支援行動裝置）
  - `src/output/db_logger.py`：推論結果寫入 SQLite（每秒 1 筆，30 天自動清理）
  - 所有 API 端點 + WebSocket 即時推送已驗證通過
- **情緒類別 7→4 縮減**：移除 fear/surprise/disgust，僅保留 neutral/happy/sad/anger
  - 推論層：`EMOTIONS` tuple 改為 4 類，ONNX 模型 7 類輸出取前 4 再正規化
  - 前端：charts.js / app.js / index.html / style.css 同步更新
  - 資料庫：舊資料清空，重新開始
  - 訓練腳本已同步更新，未來可重新訓練 4 類模型

## 專案結構
```
Multimodal-Recognition/
├── app.py                         # 入口點
├── src/
│   ├── main.py                    # 主循環、CLI 參數
│   ├── config.py                  # 設定載入 + 預設值
│   ├── shared_types.py            # 共用型別、create_ort_session()
│   ├── video/                     # 視覺管線
│   │   ├── capture.py             # OpenCV 相機初始化
│   │   ├── face_detection.py      # MediaPipe / Haar 人臉偵測
│   │   ├── model.py               # Mini-Xception ONNX 載入 + 推論
│   │   └── inference.py           # Frame skip 封裝
│   ├── audio/                     # 音訊管線
│   │   ├── features.py            # 輕量 MFCC（無 librosa）
│   │   ├── model.py               # Tiny 1D-CNN ONNX 載入 + 推論
│   │   └── inference.py           # sounddevice 擷取 + 推論執行緒
│   ├── fusion/
│   │   └── late_fusion.py         # 信心度加權融合
│   └── output/
│       ├── display.py             # OpenCV 疊加顯示
│       ├── logger.py              # CSV 日誌
│       └── db_logger.py           # SQLite 資料庫寫入
├── backend/                       # FastAPI 後端伺服器
│   ├── __init__.py
│   ├── app.py                     # REST API + WebSocket + 靜態檔案服務
│   └── database.py                # SQLite CRUD（WAL mode）
├── frontend/                      # 前端 SPA 網頁
│   ├── index.html                 # 暗色系儀表板（Live/Replay/History/Analytics）
│   ├── css/
│   │   └── style.css              # RWD 響應式樣式
│   └── js/
│       ├── app.js                 # 主邏輯、路由切換
│       ├── api.js                 # REST API 客戶端
│       ├── websocket.js           # WebSocket 客戶端（自動重連）
│       └── charts.js              # Chart.js 圖表設定
├── config/config.yaml             # 可調參數（含 backend 段落）
├── models/                        # ONNX 模型放這裡（.gitkeep）
├── scripts/                       # 訓練 + 環境初始化腳本
│   ├── train_video_mini_xception.py   # TensorFlow 訓練視覺模型
│   ├── train_audio_tiny_cnn.py        # TensorFlow 訓練音訊模型
│   ├── download_fer2013.py            # 下載 FER2013 資料集
│   ├── download_audio_datasets.py     # 下載音訊資料集
│   ├── run_train_vision.sh            # WSL 背景執行視覺訓練
│   ├── run_train_audio.sh             # WSL 背景執行音訊訓練
│   ├── run_download_audio.sh          # WSL 背景執行音訊下載
│   ├── setup_windows.ps1
│   ├── setup_wsl.sh
│   ├── run_vision.sh / .ps1
│   └── run_smoke_test.sh / .ps1
├── AGENTS.md                      # 本檔案
├── ARCHITECTURE.md                # 完整設計文件
├── TRAINING.md                    # 訓練說明
├── requirements.txt               # PC 依賴
├── requirements_rpi.txt           # RPi 5 依賴（無 mediapipe）
├── requirements_train.txt         # 訓練用依賴（tensorflow + tf2onnx）
└── requirements_backend.txt       # 後端依賴（fastapi + uvicorn）
```

## 目前限制
- 音訊最佳模型使用 RAVDESS（1,440）+ TESS（2,800）= 4,240 樣本（val_acc 76.1%）
- CREMA-D（7,442 筆）判定為雜訊，含入後 val_acc 降至 53%
- MFCC 特徵改用真實 mel-scale filterbank + DCT（numpy-only），無額外相依
- WSL 無 webcam，無法在本機執行 app.py（需 Windows 實體機）
- protobuf/mpl-dtypes 版本衝突（mediapipe vs tf2onnx/tensorflow），訓練仍可正常運作

## 待辦事項
- [x] 下載 FER2013 資料集到 data/datasets/fer
- [x] 訓練 mini_xception_fp32.onnx（int8 版本因 ConvInteger 限制改用 FP32）
- [x] 下載 RAVDESS + TESS 音訊資料集
- [x] 訓練 tiny_cnn_audio_fp32.onnx（Phase 2：peak norm + augmentation + LR schedule，val_acc 76.1%）
- [x] 驗證：模型載入 + dummy 推論通過
- [x] 下載 CREMA-D 並測試 → 判定為雜訊，含入後從 76%→53%
- [x] Phase 3 測試（MFCC 26、delta、bigger model、dropout 0.5+L2）→ 無效，全數 < 54%
- [x] Web 前後端 + SQLite 資料庫（FastAPI + SPA + WebSocket 即時推送）
- [x] 情緒類別 7→4 縮減：shared_types.py / 推論管線 / 前端 / 資料庫 / 訓練腳本
- [ ] 在 Windows 本機 Python 執行 app.py 測試 webcam
- [x] **音訊權重優化（Phase 7：face-gating + modality 權重調降）**
  - [x] `config.yaml` 新增 `fusion.audio_weight`（0.5）、`fusion.neutral_penalty`（0.3）、`fusion.gate_audio_with_face`（true）
  - [x] `late_fusion.py`：新增 `audio_weight` 全域調降語音信心度、`neutral_penalty` 對高信心 neutral 再折扣
  - [x] `main.py`：人臉消失時將 `audio` 設為 None（face-gating），fusion 呼叫傳遞新參數
  - [x] `display.py`：`draw_overlay` 新增 `audio_gated` 參數，顯示 `(gated)` 標記
- [x] **音訊推論精度改善（Phase 4：inference 管線修正）**
  - [x] 推論前加入 peak normalization（matching training）
  - [x] 加入 voice activity detection（VAD，提高 energy threshold）
  - [x] 降低 temporal smoothing alpha（0.7→0.4）
  - [x] 加入高通濾波器（high-pass filter，去除低頻噪音）
  - [x] 調整 window_seconds（1.5→2.0）與 fusion min_confidence（0.1→0.3）
- [x] **音訊延遲改善（Phase 5：非阻塞 InputStream + 重疊滑窗）**
  - [x] 改用 sd.InputStream() 持續錄音 + ring buffer
  - [x] 每 hop_seconds（0.5s）觸發一次推論
  - [x] window_seconds 設回 1.5 匹配模型
- [x] **音訊 bias 修正（Phase 6：logit adjustment + confidence floor）**
  - [x] 調降 anger logit 權重（×0.6）
  - [x] 調升 neutral logit 權重（×1.4），後續因 neutral 壓過真人語音改回 ×1.0
  - [x] 加入 confidence floor（max(probs) < 0.35 時輸出 neutral）
  - [x] VAD 未通過時 state.audio = None，避免舊 neutral 持續扭曲 fusion
  - [x] 加入 sad logit 加權（×1.5），改善 sad 被 neutral 壓制的問題
- [x] **Phase A：修復 `gate_audio_with_face` config 讀取錯誤**
  - [x] `src/main.py`：`config.get("audio", {}).get("gate_with_face", True)` → `config.get("fusion", {}).get("gate_audio_with_face", True)`
- [x] **Phase B：調降音訊推論超參數，讓真實語音更容易被辨識**
  - [x] `config.yaml`：`audio_weight` 0.5 → 0.8，`neutral_penalty` 0.3 → 0.5
  - [x] `config.yaml`：`confidence_floor` 0.35 → 0.20
  - [x] `config.yaml`：`sad_logit_scale` 1.5 → 1.0，`anger_logit_scale` 0.6 → 0.8
  - [x] `config.yaml`：`temporal_smoothing.alpha` 0.4 → 0.7
  - [x] `config.yaml`：`vad_threshold` 0.02 → 0.01
- [ ] **Phase C：下載 EmotionTalk 中文語音資料集並重新訓練**
  - [x] `requirements_train.txt` 新增 `huggingface_hub`，供 HuggingFace gated dataset 下載使用
  - [x] 建立 `scripts/download_emotiontalk.py`：從 HuggingFace 下載 EmotionTalk `Audio.tar`（需登入並同意 CC BY-NC-SA 4.0）
  - [x] 建立 `scripts/prepare_emotiontalk.py`：
    - 從 .tar 中解出 audio WAV（44.1kHz → 16kHz resample）
    - 7 類情緒映射到 4 類：neutral→neutral、happy/surprise→happy、sad→sad、anger/disgust/fear→anger
    - 輸出到 `data/datasets/emotiontalk/{neutral,happy,sad,anger}/`
  - [x] `train_audio_tiny_cnn.py` 新增 `--class-weight`，支援依類別數量反向加權
  - [x] Windows `.venv` 已安裝 `requirements_train.txt` 訓練依賴，可執行 EmotionTalk 下載與音訊模型訓練
  - [x] Hugging Face 權限檢查：目前 `.venv` 尚未登入或設定 token，因此 EmotionTalk 下載尚未開始
  - [x] Hugging Face 已登入並確認帳號 `QAQBlaze`，可繼續嘗試 gated dataset 下載
  - [x] `hf_hub_download` 下載嘗試未實際寫入大檔，已停止程序並保留 `data/datasets/emotiontalk_raw*` cache/lock 檔，未刪除任何資料
  - [x] 建立 `scripts/download_emotiontalk_stream.py`：支援續傳下載，先寫入 `Audio.tar.download`，完成後才改名為 `Audio.tar`
  - [x] 已下載 EmotionTalk `Audio.tar` 到 `data/datasets/emotiontalk_raw_stream/Audio.tar`，檔案大小 14,811,722,752 bytes
  - [x] 已整理 EmotionTalk 到 `data/datasets/emotiontalk/`，共 19,250 筆：neutral 9,378、happy 3,468、sad 1,110、anger 5,294，skipped 0
  - [x] 本機目前沒有 `data/datasets/audio/ravdess` 或 `tess` WAV；且既有下載腳本會刪除 `_extracted` 暫存資料夾，因此先不執行 RAVDESS/TESS 補下載
  - [x] 第一次 EmotionTalk-only 訓練未開始即失敗：TensorFlow import 時被 `.venv` 內既有 `jax` / `ml_dtypes` 版本衝突中斷
  - [x] `train_audio_tiny_cnn.py` 在 import TensorFlow 前遮蔽 optional `jax` 匯入，避免不需要的 JAX 相依影響訓練
  - [x] 先執行 EmotionTalk-only 訓練，輸出新模型檔，不覆蓋既有 `tiny_cnn_audio_fp32.onnx`
    - FP32：`models/tiny_cnn_audio_emotiontalk_fp32.onnx`，ONNX Runtime 與 `AudioEmotionModel` 載入成功
    - INT8：`models/tiny_cnn_audio_emotiontalk_int8.onnx` 已輸出，但 CPUExecutionProvider 無法執行 `ConvInteger`，不建議使用
    - 訓練結果：50 epochs，best val_accuracy 0.5166，final val_accuracy 0.5083
  - [x] 建立 `scripts/prepare_emotiontalk_clean.py`：只保留原始 `neutral/happy/sad/angry`，排除 `surprised/disgusted/fearful`，支援每類 balanced 輸出
  - [x] 建立 `scripts/train_audio_emotiontalk_mel_cnn.py`：使用 2 秒 log-mel spectrogram + 2D CNN 訓練 EmotionTalk 專用 FP32 ONNX
  - [x] 產生 `data/datasets/emotiontalk_clean/` balanced clean 訓練資料，每類 1,110 筆，共 4,440 筆，skipped 0
  - [x] 第一次 log-mel CNN 訓練在 `spec_augment` shape 處理失敗，已修正 2D log-mel mask 邏輯，未產生半成品模型
  - [x] 訓練 `models/audio_emotiontalk_mel_cnn_fp32.onnx`，不覆蓋既有音訊模型
    - 使用 `data/datasets/emotiontalk_clean/` 4,440 筆 balanced clean 資料
    - EarlyStopping 於 epoch 22 停止，還原 best epoch 12
    - best val_accuracy 0.5420，final val_accuracy 0.5240
    - ONNX Runtime 載入成功，輸入 shape `(1, 64, 122, 1)`，輸出 shape `(1, 4)`
  - [x] 建立 `scripts/evaluate_audio_models.py`：比較 MFCC 與 log-mel 音訊模型，輸出 accuracy、macro F1、confusion matrix、各類 precision/recall/F1、預測分布與錯誤範例
    - 補上 repo root 到 `sys.path`，確保直接執行 `python scripts/evaluate_audio_models.py` 可匯入 `src`
    - MFCC 模型評估時會 pad/truncate 到 93 frames，對齊正式 `AudioEmotionModel` 推論行為
  - [x] 執行模型診斷，輸出報告：
    - Markdown：`data/reports/audio_model_diagnostics_20260723_172740.md`
    - JSON：`data/reports/audio_model_diagnostics_20260723_172740.json`
    - `current_mfcc`：accuracy 0.2473，macro F1 0.1924，在 EmotionTalk clean 上幾乎不可用
    - `emotiontalk_mfcc`：accuracy 0.6435，macro F1 0.6447，是目前 clean set 表現最好的音訊模型
    - `emotiontalk_mel_cnn`：accuracy 0.5050，macro F1 0.4752，主要問題是 anger recall 僅 0.1171，幾乎不預測 anger
  - [ ] 執行 `train_audio_tiny_cnn.py` 合併訓練（emotiontalk + ravdess + tess），使用 `--class-weight` 處理類別不平衡
  - [ ] 用真實語音測試新模型效果

## 音訊精度改善結果
| Phase | 變更 | 訓練資料 | val_acc |
|-------|------|----------|---------|
| 初始 | 原始 tiny_cnn | ravdess 1,440 | 62% |
| Phase 1 | temporal smoothing (推論) | ravdess 1,440 | inference 改善 |
| Phase 2 | peak norm + augment + ReduceLROnPlateau | ravdess+tess 4,240 | **76.1%** |
| Phase 2 | 同上 | ravdess+tess+crema_d 11,682 | 53% |
| Phase 3 | N_MFCC=26 + delta + Conv1D 64→64 + Dense 128 + Dropout 0.5+L2 | ravdess+tess 4,240 | 54% |
| Phase 3 | N_MFCC=26 + Conv1D 64→64 + Dense 128 + Dropout 0.3 | ravdess+tess 4,240 | 52% |
| Phase 3 | N_MFCC=13 + Conv1D 32→64 + Dense 64 + Dropout 0.3 | ravdess+tess 4,240 | 54% |
| Phase C | EmotionTalk-only + class_weight | emotiontalk 19,250 | best 51.66%, final 50.83% |
| Phase C | EmotionTalk-clean + log-mel 2D CNN | emotiontalk clean 4,440 | best 54.20%, final 52.40% |

**結論**：Phase 2 為最佳配置（N_MFCC=13, Conv1D 32→64, Dense 64, Dropout 0.3, augment=True），另外 CREMA-D 資料集與 RAVDESS/TESS 不相容導致 val_acc 雪崩，已排除。Phase 3 所有嘗試均無正向效果。Phase C 的 EmotionTalk-only 與 EmotionTalk-clean log-mel CNN 訓練 val_acc 均低於既有 RAVDESS+TESS 模型，因此不能只憑訓練結果直接判定正式取代原模型。不過完整診斷報告顯示 `emotiontalk_mfcc` 在 `emotiontalk_clean` 上 accuracy 0.6435、macro F1 0.6447，是目前中文 clean set 表現最好的音訊模型；它可以作為實機中文語音測試候選，但仍需要用真實麥克風情境驗證。若實機效果仍不足，下一步應改用 pretrained speech encoder 或重新設計切片/標註策略。

## WSL 開發環境
```bash
# 安裝 Node.js 20（opencode 需要）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # v20.x

# 安裝 opencode
npm install -g opencode-ai

# 啟動推論（WSL 無 webcam，僅 smoke test）
source .venv/bin/activate
python app.py --vision-only --no-display --max-frames 10

# 啟動後端伺服器（瀏覽器可開 http://localhost:8000）
pip install -r requirements_backend.txt
python backend/app.py --host 0.0.0.0 --port 8000
```

## Windows 開發環境
```powershell
# 第一次：安裝依賴
.\scripts\setup_windows.ps1

# 每次執行
.\.venv\Scripts\Activate.ps1
python app.py --vision-only
```

## 環境狀態（2026-07-06）
- Python 3.11.15 + .venv（位於專案根目錄）
- 訓練依賴已安裝（tensorflow 2.16.2、tf2onnx 1.16.1、onnxruntime 1.17.3）
- 執行依賴已安裝（opencv 4.11.0、mediapipe 0.10.14、sounddevice 0.4.6）
- 後端依賴已安裝（fastapi 0.139.0、uvicorn 0.50.0、pydantic 2.13.4）
- ⚠️ sounddevice 需要 `libportaudio2` 系統套件：`sudo apt-get install -y libportaudio2`
- ⚠️ TensorFlow 以 CPU 模式執行（WSL 內無 CUDA 驅動）
- ⚠️ protobuf/ml-dtypes 版本衝突（mediapipe vs tf2onnx/tensorflow），訓練仍可正常運作
