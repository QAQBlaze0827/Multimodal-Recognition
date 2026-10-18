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

## 音訊推論修正（2026-10-13）
- 修正 `src/audio/model.py` 重複 softmax：目前 EmotionTalk 與舊版 MFCC FP32 ONNX 的輸出已是 softmax 機率，推論端不應再次取 exp，否則會壓縮各類分數差距。
- 改為複製模型輸出的前四類機率，套用既有類別權重後除以總和。保留 `neutral_logit_scale`、`sad_logit_scale`、`anger_logit_scale` 設定名稱以相容現有配置；此處實際調整的是機率權重，不是 logits。
- 保留 confidence floor 的既有行為：低於門檻時回傳 confidence=0 的 neutral fallback。未修改模型檔、訓練方式、麥克風設定、時間平滑、融合邏輯或前端。
- 新增 `tests/test_audio_model.py`，涵蓋原始機率保留、類別加權正規化、高信心度通過門檻、低信心度 fallback、舊版七類輸出保留四類後正規化；亦檢查不修改原始輸出陣列。
- 本機 `.venv` 驗證通過：
  - `python -m unittest discover -s tests -v`：5 項測試通過。
  - `python -m py_compile src/audio/model.py tests/test_audio_model.py`：語法檢查通過。
  - 載入 `tiny_cnn_audio_emotiontalk_fp32.onnx` 與 `tiny_cnn_audio_fp32.onnx`，確認末端為 Softmax；分別以不加權／目前 config 權重及 50、93、120 格合成特徵測試，共 12 組 ONNX 推論比對通過，包含補齊與截短路徑。
- 驗證限制：合成特徵僅驗證推論計算，並非準確率測試；尚未測試真人麥克風、樹莓派或完整前端流程。後續可執行 `python app.py --audio-only` 做實機測試；完整程式會沿用既有日誌與資料保留設定。
- 行為影響：信心度不再被重複 softmax 壓縮，可能改變融合權重及門檻判斷；分數升高不代表準確率提高。未加權時最高分類不因移除 softmax 改變；目前配置仍會對 anger 乘以 0.8，時間平滑也會影響即時畫面的分數。
- Git：本次僅修改工作目錄，未執行 git add、commit 或 push。

## EmotionTalk 四類資料清單與切分（2026-10-14）
- 在現有 repo 新增 `scripts/prepare_emotiontalk_manifests.py`，只讀取原始 `data/datasets/emotiontalk_raw_stream/Audio.tar`，產生清單與檢查報告，不解壓、刪除、轉檔或訓練。只需 Python 標準函式庫。
- 保留 Audio 標註原始四類：neutral 9,378、happy 2,105、sad 1,110、angry 3,820，合計 16,413 筆；angry 僅統一命名為 anger。其餘 2,837 筆不列入新清單，原始資料仍保留。
- 清單保留 archive 內 WAV/JSON 路徑、原始標籤、speaker_id、錄音組別、對話來源、時間、WAV 格式、標註票數與 JSON SHA-256；不包含逐字稿。SHA-256 僅針對個別標註，未計算整份 archive 或音訊內容雜湊。
- 使用固定自訂切分 `emotiontalk-audio-four-class-speaker-disjoint-v1`，不是官方切分。依上一輪盤點的共用說話者關係綁定來源組別，再將其分配為接近 70/15/15 的比例；未使用模型測試分數挑選組別。
  - train：G00002、G00004 至 G00011、G00015；11,714 筆，11 位說話者，71.37%。
  - val：G00001、G00003、G00012；2,194 筆，4 位說話者，13.37%。
  - test：G00013、G00014；2,505 筆，4 位說話者，15.26%。
  - 類別順序固定為 neutral / happy / sad / anger；三組數量依序為 train 6729/1533/779/2673、val 1148/287/152/607、test 1501/285/179/540。
- 檢查跨集合 speaker_id、組別、對話來源與 WAV 路徑沒有重疊；說話者檢查包含被排除情緒的原始標註。缺 WAV、重複 archive 路徑、不明標籤、不明組別、無效來源／時間或缺少任一類時停止，不輸出成功清單。
- 少於 4 位標註者同意最終標籤的 606 筆標為 `review_recommended`（train 414、val 100、test 92），尚未人工審核，不自動排除。
- 本次輸出：`data/reports/emotiontalk_four_class_20261014_v1/`，內含 `train.csv`、`val.csv`、`test.csv`、`summary.json`、中文 `report.md`。CSV 路徑指向 tar 內的原始資料，不是已解壓路徑。
- 執行方式（專案根目錄，已啟用 Python 3.11 環境）：
  - `python scripts/prepare_emotiontalk_manifests.py`：每次建立新的時間戳輸出資料夾，不覆蓋舊結果。
  - `python scripts/prepare_emotiontalk_manifests.py --output data/reports/emotiontalk_four_class_20261014_v1`：本次使用的固定輸出位置；已存在，重跑會拒絕覆蓋。
  - 可用 `--input` 指定原始 Audio.tar；無需重新下載資料。不要因重跑被拒絕而刪除既有輸出，改用新的輸出位置即可。
- 驗證：新增 `tests/test_emotiontalk_manifests.py` 的 7 項記憶體內 tar 測試（不建立或清除測試音檔），連同既有音訊測試共 12 項通過。實際 archive 掃描及產出成功，另獨立讀回 CSV 核對 16,413 筆、各類數量及 12 組跨集合重疊檢查；重跑覆蓋拒絕測試通過，原輸出雜湊不變。
- 限制與下一步：未完整解碼、聽音或檢查相同音訊內容，不以 speaker_id 之外的方式驗證身分。現有訓練腳本尚未接入清單；後續先規劃切片、聲道、重取樣及特徵一致性，再決定模型與訓練。類別自然分布不平衡，後續在訓練集處理，不改測試集來提高成績。舊模型可能看過本次 test 來源，不能據此宣稱公平的未見資料比較；真人耳機測試仍待進行。
- 本次未修改正式推論或模型，保留前次未提交的 softmax 修正；未新增 repo、未執行 git add、commit 或 push。

## EmotionTalk 音訊完整讀取檢查（2026-10-14）
- 新增 `scripts/audit_emotiontalk_audio.py`，沿用四類 train/val/test 清單，只讀取原始 `Audio.tar`，在記憶體中完整解碼 PCM16 音訊，不解壓寫入、不轉檔、不刪資料、不訓練。
- 逐筆核對標註 SHA-256、WAV 格式與長度，檢查音訊截斷、缺檔、重複路徑，計算左右聲道與混音 RMS、相關係數、峰值、滿刻度比例、全零與長度統計。以音訊格式及完整 PCM 的 SHA-256 檢查完全相同的錄音；發現讀取錯誤或跨集合完全重複時回傳非零結束碼並保留報告。
- 低 RMS、聲道不平衡、混音抵銷及滿刻度比例只是複核提示，不能單獨判斷無人聲、雜訊或錄音品質，不自動排除資料或選擇聲道。
- 執行方式：`python scripts/audit_emotiontalk_audio.py`，預設建立新的時間戳報告資料夾；可用 `--input`、`--manifests`、`--output` 指定來源與新輸出位置。拒絕覆蓋既有資料夾，輸出逐筆 `audio_metrics.csv`、`summary.json` 與中文 `report.md`。
- 新增 `tests/test_emotiontalk_audio_audit.py` 的 8 項記憶體內測試，涵蓋 PCM 指標、相同／反相／單邊靜音聲道、全零、截斷、雜湊格式差異與跨集合重複阻擋；連同既有測試共 20 項通過。
- 實際輸出：`data/reports/emotiontalk_audio_audit_20261014_v1/`。16,413 筆全部讀取成功（train 11,714、val 2,194、test 2,505），錯誤與完全相同 PCM 群組均為 0；狀態 `decoded_review_required` 代表可解碼但仍需品質複核，不代表標註正確或模型準確。
- 統計：低整段 RMS 113 筆（train 81、val 10、test 22）；聲道不平衡、混音抵銷、全零、滿刻度樣本超過 0.1% 的提示均為 0。13,296 筆長於 1.5 秒、3,115 筆短於 1.5 秒、2 筆恰為 1.5 秒。四類來源包含 15,861 筆 44.1 kHz 雙聲道、441 筆 44.1 kHz 單聲道、111 筆 16 kHz 單聲道，不能假設所有音檔規格一致。
- 左右聲道完全相同的數量為 train 9,890、val 1,438、test 0；來源錄音形式存在差異，後續須採固定聲道處理規則，不能只根據其中一個集合推定所有來源相同。
- 另獨立讀回逐筆 CSV，核對筆數、集合分布、PCM 雜湊唯一性與低音量／長度統計；三份來源清單 SHA-256 前後一致。相同輸出位置重跑會拒絕，拒絕後三份報告雜湊不變。未改寫原始 archive。
- 前處理建議尚未實作：先用 train 複核音訊提示，再固定單聲道規則、帶抗混疊的 16 kHz 重取樣、切片與補齊方式，以及 VAD／正規化／高通／MFCC 順序。長錄音不應永遠只取開頭；先依來源切分再切片，驗證按原錄音彙整，不以 test 表現調整規則。
- 限制：未檢查近似重複、重取樣後重複或片段重疊，未聽音、測試真人麥克風或評估準確率。舊模型可能見過本次 test 來源，仍不能宣稱公平的新舊模型未見資料比較。
- 本次未修改正式推論、設定、模型或既有資料清單；未執行 git add、commit 或 push。

## EmotionTalk 前處理基準模組（2026-10-14）
- 新增 `src/audio/preprocessing.py`，作為後續 Tiny CNN 訓練／評估可共用的候選前處理。尚未接入訓練腳本或 `src/audio/inference.py`，目前 `python app.py` 的行為不變，也尚未產生新模型。
- `to_mono_16k()` 接受浮點音訊（單聲道或 frames × 1/2 聲道），以左右算術平均轉單聲道，再以 `scipy.signal.resample_poly` 帶抗混疊濾波重取樣為 16 kHz；不使用直接每隔三點取樣。拒絕空資料、非有限值、整數 PCM 與不支援的形狀，避免把原始整數誤當正規化音訊。
- 本次平均聲道規則用於 EmotionTalk 資料，不代表 INMP441 應混合左右聲道。樹莓派若仍有單邊雜訊，須在擷取端先選取有效聲道，再傳入單聲道音訊；未修改麥克風擷取或裝置設定。
- 窗口固定 1.5 秒（24,000 點）：`training_window()` 使用呼叫端提供的 NumPy RNG，從單筆錄音均勻隨機取一個窗口，不再永遠只取開頭；`evaluation_windows()` 固定每 0.5 秒取窗並包含末尾，不重複加入整齊對齊的尾窗。短錄音右側補零至 1.5 秒，保留原始來源不寫回音檔。
- `window_features()` 沿用目前配置的順序：補齊後原始 RMS VAD（0.01）→ peak normalization → 80 Hz 高通 → 共用 `src/audio/features.py` 的 MFCC（13 維、FFT 512、hop 256、40 mel、無 delta）。1.5 秒實際得到 92 格 MFCC，右側補零至既有 Tiny CNN 輸入的 `(13, 93)`。未通過 VAD 回傳 `(None, rms)`，不捏造 neutral，也不刪除錄音。
- `preprocessing_spec()` 回傳版本 `emotiontalk-mfcc-v1` 的規格與 neutral/happy/sad/anger 類別順序，供未來隨新模型保存；本次未修改現有模型規格檔。這是 Tiny CNN 基準，不是已選定更大型或預訓練模型。
- `requirements_train.txt` 明列直接使用的 `scipy>=1.10,<2`；本機已有 SciPy 1.17.1，未安裝或升級套件。尚未將 SciPy 加入正式 PC/RPi 執行依賴，因為目前正式推論未呼叫此模組。
- 新增 `tests/test_audio_preprocessing.py` 的 13 項測試，涵蓋混音、44.1/48 kHz 重取樣長度、12 kHz 訊號降採樣的抗混疊、來源不變、輸入檢查、固定／隨機切片、短音補零、靜音先攔截、與目前推論特徵及模型補齊的計算比對。執行 `python -B -m unittest discover -s tests -v`，全專案 33 項測試通過。
- 真實資料抽測只使用 train：取每類第一筆、44.1 kHz 單聲道／16 kHz 單聲道、短於 1.5 秒及低整段 RMS 的第一筆，去重後共 7 筆。原始 tar 只讀、記憶體內解碼，隨機窗口 seed=42；共 27 個有效評估窗口通過有限值與 `(13,93)` 檢查並可送入既有 EmotionTalk MFCC ONNX，1 個窗口未通過 VAD。未計算準確率，亦未使用 val/test 調參或產生轉檔資料。
- 後續仍需將 train/val/test 清單接入訓練與評估，保留來源／說話者切分，再按原錄音彙整窗口預測；這些不是本次已完成的功能。整段無有效窗口時應另外記錄拒判及覆蓋率，不能靜默剔除驗證／測試樣本以提高分數。短片段繼承整段情緒標籤的適切性及真人麥克風效果仍待驗證。
- 本次未重訓、未改正式推論／config／模型、未刪除或覆蓋既有資料，未執行 git add、commit 或 push。

## EmotionTalk 清單式訓練入口與小規模試跑（2026-10-15）
- 新增 `scripts/train_audio_emotiontalk_manifest.py`，接入既有四類清單與 `src/audio/preprocessing.py`，沿用 `train_audio_tiny_cnn.build_model()` 的 Tiny CNN 架構與 `(13,93)` 輸入，不改舊訓練入口或正式辨識流程。電腦負責訓練，ONNX 候選格式供後續電腦與樹莓派共同使用；本次沒有測試樹莓派或宣稱兩平台部署完成。
- 讀取完整 train/val/test CSV 檢查集合重疊並記錄 SHA-256；只選取 train/val 音訊，不解碼 test，不重新隨機分組。選中樣本另比對原始標註 SHA-256、情緒、speaker_id、來源路徑及固定組別分配；WAV 完整解碼並比對格式／長度，保留 PCM SHA-256。原始 tar 只讀，不解壓寫回。
- 預設為小規模試跑：train 與 val 各類最多 8 筆（各 32 筆）、seed=42、batch size=16、2 epochs、learning rate=0.001。抽樣不移動集合，限制為 0 時才使用完整 train/val 自然分布；本次未執行完整資料訓練。
- 每個 epoch 對每筆 train 重新隨機切一個窗口，VAD 未通過不進入梯度更新，但保留於當輪 crop CSV；若任何類別沒有有效窗口則停止，不能假裝 neutral。類別權重只依當輪有效 train 窗口數反向計算。不沿用舊腳本的音量、雜訊或 SpecAugment，這次只以隨機切片建立基準。
- val 使用固定窗口，平均同一原始錄音的有效窗口機率後判斷一個情緒，不讓長錄音因窗口多而增加評分權重。整段無有效窗口保留為 abstain，計入 accuracy 分母與該類 recall 的漏判；回報 coverage、拒判數、四類 precision/recall/F1、macro-F1 與含 abstain 欄的 4×5 confusion matrix。每輪僅依 val 的錄音層級 macro-F1 選最佳權重，不依 test 選模型。
- 每次建立新輸出目錄，拒絕覆蓋；保留 `selected_records.csv`、`preprocessing.json`、`run_config.json`、每輪 train crop CSV／評估 JSON、`val_audio.csv`、`val_predictions.json`、`result.json` 與 FP32 ONNX。僅在全部完成後寫入 `result.json`；缺少此檔表示該次未完成。小規模模型命名為 `smoke_only.onnx`，不可作為正式辨識模型。
- ONNX 用 `tf2onnx.convert.from_function()` 匯出以配合目前 Keras 3；重新載入 ONNX Runtime CPUExecutionProvider，與 TensorFlow 的同批真實特徵輸出比對（rtol=1e-4、atol=1e-5）。不產生舊環境無法執行 ConvInteger 的 INT8 模型。
- 初次 `models/emotiontalk_manifest_smoke_20261015_v1/` 在訓練前失敗：舊式 `sys.modules['jax']=None` workaround 影響 SciPy 1.17.1 的 array backend 檢查。改為僅在 TensorFlow/Keras 初始化期間暫時遮蔽，之後還原 sys.modules；未更換依賴版本。v1 的三份前置紀錄保留，未刪除，沒有模型或完成結果。
- 第二次 `models/emotiontalk_manifest_smoke_20261015_v2/` 完成：32 train／32 val，2 epochs；每輪 31 個 train 窗口有效、1 個未過 VAD；val 共 320 個有效窗口，32 筆錄音皆有預測。選中 epoch 2，小樣本 val accuracy=0.34375、macro-F1=0.245798、coverage=1.0；這些僅是試跑數字，不能當正式模型準確率或與舊模型比較。
- v2 的 ONNX 為 106,880 bytes，8 個真實特徵窗口的匯出比對最大絕對差約 3.28e-7。另以既有 `AudioEmotionModel` 測試 50／93／120 格合成特徵，載入、補齊／截短及四類機率輸出正常；沒有修改 config 指向此測試模型。
- 新增 `tests/test_emotiontalk_training.py` 的 13 項測試：固定集合／可重現抽樣、自然分布、test 擋入訓練／選模、來源核對、靜音保留與缺類停止、逐錄音平均、拒判分母與 F1、錯誤機率、train 類別權重。`python -B -m unittest discover -s tests -v` 共 46 項通過。另讀回試跑結果重算指標、檢查 train/val 說話者不重疊、來源清單 SHA-256 不變、相同輸出目錄重跑拒絕且產物雜湊不變；舊 models 根目錄檔案雜湊全部不變。
- 重現小規模試跑（專案根目錄，啟用既有 Python 環境）：`python scripts/train_audio_emotiontalk_manifest.py --epochs 2 --limit-per-class 8`，會建立新的時間戳目錄；不要重用 v1/v2 位置，也不要刪除它們以便重跑。測試指令為 `python -B -m unittest discover -s tests -v`。
- 後續才考慮完整 train/val 訓練（`--limit-per-class 0`，另確認 epoch 數與 CPU 時間），再另行接入最終 test 評估及真人收音驗證。目前沒有 test 評估入口、沒有正式準確率，也沒有自動接管 PC/RPi 推論；試跑的 JSON 不等於可部署的合格證明。
- 未修改 `config/config.yaml`、`src/audio/inference.py`、前端或舊模型；保留原始資料與失敗試跑紀錄，未執行 git add、commit 或 push。

## 訓練前處理加速與進度顯示（2026-10-15）
- 使用者啟動完整 30 epochs 後因長時間沒有進度提示而中止，原目錄 `models/emotiontalk_manifest_train_20261015_075950_570717/` 保留。內含三份前置紀錄與 08:17 產生的 `val_audio.csv`，沒有第一輪 crop／epoch 檔、ONNX 或 `result.json`；依程式順序判斷中止於第一輪訓練特徵準備，不能當作已完成訓練。中止後確認 Python 程序已結束。
- 只讀量測 8 筆 train 錄音，原高通濾波的 Python 逐點迴圈耗時約 0.40 秒，是明顯瓶頸之一；完整音訊流程還包括讀檔、解碼、音訊檢查、重取樣與 MFCC，不將所有耗時歸因於濾波。
- 修改 `src/audio/preprocessing.py`：新增 `_highpass_filter_fast()`，以既有 SciPy 的 `signal.lfilter` 計算原一階高通遞迴。保留 80 Hz、16 kHz、float32 輸入差分及 `y[0]=x[0]`；內部累積改為 float64，輸出仍為 float32。不改 VAD、正規化、切片、MFCC 設定或標籤。`src/audio/features.py` 與正式 `src/audio/inference.py` 未修改。
- 浮點捨入方式不同，因此不是逐位元一致。規格改記為 `emotiontalk-mfcc-v1.1`，在 `preprocessing.json` 明列高通實作；舊模型與既有規格檔不覆蓋。後續部署須保留新模型的規格並驗證實機，不能把此次小量比對視為所有輸入都完全相同。
- 合成測試涵蓋靜音、正負 DC、脈衝、純音、隨機訊號、單取樣值，波形絕對差門檻 1e-6；廣頻 MFCC 門檻 1e-4。純音的近靜音頻帶會放大捨入差異，MFCC 絕對差約 0.00412，該測試明列門檻 0.005，不再宣稱達到舊版的 1e-6 特徵比對精度。
- 真實資料只用 train 抽樣 32 筆（seed=42），以固定 crop seed=43 取得 31 個有效窗口。初始 MFCC 絕對差 0.001 檢查未通過（62/37,479 個元素超過），進一步量測最大差 0.00165749、99 百分位差 0.000442505，符合純音測試採用的 0.005 絕對差範圍。以既有 EmotionTalk MFCC ONNX 比對，機率最大絕對差 1.0312e-5（檢查門檻 1e-4），31 個窗口分類均相同，VAD、裁切位置及來源紀錄相同；未計算準確率或讀取 test 音訊。
- 速度比較採預熱後交錯執行新舊完整 `training_arrays()` 各三次，包含讀檔／檢查／重取樣／特徵：32 筆的中位耗時由 2.5774 秒降為 1.0931 秒，約 2.36 倍。這是該批資料與當時環境的量測，不是完整 epoch 或 30 輪的時間保證；磁碟快取、錄音長度及系統負載會影響結果。
- 修改 `scripts/train_audio_emotiontalk_manifest.py`：顯示 TensorFlow 初始化、archive 索引、標註檢查、驗證特徵、每輪訓練特徵、模型擬合、驗證及 ONNX 匯出階段。逐筆處理的階段顯示已處理／總筆數與 elapsed 秒數；開始、完成立即輸出，期間每完成一筆檢查是否已間隔 5 秒才更新，並 flush。它不是背景心跳，不保證單筆耗時時每 5 秒都有訊息；沒有虛構 ETA。
- 更新 `tests/test_audio_preprocessing.py` 與 `tests/test_emotiontalk_training.py`，新增 2 項濾波／特徵及 3 項進度測試；涵蓋原始資料不變、初始狀態、定時進度、完成筆數、空清單與中斷不誤報完成。`python -B -m unittest discover -s tests -q` 共 51 項通過。
- 只執行新的小規模試跑：`python scripts/train_audio_emotiontalk_manifest.py --epochs 1 --limit-per-class 8 --output-dir models/emotiontalk_manifest_smoke_fast_20261015_v1`。32 train／32 val、1 epoch 完成；驗證 320 個有效窗口準備約 3.8 秒，訓練特徵準備約 1.2 秒，ONNX 與 TensorFlow 的 8 個窗口比對最大絕對差約 4.17e-7。`smoke_only.onnx` 仍僅供流程測試，不可替換正式模型。
- 另確認修改前既有 models 目錄內的 27 個檔案雜湊均不變（包含舊模型、先前試跑與中止紀錄）、三份來源清單 SHA-256 不變，重用新試跑輸出位置會拒絕且產物雜湊不變。沒有刪除資料、安裝套件或執行 git add／commit／push。
- 未自動重跑 30 epochs，未增加中斷續訓。使用者準備好後仍可自行執行 `python scripts/train_audio_emotiontalk_manifest.py --epochs 30 --limit-per-class 0`，會從頭訓練並建立新目錄，不會接續中止那次。保持電腦開機、不休眠，依新的階段與筆數觀察進度；完整訓練耗時和實機準確率仍待驗證。

## EmotionTalk 完整訓練執行（2026-10-15）
- 使用者授權由助理執行完整 30 epochs：`python -B scripts/train_audio_emotiontalk_manifest.py --epochs 30 --limit-per-class 0 --output-dir models/emotiontalk_manifest_train_20261015_full_v1`。新目錄拒絕覆蓋既有資料；不接續先前中止的訓練。
- 啟動前 51 項單元測試通過，確認沒有其他 Python 訓練程序；保留既有模型檔案雜湊供完成後檢查。
- 使用既有 speaker-disjoint train 11,714／val 2,194，依 val recording macro-F1 選最佳 epoch，不解碼 test 音訊。沿用前處理 v1.1 與 Tiny CNN，不修改正式推論、config 或資料清單。
- 已完成 30 epochs，程序 exit code 0，`result.json` 狀態為 `complete`。檔案時間為台灣時間 14:26:35 至 17:27:18，牆鐘時間約 3 小時 43 秒，包含中途停頓，不能當成純運算耗時或其他電腦的時間保證。
- 驗證特徵準備 206.1 秒，共 16,766 個有效窗口；每輪重新切片／讀取／計算 train 特徵，通常約 295～339 秒，模型擬合約 1～3 秒。第一輪 train 11,446 個有效窗口；最後一輪 11,451 個有效、263 個未過 VAD。沒有變更切片策略、訓練參數或增加 checkpoint。
- 第 25 輪期間等待工具連線逾時，重連後確認原 Python PID 32884 仍在且進度繼續，沒有重開訓練；該輪特徵準備 elapsed=1397.1 秒。未查閱 Windows 電源事件，因此不斷言停頓一定由休眠造成。使用者詢問帶走電腦時，僅以 `powercfg /a` 確認支援休眠，沒有修改電源設定或代為執行休眠／關機。
- 最佳選模輪次為第 30 輪：val recording accuracy **0.475843（47.58%）**、macro-F1 **0.408865**、coverage **0.992707**；2,194 筆中有 16 筆整段無有效窗口，仍計入 accuracy 分母及 recall 漏判。第 13 輪曾為最佳（accuracy=0.465816、macro-F1=0.408482），最後僅小幅超過，不能宣稱明顯改善。
- 最終各類指標如下，全部為驗證集錄音層級，非 test 或耳機／樹莓派實測：

  | 類別 | Precision | Recall | F1 | 筆數 |
  |---|---:|---:|---:|---:|
  | neutral | 78.88% | 50.44% | 0.6153 | 1,148 |
  | happy | 23.76% | 38.33% | 0.2933 | 287 |
  | sad | 18.93% | 42.11% | 0.2612 | 152 |
  | anger | 45.26% | 47.94% | 0.4656 | 607 |

- 驗證集全部猜 neutral 的 accuracy 基準為 52.32%，高於本次總 accuracy，但這種基準完全辨識不到其他三類，因此同時保留 macro-F1 與逐類指標。新模型 happy／sad precision 仍低，不應當成可靠四類辨識模型，也沒有證據顯示能取代舊模型；本次音檔驗證不經過真人麥克風，不能把這些誤判歸因於使用者耳機收音。原因仍需後續診斷，不能保證多跑 epochs 就會改善。
- 模型輸出 `models/emotiontalk_manifest_train_20261015_full_v1/model.onnx`（106,903 bytes），輸入 `[batch,13,93]`、輸出 `[batch,4]`，前處理 `emotiontalk-mfcc-v1.1`。SHA-256：`82fd30e480aabdde1eb719f5a7b1f80e47ad88abe4bdb336a0a3cc352d08ad1b`。8 個真實 val 特徵窗口的 TensorFlow／ONNX 最大機率絕對差為 3.8743e-7；匯出有 TensorFlow deprecated API 警告，但完成且數值比對通過。
- 完成後獨立讀回 2,194 筆 `val_predictions.json`，重算混淆矩陣、accuracy、逐類 F1 與 macro-F1；核對全部 30 份 epoch JSON、30 份 train crop CSV 的筆數與有效數、最佳輪次、三份來源 manifest SHA-256 及新模型 SHA-256，均通過。既有 `AudioEmotionModel` 載入新模型，50／93／120 格合成特徵的補齊／截短及四類輸出通過；這些相容性檢查不是實機準確率測試。
- `git diff HEAD -- models config src scripts tests` 無差異，原有已追蹤模型／程式／設定未改；兩個既有完成試跑模型也通過其原始 `result.json` SHA-256 核對。工具重連導致啟動前暫存在工具記憶體的全檔雜湊快照遺失，因此未宣稱已完成所有未追蹤舊檔的前後雜湊比對。沒有刪除原始資料、舊模型或任何試跑／中止目錄。
- 本次只新增完整訓練輸出並更新本節，未修改正式推論或 config 指向；未使用 test 音訊、未測真人麥克風／樹莓派、未執行 git add、commit 或 push。下一步先分析錯誤與模型／輸入限制，不因訓練流程成功就自動部署；若要再長時間訓練，應先另行加入可驗證的 checkpoint／續訓能力。

## EmotionTalk 驗證集誤判診斷（2026-10-15）
- 新增 `scripts/diagnose_emotiontalk_validation.py`，只分析完整訓練的 `val_predictions.json`／`result.json`，結合既有 `val.csv` 與音訊 audit 的 val 列，不重訓、不重新推論、不讀取 test 音訊或 test.csv，不修改正式模型與設定。audit CSV 原本含三個集合，診斷只選取 val 列進行配對與統計。
- 核對 run／audit 的 val manifest SHA-256、來源集合、重複／缺少路徑、標籤、說話者、PCM SHA-256、格式／長度、機率與有效窗口數，再重算 confusion matrix、accuracy、macro-F1 與拒判分母，避免把不同模型或資料的紀錄混在一起。原始音量採先前 audit 的整段混音 RMS，不是推論窗口補零／重取樣後的 RMS。
- 新報告：`data/reports/emotiontalk_validation_diagnostics_20261015_v1/`，包含中文 `report.md`、`summary.json`、2,194 筆 `validation_diagnostics.csv` 及人工填寫用 `listening_cases.csv`。每筆紀錄包含原標籤、預測、平均機率最大值、長度、RMS、有效／無效窗口數、說話者與標註複核旗標；不包含逐字稿。
- 統計確認：正確 1,044 筆、已有有效窗口但判錯 1,134 筆、整段拒判 16 筆。沿用最佳 epoch 30，accuracy=47.58%、macro-F1=0.408865。這是既有結果的診斷，並非新一輪準確率改善或 test 評估。
- 常見混淆：neutral → anger 217 筆、neutral → sad 202 筆、anger → happy 173 筆、neutral → happy 136 筆、happy → anger 100 筆。happy 被判為 anger 占原始 happy 的 34.84%；anger 被判為 happy 占原始 anger 的 28.50%。
- 正確／誤判的整段 RMS 中位數為 0.06177／0.06336，長度中位數為 3.378／4.227 秒，有效窗口中位數為 5／7；拒判錄音長度中位數 0.878 秒。僅 10 筆整段 RMS < 0.01（均拒判），另外 6 筆整段 RMS >= 0.01 仍無有效窗口，不能把整段 RMS 與窗口 VAD 判斷混為一談。這些統計不足以認定調高麥克風音量就會改善分類。
- 四位 val 說話者 accuracy 為 44.83%、30.72%、67.01%、38.71%，報告同時保留每組四類筆數；資料分布不同，不據此認定是特定說話者、音量或錄音長度造成誤判。短於 1.5 秒組也有大量 neutral，因此不能只看分桶總 accuracy 就斷言短片段更好。
- 人工聽音案例：每類一筆中位信心度正確案例與最高信心度誤判，另選 happy／sad 誤報、短錄音誤判、拒判，來源去重後共 10 筆。這是刻意選擇的診斷案例，不是隨機代表性抽樣；高分誤判的分數未經機率校準，不代表真實正確率。
- 音檔為原始完整 WAV 的逐位元副本，沒有轉檔、增益或裁切，存於已被 Git 忽略的 `data/datasets/emotiontalk_review_20261015_v1/`，合計 4,046,120 bytes。只從 tar 讀出這 10 筆 val WAV 與其標註，先驗證標註及 PCM 雜湊，再用固定檔名寫入新目錄，不使用 extractall；原 archive 不改寫。音檔仍受原資料集授權限制，不隨報告提交。報告內可點選音檔連結；`listening_cases.csv` 留有 `heard_label`、`label_agreement`、`audio_issue`、`notes` 空欄供組員記錄。
- 助理尚未人工聽音，沒有宣稱標籤錯誤、音檔無人聲或已找出模型不準的原因。原錄音的預測是有效窗口平均，不等於逐瞬間情緒；聽音後仍不確定可以保留不確定，不因模型判錯就修改原標籤。
- 執行方式：`python -B scripts/diagnose_emotiontalk_validation.py --run-dir models/emotiontalk_manifest_train_20261015_full_v1`。預設建立新時間戳報告／聽音目錄；可用 `--output-dir` 與 `--audio-dir` 指定新位置，聽音目錄限制在 `data/datasets`。同一位置重跑會拒絕，不刪除舊輸出以重跑；完整成功後才寫入 `summary.json`。
- 新增 `tests/test_emotiontalk_validation_diagnostics.py` 的 10 項測試，涵蓋拒判分母、test 擋入、缺失／重複來源、標籤／格式／PCM 不一致、無效機率／窗口數、空分組、分桶邊界、可重現去重案例、輸出拒絕覆蓋及 val 音訊標註核對。全專案 61 項測試通過；另驗證報告可重現、2,194 筆明細與 10 個 WAV 雜湊／連結有效、人工欄位空白、未載入 TensorFlow。
- 執行前後既有 models／data/reports／config 共 116 個檔案 SHA-256 不變；拒絕覆蓋後本次 4 份報告與 10 個 WAV 雜湊不變。沒有刪除資料、安裝套件、修改 .gitignore、切換模型或執行 git add／commit／push。下一步先由使用者與組員聽這批案例，再決定是否需要窗口層級診斷或耳機實測新舊模型，不自動重訓。

## 使用者人工聽音回饋（2026-10-15）
- 將使用者在對話中提供的 10 筆回饋填入 `data/reports/emotiontalk_validation_diagnostics_20261015_v1/listening_cases.csv` 的 `heard_label`、`label_agreement`、`audio_issue`、`notes` 四欄。原始來源、資料集標籤、模型預測、分數、窗口統計及雜湊欄位全部保留，不重標訓練資料。
- 本次是一位使用者對刻意挑選案例的主觀回饋，案例名稱與報告可見原標籤及預測，不能當作盲測、資料集品質比例、全體準確率或多人標註共識。助理未自行聽音，也未將使用者意見改寫為已證實的原因。
- 01、02、03、05、06、07、09 共 7 筆同意原標籤（05 保留「非常同意」）；04、08、10 共 3 筆保留「不確定」。04 的聽感「像開心」與對標籤「不確定」分欄保留，沒有強制統一為確定情緒。
- 02（neutral → sad）、06（sad → happy）、09（anger → happy）是使用者同意原標籤但模型判錯的案例，可優先檢查模型與輸入處理；01、03、05、07 是使用者同意且模型判對的對照。04、08 難辨認，暫不判定原標籤錯誤；10 使用者聽不清且模型拒判，拒判未必是不適當行為，仍保留原本評估分母。
- 06「太大聲所以模型判斷開心」記為使用者推測、尚未驗證因果。現有前處理包含每窗口 peak normalization，不能僅據原音檔聽感就認定絕對音量是原因；聲音起伏、音色、片段內容等仍需另行實驗，不直接改音量或分類權重。
- 此節是前一節「人工欄位空白」之後的新進度。原 `report.md` 與 `summary.json` 保留為自動診斷產物；其 `listening_performed=false` 記錄產出當時尚未人工聽音，不代表後續無使用者回饋。後續人工回饋以更新的 CSV 與本節為準，不重新產生報告覆寫人工欄位。
- 驗證：CSV 仍為 10 筆、22 欄，僅最後四個人工欄位改動，前 18 欄逐筆與修改前一致；檢查 10 個音檔雜湊與原模型 `result.json` 的模型雜湊一致。純紀錄更新，未重跑模型訓練或將使用者判斷作為新準確率。
- 待後續另行確認時間卷積方向的候選設計與小規模試跑計畫；本次只完成紀錄，沒有修改程式、模型、設定或原始音檔，沒有啟動訓練、刪除資料、執行 git add／commit／push。

## EmotionTalk 時間卷積候選與小規模對照（2026-10-15）
- 根據使用者同意的計畫，只新增時間卷積候選並跑小規模對照，不跑完整 30 epochs、不使用 test 音訊、不替換正式模型。先前人工聽音僅提供診斷線索，不能證明卷積方向是誤判唯一原因。
- 檢查既有 Tiny CNN 的外部輸入為 `[batch,13 MFCC,93 time]`，channels-last Conv1D 沿 MFCC 軸卷積，時間位置作為通道；不是沿時間軸的局部卷積，但不能說完全無法學習時間資訊。
- `scripts/train_audio_tiny_cnn.py` 的 `build_model()` 新增預設為 false 的 keyword-only `temporal_axis`。只有候選版會在模型內加入 `Permute((2,1))`，使 Conv1D 收到 `[batch,93 time,13 MFCC]`；原版層序與預設行為保留。兩版外部輸入仍為 `[batch,13,93]`，輸出四類順序不變，不改正式推論的輸入轉置。
- `scripts/train_audio_emotiontalk_manifest.py` 新增 `--architecture legacy|temporal`，預設 legacy。新增結構檢查與 `model_structure.json`，記錄輸入／輸出、卷積方向、kernel 與中間輸出形狀、總參數量；run config 與完成結果也記錄架構。其餘抽樣、前處理、類別權重、選模與匯出流程不變。
- 實際完成兩個新輸出目錄：`models/emotiontalk_axis_legacy_smoke_20261015_v1/`、`models/emotiontalk_axis_temporal_smoke_20261015_v1/`。兩版皆為 2 epochs、train／val 每類 8 筆（各 32 筆）、seed=42、batch size=16、learning rate=0.001；每輪 31 個 train 窗口有效、1 個拒絕，val 320 個有效窗口，32 筆錄音 coverage=1.0。選模皆選中 epoch 2。
- 小樣本結果：legacy accuracy=0.34375、macro-F1=0.245798；temporal accuracy=0.28125、macro-F1=0.158120。候選版本次沒有優於原版；只訓練 2 輪與單一小樣本／種子，不能當正式準確率或宣稱架構改善成功。轉置同時使第一層通道數改變，總參數由 25,828 降至 13,028，並非參數量配平的消融實驗。
- 兩版 `selected_records.csv`、`preprocessing.json`、`val_audio.csv` 與兩輪 train crop CSV 逐位元相同；run config 僅架構兩欄不同。沒有調整標籤、人工聽音回饋、VAD 或切片規則以提高候選分數。
- 新增 `tests/test_audio_temporal_architecture.py` 的 5 項測試，涵蓋原版預設／顯式 false 的層序與 seeded weights、轉置值及外部形狀、kernel／中間形狀與參數差、四類正規化機率、錯誤架構拒絕。`python -B -m unittest discover -s tests -q` 全專案 66 項測試通過。
- 兩版 `smoke_only.onnx` 均通過 8 個真實 val 特徵窗口的 TensorFlow／ONNX 比對，最大絕對差分別約 3.5763e-7、2.3842e-7。另驗證 ORT batch=1／3、既有 `AudioEmotionModel` 的 50／93／120 格特徵補齊／截短；ONNX 第一層 Conv 權重通道數分別為 93 與 13。這是格式與運算相容性驗證，不是耳機或樹莓派準確率測試。
- 完整比較、模型 SHA-256、限制與重現指令記錄於 `data/reports/emotiontalk_axis_smoke_comparison_20261015_v1.md`。重跑可使用 `python -B scripts/train_audio_emotiontalk_manifest.py --architecture temporal --epochs 2 --limit-per-class 8 --seed 42 --batch-size 16`；改為 `--architecture legacy` 可跑原版。省略 output-dir 會建立新時間戳目錄，不刪除舊輸出以重跑。
- 兩個輸出目錄重跑皆拒絕覆寫，各 12 個檔案雜湊不變；修改前既有 models／data/reports／config／聽音 WAV 共 130 個檔案 SHA-256 不變。未刪除原始資料或中止／試跑目錄，未安裝套件、修改正式 config 或模型指向，未執行 git add、commit 或 push。
- 訓練入口仍讀三份 manifest 做切分檢查，但只解碼選中的 train／val；沒有 test 音訊評分。下一步需另行確認 checkpoint／續訓功能與較充分的架構對照預算，再決定是否完整重訓。這兩個小規模模型不能直接作為正式部署模型。

## 本機耳機錄音診斷工具（2026-10-15）
- 使用者回報：在相同耳機麥克風、近似距離下，完整 30 輪模型 `models/emotiontalk_manifest_train_20261015_full_v1/model.onnx` 幾乎都預測 sad，舊模型則偏 happy／anger。這是使用者的非受控即時觀察，助理未實機重現；不是兩個模型對同一份錄音的結果，尚不能判定偏差原因或正式準確率。
- 本次先完成使用者同意的「錄音工具」，不實作模型比較或 checkpoint，不重訓、不改模型權重與正式推論。新增 `scripts/record_audio_samples.py`，使用既有 sounddevice 與 soundfile；本機已安裝，未安裝或升級套件。sounddevice 列於 requirements.txt、soundfile 列於 requirements_train.txt，只有安裝執行依賴的另一台電腦不一定已有 soundfile。
- 預設中性／開心／難過／生氣（neutral/happy/sad/anger）各 3 段，每段 5 秒，16 kHz 單聲道。每段等使用者按 Enter 才開始；可用 `--device` 指定裝置索引、`--label` 只錄一類、`--takes` 指定每類段數、`--seconds` 指定秒數、`--sample-rate` 指定擷取取樣率。每次呼叫建立新的時間戳資料夾，不接續或覆寫既有錄音。
- 使用方式（專案根目錄、啟用既有 .venv，先停止 app.py）：
  - `python scripts/record_audio_samples.py --list-devices`：只列裝置，不錄音、不建立輸出。
  - `python scripts/record_audio_samples.py --device 3`：範例索引 3 須換成清單中的耳機輸入裝置，接著依提示錄 12 段。也可省略 device 使用系統預設輸入，工具會印出實際裝置索引與名稱。
  - `python scripts/record_audio_samples.py --device 3 --label neutral --takes 1 --seconds 5`：先試錄一段再聽音確認。若裝置不支援 16 kHz，會在建立輸出前報錯，可依裝置支援規格另用 `--sample-rate 48000`；工具不自動重取樣。
- 輸出於 `data/datasets/local_audio/session_時間戳/`，含 session.json、每段原始浮點 WAV 與同名 JSON；全部成功才寫 summary.json。每段 JSON 包含預期情緒、時間、取樣率／聲道／長度、RMS、peak、滿刻度比例、是否全零與 WAV SHA-256，另留 heard_label、notes 空欄供聽後記錄。expected_label 僅代表使用者打算表達的情緒，不是經獨立標註確認的答案；不儲存逐字稿、不產生準確率。
- WAV 採 32-bit FLOAT，保留音訊 API 傳回的 float32 樣本，不做 peak normalization、高通、VAD、降噪、裁切或重取樣，也不因靜音而刪除錄音。作業系統／裝置仍可能做增益、降噪或取樣率轉換，不能把這份錄音宣稱為未經硬體／系統處理的訊號。RMS／滿刻度提示僅供聽音複核，不自動判斷品質或情緒。
- 僅針對本輪筆電耳機單聲道擷取；尚不支援 INMP441 的雙聲道擷取後選左聲道，不應直接當作樹莓派收音問題已處理。比較模型前，先聽 WAV 確認是正確麥克風、內容完整可辨；使用固定距離與音量設定，可用相同幾句內容表達四種情緒，不以提高音量代替表情語調。12 段是初步診斷，不是獨立正式測試集或訓練資料。
- 輸出目錄使用 exist_ok=False、各檔案使用 x／xb 模式，拒絕覆寫。擷取期間 Ctrl+C 會停止串流且不將未完成的緩衝區寫成成功片段；此前完成的 WAV／JSON 保留。發生 overflow 等串流錯誤會停止、不靜默接受該段。寫檔中斷或磁碟錯誤可能留下不完整檔案，工具不刪除它們；缺 summary.json 代表整批未完成，完整片段仍可人工檢查。
- 新增 `tests/test_record_audio_samples.py` 的 12 項測試：預設數量、記憶體內 FLOAT WAV 逐樣本讀回（含小音量與超滿刻度）、靜音保留、無效樣本、裝置／取樣率傳遞、等待與停止串流、overflow／中斷、擷取長度檢查、覆寫拒絕、逐段確認與最後完成標記、中斷保留與 CLI 參數檢查。測試使用模擬裝置、記憶體音檔及模擬檔案操作，不開啟真實麥克風、不建立或刪除測試音檔。
- 驗證：`python -B -m unittest discover -s tests -q` 全專案 78 項通過；`python -B scripts/record_audio_samples.py --help` 正常。git check-ignore 確認該錄音資料夾的 WAV／JSON 已被既有 data/datasets/ 規則忽略，不需修改 .gitignore，也不應隨程式 commit 上傳個人錄音。尚未驗證真實裝置開啟與收音品質，需由使用者執行試錄。
- 本次只新增錄音腳本、測試並追加本節；未修改 config、既有模型／報告／人工回饋，未開啟麥克風、啟動訓練、刪除資料或執行 git add／commit／push。下一步等使用者完成聽音，再另行實作相同錄音的新舊模型比較，區分未加權未平滑與目前正式設定的結果。

## 本機同錄音新舊模型比較（2026-10-15）
- 使用者已錄製 `data/datasets/local_audio/session_20261015_220659_198289/`，共 12 段，neutral/happy/sad/anger 各 3 段，每段 5 秒、16 kHz、單聲道 FLOAT WAV。錄音工具紀錄裝置索引 1、MME 耳機輸入。已讀取完整 WAV 並核對 summary、sidecar、格式、有限值及 SHA-256；無全零或滿刻度樣本，不代表助理已聽音或確認標籤正確。
- 使用者同意後新增 `scripts/compare_local_audio_models.py`，僅比較這批相同音檔，不錄音、不重訓、不讀 EmotionTalk 原始 archive 或 test 音訊，不修改正式推論。固定模型為 old=`models/tiny_cnn_audio_emotiontalk_fp32.onnx`、new=`models/emotiontalk_manifest_train_20261015_full_v1/model.onnx`，不是兩輪時間卷積試跑模型；記錄兩個實際模型 SHA-256，模型不存在／ONNX 載入失敗／非四類 13x93 輸入會停止，不接受 heuristic fallback 冒充模型結果。
- 沿用 `evaluation_windows()` 的 1.5 秒窗口、0.5 秒步長及尾窗，每段 8 個窗口。特徵計算使用正式 `src/audio/features.py` 的 highpass_filter／MFCC，先原始 RMS VAD、peak normalization、80 Hz 高通、MFCC 13 維、FFT 512、hop 256，再右補零至 93 格。沒有用訓練模組的快速高通代替正式版本，也沒有更動原音量／音檔。入口目前只接受完整的 16 kHz 單聲道 FLOAT 錄音 session，不自行轉檔。
- 每個窗口只準備一次特徵，交給兩模型；分為 raw（未加權、未平滑、floor=0）、calibrated（目前 config 加權與 floor）、smoothed（再套用目前 alpha）。使用既有 AudioEmotionModel 處理類別權重；本次 neutral/sad 權重 1、anger 權重 0.8、floor=0.20、alpha=0.7、VAD=0.01。每段獨立重設平滑，VAD 拒判也重設；不重現即時執行緒的精確排程或跨錄音歷史，不宣稱與即時畫面逐格相同。
- 本次固定輸出 `data/reports/local_audio_comparison_20261015_v1/`：中文 report.md、576 列逐窗口 windows.csv、72 列逐錄音 recordings.csv、完成摘要 summary.json。包含各類機率、RMS、特徵 SHA-256、VAD／模型 fallback、整段平均預測與最後窗口預測、逐類指標與含拒判欄的混淆矩陣；沒有複製個人音檔、逐字稿或耳機裝置名稱到報告。
- 結果（整段有效窗口平均機率，依使用者 expected_label 計算一致率，不是正式測試準確率）：
  | 模型／模式 | 與預期一致 | Macro-F1 | 整段覆蓋率 |
  | --- | --- | --- | --- |
  | old/raw | 3/12（25%） | 0.1000 | 100% |
  | old/calibrated | 3/12（25%） | 0.1000 | 100% |
  | old/smoothed | 3/12（25%） | 0.1000 | 100% |
  | new/raw | 6/12（50%） | 0.3375 | 100% |
  | new/calibrated | 6/12（50%） | 0.3506 | 100% |
  | new/smoothed | 6/12（50%） | 0.3506 | 100% |
- 舊模型三模式的 12 段平均分類全部為 anger；新模型在三種模式均判對 neutral 3 段及 anger 3 段，但 happy 3 段、sad 3 段全未判對。new/raw 的 happy_01 為 anger，加權後改為 neutral，仍未符合預期；其餘整段類別不因加權改變，平滑後的整段類別與 calibrated 相同。不能據此宣稱新模型四類辨識已可靠，也不能把 50% 與之前資料集分數當作同一評估。
- 逐窗口共有 96 個位置、91 個通過 VAD；neutral_02 有 2 個拒判，happy_02 有 3 個拒判，其餘皆有效，全部 12 段仍有有效預測。old/raw 的有效窗口預測 neutral/happy/sad/anger=12/10/0/69；new/raw=52/1/2/36。old/smoothed=2/2/0/87；new/smoothed=50/3/2/36。這批沒有重現完整模型「幾乎都 sad」，但有少數 sad 窗口；不否定使用者先前即時觀察。
- 本次所有窗口的 confidence-floor fallback 均為 0。比較器保留目前正式平滑邏輯：若未來出現 confidence=0 的 neutral fallback，原程式可能在平滑後將它轉為有效分數，報告以 model_fallback 額外標記，不默默修掉來改變比較結果；這項既有行為不是本輪誤判原因。
- 新增 `tests/test_local_audio_comparison.py` 的 8 項測試，涵蓋平滑公式與 VAD 重設／每段隔離、既有零信心 fallback 行為、逐錄音等權與拒判分母、正式前處理計算一致／來源不變、設定錯配阻擋、錄音雜湊與 sidecar 核對、缺失／重複／不安全路徑及載入失敗拒絕。全專案 `python -B -m unittest discover -s tests -q` 共 86 項測試通過。
- 另獨立讀回兩份 CSV 重算逐錄音平均、分類、混淆矩陣及一致率，核對 96 個窗口在兩模型／三模式的特徵 SHA-256 完全相同；相同輸出目錄重跑會拒絕，4 份報告雜湊不變。執行前後既有 models／config／data/reports／本次錄音 session 共 171 個檔案 SHA-256 不變，既有未追蹤試跑模型與前輪未提交錄音工具均保留。
- 重現指令：`python -B scripts/compare_local_audio_models.py --session data/datasets/local_audio/session_20261015_220659_198289`。省略 output-dir 會建立新的時間戳報告資料夾；不能重用 v1 或刪除它以重跑。可用 --config 指定已存在且符合本次輸入規格的配置，設定原檔不改寫。
- 限制與後續：助理沒有人工聽音，expected_label 是使用者預期情緒，不是獨立盲標；這 12 段只能診斷，不能作為正式準確率或拿來反覆調參後仍稱獨立測試。加權／平滑沒有修正本批 happy／sad 的整段誤判，不宜只調類別權重就宣稱改善。即時與檔案結果不同時，先核對當時模型路徑、config、輸入裝置與平滑歷史；目前 main 的音訊串流未傳入 device 參數，使用系統預設裝置，而本批錄音工具明確指定索引 1，兩者是否相同尚未確認，不能直接斷言麥克風故障。
- 本次只新增比較腳本、測試與診斷報告並更新本節；沒有刪除資料、修改現有錄音／模型／config、開啟麥克風、啟動訓練或執行 git add、commit、push。

## 影像七類標籤核對與機率修正（2026-10-16）
- 使用者決定暫停語音模型改善，優先檢查電腦上難過／生氣表情不易辨識的問題。本次依一次一個功能原則，只完成既有影像模型標籤核對、機率計算修正及測試；未開攝影機、未收集臉部照片、未重訓或替換模型。
- 實際 `models/mini_xception_fp32.onnx` 的輸入為 `[batch,1,48,48]`、輸出為 `[batch,7]`，末端節點為 Softmax。SHA-256 為 `cc3d71c2b35a08dbc1eef4cf2fb32e699ac9dea5f77e021367a41e9d8c887d23`；目前檔案與首次加入模型的 Git commit `ea746f2` blob 一致，blob ID 為 `f3c65cfb6d4d50493c4468cc9e872fa2e1120bcb`，不是已重新訓練的四類模型。
- 核對 `ea746f2:scripts/train_video_mini_xception.py`，當時 EMOTIONS 與資料載入的 explicit class_names 順序為 neutral/happy/sad/anger/fear/surprise/disgust；當時下載腳本將 FER2013 原始編碼先存到具名資料夾，不能把 FER2013 原始數字順序直接當 ONNX 輸出順序。現有程式取前四類符合這份訓練配方，沒有發現前四類錯置。ONNX 本身未嵌入標籤 metadata，依據為同版 Git 訓練腳本及模型檔案追溯，不代表任意外部七類模型都能直接套用這個順序。
- 修正 `src/video/model.py`：先檢查完整四類／七類輸出是否為機率，再取前四類正規化。舊邏輯先截掉三類再檢查總和，當保留的機率質量不足約 95% 時，會把已經 Softmax 的機率當 logits 再做 Softmax，壓縮四類差距。保留既有 logits 分支的穩定 Softmax，不改灰階、48x48 resize、除以 255、NCHW 輸入、臉部裁切或融合規則。
- 加入有限值及四／七類長度檢查，無效輸出明確報錯，不靜默套用不明類別。七類機率中前四類總和為零時回傳 confidence=0 的既有 neutral fallback，不捏造四類有效機率；總和很小但非零仍依原始比例正規化。
- 限制：保留四類後正規化得到的是「限定四類的相對分數」，不是原始七類的絕對機率。若模型主要認為是 fear/surprise/disgust，四類相對分數仍可能很高；本次沒有新增被排除類別的拒判門檻，也沒有解決四類任務與七類訓練的差異。一般非退化輸入的四類最高排名不因這次修正改變，信心度上升不能當作辨識準確率提高，但可能影響現有融合權重／門檻。
- 新增 `tests/test_video_model.py` 的 8 項測試，涵蓋四類機率保留、七類機率正規化不重複 Softmax、前四類位置、保留質量接近 1、四／七類 logits、零／極小保留質量、無效長度／NaN／Inf；同時核對原模型輸出不被修改及 NCHW／除以 255 的輸入格式不變。全專案 `python -B -m unittest discover -s tests -q` 共 94 項測試通過。
- 使用實際 FP32 ONNX、CPUExecutionProvider，對黑／白／灰／固定種子亂數／漸層五種合成 48x48 輸入，比對修正結果與原始七類輸出的前四類直接正規化，全部通過，且修正前後最高類別一致。這不是人臉資料或準確率測試；沒有用合成圖片分數宣稱難過／生氣已改善。
- 裁切靜態檢查：現有管線使用偵測框直接裁切 BGR，再轉灰階與縮放 48x48；尚未查看使用者真正送入模型的臉部裁切，無法判定是否截掉眉毛／嘴巴或光線不佳。既有 `scripts/debug_video_model.py` 只做合成輸入，且用四類名稱索引七類 argmax，存在可能索引越界的限制；本次未執行或修改該工具，不將它當作攝影機／裁切驗證。
- 下一步先讓使用者檢視實際人臉裁切與 48x48 模型輸入，再做四種表情的固定情境測試；這項診斷介面尚未實作。模型只辨識影像表情，不能確認人的內在情緒；判不準不應直接解釋為使用者「不夠生氣或難過」。
- 本次只修改影像模型推論、增加測試並追加本節；models／config 既有 128 個檔案 SHA-256 前後一致，未修改語音、模型、設定、資料集或前端，未刪除任何資料，未執行 git add、commit 或 push。

## 人臉裁切與模型輸入預覽（2026-10-16）
- 新增獨立工具 `scripts/inspect_face_input.py`，讓使用者在電腦桌面手動開啟攝影機，並排查看 CAMERA（原畫面加偵測框）、FACE CROP（實際框內裁切）、MODEL GRAY（依目前 config 的 input_size，預設 48x48 灰階）。不接入 app.py、不執行情緒模型、不顯示情緒分數、不啟用語音、融合、後端或資料庫日誌。
- 沿用現有 `open_camera()` 與 `FaceDetector`／config 設定；每個讀到的畫面做一次人臉偵測，依相同 bbox 直接裁切，再用與 `VideoEmotionModel.predict()` 相同的 BGR→GRAY 及預設 resize 產生模型正規化前的影像。工具沒有改偵測門檻、裁切框、相機解析度設定或正式模型；不重現正式管線的 frame_skip／cached prediction。
- 預覽為固定 960x320 的單一視窗，三個區域顯示時保持各自圖片比例並補黑邊；48x48 灰階只在顯示端以 nearest-neighbor 放大，保留像素格子，不增加模型輸入細節。原 BGR frame／crop／gray 不因畫框或排版而被改寫。底部顯示實際 frame 尺寸、偵測後端、bbox 位置、裁切尺寸及灰階 min/max/mean；數值不是品質門檻或準確率。
- 無人臉時清空兩個裁切區域並顯示 No face，不保留上一張臉。偵測框無效會停止而非偷偷改框；讀取相機失敗會報錯。Q、Esc、關閉視窗或 Ctrl+C 可結束；正常結束與讀取錯誤會釋放相機並關閉偵測器。建立 OpenCV 視窗失敗時不會繼續開相機，適用有桌面與 GUI OpenCV 的電腦，不是 headless SSH 工具。
- 使用方式（專案根目錄、已啟用 .venv，先關閉 app.py 或其他占用攝影機的程式）：`python scripts/inspect_face_input.py`。可用 `--camera-id 1` 單次改相機索引，或 `--config config/config.yaml` 指定已有設定；不寫回 config。`--max-frames 30` 可限制讀取畫面數，但在真實執行時仍會開啟攝影機。`--help` 不開攝影機。
- 檢查重點：中間裁切是否完整包含眉毛、眼睛與嘴巴；轉頭／做表情時框是否跳動或切掉特徵；右側灰階中這些部位是否仍可辨認、有沒有過暗／過亮。48x48 放大後像素化是預期現象，不能單憑看起來模糊就認定模型故障；此工具也不能確認內在情緒或量測分類準確率。
- 新增 `tests/test_face_input_inspection.py` 的 11 項測試，涵蓋精確裁切／灰階、與實際 `VideoEmotionModel.predict()` 送出 tensor 的逐像素比對（灰階除以 255、NCHW）、來源陣列不變、無臉清空、固定版面、nearest-neighbor 等比例放大、無效框拒絕，以及模擬 Q／關窗／frame limit／讀取失敗／headless 的資源釋放與不開相機行為。全專案 `python -B -m unittest discover -s tests -q` 共 105 項通過，`--help` 正常。
- 已用合成色塊／線條產生預覽並目視確認三欄標題、圖片與底部數值沒有重疊；合成影像僅在記憶體中編碼傳回檢視，沒有儲存影像檔。所有測試都未啟動真實攝影機／GUI，助理尚未看到使用者的人臉裁切，也未驗證真實裝置／視窗關閉行為。
- 不儲存照片、影片、截圖或推論紀錄，不上傳影像。手動啟動時既有人臉偵測器仍可能建立 MediaPipe／matplotlib 或 Haar XML 資源快取，這些不是人臉影像。本次未啟動該偵測器，未安裝套件、刪除資料、修改模型或設定，亦未執行 git add、commit、push；保留前次尚未提交的影像機率修正。

## 手動表情擷取與七類／四類診斷（2026-10-16）
- 使用者回饋：先前裁切預覽在正常距離大致正常，太靠近並做動作時偶爾掉框。這是使用者的實機觀察，不是助理驗證或分類準確率；本次不修改人臉偵測門檻或模型。
- 新增 `scripts/capture_face_samples.py`，獨立於 app.py。依 neutral／happy／sad／anger 順序各手動擷取 3 張，共 12 張；沿用既有攝影機、FaceDetector、裁切與 48x48 灰階預覽。只在預覽視窗按空白鍵且當下偵測到臉時保存，沒有臉不保存也不推進編號；每張後暫停，按 Enter 才允許下一張，避免長按空白鍵連拍。畫面顯示下一張編號、預期表情與 READY／PAUSED，不提前顯示分類結果引導使用者選圖。
- 啟動方式（專案根目錄、啟用 .venv，先關閉占用相機的 app.py／其他工具）：`python scripts/capture_face_samples.py`。可用 `--camera-id 1` 或 `--config config/config.yaml`，不寫回設定；`--help` 不開啟攝影機。空白鍵拍攝、Enter 準備下一張、Q／Esc／關窗退出；Ctrl+C 也會保留已完成的樣本。完成 12 張自動結束，不會在預覽時自動連拍。
- 每次建立 `data/datasets/local_faces/session_時間戳/`，已受既有 `data/datasets/` Git ignore 保護。只保存人臉裁切 PNG、實際模型輸入的 48x48 灰階 PNG，不保存完整攝影機畫面或音訊，不上傳。使用新目錄及排他寫入，拒絕覆蓋；錯誤時保留已寫入資料，不刪除半成品。
- `session.json` 記錄設定、模型 SHA-256 與擷取順序；每張 JSON 記錄預期表情、時間、bbox、實際畫面尺寸、偵測後端、兩張 PNG 的 SHA-256，以及原始七類與四類分數。每張 JSON 在兩張 PNG 成功保存後才寫入；只有 PNG 而無個別 JSON 的樣本不能視為完成。結束時輸出 `summary.json` 與中文 `report.md`；未收滿或中斷標示 incomplete。若程序被強制終止或磁碟寫入失敗，可能沒有總報告，已完成的個別 JSON／圖片仍保留。
- 工具只接受前節已追溯的 `mini_xception_fp32.onnx` SHA-256 `cc3d71c2b35a08dbc1eef4cf2fb32e699ac9dea5f77e021367a41e9d8c887d23` 及 48x48 輸入，拒絕未知模型或 heuristic fallback，避免套錯七類順序。先將灰階除以 255 並整理成 NCHW，再讀原始七類 Softmax；前四類除以其總和，另外記錄 fear／surprise／disgust 的總機率，不將排除類別合併成四類。四類總和為零時記錄拒判，不捏造 neutral 100%。
- Markdown 報告比較預期表情、七類最高及原始機率、四類最高及相對分數、排除機率，附圖片連結；完整分布在 JSON。四類分數是限定四類後的相對值，排除機率很高時四類高分不能解讀為可靠。使用者指定表情不是獨立盲標，這 12 張僅供診斷，不能當正式準確率、不能推定內在情緒，也不宜拿同批照片反覆調參後宣稱獨立測試。
- 新增 `tests/test_face_sample_capture.py` 的 14 項測試，涵蓋七／四類分數與正式模型推論計算一致、零保留質量、無效輸入／未知模型拒絕、PNG 無損往返及雜湊、排他寫入、無臉不推進、按鍵防連拍、完整 12 張順序、中途停止與相機錯誤保存報告、headless 不開相機、編碼失敗保留半成品與報告拒絕覆蓋。測試使用記憶體檔案與模擬攝影機／GUI，不建立或刪除測試照片。
- 驗證：`python -B -m unittest discover -s tests -q` 全專案 119 項通過；`python -B scripts/capture_face_samples.py --help` 正常。另用實際 ONNX 對全黑／全白／固定種子亂數三組合成輸入，比對正式 `VideoEmotionModel._predict_onnx()` 四類結果通過，未計算人臉準確率。以記憶體合成色塊檢視 960x370 擷取版面，未發現標題、圖片、數值與狀態列重疊。
- 本次尚未收集使用者臉部照片，未開啟真實攝影機／GUI、未驗證實際鍵盤或樹莓派。沒有修改正式推論、模型、config、語音或前端，沒有重訓、刪除資料或執行 git add／commit／push。下一步由使用者固定距離與光線完成擷取，再檢視難過／生氣誤判是否與裁切、原始七類分布或四類正規化有關，尚不直接決定重訓。

## FERPlus 預訓練模型同照片比較（2026-10-18）
- 使用者已完成 `data/datasets/local_faces/session_20261018_190005_599916/`，四類各 3 張。原始診斷的四類符合預期數為 neutral 2/3、happy 2/3、sad 2/3、anger 0/3，共 6/12；全部使用 Haar 偵測，七類與四類最高類別相同。這是單人預期表情一致率，不是正式準確率，不代表使用者的內在情緒。
- 經使用者確認，新增獨立 `scripts/compare_face_models.py`：只讀現有照片，比較原本七類 CNN 與外部已訓練的 Emotion FERPlus，不接入 app.py、不修改 config、不重訓、不開相機或麥克風、不重新偵測裁切，也不上傳照片。正式影像模型仍維持原檔。
- FERPlus 來源為 ONNX Model Zoo，固定 Git revision `c5cae0f1942c8d7727372c7b1f6b485aa39e4421` 下的 `validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx`。核對上游 Git LFS pointer 的 SHA-256 `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c` 與長度 35,040,571 bytes；下載後雜湊／大小吻合，ONNX checker 通過。權重約 35 MB（33.42 MiB），置於既有 Git ignore 涵蓋的 `.cache/ferplus/`，沒有下載資料集或整個模型庫。
- 只有明確加 `--download-model` 才會連線下載固定來源；預設比較離線執行。下載使用新時間戳暫存檔，驗證成功才改名，失敗保留暫存不刪除。既有模型雜湊不符會停止，不覆蓋。此工作另保存上游 README、倉庫 LICENSE 及 `source.json` 來源紀錄。
- 授權說明需保留差異：模型子目錄 README 標示 MIT，ONNX Model Zoo 根 LICENSE 與 Hugging Face metadata 標示 Apache-2.0。本次保存上游原文並僅作本機比較，不自行宣稱權重再散布條件已完全釐清；若未來提交或散布第三方模型，須先確認並附上適用聲明。上游說明：https://raw.githubusercontent.com/onnx/models/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/validated/vision/body_analysis/emotion_ferplus/README.md 。
- 輸入核對：完整 12 張 session、預期順序與編號、原模型雜湊、七類順序、個別 JSON 與 summary 一致、PNG 雜湊、尺寸、檔名不越出資料夾、無重複圖片路徑。由原裁切重算 48x48 灰階並逐像素比對保存的 gray；再用原 ONNX 重算，確認最高類別及機率與擷取紀錄相符。任一項不符停止，不跳過有問題的照片以提高分數。
- 舊模型沿用 OpenCV BGR→灰階→48x48 linear resize→float32 /255、NCHW，讀取既有 Softmax。FERPlus 使用相同原始 BGR 裁切，先 OpenCV 灰階，再 Pillow LANCZOS 縮放至 64x64，float32 保持 0..255、NCHW；不把舊的 48x48 放大、不除以 255。實際 FERPlus 圖包含輸入 Sub／Div，末端 Add 輸出 logits，因此後處理只做一次穩定 Softmax。
- FERPlus 原始順序為 neutral/happiness/surprise/sadness/anger/disgust/fear/contempt；只統一 happiness→happy、sadness→sad 的名稱。四類依索引 `[0,1,3,4]` 選取後正規化，不可取前四個，也不把其他情緒硬合併；完整八類及排除的四類機率另保留。四類總和為零時記錄拒判。沒有加權、時間平滑或依這批照片調整門檻。
- 執行：`python scripts/compare_face_models.py --session data/datasets/local_faces/session_20261018_190005_599916`。第一次沒有候選權重時才加 `--download-model`；本機已下載，不必再加。預設每次建立新時間戳報告目錄；`--output` 可指定新目錄，既有目錄一律拒絕覆蓋。本次固定結果為 `data/reports/face_model_comparison_20261018_v1/`，含中文 `report.md` 與 `summary.json`，沒有複製照片。完成 JSON 最後才寫入，缺少時不能視為完整報告。
- 本次結果（完整類別及限定四類的最高類別均一致）：
  | 預期表情 | 原模型符合 | FERPlus 符合 |
  | --- | --- | --- |
  | neutral | 2/3 | 3/3 |
  | happy | 2/3 | 2/3 |
  | sad | 2/3 | 1/3 |
  | anger | 0/3 | 2/3 |
  | 合計 | 6/12（50%） | 8/12（66.7%） |
- 差異：01 中性由 sad 改為 neutral；07 難過由 neutral 改為 sad，但 08／09 難過由 sad 改為 neutral；10／11 生氣改判 anger，12 仍未符合預期。新模型改善中性／生氣，但難過退步，不能宣稱全面勝出；更不能把這批 66.7% 寫成獨立測試集準確率。這批可篩選候選，後續需新照片、其他人及樹莓派驗證，不以同批樣本反覆調參後當獨立測試。
- 本機 Windows／ONNX Runtime CPUExecutionProvider、intra threads=2／inter threads=1，預熱後 12 次 session.run 中位數：舊模型約 0.19 ms、FERPlus 約 9.92 ms。只計模型呼叫，不含讀圖／前處理／偵測／GUI，也不是嚴格效能基準或樹莓派即時效能；環境版本與逐張時間保存在 JSON。候選比舊模型慢，不能由檔案小就宣稱樹莓派部署已通過。
- 新增 `tests/test_face_model_comparison.py` 14 項測試，涵蓋 raw 灰階／LANCZOS／來源不變、八類索引、保留排除類別、穩定 Softmax 與拒判、無效輸出與雜湊、路徑安全、完整 session／sidecar／gray 一致性、重複路徑、拒絕覆蓋、拒判仍計入分母與原結果重現。測試不寫入或刪除測試照片。`python -B -m unittest discover -s tests -q` 全專案 133 項通過，CLI `--help` 通過。
- 另獨立讀回報告重算兩模型完整／四類一致數及正規化，確認重跑同輸出位置被拒絕且兩份報告不變；既有 config、models 與本次照片 session 共 167 個檔案 SHA-256 前後一致。`requirements_train.txt` 補列診斷工具直接使用的 `pillow>=9.1,<13`，本機原有 Pillow 12.2.0，沒有安裝／升級套件。
- 本次僅新增比較工具、測試、報告、候選模型快取與來源紀錄，補列 Pillow 並追加本節；沒有刪除任何資料、修改正式推論／設定或舊模型，沒有 git add、commit 或 push。尚未改成正式使用 FERPlus，下一步可另行規劃獨立即時預覽與新樣本驗證，不能假設已支援正式 PC／RPi 管線。

## 筆電 FERPlus 獨立即時預覽（2026-10-18）
- 新增 `scripts/preview_ferplus.py`，讓使用者手動在筆電啟動 FERPlus 攝影機預覽。只新增獨立工具，不接入 app.py、前後端、融合或資料庫，不修改正式模型路徑，也不切換／合併樹莓派分支。保留前次 FERPlus 比較工具及報告等尚未提交的工作。
- 啟動指令（專案根目錄、啟用 .venv，先關閉 app.py 或其他占用相機的程式）：`python scripts/preview_ferplus.py`。可用 `--camera-id 1` 單次選相機，`--config config/config.yaml` 只讀取相機／偵測設定，`--max-frames 30` 限制實際讀取畫面數；預設持續到 Q／Esc／關閉視窗／Ctrl+C。`--help` 不開相機。
- 固定使用前節已下載的 `.cache/ferplus/emotion-ferplus-8.onnx`，先驗證 SHA-256、64x64 float NCHW 輸入及八類輸出，再建立視窗／相機。缺少模型或雜湊不符會停止，不自動下載、不改用 heuristic 或舊模型。這不是任意 ONNX 模型的通用預覽器。
- 重用 `compare_face_models.py` 的 `open_session()`、`fer_tensor()`、`fer_scores()`，保留相同 CPU 執行設定、OpenCV 灰階→Pillow LANCZOS 64x64→float32 0..255 NCHW、一次 Softmax、完整八類及四類 `[0,1,3,4]` 對應。預覽右欄直接使用實際送進模型的 tensor 轉回 uint8，而非人臉預覽工具的 linear resize 結果。
- 視窗為固定 960x560，上方並排相機／人臉裁切／64x64 灰階，下方顯示 FULL (8) 原始最高類別與機率、FOUR (relative) 限定四類的相對分數、Excluded mass 排除機率及八類機率條。八類名稱為 neutral／happy／surprise／sad／anger／disgust／fear／contempt；未將其他情緒硬合併。保持既有 OpenCV 診斷視窗的英文文字，沒有改動前端中英切換。
- 每個有效畫面偵測一次，找到臉就執行一次分類；不使用正式管線的 frame_skip、舊結果快取、時間平滑、門檻或類別加權。因此逐格分數可能跳動，不能把跳動直接視為程式故障，也不能把分數當準確率。無臉時清空裁切／灰階／全部分數，不保留上一個情緒；若四類保留機率總和為零，顯示 Abstain 而非 neutral 100%。
- Model inference 顯示單次 session.run 毫秒數，不含人臉偵測、前處理、畫面繪製或相機讀取，亦沒有排除第一次模型冷啟動；不是 FPS、端到端延遲或樹莓派效能。實際可用性、即時辨識與兩平台表現仍須實機測試。
- GUI 無法建立時不開相機；相機讀取或推論出錯時會離開並釋放相機、關閉偵測器與視窗。工具沒有照片／影片／推論記錄寫入及上傳路徑，也不啟動語音、Web 後端或模型下載；使用者啟動時，既有人臉偵測器仍可能建立資源快取，不能把「不存影像」解讀為所有相依套件完全不寫快取。
- 新增 `tests/test_ferplus_preview.py` 的 12 項測試：預覽 tensor／機率與離線共用流程一致、來源影像不變、無臉不推論且清除前次資料、固定版面與文字邊界、完整類別及四類拒判顯示、Q／關窗／畫面上限／讀圖失敗／推論失敗的資源釋放、headless 不開相機，以及模型缺失／雜湊錯誤先停止。全專案 `python -B -m unittest discover -s tests -q` 共 145 項通過，CLI `--help` 正常。
- 另外用真實 FERPlus ONNX 與原 12 張裁切，在模擬偵測框下執行新 `process_frame()`，完整八類機率及最高類別與先前離線報告全部一致；不產生新照片或報告。以記憶體合成色塊目視檢查版面，沒有標題／圖片／數值／機率條重疊；未儲存該合成圖。
- 助理未開啟真實攝影機／GUI、未驗證使用者實際鍵盤及關窗操作、未測樹莓派，沒有宣稱難過辨識改善。下一步由使用者先觀察生氣是否較容易辨識、難過是否仍偏中性，以及移出畫面時是否清空結果，再決定是否值得進一步整合。
- 本次只新增預覽腳本與測試並追加本節；沒有新增依賴、下載模型、刪除資料或執行 git add／commit／push，既有模型、設定與照片保留。

## FERPlus 筆電實測回饋（2026-10-18）
- 使用者自行操作獨立即時預覽後，主觀認為整體辨識體驗改善；尚未記錄固定次數、逐次預測或其他人的結果，因此不是正式準確率，也不能據此宣稱四類都比原模型好。
- 生氣：使用者感覺與原本差不多，但同時脫掉眼鏡並加大表情時可以辨識。眼鏡與表情強度同時改變，不能認定眼鏡或反光就是唯一原因；日常戴眼鏡仍應是後續驗證情境，不要求使用者永久摘眼鏡。
- 難過：偶爾被判為 neutral，sad／neutral 混淆仍存在，尚未量化改善幅度。使用者未指明觀察的是 FULL (8) 或 FOUR (relative)，後續比較須固定讀取同一欄位。
- 無臉：使用者確認移出畫面後會清空結果。這是筆電實測回報，不代表助理已操作硬體或樹莓派已驗證。
- 暫時保留 FERPlus 為候選，不自動取代正式模型。若後續確認整合，先在筆電加入可選模型並保留原模型回退，再另行測試樹莓派；本次尚未實作整合或調整門檻。
- 本次僅追加實測觀察至 AGENTS.md，未修改程式、模型、設定或既有資料，未執行 git add／commit／push。文件更新以 git diff --check 檢查格式，不把既有單元測試或主觀體驗當準確率證明。

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
