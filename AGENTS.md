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
