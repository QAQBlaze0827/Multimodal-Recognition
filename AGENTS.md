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
- **Raspberry Pi Camera 支援（branch: `feature/rpi-camera-support`）**
  - 修改 `src/video/capture.py`
  - 原本只使用 OpenCV `VideoCapture` 開啟相機
  - 現在會先嘗試 OpenCV `VideoCapture`
  - 若 OpenCV 無法讀取相機，會 fallback 到 `picamera2`
  - Pi Camera 輸出格式指定為 `RGB888`，再轉成 OpenCV 使用的 BGR frame
  - 已在 Raspberry Pi 5 + IMX708 Camera 上測試可成功讀取影像
  - Raspberry Pi 上需使用 `python app.py --no-display`，避免 OpenCV `cv2.imshow()` GUI 錯誤

- **Raspberry Pi MediaPipe 實測**
  - Raspberry Pi 上未安裝 `mediapipe` 時，系統會 fallback 到 OpenCV Haar cascade
  - 實測 Haar cascade 在 Pi Camera 上較不穩，`face_detected` 可能持續為 `0`
  - 在 Raspberry Pi 的 `.venv` 手動安裝 `mediapipe` 後，video emotion 開始正常產生資料
  - `mediapipe` 尚未加入 `requirements_rpi.txt`，後續需確認是否要納入 RPi 安裝依賴

- **INMP441 / Google voiceHAT 音訊聲道實測**
  - Raspberry Pi 上音訊裝置顯示為 `snd_rpi_googlevoicehat_soundcar`，sounddevice device id 為 `0`
  - 硬體實際格式為 `S32_LE`、`2 channels`、`48000 Hz`
  - 實測左聲道可清楚收到人聲，右聲道為雜訊
  - 右聲道資料包含大量 32-bit 極值，例如 `-2147483648`
  - 因此後續音訊接收應指定 device、讀取雙聲道、只取左聲道，避免右聲道雜訊進入模型
  - 目前建議設定方向：`device_id: 0`、`channels: 2`、`channel_select: left`、`input_sample_rate: 48000`、`sample_rate: 16000`

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

**結論**：Phase 2 為最佳配置（N_MFCC=13, Conv1D 32→64, Dense 64, Dropout 0.3, augment=True），另外 CREMA-D 資料集與 RAVDESS/TESS 不相容導致 val_acc 雪崩，已排除。Phase 3 所有嘗試均無正向效果。

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

## 樹莓派儀表板與柴犬整合測試（2026-11-03）
- 從 `feature/rpi-camera-support`（`c6025f0`）建立筆電本機測試分支 `codex/rpi-dashboard-companion`。本次只帶入 `main`（`4d9e0cc`）的四個前端檔：`frontend/index.html`、`frontend/css/style.css`、`frontend/js/app.js`、`frontend/js/websocket.js`，以及 `tests/test_live_freshness.cjs`，不是整支 main 合併。
- 即時頁分開顯示後端連線與辨識資料是否仍在更新；有效資料停止更新達 5 秒後淡化最後判讀。「互動陪伴」頁可點擊柴犬歪頭；連續三筆新的有效開心臉部結果觸發一次輕跳，不採用語音或綜合分數。沒有接入語音轉文字。
- 筆電執行 `node --test tests/test_live_freshness.cjs`：17 項通過。比對前端與來源版本一致，並核對測試檔 Git blob；Pi Camera、INMP441 的 48 kHz 雙聲道選左、推論與融合設定、模型、後端和 RPi 相依未更動。未啟動辨識、未操作資料庫，沒有實機驗證、提交、推送或合回任何分支。
- 下一步：提交並推送測試分支後，在 Pi 確認工作目錄乾淨才切換，使用原有啟動方式，從筆電瀏覽器驗收相機、麥克風、即時更新和互動；通過後再決定是否合回 Pi 分支。正式程式仍沿用原本資料保留規則。

## 樹莓派 FERPlus 可選模型：筆電整合（2026-11-03）
- 使用者已回報 Pi 儀表板與互動可運作，並確認 Pi 的 `feature/rpi-camera-support` 已在 `ff1712a`。本次從此提交開本機 `codex/rpi-ferplus-test`；從 main 的 `4d9e0cc` 移植 `src/video/ferplus.py` 與 `src/main.py` 的模型切換，不合併整個 main。沒有更動 Pi Camera、INMP441、舊模型分數、融合、設定、資料庫或柴犬動作。
- PC 與 Pi 的執行需求補列 Pillow；只有選 FERPlus 才匯入該模組。原指令仍用 legacy；Pi 測試影像時使用 `python app.py --vision-only --no-display --video-model ferplus`，需要語音融合時移除 `--vision-only`。選 FERPlus 時缺檔、SHA-256 不符或 ONNX 契約錯誤會在開相機之前停止，不自動改用其他模型。
- 部署前在 Pi 的專案 .venv 確認 Pillow 可用，並將已驗證的模型放在專案根目錄下的 `.cache/ferplus/emotion-ferplus-8.onnx`；這個目錄被 Git 忽略，pull 不會取得模型。在專案根目錄用 `sha256sum .cache/ferplus/emotion-ferplus-8.onnx` 核對預期值 `a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c`。模型八類輸出取 neutral/happy/sad/anger 後正規化；結果是四類相對分數，不是準確率。
- 筆電：`python -B -m unittest discover -s tests -v` 通過新增的 3 項測試（預設模型、缺檔不開相機也不降級、四類索引）；`node --test tests/test_live_freshness.cjs` 通過 17 項。用已存在的模型、CPUExecutionProvider 與合成影像做實際推論，四類分數總和約 1。未開啟相機、麥克風或正式日誌；未安裝或下載、未測 Pi 的 FPS／準確率，未提交、推送或合回。

## 筆電獨立麥克風轉錄測試（2026-11-09）
- 經使用者同意，本次新增的是專題外的實驗，不是正式語音情緒辨識或網頁功能。位置：`C:\Users\QAQBl\AppData\Local\emotion-stt-test\transcribe_mic.py`；同資料夾的 `test_transcribe_mic.py` 為合成／模擬測試。兩個檔案不在本 repo 的 Git 管理範圍，換機或 clone 不會取得，正式整合前再決定納入專案的方式。本 repo 只追加本節，保留 `codex/rpi-ferplus-test` 分支與既有未追蹤產物。
- 使用外部 `.venv\Scripts\python.exe`，既有 faster-whisper 1.2.1、soundfile 0.14.0、NumPy 2.4.6；本次只在此環境新增 sounddevice 0.5.6。未更動專題 `.venv`、requirements、情緒模型、設定或原始錄音，未刪除資料。套件安裝需要連網；程式轉錄使用既有 `models` 快取的多語言 base、CPU INT8，設定 `HF_HUB_OFFLINE=1` 與 `local_files_only=True`，不下載新模型或上傳音訊。
- 程式預設使用作業系統的預設麥克風，16 kHz 單聲道 float32、每塊 100 ms。約 0.8 秒低能量停頓後轉錄，保留最多 0.3 秒前置音訊、至少 0.3 秒達門檻聲音，每段最多約 12 秒；長句會被切段，不是逐字串流或保證完整語意的斷句。原始 RMS 門檻預設 0.002，可用 `--threshold` 調整；這是粗略音量判斷，不等於語音偵測準確率，背景噪音或低音量仍可能影響切句。完成片段再由 faster-whisper `vad_filter=True` 過濾，指定中文、beam_size=5、condition_on_previous_text=False；不保證靜音／噪音永遠沒有幻覺文字。
- 模型啟動時只載入一次，錄音 callback 與轉錄 worker 分開。收音佇列最多 50 塊、轉錄待處理最多 3 段；轉錄塞滿會明確顯示略過新片段，不無限累積延遲。收音溢位／狀態錯誤、斷流、模型錯誤會停止並顯示原因，不假裝仍在正常辨識。Ctrl+C 關閉收音，等待正在執行的轉錄完成後離開，不保存待處理或未完成片段。程式沒有錄音／逐字稿寫檔功能，但終端機仍顯示文字；音訊在記憶體處理，不宣稱具備作業系統層級的安全抹除保證。
- 啟動（筆電 PowerShell，任意工作目錄）：`& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" "$env:LOCALAPPDATA\emotion-stt-test\transcribe_mic.py"`。加 `--list-devices` 只列裝置，不載入模型、不開收音；加 `--device N` 選擇當次裝置編號，不要假設編號跨電腦或重插後固定。啟動前檢查 16 kHz 單聲道支援，不支援則報錯，不偷偷改取樣率。此次唯讀列舉預設為 HyperX 耳機麥克風，格式檢查通過，但未開啟真實 InputStream。
- 驗證：在外部工具目錄執行其 Python 的 `-B -m unittest -v test_transcribe_mic`，13 項通過，涵蓋靜音界限、前置音訊、停頓、短脈衝排除、低音量、連續聲音上限、重複切句、來源不變、轉錄選項、worker 例外、佇列容量／轉錄時仍能排入下一段、模擬 Ctrl+C 與開流失敗的資源釋放、只列裝置不開模型或收音；`--help` 與 `--list-devices` 通過。
- 真實模型離線檢查：一秒合成靜音輸出空字串。只讀使用者既有 `OneDrive\文件\錄音\錄製 (3).mp3`、`錄製 (4).mp3`、`錄製 (5).mp3`，解碼到 16 kHz 並在記憶體補尾端靜音，以 100 ms 塊快速餵給相同分段器，三份各產生一段；分別轉出「你先安靜一下,我現在不想玩」「先安靜一下,我現在不想玩」「我今天有點累想休息一下」。先前 0.01 門檻漏掉第 5 份較小聲錄音，故調整為 0.002 並新增低音量測試；這批已用來調整門檻，不能當作獨立測試集。轉錄本身約 0.45～0.59 秒，不含模型載入、真實等候停頓／網路與網頁延遲；不是即時效能驗收或正式準確率。
- 下一步由使用者手動開啟程式，確認終端機 Input 名稱、真人句子與否定詞、停頓出字、連說與安靜狀態；目前沒有接樹莓派音訊、後端、網頁、狗狗文字回覆、TTS 或 LLM，不啟動 app.py、正式資料庫或歷史清理，也未操作 git add／commit／push／切換分支。

## 筆電轉錄文字與固定回覆（2026-11-09）
- 使用者回報已執行前節真人麥克風測試：輸出包含「今天有點內想休息一下」及「現在不想玩,請先安靜」，語音中的「累」有誤辨，但主要要求仍保留。貼出的三筆 queue=0.00 秒、STT=0.44～0.77 秒，僅記錄該次觀察，不等於正式準確率、端到端延遲，亦未完成長時間靜音驗收。
- 經確認，本次只在外部 `C:\Users\QAQBl\AppData\Local\emotion-stt-test\` 新增 `dog_reply.py` 與 `test_dog_reply.py`，並在 `transcribe_mic.py` 的轉錄輸出後加入 `Dog [intent]: 回覆`。原本收音、停頓、模型、佇列與耗時欄位不變；沒有安裝新套件、開真實麥克風、重訓、修改情緒辨識、連線樹莓派或接網頁。新增中文回覆與測試文字使用 UTF-8，終端機沿用既有 UTF-8 輸出。
- 規則是單次、無對話記憶的固定文字比對：company／play／rest／quiet 四類，模糊、未知或衝突回傳 clarify；空白或純標點不回覆。先做有限簡繁字元與空白正規化，再以標點分句、完整短句匹配，不以任意子字串「玩」觸發。「不想玩」不邀請玩耍；「不想玩，想休息一下」回休息；「想玩，請先安靜」或同意圖正反矛盾會詢問。未支援的句型、轉述、雙重否定等保守回詢問，並非自然語言理解的完整解法。
- 支援「有點…想休息一下」的有限描述前綴，因此使用者範例的「內」不需被硬改成「累」，仍可依明確休息要求回應；原始轉錄照常顯示，不改寫錄音或文字、不虛構 ASR 已修正。這是依目前例句設計的規則，不是獨立語意準確率測試。若 ASR 漏掉否定詞，規則仍可能回錯；未來更長或跨片段句子需另外處理。
- quiet 在這一步只顯示「好，我安靜陪著你。」等文字，不停止錄音、不建立持續靜音狀態、不控制角色動畫，後續新句子仍會獨立回覆。沒有 LLM、TTS、網頁對話框、音訊傳輸、逐字稿保存或上傳，模型仍從本機快取載入。
- 驗證：外部環境在工具目錄執行 `python -B -X utf8 -m unittest -v`，原 13 項與新增 15 項共 28 項通過。新增涵蓋四種意圖、否定、衝突、未知／轉述／雙重否定的回詢問、有限簡繁與空白、純標點、誤辨前綴、無狀態，以及模擬 worker 印出回覆／空白不回話；另直接以使用者兩句文字與陪伴、玩耍、衝突、未知例句核對結果。沒有重新測量模型準確率或真人收音。
- 啟動指令不變：先在使用者已開的舊程式按 Ctrl+C，再於筆電執行 `& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" "$env:LOCALAPPDATA\emotion-stt-test\transcribe_mic.py"`。測「可以陪我一下嗎」「我想玩」「我想休息一下」「我不想玩，請先安靜」，另測矛盾句確認會詢問；先由使用者驗收終端機回覆，之後再決定網頁整合。本 repo 僅追加本節，外部工具仍不隨 Git 提交；沒有 git add／commit／push／切換分支或刪除資料。

## 陪伴回覆句型補充（2026-11-09）
- 使用者真人轉錄已正確輸出「你可以陪我一下嗎」「我希望你可以陪我」「我想要陪伴」，卻落入 clarify；確認是固定句型涵蓋不足，不是本次 ASR 誤辨。經確認只修改外部 `emotion-stt-test/dog_reply.py` 的陪伴規則與 `test_dog_reply.py`，保留原本收音、模型、門檻、其餘意圖、分支及既有文件變更。
- 陪伴支援「你可以／你能不能」與「可以你／能不能」等語序、「希望／想要你陪我」、名詞式「想要陪伴／需要你的陪伴」；上述三句皆回 company。補列「不想要陪伴／不需要你的陪伴／不希望你陪我」等拒絕：不觸發 company，quiet 回覆改為「好，我先不打擾你。」而非仍說要陪伴。這仍僅是單句文字回覆，不實作持續靜音或停止收音。
- 保留完整短句與否定／衝突檢查，不改成單純子字串搜尋；「我不知道要不要你陪我」、轉述、雙重否定、自我疑問及未支援否定句仍詢問。陪伴與安靜同時要求仍依既有保守規則回 clarify，不宣稱能理解所有自然語句；「陪我玩」仍為 play，不被陪伴規則吞掉。
- 驗證：外部工具目錄以獨立 Python 執行 `-B -X utf8 -m unittest -q`，新增 6 項陪伴回歸與模擬 worker 測試，合計 34 項通過；另直接檢查使用者三句、拒絕、疑問、衝突與玩耍例句。未開真人麥克風、未重跑模型或測量辨識準確率，未安裝套件、修改正式程式、儲存錄音、刪除資料、提交或推送。本 repo 僅追加此節；使用者需 Ctrl+C 停止舊程序後以原指令重啟，再測原本三句及否定句。

## 心情描述與陪伴要求分開處理（2026-11-10）
- 使用者回報「我好難過可不可以陪我一下」已被 ASR 轉出文字（audio=4.40 秒、queue=0.00 秒、STT=0.46 秒），但固定回覆落入 clarify。這筆問題是規則不接受心情前綴，不是 12 秒切段上限或本次轉錄失敗；不據此推估整體準確率或陪伴成效。
- 經確認，本次僅修改專題外 `C:\Users\QAQBl\AppData\Local\emotion-stt-test\dog_reply.py` 與 `test_dog_reply.py`。先辨識句首有限的自述背景（難過、低落、沮喪、孤單、心情不好、過得不太好，搭配常見程度與時間詞），再檢查明確陪伴要求，支援同一片段有／無標點，以及「所以／但是／但／不過／可是」連接詞。只去除已辨識背景與連接詞，不任意搜尋整句中的「陪我」，不移除否定；不改原始轉錄文字，也不把自述當成新情緒模型輸出或心理診斷。
- 「我好難過可不可以陪我一下」「今天過得不太好，我希望你陪我」回 company；「我好難過，但我不想要你陪我」回 quiet／「好，我先不打擾你。」；單獨「我好難過」或明確猶豫「我不知道要不要找人陪」回 clarify／「你希望我陪你待一下嗎？」。不因難過就自動答應陪伴，同一片段明確正反要求、轉述、雙重否定、未知語句仍不硬猜。
- 範圍僅擴充陪伴：新增背景搭配玩耍／休息／安靜等其他要求時仍保守詢問，不擴充其他指令的語意；原本不含新增心情背景的四類規則回歸通過。這仍是有限固定句型，不支援任意敘事、跨片段上下文或多輪確認；單獨回答「好」尚不會自動承接前一則詢問。ASR 漏掉否定詞仍可能誤回，實際使用者體驗需以未用來調規則的新句子驗證。
- 驗證：外部工具目錄使用獨立 Python 執行 `-B -X utf8 -m unittest -v`，原 34 項加新增 9 項共 43 項通過，涵蓋無標點／有標點、拒絕優先、自述只詢問、猶豫、轉述／雙重否定與未知內容、衝突與其他意圖邊界、有限簡繁與空白，以及模擬 worker 保留原文並輸出回覆；另直接核對本次五種預期例句及矛盾句。這是文字規則與模擬整合驗證，沒有真人收音或新的模型準確率量測。
- 使用者先 Ctrl+C 停止舊程式，再以既有外部環境指令重啟 `transcribe_mic.py`。後續請組員自由說新的句子，分開記錄轉錄是否正確、回覆是否符合要求；通過後才評估接網頁。本次沒有修改收音、Whisper、情緒分數、其他正式程式或設定，沒有安裝套件、開相機／麥克風、存錄音、接樹莓派／網頁／TTS／LLM，也未刪資料或操作 git add／commit／push／切換分支；本 repo 只追加本節並保留既有未提交變更。

## 筆電語音固定回覆接入網頁（2026-11-12）
- 使用者決定暫停準確度與新語意模型的研究，保留現有 Whisper 轉錄與 `dog_reply.py` 規則，先完成網頁互動。沒有修正「好，陪我一下」等未支援句型，也沒有加入 LLM、TTS、多輪記憶、語音驅動動畫或樹莓派音訊傳輸。第一版流程是「筆電 Python 收音與轉錄 → 固定回覆 → 本機 FastAPI → 網頁對話框」，瀏覽器本身不開麥克風。
- Repo 新增 `backend/companion.py`、`frontend/js/companion.js`、`tests/test_companion_bridge.py`、`tests/test_companion_dialogue.cjs`；修改 `backend/app.py`、`frontend/index.html`、`frontend/css/style.css`、`frontend/js/app.js` 與本節。既有狗狗 SVG、點擊歪頭與連續開心表情觸發輕跳保留；`src/`、情緒模型、設定、相依清單、`backend/database.py`、原情緒 WebSocket 邏輯皆未修改。
- 外部 `C:\Users\QAQBl\AppData\Local\emotion-stt-test\` 修改 `transcribe_mic.py`，新增 `web_reply.py` 與 `test_web_reply.py`；沒有修改 `dog_reply.py`。新選項 `--web-url http://127.0.0.1:8000` 才傳送文字，未帶此參數時維持原本終端機模式。外部檔案仍不在本 repo Git 範圍，clone／提交 repo 不會取得它們；未搬移或刪除原檔、未安裝套件、未下載模型。
- 後端必須明確加 `--companion-local` 並綁定 loopback，否則拒絕啟用；未加參數時原有功能照常，新對話通道關閉。新增 `POST /api/companion/events` 與獨立 `/ws/companion`，不把私人文字廣播到原本情緒 `/ws`，也不讓文字更新情緒趨勢、分數或狗狗表情動作。啟用時不信任代理轉送的來源 IP；寫入只接受本機來源、合法 Host 與 JSON，拒絕帶 Origin 的瀏覽器寫入；文字 WebSocket 要求本機及同源 Origin。這只是單機測試邊界，不是跨裝置驗證／授權方案，也未宣稱修復原後端所有資安議題。勿直接改成公開服務或加代理對外轉送。
- 外部傳送器只在麥克風成功開啟後啟動獨立執行緒，每約 2 秒心跳，單次 HTTP 逾時 1 秒，最多排隊 3 則文字，超過 5 秒的待傳回覆略過。停用環境代理且不跟隨 HTTP 重新導向；傳送失敗會提示，終端機與轉錄繼續，不無限重送舊對話。每個來源有隨機 session ID 與遞增序號，後端忽略重複／較舊序號；同時間只接受一個仍有效的轉錄來源，另一個收到 409。Ctrl+C 關閉傳送並通知 stopped，處理中的轉錄仍依原流程收尾。
- 網頁「互動陪伴」新增柴犬回覆框、最近一則原始轉錄、後端收到回覆的時間，以及獨立的語音狀態；底部明列「表情互動」避免與語音混淆。介面標籤支援中英切換，但不翻譯使用者原文或固定中文回覆。所有轉錄／回覆使用 `textContent`，不解析成 HTML，不存 localStorage。沒有聊天歷史列表；停止／失聯清空對話，重連可取得仍有效的最近回覆，不需要先有臉部資料。
- 後端只保留最近一則文字於記憶體，透過每秒狀態快照檢查有效期；來源超過 8 秒未更新會標為逾時、清空回覆，單則回覆達 60 秒亦會在檢查時移除。沒有連線或後續請求時，不保證記憶體內物件會準時於第 60 秒清除，也不承諾安全抹除。沒有寫入 SQLite、錄音檔或逐字稿檔；原本終端機仍會顯示轉錄。網頁開啟在同一台電腦的人可看到文字，勿當成不同本機使用者之間的隔離機制。
- 驗證：專題 `.venv` 執行 `python -B -X utf8 -m unittest discover -s tests -v` 共 10 項通過（既有 FERPlus 3 項＋新後端 7 項）；`node --test tests/test_live_freshness.cjs tests/test_companion_dialogue.cjs` 共 24 項通過（既有 17＋新增 7）。外部 Python 執行其目錄 `-B -X utf8 -m unittest discover -s . -q` 共 52 項通過（既有 43＋新增 9），沒有啟動真實收音。測試涵蓋否定規則不變、選用傳送、佇列與逾時、停止、重複訊息、本機／Origin／Host／格式與大小限制。TestClient 顯示既有 httpx 棄用警告，測試仍通過，沒有為此更動相依。
- 真實整合驗證：以專題後端 `--db :memory:` 啟動於 `127.0.0.1:8000`，外部傳送器搭配既有 `choose_reply` 傳入測試文字，瀏覽器收到相同回覆。Playwright 驗證真實 HTTP → WebSocket → DOM、來源逾時恢復、停止清空、斷線重連快照、中英標籤、HTML 字串不執行、重複訊息拒絕、情緒紀錄仍為空、原點擊歪頭可用。1280px 桌面、375px 手機及 320px 長中文／連續英文字換行檢查通過，沒有橫向溢出或對話與狗狗重疊；檢查桌面與手機截圖。預期的瀏覽器寫入拒絕測試產生一筆 HTTP 403 console 訊息，不是正常功能錯誤。截圖與測試伺服器日誌位於 `output/playwright/companion-20261112/`，只含測試文字。
- 目前未測真人「收音→轉錄→網頁」完整流程、跨裝置或樹莓派；沒有新準確率／端到端延遲數字。未開正式 `app.py`、未讀寫正式 `data/emotion.db` 或執行歷史清理，未刪資料、未 git add／commit／push／切換分支。保留原 `codex/rpi-ferplus-test` 與先前未提交／未追蹤內容。本次單機預覽後端保留執行供使用者驗收，未啟動其麥克風。

### 單機語音網頁驗收啟動方式
1. 筆電專題根目錄的第一個 PowerShell：若同 port 已有本次預覽後端，不要重複開啟；以下是日後重開的指令。`:memory:` 是測試資料庫，不會顯示原有臉部／情緒歷史，也不會改正式資料。

```powershell
.\.venv\Scripts\python.exe -B backend/app.py --host 127.0.0.1 --port 8000 --db :memory: --companion-local
```

2. 另一個 PowerShell（任意目錄）：先 Ctrl+C 停止舊轉錄程序，避免同時開兩份收音，再執行：

```powershell
& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" "$env:LOCALAPPDATA\emotion-stt-test\transcribe_mic.py" --web-url http://127.0.0.1:8000
```

3. 開 `http://127.0.0.1:8000`，切換「互動陪伴」。說「我想玩」「我好難過可不可以陪我一下」「我不想要你陪我」，確認終端機與網頁的原文／回覆一致。停止轉錄後確認顯示已停止；異常斷開則約 8 秒後標為逾時。只測對話框不用開 `python app.py`；「表情互動：等待辨識資料」在這種測試下屬預期。若 8000 被其他程式占用，選其他空閒 port，後端、`--web-url` 與瀏覽器網址三處一致修改。若後端原本未啟用文字橋接，重啟帶參數後需重新整理頁面。
4. 這一步尚不支援筆電轉錄直接送到樹莓派 IP／Tailscale。未來需另外確認連線授權與傳輸設計，不能直接放寬本機限制。若之後要同時展示正式情緒紀錄，需在確認後用正式資料庫重新啟動後端；不要在目前記憶體測試模式下誤認為臉部辨識失效。

## README 語音互動啟動流程補充（2026-11-14）
- 使用者回報文字 POST 持續 404，並要求詳細啟動流程與提交方式。本次只修改 `README.md`、追加本節，保留先前尚未提交的功能程式與所有既有產物；沒有再改辨識、回覆規則、後端或前端程式。
- README 新增筆電語音互動專節與入口連結，保留原 Quick Start／一般情緒儀表板內容，並提醒其指令不會啟用語音接收。分開說明 PowerShell A 的專題 `.venv`、PowerShell B 的外部轉錄 `.venv`、工作目錄、既有環境需求、完整參數、預期訊息、瀏覽器操作、驗收例句及停止順序。
- 明列 `--companion-local` 與 `--web-url` 不能漏掉、只支援本機、`:memory:` 不讀寫正式情緒紀錄、表情等待狀態屬預期、安靜回覆不會停止收音。補上 404／409、port 占用、終端機有字而網頁沒字、模型／外部檔案缺失、麥克風選擇與逾時的排查方式。提醒外部工具不隨 Git 提交或 clone，以及 health 成功不等於語音接收已啟用。
- 驗證方式：比對目前後端與外部轉錄程式的實際參數，分別執行兩支程式的 `--help`（不開收音、不啟動伺服器、不載入模型），檢查 README 新增內容與 Markdown 格式、`git diff --check`。本次是文件更新，未重跑真人收音或瀏覽器驗收、未安裝套件、未刪資料、未操作 git add／commit／push／切換分支；提交指令僅提供給使用者手動執行，並提醒外部程式尚不受本 repo 管理。

## 樹莓派語音互動準備：工具納入專案（2026-11-16）
- 使用者確認最終方向為「Pi 麥克風收音 → 筆電 Whisper 轉文字與固定規則回覆 → 文字傳回 Pi 後端 → 原樹莓派網頁顯示」，並授權開始實作。本回合只完成第一步：從 `codex/rpi-ferplus-test` 的 `1c4367a` 建立並切換至本機 `codex/rpi-voice-companion`，整理既有筆電工具；沒有合併 `main` 或 `feature/rpi-camera-support`，沒有建立遠端分支、暫存、提交或推送。
- 使用者在前次對話已回報舊外部工具的真人網頁驗收：四句有更新、網頁與終端機一致、停止清空、重啟恢復、該次安靜觀察沒有亂出字。這補充先前「未測真人」的狀態，但只代表使用者當次功能觀察，不是一般語意準確率、長時間靜音保證或 Pi 傳音驗收。搬入後的真人測試仍需另做。
- 新增 `tools/voice_companion/`，從外部 `C:\Users\QAQBl\AppData\Local\emotion-stt-test\` 複製 `transcribe_mic.py`、`dog_reply.py`、`web_reply.py` 與對應三支 `test_*.py`。複製時六檔 SHA-256 與原檔一致；之後只調整專案內轉錄腳本的模型快取定位、增加相關測試與獨立 `requirements.txt`，外部原檔、環境、模型與錄音全部保留。沒有納入 `.venv`、模型快取、私人音訊或逐字稿。
- `transcribe_mic.py` 不再以腳本所在資料夾尋找模型，Windows 預設沿用 `%LOCALAPPDATA%\emotion-stt-test\models`；沒有 `LOCALAPPDATA` 時改用 `~/.cache/emotion-stt-test/models`。新增 `--model-dir` 可指定既有 base 模型的下載快取根目錄，並印出 `Model cache:`。仍使用多語言 base、CPU INT8、`HF_HUB_OFFLINE=1`、`local_files_only=True`；缺模型報錯，不自動下載或改用其他模型。Linux 的預設路徑只有 mock 測試，未測 Linux 收音。
- 收音、16 kHz 單聲道、切句、RMS 門檻、回覆規則、否定處理、容量與逾時、原本本機限定的 `--web-url` 都保持不變。新增 `requirements.txt` 記錄現有 Python 3.11 語音環境的 faster-whisper 1.2.1、ctranslate2 4.8.5、numpy 2.4.6、sounddevice 0.5.6、soundfile 0.14.0；未鎖定所有間接依賴，未執行新環境安裝，不宣稱全新機器安裝已驗證。不要把這份清單裝進正式情緒辨識 `.venv` 或 Pi。
- 更新根目錄 `README.md` 的語音啟動與懶人包，保留既有章節：改執行專案內腳本、說明新機器的兩套環境與首次模型下載、列裝置與獨立測試方式。已備妥這台筆電可直接沿用外部 `.venv` 和模型，不須安裝或下載。兩個 PowerShell 都從專題根目錄執行；後端仍用 `--host 127.0.0.1 --db :memory: --companion-local`，語音指令為 `& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" -B .\tools\voice_companion\transcribe_mic.py --web-url http://127.0.0.1:8000`。若選 8001，後端、轉錄網址與瀏覽器三處同步修改。
- 驗證：搬入前外部原版 52 項測試通過；搬入後用外部語音 Python 執行 `-B -X utf8 -m unittest discover -s .\tools\voice_companion -v` 共 57 項通過，新增 5 項涵蓋快取預設路徑、明確覆寫、無 LOCALAPPDATA、缺檔不改線上下載及 CLI 傳遞。專題 Python 的 `-B -X utf8 -m unittest discover -s tests -v` 10 項通過；`node --test tests/test_live_freshness.cjs tests/test_companion_dialogue.cjs` 24 項通過。既有 TestClient httpx 棄用警告仍在，未為此更新套件。
- 另確認專案內 `--help` 可用，實際載入外部既有 Whisper 快取並轉錄一秒合成靜音，結果為空字串；沒有讀私人錄音或開真實麥克風。README 的 22 段 PowerShell 通過語法解析、74 行程式碼圍欄成對、`git diff --check` 通過。這次未重跑瀏覽器、未啟動後端或正式 `app.py`，也未接觸正式資料庫與自動清理。
- 近期 Pi 排查背景：使用者曾回報網頁固定約 10 秒前，瀏覽器紀錄 ID 持續增加、接收間隔小於一秒；校時排查後畫面恢復「資料持續更新／0 秒前」，支持跨裝置時鐘落差引發過期判定，未在本回合修正 freshness 邏輯。畫面仍曾顯示未偵測到臉，不能宣稱影像辨識已完整恢復。使用者另回報 Pi 無散熱片／風扇、辨識時約 70°C，未由工具量測；傳音整合後需比較溫度、FPS 與延遲，不預先承諾效能不變。
- 下一步才是 Pi 左聲道單次收音分流與有驗證的跨裝置傳輸，先確認筆電終端機轉錄，再接回 Pi 網頁。此次沒有修改 `src/`、後端、前端、現有 config／requirements、情緒模型或語意規則；沒有 Tailscale 設定、網路安全限制放寬、LLM、TTS、語音控制動畫。既有未追蹤訓練產物與日誌未刪除或暫存。

## 網頁轉錄統一繁體顯示（2026-11-16）
- 使用者回報 Whisper 有時輸出簡體，並同意先只調整網頁顯示。維持 `codex/rpi-voice-companion`；修改 `frontend/js/companion.js`、`frontend/index.html`、`tests/test_companion_dialogue.cjs`、`README.md` 與本節，新增 `frontend/vendor/opencc-js-1.4.2/`。沒有修改 Whisper、收音、原始轉錄、傳輸格式、後端、情緒模型、固定回覆或語意規則，也沒有進行 Pi 音訊整合。
- 使用 OpenCC JS 1.4.2 的 `cn2t` UMD bundle，初始化 `OpenCC.Converter({ from: 'cn', to: 'tw' })` 一次，僅在寫入 `companion-transcript.textContent` 時轉換。`companionReply.text`、收到的原始事件、終端機文字與回覆決策保持原樣；英文介面只切換標籤，轉錄仍顯示繁體。既有繁體與英文例句回歸通過，但不宣稱所有歧義字都轉換正確；不採 `twp` 額外台灣用詞轉換，不修正 ASR 錯字。若轉換腳本缺失，主控台警告並顯示原文，對話與清空機制仍可運作。
- 從 npm 官方 registry 下載固定版本封包到被忽略的 `.cache/vendor/opencc-js-1.4.2/`，比對封包 SHA-512 與版本 metadata 一致；只將未修改的 `dist/umd/cn2t.js`（1,100,656 bytes，約 1.1 MB）、MIT `LICENSE`、`LICENSES/Apache-2.0.txt` 與 `THIRD_PARTY_LICENSES.md` 複製到前端 vendor。另新增來源／版本／雜湊 README 及 `cn2t.js text eol=lf` 屬性，避免跨平台換行影響 SHA-256 檢查。套件來源 https://github.com/nk2028/opencc-js ，封包 https://registry.npmjs.org/opencc-js/-/opencc-js-1.4.2.tgz ；轉換本身由瀏覽器離線字典完成，不將轉錄送到外部轉換服務，也不在執行時從 CDN 下載字典。未執行 npm install 或安裝 Python 套件。
- 測試：`node --test tests/test_live_freshness.cjs tests/test_companion_dialogue.cjs` 共 30 項通過（原 24＋新增 6），涵蓋本機腳本順序與 bundle SHA-256／授權檔、簡轉繁且原文與回覆不變、繁體／英文／數字／標點／歧義詞例句／ASR 錯字、中英切換、HTML 安全顯示，以及缺少字典時降級。`node --check frontend/js/companion.js` 與 `git diff --check` 通過；專題 Python `-B -X utf8 -m unittest discover -s tests -q` 10 項通過。既有 TestClient httpx 棄用提示未處理；字典缺失測試中的警告是預期輸出。
- 實際瀏覽器驗證：另開 `127.0.0.1:8011` 的 `--db :memory: --companion-local` 後端，不占用既有 8000 程序。Playwright 透過 HTTP 傳入測試文字，再經真實 WebSocket 到頁面；「我好难过，可以陪我一下吗？」顯示「我好難過，可以陪我一下嗎？」，前端記憶體保留簡體、回覆與意圖不變、情緒資料仍為空。確認 OpenCC 資源由同一個本機後端載入、中英切換不改原文、HTML 字串不產生元素、停止後清空；1280px、375px 及 320px 長中文／英文測試沒有橫向溢出，已檢視桌面與手機截圖，瀏覽器 console 無錯誤或警告。
- 截圖與測試伺服器日誌位於 `output/playwright/companion-traditional-20261116/`，只含合成測試文字；本機 8011 預覽後端保留供驗收。使用者原本筆電網頁按 `Ctrl+F5` 亦可載入新靜態檔；若改用 8011，轉錄 `--web-url` 要對應 8011。終端機仍出現簡體是保留原始轉錄的預期行為。未啟動真實相機／麥克風、未測 Pi、未寫正式情緒資料庫、未刪除任何資料，未操作 git add／commit／push／切換分支。下一步仍是樹莓派音訊到筆電的獨立傳輸驗證。

## Pi 音訊傳到筆電終端機：第一階段（2026-11-16）
- 使用者確認繼續下一步；維持 `codex/rpi-voice-companion`，只做「Pi 單次收音分流 → 筆電 Whisper → 固定回覆印在終端機」。未把回覆接回 Pi 網頁，未放寬原 `--companion-local` 的 loopback／Origin 限制，也未新增 LLM、TTS、語意規則、動畫或模型訓練。
- 修改 `src/main.py`、`src/audio/inference.py`；新增 `src/audio/remote.py`。明確加 `--stt-url http://筆電TailscaleIPv4:8765` 才建立傳送器，與 vision-only／停用 audio 不相容，URL／金鑰／16 或 48 kHz 驗證在開硬體前完成。沿用同一個 InputStream、原 device 與聲道設定；選好的左聲道在 peak normalization 前分流，網路在獨立執行緒，不另外搶麥克風。未改 config、原音訊重採樣／VAD／模型／平滑／融合、Pi Camera 或 FERPlus；關閉時加入 stream 與 sender 的 finally 清理，audio-only 的收音執行緒結束後不再留下空轉主循環。
- 啟用傳送時讀取 100 ms 塊、每 0.5 秒送一包 mono signed 16-bit little-endian PCM。收音待傳佇列最多 10 塊，超過 1 秒的塊捨棄；序號、擷取時間停頓、狀態錯誤或遺失觸發半句重設。網路失敗不重送舊音訊，原情緒辨識繼續；401／403 停止此傳送器，修正設定後需重啟。48 kHz 音訊本體約 96 KB/s，不含協定開銷，這是格式計算而非實測頻寬。
- 新增 `tools/voice_companion/receive_audio.py`，僅在筆電獨立 STT 環境執行；不開筆電麥克風。使用已存在 PyAV 18.1.0 的連續 AudioResampler 將 48 kHz 濾波重採樣成 16 kHz，再沿用 PhraseBuffer、base CPU INT8 離線快取、Whisper VAD 與 choose_reply。這個新路徑不使用 `samples[::3]`；原情緒管線保留。工具 requirements 補列直接使用的 `av==18.1.0`，本次未安裝套件或下載模型，Pi 不需要這份 Whisper／PyAV 依賴。
- 傳輸為標準函式庫 HTTP，僅允許明確 `100.64.0.0/10` IPv4 或兩端 127.0.0.1 測試；接收器綁筆電指定地址，驗證 Pi peer IP、64 hex 環境金鑰 `COMPANION_AUDIO_TOKEN`、Host、Origin、長度／格式、UUID session 與序號。拒絕瀏覽器 Origin 與代理替代 IP，不跟隨 redirect／環境 proxy。IP 範圍檢查不能證明 Tailscale 已運作；HTTP 本身無 TLS，部署必須確認實際走可信 Tailscale 加密連線及存取規則，禁止公網轉發。不是公開網路服務或完整資安認證；未自動改防火牆或 Tailscale。
- 接收器只接受一個有效來源；3 秒來源逾時、停止、斷包、來源／取樣率改變會清除半句與待轉錄內容；正在轉錄的舊世代結果也不再顯示。轉錄佇列最多 3 段，超過 5 秒的結果捨棄，不累積無限延遲。單執行緒 HTTP 有 socket 等待上限，慢速請求可能延後逾時檢查，不是硬即時保證。Windows 上拒絕有 body 的請求可能產生 TCP reset、遮住 401，故加入有大小與時間上限的拒絕內容排空，並以實際 loopback 驗證錯誤金鑰會停傳。
- 此功能不寫錄音、逐字稿、模型或對話資料庫；終端機保留原始 ASR，仍可能簡體。啟用後靜音／背景聲亦持續傳到指定筆電，不受 face-gating 或情緒 VAD 控制，需取得在場者同意；停筆電接收器不等於停 Pi 麥克風。記憶體、環境變數與終端機不宣稱可抵禦本機管理員或保證安全抹除。完整 app 模式原有 CSV／SQLite／歷史保留清理政策不變，本次未啟動正式辨識或觸碰正式資料庫。
- 新增 `tests/test_audio_remote.py`、`tools/voice_companion/test_receive_audio.py`。專題 Python unittest 共 21 項、獨立 STT Python unittest 共 73 項、Node 即時頁／對話框共 30 項通過。涵蓋 opt-in、單次取左聲道、預設 blocksize 不變、開流失敗清理、容量／丟塊／過期、PCM clipping、16/48 kHz 切句與抗混疊、來源與金鑰／Host／Origin／長度限制、重複／第二來源、斷線及 in-flight 結果清除。真實 loopback HTTP 使用合成 PCM、真實 PyAV、模擬轉錄字串接到既有回覆規則，沒有真人收音或 Tailscale 實機。既有 httpx 棄用警告未處理。
- 另外用筆電真實既有 Whisper 快取轉錄 2 秒合成靜音，輸出空字串；這不代表語音準確率或長期無幻覺保證。兩個 CLI 的 --help 通過；未讀私人錄音、未開真實相機／麥克風，所有本次 loopback 測試伺服器均已停止。Pi 溫度、FPS、48 kHz 實際 callback 穩定性與端到端延遲仍待使用者實機驗收，不因原本約 70°C 就預先宣稱有足夠散熱。
- README 新增模式入口與說明，另新增 `tools/voice_companion/REMOTE_AUDIO.md` 詳列兩台分支／環境前置、Tailscale IP、臨時共享金鑰、筆電接收器先開、Pi audio-only、四句／靜音／停止／恢復驗收，再選擇加回影像。先由使用者提交推送並更新 Pi 後才可測；此回合未 git add／commit／push／切換／合併分支，未刪除任何資料，既有未追蹤產物保留。終端機驗收通過後，下一步才設計文字回傳 Pi 網頁。

## 語音傳輸簡化啟動與保存設定（2026-11-16）
- 使用者回報已在 Pi 說「我想玩」，筆電成功顯示相同轉錄及 `Dog [play]`。這證明當次 Pi 收音 → 筆電轉錄 → 固定回覆跑通，不是正式準確率、完整靜音／斷線驗收；使用者指出效果差的是語音情緒，與 ASR 分開保留問題。本次不重訓或修改情緒模型，也不接回 Pi 網頁。
- 經同意，在 `codex/rpi-voice-companion` 新增 `tools/voice_companion/start_receiver.ps1`、`start_sender.sh`、`start_sender.py`，只包裝原來的啟動方式，不改傳輸、收音、驗證、回覆、後端、前端、config 或依賴。新增同目錄 `.gitattributes` 固定 Bash wrapper 為 LF；首次設定、日常啟動與文件一併處理，沒有新增常駐服務、開機自動收音、LLM、TTS 或自動 Git 操作。
- 筆電在原 PowerShell 以 `& .\tools\voice_companion\start_receiver.ps1 -SetupOnly` 保存／檢查，直接繼承普通變數 `$laptopIp`、`$piIp` 與環境金鑰。Pi 在原 Bash 用 `. ./tools/voice_companion/start_sender.sh --setup-only`，必須保留開頭點與空格，才能繼承未 export 的 `laptop_ip`；wrapper 在子 shell 中轉交給專案 `.venv/bin/python`，不退出原 SSH 或改變原工作目錄。只有缺少值才詢問；金鑰隱藏輸入，不產生新金鑰、不印出金鑰，也不需要使用者貼給 AI。
- 筆電保存於 `%LOCALAPPDATA%\emotion-stt-test\connection\receiver.json`，金鑰用 Windows 當前使用者 DPAPI（ConvertFrom-SecureString，不另帶加密 key），專用資料夾／檔案限制目前使用者；既有目錄權限過寬或非本人擁有時停止，不擅自修改既有權限。Pi 保存於 `~/.config/multimodal-voice/sender.json`，資料夾 700、檔案 600、檢查擁有者與連結；Pi 金鑰仍是明文，不宣稱磁碟加密、能防 root／同帳號惡意程式或 SD 卡離線讀取。預設位置在專案外，Windows 不放進本專案 OneDrive 目錄，禁止將設定另加入 Git／同步。
- 已保存時以檔案為準，不被當次舊環境變數覆蓋；損毀、解密失敗、權限／格式錯誤停止，不刪除、不自動重設、不覆寫原設定。首次建立採 CreateNew／O_EXCL；兩台與格式驗證通過才保存，固定此階段 port 8765，不新增設定編輯器。IP／金鑰更換或非標準 port 先用舊手動流程驗證，再明確處理保存設定更新；不能誤以為改環境變數會更新已有檔案。
- 日常只需筆電 `& .\tools\voice_companion\start_receiver.ps1`，等 Listening 後 Pi `. ./tools/voice_companion/start_sender.sh`。筆電固定使用既有外部 STT Python，Pi 固定此 repo 的 .venv，不須 activate；Pi 仍 audio-only，不開相機／正式資料庫。Windows 程序執行時暫設金鑰環境並於返回後還原，Pi 透過子程序環境傳遞金鑰，不放進命令參數。結束先停 Pi，再停筆電；接收器關閉本身不停止 Pi 麥克風。
- 新增 `test_start_receiver.ps1`、`test_start_sender.py`。Windows PowerShell 真實 DPAPI／ACL 的 12 個檢查通過：沿用父層 IP／金鑰、密文可還原、磁碟與輸出不含明文 token、保存後無提示及不覆寫、非法 IP／token／過寬權限拒絕、缺值才詢問。WSL Ubuntu 的 Linux 測試 12 項通過，含實際 POSIX 700／600、損毀保留、連結與權限拒絕、缺值／保存優先、mock exec 不把金鑰放進參數，以及真實 Bash source + 合成子程式確認普通變數繼承、原 shell 繼續且 cwd 不變；不是 Pi 硬體驗收。
- 回歸：Windows STT unittest 共 85 項（75 通過、10 項 Linux 專用略過，已另在 WSL 執行）；專題 unittest 21 項、Node 即時／對話 30 項通過；Bash 語法檢查通過。既有 TestClient 棄用警告及缺字典模擬警告保持原樣。測試只使用臨時目錄的合成憑證，Windows `Temp/voice-launcher-test-*` 與 WSL `/tmp/voice-*-test-*` 保留、不刪除；沒有讀取或建立使用者真實連線設定、開真實麥克風／相機、啟動真實模型／接收器、改防火牆／執行原則或安裝套件。
- README 最上方新增兩個日常指令；REMOTE_AUDIO.md 新增首次沿用設定與保存位置，保留既有手動流程並清楚區隔。使用者需先自行提交、推送並讓 Pi 更新新增腳本，才能在原終端機匯入設定；本次沒有 git add／commit／push／切換分支、刪檔或修改既有未追蹤產物。下一步先驗收保存後重啟，再處理文字回傳 Pi 網頁。

## 相機預覽分離影像與資訊（2026-11-17）
- 使用者已回報 Pi 桌面與 VNC 可開啟，OpenCV 4.11.0 顯示 QT5 支援，但原 320×240 預覽被六行除錯文字遮住。經確認，在既有 `codex/rpi-voice-companion` 修改 `src/output/display.py`、`src/main.py`，新增 `tests/test_preview_display.py`；只調整顯示，不改相機解析度、模型、音訊、融合、網頁、設定或資料庫行為。
- 預覽建立 960×700 顯示副本：上方 960×540 區域等比例放置影像、留黑邊，依縮放比例畫人臉框；下方 160px 分兩欄顯示 Face／Video／Audio／Fused／FPS／溫度與分數非準確率提示。移除座標、模型後端等冗長疊字，原相機 frame 不被修改。使用英文標籤避免 OpenCV 內建字型不支援中文字；沒有新增中文字型或套件。
- `draw_overlay()` 現在回傳預覽影像，主循環將回傳值交給 imshow；只在顯示模式建立 WINDOW_NORMAL／WINDOW_KEEPRATIO／WINDOW_GUI_NORMAL 視窗並設定初始尺寸，Qt 模式不顯示原工具列。沒有臉或零信心度時不顯示預設中性為有效影像結果；無有效輸入時 Fused 顯示 `--`。這只是預覽表達，不修改原模型分數、融合、正式日誌或網頁顯示。
- 驗證：專題 `.venv` 執行 `python -B -m unittest discover -s tests -v` 共 28 項通過（原 21＋新增 7），覆蓋原 frame 不變、人臉框位置、橫直影像比例、文字邊界、缺臉／零信心度／缺溫度／audio gating、模擬顯示初始化與 headless 不呼叫 GUI。以真實 OpenCV 渲染合成預覽 PNG 並人工檢視，未見重疊／裁字，保留於 Windows Temp 的 `emotion-preview-h9whnm0x/preview.png`。`git diff --check` 通過；既有 httpx 棄用警告未處理。
- 未開實體相機／麥克風、未啟動正式後端或資料庫、未操作真實 VNC 視窗；Pi 拖曳縮放、FPS、溫度尚待實機確認，顯示放大仍有額外繪製／傳輸成本，不宣稱效能不變。維持 `--no-display` 原行為，SSH 無桌面時仍可使用；本次不更動系統 VNC、Qt、桌面或相依套件。
- 部署由使用者確認指定四檔後自行提交並 push `codex/rpi-voice-companion`；Pi 先確認同分支且工作目錄乾淨，再 `git pull --ff-only origin codex/rpi-voice-companion`。停止原相機程序，在 VNC 桌面終端機執行 `./.venv/bin/python app.py --vision-only --video-model ferplus`（不加 `--no-display`），確認人臉框、資訊區、拖曳縮放與按 q 離開；需語音時移除 `--vision-only`。本次沒有 git add／commit／push／切換分支、刪除資料或處理既有未追蹤產物。

## 統計分析：有效紀錄與分布分母（2026-11-22）
- 經使用者同意，只完成統計分析第一步，維持 `codex/rpi-voice-companion`。修改 `backend/database.py`、`frontend/index.html`、`frontend/css/style.css`、`frontend/js/app.js`、`frontend/js/charts.js`，新增 `tests/test_analytics.py`、`tests/test_analytics.cjs`，並追加本節。未修改即時頁、歷史、回放、狗狗互動、辨識模型、融合、擷取、寫入格式或資料庫 schema；沒有加入來源切換、重新訓練或刪除舊紀錄。
- `/api/analytics` 保留 days 參數與既有回傳欄位，distribution 改為僅統計有效綜合紀錄。來源有效定義為「face_detected=1 且臉部類別為四類之一、數值分數在 (0,1]」，或「音訊類別為四類之一、數值分數在 (0,1]」；另外綜合結果也須有四類標籤與 (0,1] 數值分數。這是現有紀錄可支持的有效性判斷，不證明真的有人聲／模型判對，也不重新推論歷史資料。
- 無有效來源（包含舊紀錄來源欄位不足）與有來源但綜合結果無效分開計數；兩者皆不進入情緒分布。低分數定義為有效綜合紀錄的分數 <0.5，是有效紀錄的子集合，仍納入分布與平均模型分數。50% 僅為顯示分類門檻，沿用原介面中低分數分界，不是新推論門檻或準確率校準；等於 50% 不屬於低分數。原始紀錄不刪除、不改寫。
- SQL 在選取期間到現在的範圍內分類與聚合，不將所有逐筆資料載入前端；未來時間不計入。新增有效／排除／無來源／綜合無效／低分數筆數、固定分類門檻與實際最早／最晚紀錄。session_count 改為此期間實際有紀錄的不同 session 數，避免忽略在區間前開始但仍有新紀錄的 session。現有本地無時區時間格式保留，未實作跨時區遷移；舊資料缺少來源欄位時不能還原當時感測狀態。
- 網頁改為「綜合辨識統計」：四項摘要、實際資料範圍、排除原因、分布分母，以及四類情緒筆數／紀錄占比／平均模型分數表格與圓環圖。強調分數不是準確率、紀錄占比不是情緒持續時間或心理評估。只在統計頁改用模型分數稱呼，其他頁文案保持原樣。中英與手機排版同步；Chart.js 圖例移到底部並停用動畫，避免窄畫面壓縮與切換過渡誤解。
- 載入開始即清空舊統計與圖表；空資料、全部排除、請求失敗與舊後端缺少新欄位分別顯示狀態。遞增請求序號避免快速切期間／語言時舊回應覆蓋新結果。Chart.js CDN 未載入時仍顯示數值表格；沒有新增外部相依，既有 CDN 依賴未改成本機。時間範圍透過 textContent 顯示，不當 HTML 解析。
- 驗證：Python unittest 共 35 項通過（新增 7），Node 即時／對話／統計共 36 項通過（新增 6）。涵蓋無訊號中性排除、來源與分數合法性、低分數保留及 0.5 邊界、僅音訊來源、空資料、期間／未來資料／session 範圍、API 格式與 days 驗證、查詢不改資料、舊圖清除、中英、請求競爭、錯誤恢復、缺圖表元件與文字安全。既有 httpx 棄用與缺字典模擬警告未改。
- 真實瀏覽器：僅綁 `127.0.0.1:8012` 的獨立預覽後端，資料庫位於 Windows Temp `analytics-preview-2ab85eb06e1c4adeaaaee35904555a86/synthetic.db`，非 `data/emotion.db`。八筆合成資料經真實 POST／統計 API 得到總數 8、有效 5、低分數 2（有效子集）、排除 3；Playwright 使用現有 Edge headless 驗證 1440／375／320px、中英、canvas 非空、表格邊界、空期間及模擬全排除／500 後恢復，無 pageerror。檢視桌面與手機截圖；腳本與圖保留於既有被忽略的 `output/playwright/`，不應當作正式模型成效。未安裝套件；CLI 未安裝，改用 bundled Playwright library，未使用 @playwright/test。
- 沒有開相機／麥克風、讀取或更動正式情緒資料、執行歷史清理、刪除任何檔案或操作 git add／commit／push。預覽服務留供使用者查看合成資料；樹莓派尚未驗收。部署時前後端需一起更新、重啟後端並強制重新整理網頁；辨識程式無需因本次更新重啟。歷史／回放依然沿用舊規則，和新統計的數量可能不同；模型／使用者未分組的舊紀錄不能解讀為單一人的長期心理狀態。

## 歷史紀錄：每次使用摘要與辨識明細（2026-11-22）
- 經使用者確認，接續統計頁改善歷史紀錄，目標是先看懂每次使用的辨識概況，再查逐筆明細。維持 `codex/rpi-voice-companion`；修改 `backend/app.py`、`backend/database.py`、`frontend/index.html`、`frontend/css/style.css`、`frontend/js/app.js`，新增 `frontend/js/history.js`、`tests/test_history.py`、`tests/test_history.cjs`，追加本節。未更動模型、融合、擷取、寫入、資料庫 schema、即時／回放／陪伴邏輯。
- 歷史頁由最近 500 筆平鋪與情緒／session 篩選，改為期間內的每次使用摘要及明細；期間為 7／30／90／365 天。摘要每頁 10 次使用、明細每頁 20 筆，由伺服器分頁，不把完整歷史載入瀏覽器。新增唯讀 `/api/history/sessions` 與 `/api/history/records`；後者使用 query parameter 傳 session ID。原 sessions API 保留供回放使用。
- 歷史與統計共用同一段來源／綜合結果有效性 SQL。摘要顯示實際首末紀錄、可參考／總筆數、其中模型分數低於 50% 的筆數、無有效來源及綜合結果無效筆數；低分數仍屬有效子集。情緒占比只使用有效紀錄，並列最多會列出所有同票類別。沒有有效結果時不顯示中性為主要結果。明細保留無效紀錄的時間與狀態，但不把其預設情緒／分數顯示成有效判讀；異常非有限分數回傳 null，原資料不改寫。
- 所有數字僅涵蓋選定期間到查詢當下，跨界 session 只計區間內紀錄，未來時間排除。每次使用不等於同一個人，紀錄占比不是情緒持續時間、心理狀態或準確率；「可參考」僅代表符合資料有效規則，不代表模型判對。舊資料不足以區分使用者或模型，未自行補上不存在的資訊。
- 介面支援中英、桌面／手機、摘要及明細分頁；動態文字使用 textContent。載入時移除舊結果，空資料／錯誤各有狀態，請求序號防止舊摘要／明細覆蓋新期間；點明細移動焦點到明細標題。資料查詢仍沿用現有本地時間格式，未修改跨裝置時間同步或加入新的身分驗證。
- 驗證：Python unittest 41 項通過（新增 6），Node 即時／對話／統計／歷史 40 項通過（新增 4），git diff --check 通過。涵蓋和統計計數一致、低分數邊界、只有音訊、無訊號、非有限分數、空資料、期間、分頁順序、特殊 session ID、API 參數限制、查詢無寫入、並列結果、錯誤與請求競爭。既有 httpx 棄用警告未處理。
- Playwright 使用既有 Edge headless，真實 HTTP／DOM 驗證 12 次使用的合成紀錄、兩層分頁、無效紀錄不顯示中性、1440／375／320px、中英、空期間、500 錯誤恢復，無 pageerror 或橫向溢出；檢視桌面與手機截圖。測試腳本／截圖在未追蹤的 `output/playwright/`，不是正式成效資料。獨立預覽後端 `http://127.0.0.1:8013/` 使用 Windows Temp `history-preview-2f5cb5ce5736436588a555ba98a13c84/synthetic.db`，不使用正式 DB。
- 未開相機／麥克風、未讀寫正式情緒資料或金鑰、未清理資料、刪檔、安裝套件、git add／commit／push／切換分支。保留原有未追蹤模型與日誌。Pi 尚未實機驗收；部署時前後端需一起更新並重啟後端、強制重新整理網頁，辨識程式不需因本次修改重啟。回放仍沿用舊統計規則，數量可能和新歷史／統計不同，留待下一步處理。
