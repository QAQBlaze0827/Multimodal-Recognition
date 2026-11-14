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
