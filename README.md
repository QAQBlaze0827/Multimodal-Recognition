# Multimodal Emotion Recognition

多模態情緒辨識原型：即時影像 + 音訊情緒 + confidence-based late fusion。  
目前主線架構依照 `ARCHITECTURE.md`，程式入口是 `app.py`，實作放在 `src/`。

要測試「說話後，網頁狗狗顯示文字回覆」，請先看下方的[筆電語音互動網頁啟動流程](#laptop-voice-start)。它與一般的影像／語音情緒辨識是不同啟動流程。

<a id="laptop-voice-start"></a>

## 筆電語音互動網頁：啟動流程

### 目前的使用範圍

這一版在同一台 Windows 筆電執行：

```text
耳機／筆電麥克風 → Python 語音轉文字 → 固定回覆規則
                                      ↓ 只傳文字
                              本機後端 → 網頁狗狗對話框
```

- 本節適用於已備妥環境的開發筆電；不是新電腦 clone 後的完整安裝流程。
- 瀏覽器不負責收音，不需要開啟網頁的麥克風權限。
- 不用先開 `python app.py`，也不用開樹莓派，即可測試文字回覆。
- 沒有接 LLM、語音朗讀或多輪對話記憶；未支援的說法仍會回詢問。
- 目前只支援本機網址，不支援把文字直接送往樹莓派 IP 或 Tailscale 網址。

### 先分清楚兩個環境

| 視窗 | 執行內容 | 使用的 Python | 工作目錄 |
|---|---|---|---|
| PowerShell A | 網頁後端 | 專題內 `.venv\Scripts\python.exe` | 專題根目錄 |
| PowerShell B | 收音、轉錄、固定回覆及傳送文字 | `$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe` | 任意目錄 |
| 瀏覽器 | 顯示狗狗與文字 | 不需 Python | 開啟本機網址 |

以下都明確指定 Python 路徑，**不需要另外 Activate，也不用依賴終端機前面是否顯示 `(.venv)`**。兩個 PowerShell 必須同時保持執行，不要在同一個正在運作的視窗內貼入第二個程式的指令。

外部工具位於這台筆電的 `C:\Users\QAQBl\AppData\Local\emotion-stt-test\`，需要既有的 `.venv`、`models` 模型快取，以及 `transcribe_mic.py`、`dog_reply.py`、`web_reply.py`。

**注意：外部工具不在本 repo 的 Git 管理範圍。** 只 clone 或 pull 專題不會取得這些檔案、虛擬環境或 Whisper 模型；換電腦時需另外準備，不能直接照本節就視為已安裝完成。專題內的 `requirements_backend.txt` 也不是外部轉錄環境的安裝清單。

### 1. 停止先前開啟的程式

如果已有舊轉錄程式，先在它的終端機按 `Ctrl+C`，等待它結束。接著停止你先前開啟的舊後端，尤其是使用 `python backend/app.py` 啟動、沒有語音互動參數的那一份。

不要同時開兩份收音程式。也不要只關掉網頁就以為麥克風已停止，收音由 PowerShell B 的程式控制。

### 2. PowerShell A：進入專題根目錄

專題根目錄就是能看到 `app.py`、`backend`、`frontend`、`README.md` 的資料夾。這台筆電的路徑如下；其他電腦請換成自己的專題位置：

```powershell
Set-Location -LiteralPath "C:\Users\QAQBl\OneDrive\桌面\大學的學習歷程\大三\下學期\專題\Multimodal-Recognition"
Get-Location
Test-Path .\backend\app.py
Test-Path .\.venv\Scripts\python.exe
```

兩個 `Test-Path` 都應顯示 `True`。如果是 `False`，先確認路徑或既有環境，不要在不確定的位置重新建立環境。

### 3. PowerShell A：啟動後端

```powershell
.\.venv\Scripts\python.exe -B backend/app.py --host 127.0.0.1 --port 8000 --db :memory: --companion-local
```

參數說明：

| 參數 | 作用 |
|---|---|
| `--host 127.0.0.1` | 只讓同一台筆電連線；語音互動測試不要改成 `0.0.0.0` |
| `--port 8000` | 後端與網頁使用的連接埠 |
| `--db :memory:` | 使用暫時的記憶體情緒資料庫，不讀寫正式的 `data/emotion.db` |
| `--companion-local` | **啟用本機語音文字接收功能，不能漏掉** |
| `-B` | 不產生 Python bytecode 快取，不影響互動功能 |

啟動成功應看到：

```text
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

保持這個視窗開著。若缺少 `fastapi` 或 `uvicorn`，才在專題根目錄用專題的 Python 安裝後端依賴，再重跑啟動指令：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_backend.txt
```

`--db :memory:` 不會刪除原有資料，但測試網頁不會顯示正式的情緒歷史，也不會看到另一個 `app.py` 寫入正式資料庫的結果。這不是攝影機或情緒模型故障。本流程先驗收語音對話框，暫時不要混用一般情緒辨識流程。

### 4. PowerShell B：啟動收音與轉錄

另開一個 PowerShell，在任意目錄執行：

```powershell
& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" "$env:LOCALAPPDATA\emotion-stt-test\transcribe_mic.py" --web-url http://127.0.0.1:8000
```

`&` 是 PowerShell 執行引號內程式路徑的語法，請一起貼上。`--web-url` 決定文字要送到哪裡；沒有它時只會顯示在終端機，不會傳到網頁。

啟動時會顯示 `Input:` 麥克風名稱，再載入本機快取的 Whisper base 模型。等待出現以下訊息後再說話：

```text
Listening. Speak a short sentence, then pause about 0.8 seconds.
[web] Connected to local dashboard.
```

兩行的出現順序可能不同。確認 `Input:` 是預期的耳機／筆電麥克風；這裡讀的是筆電裝置，不是樹莓派的 INMP441。模型由本機快取載入，不會在啟動時自動下載缺少的模型。

連線成功後，PowerShell A 會定期顯示：

```text
POST /api/companion/events HTTP/1.1" 200 OK
```

沒有說話時仍可能約每 2 秒出現一筆，這是連線狀態心跳，不代表一直辨識出新句子。

### 5. 瀏覽器：開啟互動陪伴頁

開啟 [http://127.0.0.1:8000](http://127.0.0.1:8000)，點選上方「互動陪伴」。先前已開著網頁的話，請重新整理，必要時用 `Ctrl+F5` 重新載入前端檔案。

確認「語音互動」顯示「筆電收音中」，然後說一句話，結束後停頓約 0.8 秒，再等待轉錄與傳送完成。例如：

| 說的話 | 現有規則預期回覆 |
|---|---|
| 我想玩 | 好呀！我們一起玩！ |
| 我好難過可不可以陪我一下 | 我在這裡，陪你待一下。 |
| 我好難過，但我不想要你陪我 | 好，我先不打擾你。 |
| 我好難過 | 你希望我陪你待一下嗎？ |

以終端機實際轉出的文字為準，ASR 錯字可能影響規則；這些是功能檢查例句，不是準確率測試集。

網頁應顯示「你的話（轉錄）」與狗狗回覆，內容和 PowerShell B 一致。只顯示最近一則，不是聊天歷史；轉錄文字與回覆不寫入 SQLite。切換英文介面只改標籤，不會翻譯原文或固定中文回覆。

底下「表情互動：等待辨識資料」在本流程中是正常的，因為沒有開臉部辨識。文字回覆不會觸發新的狗狗動作；摸摸歪頭仍可用。「先安靜」的回覆也只是文字，**不會停止收音**。

### 6. 測試結束：先停收音，再停後端

1. 在 PowerShell B 按 `Ctrl+C`，等待目前轉錄收尾，看到 `Stopped` 再關閉視窗。
2. 後端仍開著時，網頁應顯示「筆電收音已停止」並清空對話；若程式異常中斷，約 8 秒未收到心跳後會顯示逾時。
3. 在 PowerShell A 按 `Ctrl+C` 關閉後端，最後關閉網頁。

來源持續連線但沒有新回覆時，最近一則對話約 60 秒後會清空。下次使用重新執行步驟 3、4，再開網頁即可；不要假設先前的預覽後端在重開機後還會運作。

## 賴彥光懶人包
11/14 筆電端
1. PowerShell A：在專題根目錄開後端
``` text 
  .\.venv\Scripts\python.exe -B backend/app.py --host 127.0.0.1 --port 8000 --db :memory: --companion-local
```
2. PowerShell B：另開視窗，啟動轉錄
``` powershell
& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" "$env:LOCALAPPDATA\emotion-stt-test\transcribe_mic.py" --web-url http://127.0.0.1:8000
```
3. 開網頁
``` text
  開啟 http://127.0.0.1:8000，重新整理並點「互動陪伴」，再開始說話。
```


### 常見問題

| 現象 | 檢查與處理 |
|---|---|
| 後端一直出現 `POST /api/companion/events ... 404 Not Found` | 最常見是漏掉 `--companion-local`。停止後端，使用步驟 3 的完整指令重開，再重新整理網頁。若仍是 404，檢查是否開到舊版專題或不同 port；不是調整語音準確度能解決的問題。 |
| 只看到 `python backend/app.py` 可以開網頁，但不能收文字 | 一般情緒儀表板不會自動啟用語音互動，必須使用本節的參數。 |
| 網頁顯示「語音網頁連線未啟用」 | 後端先加 `--companion-local` 重啟，再重新整理頁面。 |
| 終端機有 `Dog [...]`，網頁沒有回覆 | 確認轉錄指令帶 `--web-url`，並看到 `[web] Connected to local dashboard.`；後端與瀏覽器網址也要對應同一個 port。失敗的舊回覆不會補送，恢復連線後重新說一句。 |
| `WinError 10048`／address already in use | 該 port 已被使用。只停止你確認是自己開的舊後端；不確定時可改用另一個空閒 port，例如 8001，並同步修改後端 `--port`、轉錄 `--web-url` 與瀏覽器網址。 |
| `409 Conflict` | 已有另一份有效的轉錄來源。停止多餘的轉錄程式；若它是異常退出，等約 8 秒後再確認連線。 |
| 網頁顯示「轉錄程式連線逾時」 | 檢查 PowerShell B 是否停止或有錯誤。後端連線正常，不代表轉錄程式仍正常。 |
| 找不到外部 Python／`transcribe_mic.py`／模型快取 | 外部工具不隨 repo clone，確認這台電腦的 `emotion-stt-test` 已備妥；不要把套件誤裝到專題 `.venv`。 |
| 麥克風不是預期裝置 | 先停舊收音程式，用下方指令列出裝置，再於原啟動指令加 `--device N`；N 要用當次列出的輸入裝置編號。 |
| 只有「表情互動：等待辨識資料」 | 這是純語音測試下的預期狀態，先看「語音互動」那一列，不要因此再開一份收音程式。 |

只列出裝置、不開收音或模型：

```powershell
& "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe" "$env:LOCALAPPDATA\emotion-stt-test\transcribe_mic.py" --list-devices
```

若要確認 8000 上是否有後端，可在另一個空閒 PowerShell 執行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

本測試模式預期 `status` 是 `ok`、`db_path` 是 `:memory:`。這只能確認後端與資料庫選項，**不能單靠 health 成功就判斷語音互動已啟用**；仍需看文字 POST 的 `200 OK` 與網頁回覆。

## Quick Start

### 1. Clone 或取得專案

進入專案資料夾後，先確認電腦有 Python 3.11：

```powershell
py -0p
```

### 2. 建立環境

Windows 可以直接跑：

```powershell
.\scripts\setup_windows.ps1
```

如果 PowerShell 擋腳本，可以改手動執行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 測試攝影機

```powershell
python app.py --vision-only
```

或：

```powershell
.\scripts\run_vision.ps1
```

無視窗 smoke test，讀 10 幀後停止：

```powershell
python app.py --vision-only --no-display --max-frames 10
```

或：

```powershell
.\scripts\run_smoke_test.ps1
```

## How To Check It Is Working

畫面左下角會顯示：

```text
Face: YES [haar] bbox=...
Video: neutral 0.84 [heuristic]
Fused: neutral 0.84
```

判斷方式：

- `Face: YES`：有偵測到臉。
- 綠色框框：偵測到的人臉位置。
- `bbox=x,y,w,h`：人臉 bounding box。
- `[haar]`：目前使用 OpenCV Haar detector。
- `[heuristic]`：目前沒有正式 ONNX 情緒模型，正在使用 fallback。
- `neutral 0.84`：fallback 的預設分數，不代表正式模型真的判斷為 neutral。

每次執行也會寫 CSV log：

```text
data/logs/session_*.csv
```

可檢查欄位：

```csv
face_detected,face_backend,face_bbox,video_emotion,video_conf
```

## Web Dashboard（新功能）

提供暗色系網頁儀表板，便於即時監控與歷史分析。

### 啟動後端伺服器

以下是一般情緒儀表板的啟動方式，不啟用狗狗語音文字接收。測試狗狗對話框請使用上方[筆電語音互動流程](#laptop-voice-start)，不要同時在相同 port 開兩份後端。

```bash
pip install -r requirements_backend.txt
python backend/app.py --host 0.0.0.0 --port 8000
```

### 開啟瀏覽器

```
http://localhost:8000
```

### 網頁功能

| 頁面 | 功能 |
|------|------|
| **Live** | 即時 WebSocket 顯示 Fused/Video/Audio 三模態情緒 + 長條圖 |
| **Replay** | 選擇 session → 回放情緒變化（可調速度、時間軸拖曳） |
| **History** | 歷史紀錄分頁瀏覽 + 篩選（情緒、session） |
| **Analytics** | 情緒分布圓餅圖 + 統計摘要（7/30/90/365 天區間） |

### 搭配推論執行（記錄到資料庫）

在 `config/config.yaml` 中設定：

```yaml
backend:
  enabled: true          # 寫入 SQLite
  db_path: "data/emotion.db"
  log_interval: 1.0      # 每秒記錄一筆
  retention_days: 30     # 自動清理 30 天前資料
```

然後執行 `python app.py` 即會自動將推論結果寫入 SQLite，後端可即時查詢與推送。

## Project Structure

```text
config/config.yaml              # Runtime config (含 backend 段落)
src/main.py                     # Main loop
src/video/                      # Camera, face detection, video emotion
src/audio/                      # Audio feature and audio emotion
src/fusion/                     # Late fusion
src/output/                     # OpenCV overlay, CSV logger, DB logger
backend/                        # FastAPI server (REST + WebSocket)
  ├── app.py                    #   API routes + static file serving
  └── database.py               #   SQLite CRUD (WAL mode)
frontend/                       # Web dashboard SPA
  ├── index.html                #   Dark-theme dashboard
  ├── css/style.css             #   RWD styles
  └── js/                       #   app.js, api.js, websocket.js, charts.js
models/                         # ONNX models
data/logs/                      # Runtime CSV logs
scripts/                        # Setup, run, training scripts
```

## Models

Expected model paths:

```text
models/mini_xception_int8.onnx
models/tiny_cnn_audio_int8.onnx
```

If the model file exists, the app uses ONNX Runtime.  
If the model file does not exist, the app uses a lightweight fallback so the pipeline can still run.

Recommended order:

1. Finish video model first.
2. Verify `python app.py --vision-only`.
3. Add audio model later.

## Train Your Own Video Model

See [TRAINING.md](TRAINING.md).

Short version:

```powershell
pip install -r requirements_train.txt
python scripts/train_video_mini_xception.py --data-dir data/datasets/fer --epochs 30
```

Expected dataset layout:

```text
data/datasets/fer/
  train/
    neutral/
    happy/
    sad/
    anger/
  val/
    neutral/
    happy/
    sad/
    anger/
```

## Docker

Docker is useful for reproducing Python 3.11.2 and dependencies.  
On Windows, webcam and microphone access through Docker Desktop can be awkward, so local `.venv` is recommended for live testing.

```powershell
docker build -t multimodal-emotion:py311 .
docker run --rm multimodal-emotion:py311
```

If your Docker supports compose:

```powershell
docker-compose build
docker-compose run --rm multimodal-emotion
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | 伺服器狀態 |
| `POST` | `/api/emotions` | 寫入一筆情緒紀錄 |
| `GET` | `/api/emotions` | 查詢紀錄（支援 session_id、emotion、limit、offset） |
| `GET` | `/api/emotions/recent?n=100` | 最近 N 筆 |
| `GET` | `/api/sessions` | 列出所有 session |
| `GET` | `/api/sessions/{id}` | 單一 session 統計 |
| `GET` | `/api/sessions/{id}/logs` | session 完整紀錄（回放用） |
| `GET` | `/api/sessions/{id}/timeline` | session 時間序列 |
| `GET` | `/api/analytics?days=30` | 分析摘要（情緒分布） |
| `POST` | `/api/cleanup?days=30` | 手動清理舊資料 |
| `WS` | `/ws` | WebSocket 即時推播 |

## Database

使用 SQLite（WAL mode），無需額外資料庫伺服器。  
資料表：

- **sessions**：記錄每次執行的 session（start_time、end_time、total_frames）
- **emotion_logs**：每秒一筆的情緒紀錄（含 face/video/audio/fused 各模態結果）
- **emotion_summary**：session 層級的快取摘要

自動保留最近 30 天資料（可在 `config.yaml` 調整）。

## Known Notes

- MediaPipe can fail to load `.binarypb` resources under Chinese paths on Windows. The app automatically falls back to OpenCV Haar detection.
- Current emotion output is not meaningful until a real ONNX model is placed in `models/`.
- Logs under `data/logs/*.csv` are ignored by git.
- Emotion history database (`data/*.db`) is also ignored by git.
