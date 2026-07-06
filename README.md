# Multimodal Emotion Recognition

多模態情緒辨識原型：即時影像 + 音訊情緒 + confidence-based late fusion。  
目前主線架構依照 `ARCHITECTURE.md`，程式入口是 `app.py`，實作放在 `src/`。

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
    fear/
    surprise/
    disgust/
  val/
    neutral/
    happy/
    sad/
    anger/
    fear/
    surprise/
    disgust/
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
